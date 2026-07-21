"""
populate_config.py
==================
Interactive script to populate the R2 entries of a CSAR/FastCSAR TOML
configuration file from experimental NMR data.

Workflow
--------
(A) Select workflow type interactively.
(B) Parse the existing config — reads the compound list and the workflow
    parameters (field strengths, protein concentrations, …).
(C) For each compound, ask the user to provide the path(s) to the relevant
    experiment file(s).  Compounds with no data provided are removed from
    the final config.
(D) Load and fit the experimental data → R2 values.
(E) Write the fitted R2 values back into the ``[compound.NAME.R2]`` blocks.

Usage
-----
    python populate_config.py experiment.toml

The script modifies the file in-place.  A timestamped backup is created
automatically before any changes are written.
"""

from __future__ import annotations

from importlib.resources import files
import os
from pathlib import Path
import re
import shutil
import sys
import tomllib
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import lmfit

import numpy as np
try:
    import klassez as kz
    from f_fit import fit_exponential, fit_exponential_Jmod, extract_R2_uncertainty, plot_exponential_fit
except ImportError:
    kz = None   # klassez / f_fit not installed; fitting functions unavailable

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

WORKFLOWS = ("csar", "fastcsar", "reporter", "titration")


@dataclass
class ExperimentPaths:
    """
    All file paths needed to fit R2 for one compound under one workflow.

    Attributes
    ----------
    compound_name : str
    workflow : str
    fields : dict
        Keyed by magnetic field in Tesla (float).
        Values are dicts with string keys describing the sample condition
        (e.g. ``"free"``, ``"protein"``) mapped to file path strings.

        Structure per workflow:

        ``csar``
            ``{B_high: {"free": path, "protein": path},
               B_low:  {"free": path, "protein": path}}``

        ``fastcsar``
            ``{B: {"free": path, "protein": path}}``

        ``reporter``
            ``{0.0: {"reporter_with_compound": path}}``
            (field key is 0.0 as a sentinel — not used physically)

        ``titration``
            ``{B: {"P_{i}uM": path, …}}``
            one key per protein concentration ``i``
    """
    compound_name: str
    workflow: str
    fields: Dict[float, Dict[str, str]] = field(default_factory=dict)


@dataclass
class FittedR2:
    """
    Fitted results for one compound ready to be written into the config.

    Attributes
    ----------
    compound_name : str
    r2 : dict
        Keyed by magnetic field (float, Tesla). Despite the name, this
        holds whatever :func:`fit_experiment` returned as ``payload`` —
        for ``fastcsar`` in its default ``"raw"`` mode that's the raw
        ``[delays, I_free, I_protein]`` arrays (no R2 is fitted there),
        not an R2 value; in ``"r2"`` mode it's an R2 pair, same shape as
        ``csar``. Which shape it is depends on ``workflow`` plus, for
        ``fastcsar``, the ``fastcsar_mode`` returned alongside ``fitted``
        by :func:`run_fitting` — the caller (``update_config``) needs
        that to pick the right writer.

        ``csar``
            ``{B: [R2_free, R2_protein]}``  — two-element list (s⁻¹)

        ``fastcsar``, mode ``"raw"`` (default)
            ``{B: [delays, I_free, I_protein]}`` — raw relaxation data
            for the analytical point-ratio formula (Eq. 11, Rüdisser
            2020 — see ``workflow_FastCSAR`` downstream) to consume
            directly; see :func:`_write_relaxation_data_block` for
            exactly how this is serialized into the config.

        ``fastcsar``, mode ``"r2"``
            ``{B: [R2_free, R2_protein]}``  — two-element list (s⁻¹),
            same shape as ``csar``; written into ``[compound.NAME.R2]``
            via :func:`_write_r2_block` so ``workflow_FastCSAR`` can
            take the fitted-R2 path instead.

        ``reporter``
            ``{0.0: [reporter_signal_plus]}``
            Stored as ``reporter_signal_plus`` in the compound block,
            not in the R2 sub-table (handled specially by the writer).

        ``titration``
            ``{B: [R2_P0, R2_P1, …]}``  — one value per protein concentration

    r2_err : dict, optional
        Same structure as ``r2`` but carries 1σ uncertainties.
        ``csar``              →  ``{B: [σ_R2_free, σ_R2_protein]}``
        ``titration``         →  ``{B: [σ_R2_P0, σ_R2_P1, …]}``
        ``fastcsar`` "r2"     →  ``{B: [σ_R2_free, σ_R2_protein]}``
        ``fastcsar`` "raw"    →  empty (nothing to report at this stage)
        ``None`` or empty means no uncertainty was estimated.
    """
    compound_name: str
    r2: Dict[float, list] = field(default_factory=dict)
    r2_err: Dict[float, list] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# (A) Workflow selection
# ---------------------------------------------------------------------------

def select_workflow(config_path: str) -> str:
    """
    (A) Ask the user which workflow to run.

    If the config already has a ``[workflow]`` section, its ``name`` is
    offered as the default.

    Parameters
    ----------
    config_path : str
        Path to the TOML config (read for the existing workflow name).

    Returns
    -------
    str
        One of ``"csar"``, ``"fastcsar"``, ``"reporter"``, ``"titration"``.
    """
    default_wf = _read_workflow_name(config_path)

    print("\n── (A) Workflow selection ──────────────────────────────────────")
    for i, wf in enumerate(WORKFLOWS, 1):
        marker = "  ← current" if wf == default_wf else ""
        print(f"  [{i}] {wf}{marker}")
    print("────────────────────────────────────────────────────────────────")

    while True:
        raw = input(
            f"Select workflow [1-{len(WORKFLOWS)}]"
            + (f"  [{WORKFLOWS.index(default_wf) + 1}]" if default_wf else "")
            + ": "
        ).strip()
        if raw == "" and default_wf:
            return default_wf
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(WORKFLOWS):
                chosen = WORKFLOWS[idx]
                print(f"  → {chosen}\n")
                return chosen
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(WORKFLOWS)}.")


# ---------------------------------------------------------------------------
# (B) Parse existing config
# ---------------------------------------------------------------------------

def parse_config(config_path: str) -> Tuple[dict, List[str]]:
    """
    (B) Load the TOML config and return the full dict and the list of
    compound names.

    Parameters
    ----------
    config_path : str

    Returns
    -------
    cfg : dict
        Full parsed config.
    compound_names : list of str
        Names in the order they appear in the ``[compound]`` table.
    """
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)

    compound_names = list(cfg.get("compound", {}).keys())
    if not compound_names:
        sys.exit(f"No compounds found in {config_path}.")

    print(f"── (B) Found {len(compound_names)} compound(s) ─────────────────────────────")
    for name in compound_names:
        ftype = cfg["compound"][name].get("fluorine_type", "?")
        ds    = cfg["compound"][name].get("delta_sigma", float("nan"))
        print(f"  {name:<25}  [{ftype}]  δσ = {ds:.1f} ppm")
    print()

    return cfg, compound_names


# ---------------------------------------------------------------------------
# (C) Collect experiment paths
# ---------------------------------------------------------------------------

def collect_paths(
    compound_names: List[str],
    workflow: str,
    cfg: dict,
    config_path: str,
) -> Tuple[List[ExperimentPaths], List[str]]:
    """
    (C) Interactively ask the user for the path to each experiment file.

    Any ``[compound.NAME.experiment_paths]`` already present in the
    config (e.g. saved during a previous, interrupted run) is read back
    and offered as the default for each prompt — pressing Enter reuses
    it, so restarting the script does not silently wipe out compounds
    that were already answered. Typing '-' explicitly clears a path.

    Compounds for which no path is provided/kept at all are collected in
    ``skipped`` and will be removed from the config.

    Parameters
    ----------
    compound_names : list of str
    workflow : str
    cfg : dict
        Full parsed config (used to read field strengths / protein
        concentrations from the ``[workflow]`` section).
    config_path : str
        Path to the TOML config file. Used so that field strengths and
        other workflow parameters discovered/collected interactively —
        ``B_high``/``B_low`` for ``csar``, ``B`` for ``fastcsar``, the
        ``[reporter]`` reference measurement for ``reporter``, and
        protein concentrations for ``titration`` — can be persisted
        back into the config on disk as soon as they're known.

    Returns
    -------
    experiments : list of ExperimentPaths
        One entry per compound for which at least one path was provided.
    skipped : list of str
        Compound names excluded by the user (no paths given).
    """
    wf_cfg = cfg.get("workflow", {})
    experiments: List[ExperimentPaths] = []
    skipped: List[str] = []

    print(f"── (C) Experiment paths  [{workflow}] ──────────────────────────────")
    print("   Leave a path blank on a fresh entry to skip it. If a path was")
    print("   already saved in a previous run, it's shown as [default]:")
    print("   press Enter to keep it, type a new path to replace it, or")
    print("   '-' to clear it.\n")

    # Snapshot the file once before we start writing to it incrementally.
    if compound_names:
        _backup(Path(config_path))

    # For titration, resolve the protein-concentration list once, up front,
    # rather than per compound. If it's missing from the config, ask the
    # user interactively and persist it to the config file instead of
    # exiting.
    prot_concs: Optional[List[float]] = None
    if workflow == "titration":
        ligand_conc = wf_cfg.get("ligand_concentration", None)
        if ligand_conc is None:
            print("  'ligand_concentration' not found in the [workflow] section.")
            conc_input = input(
                "  Enter ligand concentration (µM): "
            )
            try:
                ligand_conc = float(conc_input.strip())
            except ValueError:
                sys.exit(
                    "No valid ligand concentration provided — cannot continue "
                    "the titration workflow."
                )
            wf_cfg["ligand_concentration"] = ligand_conc
            _write_ligand_concentration(config_path, ligand_conc)
            print()
        prot_concs = [float(p) for p in wf_cfg.get("protein_concentrations", [])]
        if not prot_concs:
            print("  'protein_concentrations' not found in the [workflow] section.")
            conc_input = input(
                "  Enter protein concentrations (µM), comma-separated: "
            )
            prot_concs = [
                float(p.strip()) for p in conc_input.split(",") if p.strip()
            ]
            if not prot_concs:
                sys.exit(
                    "No protein concentrations provided — cannot continue "
                    "the titration workflow."
                )
            wf_cfg["protein_concentrations"] = prot_concs
            _write_protein_concentrations(config_path, prot_concs)
            print()

    # For reporter, resolve the reference reporter-compound measurement
    # (name + free/protein signal) once, up front, rather than per
    # compound — it's a single global reference shared by every ligand,
    # not per-compound data. If it's missing from the config, ask the
    # user interactively and persist it to the config file instead of
    # exiting. csar_workflows.py (run_from_config / workflow_reporter)
    # reads this as its own top-level ``[reporter]`` section, distinct
    # from ``[workflow]``.
    if workflow == "reporter":
        rep_cfg = cfg.get("reporter", {})

        reporter_name = rep_cfg.get("name", None)
        if not reporter_name:
            print("  'name' not found in the [reporter] section.")
            reporter_name = input("  Enter reporter compound name: ").strip()
            if not reporter_name:
                sys.exit(
                    "No reporter compound name provided — cannot continue "
                    "the reporter workflow."
                )
            rep_cfg["name"] = reporter_name

        signal_free = rep_cfg.get("signal_free", None)
        if signal_free is None:
            print("  'signal_free' not found in the [reporter] section.")
            sf_input = input("  Enter reporter signal, free (no protein): ")
            try:
                signal_free = float(sf_input.strip())
            except ValueError:
                sys.exit(
                    "No valid free-reporter signal provided — cannot "
                    "continue the reporter workflow."
                )
            rep_cfg["signal_free"] = signal_free

        signal_protein = rep_cfg.get("signal_protein", None)
        if signal_protein is None:
            print("  'signal_protein' not found in the [reporter] section.")
            sp_input = input(
                "  Enter reporter signal, + protein (no compound): "
            )
            try:
                signal_protein = float(sp_input.strip())
            except ValueError:
                sys.exit(
                    "No valid protein-reporter signal provided — cannot "
                    "continue the reporter workflow."
                )
            rep_cfg["signal_protein"] = signal_protein

        cfg["reporter"] = rep_cfg
        _write_reporter_section(
            config_path, reporter_name, float(signal_free), float(signal_protein)
        )
        print()

    for name in compound_names:
        print(f"  ▸ {name}")
        ep = ExperimentPaths(compound_name=name, workflow=workflow)
        existing = _load_existing_experiment_paths(cfg, name)
        if existing:
            print(f"    (found {len(existing)} previously-saved field "
                  f"entry/entries for '{name}' — Enter reuses a path, "
                  f"'-' clears it)")

        if workflow == "csar":
            e1 = existing[0] if len(existing) > 0 else None
            e2 = existing[1] if len(existing) > 1 else None
            B1, paths1 = _ask_free_protein_paths(name, workflow, label="Field point 1", existing=e1)
            B2, paths2 = _ask_free_protein_paths(name, workflow, label="Field point 2", existing=e2)
            if paths1["free"] or paths1["protein"]:
                ep.fields[B1] = paths1
            if paths2["free"] or paths2["protein"]:
                ep.fields[B2] = paths2
            # Persist the two field strengths to [workflow] as soon as
            # both are known — csar_workflows.py (workflow_CSAR) reads
            # B_high/B_low from there to run, and the field order the
            # user entered them in needn't match high/low, so sort them.
            if B1 and B2:
                B_high, B_low = (B1, B2) if B1 >= B2 else (B2, B1)
                _write_field(config_path, B_high, key="B_high")
                _write_field(config_path, B_low, key="B_low")

        elif workflow == "fastcsar":
            e0 = existing[0] if existing else None
            B, paths = _ask_free_protein_paths(name, workflow, existing=e0)
            if paths["free"] or paths["protein"]:
                ep.fields[B] = paths
            # Persist the field strength to [workflow] as soon as it's
            # known — csar_workflows.py (workflow_FastCSAR) reads B
            # from there to run.
            if B:
                _write_field(config_path, B)

        elif workflow == "reporter":
            existing_B, existing_paths = existing[0] if existing else (None, {})
            default_path = existing_paths.get("reporter_with_compound", "")
            path = _ask_path(f"    Reporter spectrum with {name}: ", default=default_path)
            if path == default_path and existing_B is not None:
                B = existing_B
            elif path:
                B = _read_field_from_experiment(path, workflow)
                print(f"      → B0 = {B} T (read from experiment)")
            else:
                B = 0.0
            ep.fields[B] = {"reporter_with_compound": path}

        elif workflow == "titration":
            e0 = existing[0] if existing else None
            B, paths = _ask_titration_paths(name, prot_concs, workflow, existing=e0)
            ep.fields[B] = paths
            #write the field strength to the config file for titration workflow
            _write_field(config_path, B)
            #ask the user whether to fit the KD value and write it to the config file
            fit_KD_input = input("  Fit KD value? (y/n) [n]: ").strip().lower()
            DoFit = fit_KD_input == "y"
            _write_fit_KD(config_path, DoFit)

        # Check whether the user provided at least one non-empty path
        any_provided = any(
            path
            for condition_paths in ep.fields.values()
            for path in condition_paths.values()
        )
        if any_provided:
            experiments.append(ep)
            write_experiment_paths(config_path, ep)
        else:
            skipped.append(name)
            print(f"    ⚠  No paths provided — {name} will be removed.\n")

    print()
    return experiments, skipped


# -- helpers for (C) ---------------------------------------------------------

def _toml_str(s: str) -> str:
    """
    Render ``s`` as a TOML string literal, safe for Windows paths.

    Prefers a TOML *literal* string (single-quoted, e.g. ``'C:\\Users\\x'``)
    because literal strings do not process escape sequences at all — a
    backslash is just a backslash, so Windows paths need no
    transformation whatsoever. Falls back to an escaped basic
    (double-quoted) string only if ``s`` contains a character a literal
    string can't hold (a single quote or a newline).
    """
    if "'" not in s and "\n" not in s and "\r" not in s:
        return f"'{s}'"
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _experiment_paths_content(fields: Dict[float, Dict[str, str]]) -> str:
    """
    Render the body of a ``[compound.NAME.experiment_paths]`` block: one
    inline table per field, keyed by field strength as a quoted string
    (mirroring the ``[compound.NAME.R2]`` convention), mapping condition
    name -> file path.

    Condition keys (``free``, ``protein``, ``P_0.0uM``, ...) are always
    quoted. Some of them (e.g. the titration ``P_<conc>uM`` keys) contain
    a literal ``.``; an *unquoted* TOML key with a dot is parsed as a
    dotted/nested key (``a.b = x`` → ``{"a": {"b": x}}``), silently
    turning the condition dict into nested subtables instead of the flat
    mapping we want. Quoting avoids that entirely.
    """
    lines = []
    for B in sorted(fields.keys()):
        conditions = fields[B]
        inline = ", ".join(
            f'"{k}" = {_toml_str(v)}' for k, v in conditions.items()
        )
        lines.append(f'"{B:.4g}" = {{ {inline} }}')
    return "\n".join(lines)


def _compound_section_end(text: str, compound_name: str) -> int:
    """
    Return the text index marking the end of ``[compound.NAME]`` and all
    of its subtables (e.g. ``.R2``, ``.experiment_paths``) — i.e. right
    before the next section header that does *not* belong to this
    compound.

    New subtables must be inserted here (not right after the opening
    ``[compound.NAME]`` header), otherwise any scalar keys such as
    ``delta_sigma``/``eta`` that follow the header would end up
    misparsed as belonging to the newly-inserted subtable instead of
    ``[compound.NAME]``.
    """
    compound_header = f"[compound.{compound_name}]"
    prefix = f"[compound.{compound_name}."
    start  = text.index(compound_header)
    pos    = start + len(compound_header)

    for m in re.finditer(r"\n\[([^\]]+)\]", text[pos:]):
        header_name = "[" + m.group(1) + "]"
        if header_name == compound_header or header_name.startswith(prefix):
            continue
        return pos + m.start()
    return len(text)


def _write_experiment_paths_block(
    text: str,
    compound_name: str,
    fields: Dict[float, Dict[str, str]],
) -> str:
    """
    Insert or replace ``[compound.NAME.experiment_paths]`` in the raw TOML
    text with the paths currently held in ``fields``.
    """
    header  = f"[compound.{compound_name}.experiment_paths]"
    content = _experiment_paths_content(fields)

    if header not in text:
        compound_header = f"[compound.{compound_name}]"
        if compound_header not in text:
            raise ValueError(
                f"Cannot find '{compound_header}' in the config to append "
                "experiment_paths block."
            )
        insert_at = _compound_section_end(text, compound_name)
        block = f"\n{header}\n{content}\n"
        text = text[:insert_at] + block + text[insert_at:]
    else:
        parts  = re.split(r"(\n\[[^\n]+\])", text)
        result = [parts[0]]
        i = 1
        while i < len(parts):
            sec_header = parts[i]
            sec_body   = parts[i + 1] if i + 1 < len(parts) else ""
            stripped   = sec_header.lstrip("\n")
            if stripped == header:
                result.append(sec_header)
                result.append("\n" + content + "\n")
            else:
                result.append(sec_header)
                result.append(sec_body)
            i += 2
        text = "".join(result)

    return text


def write_experiment_paths(config_path: str, ep: ExperimentPaths) -> None:
    """
    Persist one compound's experiment file paths into
    ``[compound.NAME.experiment_paths]`` in the TOML config, immediately
    after the user enters them in (C).

    Writing this as soon as each compound's paths are collected — rather
    than waiting for fitting/(E) to finish — means the config reflects
    what was entered even if the run is interrupted (e.g. during the
    load/fit step) before ``update_config`` runs.

    Parameters
    ----------
    config_path : str
    ep : ExperimentPaths
        Paths for a single compound, as built in ``collect_paths``.
    """
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    text = _write_experiment_paths_block(text, ep.compound_name, ep.fields)
    path.write_text(text, encoding="utf-8")
    print(f"    [config] Saved experiment paths for '{ep.compound_name}'")


def _write_toml_scalar(
    config_path: str, header: str, key: str, value_str: str
) -> None:
    """
    Persist a single ``key = value_str`` pair into the ``header`` section
    (e.g. ``"[workflow]"`` or ``"[reporter]"``) of the TOML config file
    on disk.

    This is the generic engine behind all the small ``_write_*``
    workflow-parameter helpers below — csar (``B_high``/``B_low``),
    fastcsar/titration (``B``), titration (``fit_KD``,
    ``ligand_concentration``, ``protein_concentrations``), and reporter
    (``name``, ``signal_free``, ``signal_protein``) all reduce to
    "replace or append this one key in this one section".

    If ``key`` already exists in the section it is replaced; otherwise
    it is appended to the section. If ``header`` itself doesn't exist
    yet in the file, it is created at the top.

    Parameters
    ----------
    config_path : str
    header : str
        TOML section header, e.g. ``"[workflow]"`` or ``"[reporter]"``.
    key : str
        Key name within that section.
    value_str : str
        Pre-rendered TOML value (caller is responsible for formatting/
        quoting, e.g. via ``_toml_str`` for strings).
    """
    path = Path(config_path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    if header not in text:
        new_section = f"{header}\n{key} = {value_str}\n\n"
        text = new_section + text
        path.write_text(text, encoding="utf-8")
        print(f"  [config] Created {header} section with {key} in {config_path}")
        return

    start = text.index(header)
    body_start = start + len(header)
    next_header = re.search(r"\n\[", text[body_start:])
    section_end = body_start + next_header.start() if next_header else len(text)
    section = text[start:section_end]

    kv_pattern = re.compile(
        r"(^" + re.escape(key) + r"\s*=\s*)[^\n]*", re.MULTILINE
    )
    if kv_pattern.search(section):
        section = kv_pattern.sub(lambda m: m.group(1) + value_str, section)
    else:
        section = section.rstrip("\n") + f"\n{key} = {value_str}\n"

    text = text[:start] + section + text[section_end:]
    path.write_text(text, encoding="utf-8")
    print(f"  [config] Saved {key} to {config_path}")


def _write_fit_KD(
    config_path: str, DoFit: bool
) -> None:
    """
    Persist the option to fit the KD value into the ``[workflow]``
    section of the TOML config file on disk for the titration workflow.
    See :func:`_write_toml_scalar` for the replace-or-append semantics.

    Parameters
    ----------
    config_path : str
    DoFit : bool
        Whether to fit the KD value (True) or not (False).
    """
    _write_toml_scalar(config_path, "[workflow]", "fit_KD", f"{DoFit}")


def _write_field(
    config_path: str, B: float, key: str = "B"
) -> None:
    """
    Persist a magnetic field strength into the ``[workflow]`` section of
    the TOML config file on disk.

    ``key`` defaults to ``"B"`` (single-field workflows: fastcsar,
    titration). csar needs two field points instead, so it calls this
    with ``key="B_high"`` and again with ``key="B_low"``. See
    :func:`_write_toml_scalar` for the replace-or-append semantics.

    Parameters
    ----------
    config_path : str
    B : float
        Magnetic field strength in Tesla.
    key : str, default "B"
        Which ``[workflow]`` key to write — ``"B"``, ``"B_high"``, or
        ``"B_low"``.
    """
    _write_toml_scalar(config_path, "[workflow]", key, f"{B:.2f}")



def _write_ligand_concentration(
    config_path: str, ligand_concentration: float
) -> None:
    """
    Persist the ligand concentration into the ``[workflow]`` section of
    the TOML config file on disk. See :func:`_write_toml_scalar` for the
    replace-or-append semantics.

    Parameters
    ----------
    config_path : str
    ligand_concentration : float
        Concentration in µM.
    """
    _write_toml_scalar(
        config_path, "[workflow]", "ligand_concentration",
        f"{ligand_concentration:.6g}",
    )


def _write_protein_concentrations(
    config_path: str, protein_concentrations: List[float]
) -> None:
    """
    Persist newly-collected ``protein_concentrations`` into the
    ``[workflow]`` section of the TOML config file on disk. See
    :func:`_write_toml_scalar` for the replace-or-append semantics.

    Parameters
    ----------
    config_path : str
    protein_concentrations : list of float
        Concentrations in µM, in the order the user entered them.
    """
    conc_str = "[" + ", ".join(f"{c:.6g}" for c in protein_concentrations) + "]"
    _write_toml_scalar(
        config_path, "[workflow]", "protein_concentrations", conc_str
    )


def _write_reporter_section(
    config_path: str, name: str, signal_free: float, signal_protein: float
) -> None:
    """
    Persist the reporter-compound reference measurement into the
    ``[reporter]`` section of the TOML config file on disk.

    Unlike the per-field/per-concentration parameters above, this is
    its own top-level section (not nested under ``[workflow]``) —
    ``run_from_config`` in ``csar_workflows.py`` reads it as
    ``cfg["reporter"]`` and requires all three keys (``name``,
    ``signal_free``, ``signal_protein``) to run the reporter workflow.
    See :func:`_write_toml_scalar` for the replace-or-append semantics;
    each key is written independently so a partial ``[reporter]``
    section from an earlier run is topped up rather than clobbered.

    Parameters
    ----------
    config_path : str
    name : str
        Name of the reference reporter compound.
    signal_free : float
        Reporter signal intensity with no protein present (a.u.).
    signal_protein : float
        Reporter signal intensity with protein present, no ligand
        compound (a.u.).
    """
    _write_toml_scalar(config_path, "[reporter]", "name", _toml_str(name))
    _write_toml_scalar(
        config_path, "[reporter]", "signal_free", f"{signal_free:.6g}"
    )
    _write_toml_scalar(
        config_path, "[reporter]", "signal_protein", f"{signal_protein:.6g}"
    )



def _ask_path(prompt: str, default: str = "") -> str:
    """
    Return the path string entered by the user.

    If ``default`` is non-empty (i.e. a path already exists in the config
    from a previous run), it is shown in the prompt and pressing Enter
    with no input reuses it — it is NOT treated as "skip". To
    deliberately clear a previously-saved path, type ``-``.
    """
    if default:
        raw = input(f"{prompt}[{default}] ").strip()
        if raw == "":
            return default
        if raw == "-":
            return ""
        if not Path(raw).exists():
            print(f"    ⚠  File not found: {raw}")
        return raw

    raw = input(prompt).strip()
    if raw and not Path(raw).exists():
        print(f"    ⚠  File not found: {raw}")
    return raw


def _flatten_conditions(conditions: dict, prefix: str = "") -> Dict[str, str]:
    """
    Flatten a condition dict, rejoining any dotted-key artifacts back
    into a single string key.

    Config files written before the quoting fix in
    ``_experiment_paths_content`` may have a key like ``P_0.0uM`` parsed
    by ``tomllib`` as a nested table (``{"P_0": {"0uM": path}}``) instead
    of the intended flat ``{"P_0.0uM": path}``, because an unquoted TOML
    key containing a ``.`` is a dotted key. This reconstructs the
    original flat key so those older files still yield usable defaults.
    """
    flat: Dict[str, str] = {}
    for k, v in conditions.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_conditions(v, full_key))
        else:
            flat[full_key] = v
    return flat


def _load_existing_experiment_paths(
    cfg: dict, compound_name: str
) -> List[Tuple[float, Dict[str, str]]]:
    """
    Read back any ``[compound.NAME.experiment_paths]`` already present in
    the parsed config (e.g. from a previous, interrupted run), so that
    re-running the script offers those paths as defaults instead of
    silently discarding the compound when the user just presses Enter.

    Returns a list of ``(B, {condition: path, ...})`` sorted by B
    ascending — matching the order they were originally written in.
    """
    comp      = cfg.get("compound", {}).get(compound_name, {})
    exp_paths = comp.get("experiment_paths", {})
    entries: List[Tuple[float, Dict[str, str]]] = []
    for key, conditions in exp_paths.items():
        try:
            B = float(key)
        except (TypeError, ValueError):
            continue
        if isinstance(conditions, dict):
            entries.append((B, _flatten_conditions(conditions)))
    entries.sort(key=lambda item: item[0])
    return entries


def _read_field_from_experiment(path: str, workflow: str) -> float:
    """
    Read the magnetic field strength (B0) directly from an experiment's
    acquisition parameters, since at this stage of the pipeline the
    config file does not yet carry field-strength information.

    - ``workflow == "reporter"`` → the file is a 1D spectrum, loaded as
      ``kz.Spectrum_1D``.
    - every other workflow (csar, fastcsar, titration) → the file is a
      pseudo-2D relaxation experiment, loaded as
      ``kz.Spectrum_Pseudo_2D``.

    In both cases the field is read from ``s.acqus["B0"]``.
    """
    if kz is None:
        raise RuntimeError(
            "klassez is required to read the magnetic field from an "
            "experiment file, but is not installed/importable."
        )
    if workflow == "reporter":
        s = kz.Spectrum_1D(path)
    else:
        s = kz.Pseudo_2D(path)
    return float(s.acqus["B0"])


def _ask_free_protein_paths(
    compound_name: str,
    workflow: str,
    label: str = "Field",
    existing: Optional[Tuple[float, Dict[str, str]]] = None,
) -> Tuple[float, Dict[str, str]]:
    """
    Ask for the free-ligand and protein experiment paths for one field
    point, then read the actual field strength B0 from whichever of the
    two experiments was provided (free-ligand preferred; falls back to
    the protein experiment if only that one was given).

    Parameters
    ----------
    existing : (float, dict) or None
        A previously-saved ``(B, {"free": ..., "protein": ...})`` entry
        for this field point, read back from the config. When given, its
        paths are offered as defaults (Enter reuses them) instead of
        forcing the user to re-enter everything on every run, and its B
        is reused as-is when neither path is changed.

    Returns
    -------
    B : float
        Field strength (0.0 if neither path was given/kept).
    paths : dict
        ``{"free": ..., "protein": ...}``
    """
    existing_B, existing_paths = existing if existing else (None, {})
    print(f"    {label}")
    free    = _ask_path("      Free ligand experiment: ", default=existing_paths.get("free", ""))
    protein = _ask_path("      + Protein experiment  : ", default=existing_paths.get("protein", ""))

    source = free or protein
    if existing_B is not None:
        B = existing_B
    elif source:
        B = _read_field_from_experiment(source, workflow)
        print(f"      → B0 = {B} T (read from experiment)")
    else:
        B = 0.0
    print()
    return B, {"free": free, "protein": protein}


def _ask_titration_paths(
    compound_name: str,
    protein_concentrations: List[float],
    workflow: str,
    existing: Optional[Tuple[float, Dict[str, str]]] = None,
) -> Tuple[float, Dict[str, str]]:
    """
    Ask for one experiment path per protein concentration. The field
    strength is read from the acquisition parameters of only the first
    (newly provided) experiment — all concentration points are assumed
    to be recorded at the same field.

    Parameters
    ----------
    existing : (float, dict) or None
        Previously-saved ``(B, {"P_<conc>uM": path, ...})`` for this
        compound, offered as defaults so a re-run doesn't force
        re-entering paths that were already provided.

    Returns
    -------
    B : float
        Field strength (0.0 if no paths were given/kept at all).
    paths : dict
        ``{"P_<conc>uM": path, ...}``
    """
    existing_B, existing_paths = existing if existing else (None, {})
    print("    One file per [protein] concentration")
    paths: Dict[str, str] = {}
    B: Optional[float] = existing_B
    for conc in protein_concentrations:
        key      = f"P_{conc:.1f}uM"
        default  = existing_paths.get(key, "")
        path     = _ask_path(f"      [P] = {conc:.1f} µM: ", default=default)
        paths[key] = path
        if B is None and path:
            # Field not yet known from a previous run — read it from the
            # first newly-provided experiment.
            B = _read_field_from_experiment(path, workflow)
            print(f"      → B0 = {B} T (read from first experiment)")
    B = B if B is not None else 0.0
    print()
    return B, paths


# ---------------------------------------------------------------------------
# (D) Load and fit
# ---------------------------------------------------------------------------

def _sanitize_dirname(name: str) -> str:
    """
    Turn an arbitrary protein name into a filesystem-safe directory
    component: whitespace → underscore, anything else that isn't
    alphanumeric/underscore/hyphen stripped out.

    Shared by :func:`_basedir` here and by the equivalent helper in
    ``csar_workflows.py`` — both must produce the *identical* string
    from the same ``protein_name`` so that populate_config's per-field
    fit results and csar_workflows' ranking plot land in the same
    directory.
    """
    safe = re.sub(r"\s+", "_", name.strip())
    safe = re.sub(r"[^A-Za-z0-9_-]", "", safe)
    return safe or "protein"


def _basedir(protein_name: Optional[str]) -> Path:
    """
    Return the ``fit_results`` directory to write into, namespaced by
    protein name when one is known.

    ``fit_results/`` on its own (``protein_name`` is ``None`` or empty)
    for backward compatibility with configs that predate the
    ``[workflow].protein_name`` field; ``fit_results/{protein_name}/``
    once a protein name is available, so results from different
    proteins run through the same working directory don't collide or
    get mixed together. Creates the directory if it doesn't exist yet.
    """
    basedir = Path("fit_results")
    if protein_name:
        basedir = basedir / _sanitize_dirname(protein_name)
    basedir.mkdir(parents=True, exist_ok=True)
    return basedir


def _get_or_ask_protein_name(cfg: dict, config_path: str) -> str:
    """
    Return ``[workflow].protein_name``, prompting the user for it and
    persisting it to the config if it isn't already there.

    Asked once per run (like ``fastcsar_output``) rather than per
    compound — it's a single workflow-wide value, and every
    compound/field's fit results should land under the same
    ``fit_results/{protein_name}/`` directory, plus
    ``csar_workflows.py``'s ranking plot needs the identical value to
    save alongside them.
    """
    wf_cfg = cfg.get("workflow", {})
    protein_name = wf_cfg.get("protein_name", None)
    if not protein_name:
        protein_name = input("  Protein name (used to name the results folder): ").strip()
        if not protein_name:
            protein_name = "protein"
        wf_cfg["protein_name"] = protein_name
        cfg["workflow"] = wf_cfg
        _write_toml_scalar(config_path, "[workflow]", "protein_name", _toml_str(protein_name))
        print()
    return protein_name



    """
    Build the filename fragment that disambiguates one field point from
    another (e.g. csar's field point 1 vs. field point 2), shared by
    every function that writes or caches per-field files:
    :func:`load_experiment` (igrl/fvf/ivf caching) and
    :func:`_fit_free_protein_R2` (.inp dumps and the fit .png).

    ``field_index`` (1-based ordinal, in the order the field points were
    collected) is what actually guarantees uniqueness — two distinct
    field points can otherwise format to the same string (e.g. 16.401
    and 16.404 both round to "1.6e+01" under a coarse format), which
    silently overwrote/reused one field point's files with another's.
    The field value itself is included too, at 6 significant figures,
    purely so filenames stay human-readable/traceable to a field
    strength — it is not relied on for uniqueness.

    Parameters
    ----------
    field_T : float
        Magnetic field in Tesla.
    field_index : int, optional
        1-based ordinal of this field point. ``None`` (e.g. workflows
        with no notion of multiple field points) omits the "_ptN" part.

    Returns
    -------
    str
        e.g. ``"_pt1_16p401T"``, or ``"_16p401T"`` if ``field_index`` is
        ``None``. The decimal point is replaced with "p" — load_experiment
        repeatedly calls ``.with_suffix(...)`` on a Path built from this
        tag (to try ``.igrl``/``.fvf``/``.ivf`` variants), and
        ``with_suffix`` replaces everything after the *last* dot in the
        name; a literal "16.401T" would corrupt to "16" the moment
        ``.with_suffix(".igrl")`` was called.
    """
    point_tag = f"_pt{field_index}" if field_index is not None else ""
    field_str = f"{field_T:.6g}".replace(".", "p")
    return f"{point_tag}_{field_str}T"


def load_experiment(
    paths: Dict[str, str],
    workflow: str,
    fittingmode: str,
    compound_name: str,
    field_T: Optional[float] = None,
    field_index: Optional[int] = None,
    protein_name: Optional[str] = None,
) -> Dict[str, object]:
    """
    Load raw experimental data from the file paths collected in step (C).

    .. todo::
        Implement this function to read your spectral data format.

    This function is called once per compound per workflow.  It receives the
    path dictionary built by :func:`collect_paths` and must return a dict of
    raw data objects that :func:`fit_experiment` knows how to process.

    Parameters
    ----------
    paths : dict
        ``{"free": "/path/to/free.dat", "protein": "/path/to/protein.dat"}``
        for CSAR/FastCSAR; see :class:`ExperimentPaths` for other workflows.
    workflow : str
        One of ``"csar"``, ``"fastcsar"``, ``"reporter"``, ``"titration"``.
    fittingmode : str
        Either ``"i"`` (integrate) or ``"f"`` (fit) — passed from the user prompt.
    protein_name : str, optional
        Used to namespace the ``.igrl``/``.fvf``/``.ivf`` cache files
        under ``fit_results/{protein_name}/`` instead of bare
        ``fit_results/`` — see :func:`_basedir`.

    Returns
    -------
    dict
        Workflow-dependent raw data.  Suggested conventions:

        ``csar`` / ``fastcsar``::

            {
                "free":    (delays_s: np.ndarray, intensities: np.ndarray),
                "protein": (delays_s: np.ndarray, intensities: np.ndarray),
            }

        ``reporter``::

            {"reporter_intensity": float}

        ``titration``::

            {
                "P_0.0uM":  (delays_s, intensities),
                "P_5.0uM":  (delays_s, intensities),
                ...
            }


    """
    if workflow != "reporter":
        dict_of_delays_intensities = {}
        basedir = _basedir(protein_name)

        # Same collision this workflow's .inp/.png output had (see
        # _field_tag's docstring): with no field disambiguation here,
        # csar's field point 2 wrote its .igrl/.fvf cache under the
        # exact same "{compound_name}{condition}" base as field point 1,
        # so the second field point's run would find field point 1's
        # cached integrals/fit already sitting there (same filename) and
        # silently reuse them instead of processing its own data.
        tag = _field_tag(field_T, field_index) if field_T is not None else ""

        for condition, path in paths.items():
            filename = basedir / (compound_name + condition + tag)
            if not path:
                raise ValueError(f"Missing path for condition '{condition}' in workflow '{workflow}'.")
            #check for the existence of igrl or ivf file in the fit_results folder.

            s = kz.Pseudo_2D(path)
 
            vclistpath = Path(path) / 'lists' / 'vc'
            #vclistpath is a folder. Check if a file exists in the folder.
            files = [f for f in vclistpath.iterdir() if f.is_file()]

            if len(files) == 1:
                vclistfile = files[0]
                print(f"Success! Found: {vclistfile.name}")
            else:
                print(f"Error: Found {len(files)} files instead of 1")
            #   Print a notification
            #   Actual loading and storage in an attribute
            vclist = np.loadtxt(vclistpath / vclistfile.name)
            #vclist = np.roll(vclist, -1)  # roll the list to the right by one position 

            p2 = s.ngdic['acqus']['P'][2]*1e-6
            d20 = s.ngdic['acqus']['D'][20]
            print(f'Found P2 = {p2} and D20 = {d20} in the acquisition parameters')
            print(f'Found {len(vclist)} delays in the vc list')
            # for i, counter in enumerate(vclist):
            #     print(f'vclist[{i}] = {counter} - {counter * (p2 + 2 * d20):.2g} s')
            delays_s = vclist * (p2 + 2 * d20)  # convert to seconds
            # for i, delay in enumerate(delays_s):
            #     print(f'delays_s[{i}] = {delay:.2g} s')
            # exit()
            if fittingmode == "i":
                if (filename.with_suffix(".igrl")).exists():
                    s.read_integrals(filename=filename.with_suffix(".igrl"))
                    maxtrace_idx = 0
                else:
                    s.procs["wf"]["mode"] = "em"
                    s.procs["wf"]["lb"] = 5
                    s.procs["zf"] = 2*s.fid.shape[-1]
                    s.process()
                    s.pknl()
                    s.adjph(update=False) 
                    maxtrace_idx = np.argmax(kz.processing.integrate(np.abs(s.S)))
                    s.integrate(filename=filename)
                N = len(delays_s)
                try:
                    key = next(
                        key for key, val in s.integrals.items()
                        if isinstance(val, np.ndarray) and len(val) == N
                    )
                    print(f"First matching key: {key}")
                except StopIteration:
                    print(f"No list of length {N} found")
                intensities = np.array(s.integrals[key])
            else:
                if (filename.with_suffix(".fvf")).exists():
                    s.F.load_fit(output_file=filename.with_suffix(".fvf"))
                else:
                    s.F.iguess(input_file=filename, expno=maxtrace_idx)
                    s.F.dofit(filename = filename)
                intensities = np.array([s.F.result[0][i][1]['k'] for i in range(len(s.F.result[0]))])
            dict_of_delays_intensities[condition] = (delays_s, intensities)

        return dict_of_delays_intensities
    else:
        dict_of_reporter_signal = {}
        basedir = _basedir(protein_name)

        for condition, path in paths.items():
            filename = basedir / (compound_name + condition)
            if not path:
                raise ValueError(f"Missing path for condition '{condition}' in workflow '{workflow}'.")
            s = kz.Spectrum_1D(path)

            if fittingmode == "i":
                if (filename.with_suffix(".igrl")).exists():
                    s.read_integrals(filename=filename.with_suffix(".igrl"))
                else:
                    s.procs["wf"]["mode"] = "em"
                    s.procs["wf"]["lb"] = 5
                    s.procs["zf"] = 2*s.fid.shape[-1]
                    s.process()
                    s.pknl()
                    s.adjph(update=False)
                    s.integrate(filename=filename)

                pattern = re.compile(r'^\d+\.\d{2}:\d+\.\d{2}$')
                first_key = None
                for key, value in s.integrals.items():
                    if isinstance(key, str) and pattern.match(key):
                        first_key = key
                        break
                if first_key is None:
                    raise ValueError(
                        f"No integral region matching the expected label "
                        f"was found for '{compound_name}{condition}'."
                    )
                intensity = np.array(s.integrals[first_key])
            else:
                if (filename.with_suffix(".ivf")).exists():
                    s.F.load_fit(output_file=filename.with_suffix(".ivf"))
                else:
                    s.procs["wf"]["mode"] = "em"
                    s.procs["wf"]["lb"] = 5
                    s.procs["zf"] = 2*s.fid.shape[-1]
                    s.process()
                    s.pknl()
                    s.adjph(update=False)
                    s.F.iguess(filename=basedir / (compound_name + condition), ext='ivf')
                    s.F.dofit(filename = basedir / (compound_name + condition))
                print(s.F.result)
                intensity = s.F.result[0]['I'] * s.F.result[0][1]['k']
            dict_of_reporter_signal["reporter_intensity"] = float(intensity)

        return dict_of_reporter_signal



def _fit_free_protein_R2(
    raw_data: Dict[str, object],
    compound_name: str,
    field_T: float,
    experiment_type: str,
    basedir: Path,
    field_index: Optional[int] = None,
) -> Tuple[list, list]:
    """
    Fit R2 (free-ligand and +protein) from a two-condition raw-data dict
    ``{"free": (delays, intensities), "protein": (delays, intensities)}``
    at a single field, saving one two-panel figure.

    Shared by the ``csar`` workflow and the ``fastcsar`` workflow's
    ``"r2"`` output mode — both start from the same free/protein
    delay-intensity pair at a single field and differ only in what
    happens to the resulting R2 values downstream (csar subtracts two
    fields' worth of these; fastcsar's "r2" mode feeds a single field's
    pair straight into the analytical point-ratio formula).

    Parameters
    ----------
    field_index : int, optional
        1-based ordinal of this field point among the compound's
        ``ep.fields`` (in insertion order — field point 1, field point
        2, …). csar has exactly two field points per compound; without
        this, the only thing distinguishing their ``.inp``/``.png``
        filenames was the formatted ``field_T`` value itself, and a
        too-coarse format (or two field points that happen to render
        the same) silently overwrote field point 1's files with field
        point 2's, so field point 1 was effectively never fit/plotted.
        Folding in ``field_index`` makes the filename collision-proof
        regardless of how ``field_T`` happens to format.

    Returns
    -------
    (values, errors) : ([R2_free, R2_protein], [σ_R2_free, σ_R2_protein])
    """
    values, errors = [], []
    tag = _field_tag(field_T, field_index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharex=False, sharey=False, squeeze=False)
    axes_flat = axes.flatten()
    for condition in ('free', 'protein'):
        ax = axes_flat[0] if condition == 'free' else axes_flat[1]
        ax.set_title(f"{condition.capitalize()} ligand")
        delays, intensities = raw_data[condition]
        filename = basedir / f"{compound_name}_{condition}{tag}.inp"
        open(filename, 'w').write('\n'.join(f"{d:.6e} {i:.6e}" for d, i in zip(delays, intensities)))
        if experiment_type == "s":
            result = fit_exponential(delays, intensities, multi=1)
        elif experiment_type == "n":
            result = fit_exponential_Jmod(delays, intensities, multi=1)
        plot_exponential_fit(ax, delays, intensities, result)
        R2, sigma_R2 = extract_R2_uncertainty(result, multi=1)
        values.append(R2)
        errors.append(sigma_R2)

    # Save once, after BOTH panels are drawn — saving mid-loop (once per
    # condition) meant the "free" panel had to be complete before the
    # "protein" one was even plotted, and closing `fig` right after that
    # first save left the second `plt.savefig` with no current figure to
    # target, so it silently created and saved a brand-new blank one
    # instead (that's the empty "_protein_..._fit.png" you saw). Saving
    # `fig` explicitly — rather than relying on pyplot's implicit
    # "current figure" — avoids that failure mode entirely.
    fig.savefig(
        basedir / f"{compound_name}{tag}_fit.png",
        dpi=300,
    )
    plt.close(fig)
    return values, errors   # ([R2_free, R2_prot], [σ_free, σ_prot])


def fit_experiment(
    raw_data: Dict[str, object],
    workflow: str,
    compound_name: str,
    field_T: float,
    protein_concentrations: Optional[List[float]] = None,
    experiment_type: str = "s",
    fastcsar_mode: str = "raw",
    field_index: Optional[int] = None,
    protein_name: Optional[str] = None,
) -> list:
    """
    Fit the raw experimental data and return R2 values ready for the config.

    Parameters
    ----------
    raw_data : dict
        As returned by :func:`load_experiment` for this field.
    workflow : str
        One of ``"csar"``, ``"fastcsar"``, ``"reporter"``, ``"titration"``.
    compound_name : str
        Name of the compound being fitted (for logging / error messages).
    field_T : float
        Magnetic field in Tesla (for logging; the data are already
        field-specific at this point).
    protein_concentrations : list of float, optional
        Required for ``"titration"`` — the protein concentration (µM)
        corresponding to each element of the returned list.
    fastcsar_mode : str, default "raw"
        Only used for ``workflow == "fastcsar"``. ``"r2"`` fits R2
        (free/protein) exactly like ``csar`` does, so
        ``csar_workflows.py``'s analytical point-ratio formula
        (``workflow_FastCSAR``) can take the fast, fitted-R2 path
        instead of the raw-data path. ``"raw"`` (default) keeps the
        original behaviour of handing back the delay/intensity arrays
        unfit.
    field_index : int, optional
        1-based ordinal of this field point (passed straight through to
        :func:`_fit_free_protein_R2` for csar/fastcsar "r2" mode — see
        its docstring). Only matters for filename uniqueness; doesn't
        affect the returned values.
    protein_name : str, optional
        Used to namespace the ``.inp``/``.png`` output files under
        ``fit_results/{protein_name}/`` instead of bare
        ``fit_results/`` — see :func:`_basedir`. ``csar_workflows.py``
        needs the identical namespacing to save its ranking plot
        alongside these.

    Returns
    -------
    (payload, errors) : tuple
        Every workflow returns this same 2-tuple shape — ``run_fitting``
        relies on that to unpack it uniformly, regardless of workflow.

        ``csar``
            ``payload = [R2_free, R2_protein]``  (s⁻¹),
            ``errors  = [σ_R2_free, σ_R2_protein]``

        ``fastcsar``, ``fastcsar_mode="r2"``
            ``payload = [R2_free, R2_protein]``  (s⁻¹) — same shape as
            csar; ``csar_workflows.py`` reads this from ``R2_fitted[B]``.
            ``errors  = [σ_R2_free, σ_R2_protein]``

        ``fastcsar``, ``fastcsar_mode="raw"`` (default)
            ``payload = [[delays], [I_free], [I_protein]]``  — raw
            arrays (s, intensity, intensity); no R2 is fitted here.
            ``errors  = []``  (nothing to report)

        ``reporter``
            ``payload = [reporter_signal_plus]``  (intensity, a.u.)
            ``errors  = [nan]``  (uncertainty not meaningful here)

        ``titration``
            ``payload = [R2_P0, R2_P1, …]``  — one value per protein
            concentration, same order as ``protein_concentrations``.
            ``errors  = [σ_R2_P0, σ_R2_P1, …]``

    """
    basedir = _basedir(protein_name)
    if workflow == 'csar':
        return _fit_free_protein_R2(
            raw_data, compound_name, field_T, experiment_type, basedir,
            field_index=field_index,
        )

    elif workflow == 'fastcsar':
        if fastcsar_mode == "r2":
            # Fit R2 for free/protein just like csar, so
            # csar_workflows.py's workflow_FastCSAR can take the
            # analytical fitted-R2 path (Eq. 11 with R2_fitted[B])
            # instead of the raw-data path.
            return _fit_free_protein_R2(
                raw_data, compound_name, field_T, experiment_type, basedir,
                field_index=field_index,
            )

        delays, I_free, I_protein = [], [], []
        for condition in ('free', 'protein'):
            d, i = raw_data[condition]
            delays.append(d)
            if condition == 'free':
                I_free.append(i)
            else:
                I_protein.append(i)
        # No R2 is fitted here — fastcsar hands back the raw
        # delay/intensity arrays for the analytical point-ratio formula
        # downstream to consume directly. Still returned as a
        # (payload, errors) 2-tuple, like every other workflow, so
        # run_fitting has one uniform code path; errors is empty since
        # there's no fit uncertainty to report at this stage.
        return [delays, I_free, I_protein], []

    elif workflow == 'reporter':
        signal = float(raw_data['reporter_intensity'])
        return [signal], [np.nan]   # uncertainty not meaningful here

    elif workflow == 'titration':
        values, errors = [], []
        n = len(protein_concentrations)
        nrows = int(np.ceil(np.sqrt(n)))
        ncols = int(np.ceil(n / nrows))
        panel_w, panel_h = 2.6, 2.3   # inches per panel (height > width keeps each panel, and hence
                                  # the whole grid given nrows >= ncols, on the tall side too)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_w, nrows * panel_h),
                                sharex=False, sharey=False, squeeze=False)
        axes_flat = axes.flatten()

        for i, conc in enumerate(protein_concentrations):
            key = f"P_{conc:.1f}uM"
            if key not in raw_data:
                raise KeyError(
                    f"No loaded data for concentration {conc:.1f} µM "
                    f"(expected key '{key}') for compound '{compound_name}'."
                )
            ax = axes_flat[i]
            ax.set_title(f"[P] = {conc:.1f} µM")
            delays, intensities = raw_data[key]
            filename = basedir / (compound_name + key + '.inp')
            open(filename, 'w').write('\n'.join(f"{d:.6e} {ival:.6e}" for d, ival in zip(delays, intensities)))
            if experiment_type == "s":
                result = fit_exponential(delays, intensities, multi=1)
            elif experiment_type == "n":
                result = fit_exponential_Jmod(delays, intensities, multi=1)
            print(lmfit.fit_report(result))
            R2, sigma_R2 = extract_R2_uncertainty(result, multi=1)
            print(f"    [P] = {conc:.1f} µM  →  R2 = {R2:.3f} ± {sigma_R2:.3f} s⁻¹")
            plot_exponential_fit(ax, delays, intensities, result, experiment_type=experiment_type)

            values.append(R2)
            errors.append(sigma_R2)

        # n doesn't always fill the nrows x ncols grid exactly (e.g. n=5
        # needs a 3x2 grid, leaving one panel empty) — hide the leftovers
        # instead of leaving them blank with no title.
        for j in range(n, nrows * ncols):
            axes_flat[j].set_visible(False)
        
        #fig.tight_layout()
        figurename = basedir / str(compound_name + '_titration_fit.png')
        print(f"Saving figure to {figurename}")
        plt.savefig(figurename, format='png', dpi=300)
        print(f"Saved figure to {figurename}")
        plt.close()
        plt.cla()
        return values, errors
    # fit_experiment must return (values_list, uncertainties_list)

def run_fitting(
    experiments: List[ExperimentPaths],
    workflow: str,
    cfg: dict,
    config_path: str,
) -> Tuple[List[FittedR2], Optional[str]]:
    """
    Drive steps (D.1) load and (D.2) fit for all compounds.

    Parameters
    ----------
    experiments : list of ExperimentPaths
    workflow : str
    cfg : dict
        Full config dict (used to read protein_concentrations for titration).
    config_path : str
        Path to the TOML config file. Used so that, for ``fastcsar``, the
        user's choice of output (fitted R2 vs raw relaxation data) is
        persisted into the ``[workflow]`` section on disk as soon as
        it's made, and so ``[workflow].protein_name`` can be persisted
        the same way if it isn't already present.

    Returns
    -------
    fitted : list of FittedR2
        One entry per successfully fitted compound.
    fastcsar_mode : str or None
        Which output the user chose for the ``fastcsar`` workflow —
        ``"r2"`` (fit R2 values, same shape/config block as ``csar``) or
        ``"raw"`` (store raw delay/intensity arrays in
        ``[compound.NAME.relaxation_data]`` for the point-ratio
        formula). ``None`` for every other workflow. The caller needs
        this to know which block ``update_config`` should write to.
    """
    wf_cfg = cfg.get("workflow", {})
    prot_concs = [float(p) for p in wf_cfg.get("protein_concentrations", [])]

    # Resolve (and persist, if missing) the protein name — every
    # compound/field's .inp/.png output for this run goes under
    # fit_results/{protein_name}/, and csar_workflows.py's ranking plot
    # needs the identical value to land in the same directory.
    protein_name = _get_or_ask_protein_name(cfg, config_path)

    # For fastcsar, csar_workflows.py's workflow_FastCSAR accepts either
    # a pre-fitted R2 (fast, analytical, same shape as csar) or raw
    # relaxation data (fit-free, feeds the point-ratio formula
    # directly). Ask once, up front, rather than per-compound — it's a
    # workflow-wide choice — and persist it so it's visible in the
    # config and a re-run doesn't need to re-ask.
    fastcsar_mode: Optional[str] = None
    if workflow == "fastcsar":
        mode_input = input(
            "  FastCSAR: return fitted R2 values or raw relaxation data? "
            "([R2]/raw): "
        ).strip().lower()
        fastcsar_mode = "raw" if mode_input.startswith("raw") else "r2"
        _write_toml_scalar(
            config_path, "[workflow]", "fastcsar_output",
            _toml_str(fastcsar_mode),
        )
        print()

    fitted: List[FittedR2] = []

    for ep in experiments:
        fittingmode=input("Integrate or fit? ([i]/f): ",).strip().lower() or "i"
        experiment_type=input("Selective or non-selective? ([s]/n): ",).strip().lower() or "s"
        print(f"  Fitting {ep.compound_name} …")
        r2_dict:     Dict[float, list] = {}
        r2_err_dict: Dict[float, list] = {}
        failed = False

        for field_index, (field_T, condition_paths) in enumerate(ep.fields.items(), start=1):
            try:
                # (D.1) ── load ──────────────────────────────────────────────
                raw = load_experiment(
                    condition_paths, workflow, fittingmode, ep.compound_name,
                    field_T=field_T, field_index=field_index,
                    protein_name=protein_name,
                )

                # (D.2) ── fit ───────────────────────────────────────────────
                # Every workflow's fit_experiment returns the same
                # (payload, errors) 2-tuple shape (see its docstring) —
                # for fastcsar in "raw" mode, payload is the raw
                # [delays, I_free, I_protein] arrays rather than an R2,
                # and errors is empty, but the shape itself is uniform,
                # so one code path here covers every workflow.
                r2_values, r2_errors = fit_experiment(
                    raw,
                    workflow=workflow,
                    compound_name=ep.compound_name,
                    field_T=field_T,
                    protein_concentrations=prot_concs or None,
                    experiment_type=experiment_type,
                    fastcsar_mode=fastcsar_mode or "raw",
                    field_index=field_index,
                    protein_name=protein_name,
                )
                r2_dict[field_T]      = r2_values
                r2_err_dict[field_T]  = r2_errors

            except NotImplementedError as exc:
                # Propagate so the user sees the message clearly
                raise
            except Exception as exc:
                # Print the full traceback, not just str(exc) — the
                # exception could originate several calls deep (inside
                # load_experiment, fit_experiment, or a klassez/lmfit
                # call they make), and the one-line summary alone gives
                # no way to tell which line actually raised it.
                print(f"    ✗ Error fitting {ep.compound_name} at {field_T} T:")
                traceback.print_exc()
                failed = True
                break

        if not failed:
            fitted.append(FittedR2(compound_name=ep.compound_name,
                                   r2=r2_dict, r2_err=r2_err_dict))
            _print_fitted(ep.compound_name, r2_dict, workflow, fastcsar_mode=fastcsar_mode)

    return fitted, fastcsar_mode


def _print_fitted(
    name: str,
    r2_dict: Dict[float, list],
    workflow: str,
    fastcsar_mode: Optional[str] = None,
) -> None:
    """Print a compact summary of fitted/loaded results."""
    for B, vals in sorted(r2_dict.items()):
        if workflow == "csar" or (workflow == "fastcsar" and fastcsar_mode == "r2"):
            print(f"    {B:.2f} T  →  R2_free = {vals[0]:.3f} s⁻¹,  "
                  f"R2_protein = {vals[1]:.3f} s⁻¹")
        elif workflow == "fastcsar":
            delays, I_free, I_protein = vals
            n_free = len(delays[0]) if len(delays) > 0 else 0
            n_prot = len(delays[1]) if len(delays) > 1 else 0
            print(f"    {B:.2f} T  →  loaded {n_free} free / {n_prot} protein "
                  f"intensity point(s) (no R2 fit — raw data stored for "
                  f"the point-ratio formula)")
        elif workflow == "reporter":
            print(f"    reporter signal = {vals[0]:.4f}")
        elif workflow == "titration":
            summary = ", ".join(f"{v:.3f}" for v in vals)
            print(f"    {B:.2f} T  →  [{summary}] s⁻¹")
    print()


# ---------------------------------------------------------------------------
# (E) Write R2 values back into the config
# ---------------------------------------------------------------------------

def update_config(
    config_path: str,
    fitted: List[FittedR2],
    skipped: List[str],
    workflow: str,
    fastcsar_mode: Optional[str] = None,
) -> None:
    """
    (E) Write fitted R2 values into the TOML file and remove skipped compounds.

    A timestamped backup of the original file is created before any changes
    are written.

    Parameters
    ----------
    config_path : str
    fitted : list of FittedR2
    skipped : list of str
        Compound names to remove from the file.
    workflow : str
    fastcsar_mode : str, optional
        Only relevant for ``workflow == "fastcsar"`` (as returned by
        :func:`run_fitting`). ``"r2"`` writes the usual
        ``[compound.NAME.R2]`` block (same as csar) so
        ``csar_workflows.py`` takes the fitted-R2 path; anything else
        (including ``None``) writes ``[compound.NAME.relaxation_data]``
        instead, for the raw-data point-ratio path.
    """
    path = Path(config_path)
    _backup(path)

    text = path.read_text(encoding="utf-8")

    # (E.1) Remove skipped compounds
    for name in skipped:
        text = _remove_compound_text(text, name)
        print(f"  [config] Removed '{name}'")

    # (E.2) Update R2 / relaxation_data blocks for fitted compounds
    for fr in fitted:
        if workflow == "reporter":
            text = _write_reporter_signal(text, fr.compound_name, fr.r2)
            print(f"  [config] Updated reporter signal for '{fr.compound_name}'")
        elif workflow == "fastcsar" and fastcsar_mode != "r2":
            text = _write_relaxation_data_block(text, fr.compound_name, fr.r2)
            print(f"  [config] Updated relaxation data for '{fr.compound_name}'")
        else:
            text = _write_r2_block(text, fr.compound_name, fr.r2, workflow,
                                   r2_err_dict=fr.r2_err or None)
            print(f"  [config] Updated R2 for '{fr.compound_name}'")

    path.write_text(text, encoding="utf-8")
    print(f"\n  Configuration saved → {config_path}")


# -- helpers for (E) ---------------------------------------------------------

def _backup(path: Path) -> None:
    """Create a timestamped backup of the config file."""
    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(f".{stamp}.bak")
    shutil.copy2(path, backup)
    print(f"  [backup] {backup}")


def _r2_block_content(r2_dict: Dict[float, list],
                      r2_err_dict: Optional[Dict[float, list]] = None) -> str:
    """
    Render the body of ``[compound.NAME.R2]`` (and optionally
    ``[compound.NAME.R2_err]``) blocks.

    Each field maps to a TOML array: ``"B" = [v1, v2, ...]``
    Returns a tuple ``(r2_content, r2_err_content)`` where ``r2_err_content``
    is ``None`` if no error dict was provided or all values are nan.
    """
    r2_lines, err_lines, any_finite_err = [], [], False
    for B in sorted(r2_dict.keys()):
        vals    = r2_dict[B]
        val_str = ", ".join(f"{v:.6g}" for v in vals)
        r2_lines.append(f'"{B:.4g}" = [{val_str}]')
        if r2_err_dict and B in r2_err_dict:
            errs    = r2_err_dict[B]
            err_str = ", ".join(f"{e:.6g}" for e in errs)
            err_lines.append(f'"{B:.4g}" = [{err_str}]')
            if any(np.isfinite(e) for e in errs):
                any_finite_err = True
    r2_content  = "\n".join(r2_lines)
    err_content = "\n".join(err_lines) if any_finite_err else None
    return r2_content, err_content


def _write_r2_block(
    text: str,
    compound_name: str,
    r2_dict: Dict[float, list],
    workflow: str,
    r2_err_dict: Optional[Dict[float, list]] = None,
) -> str:
    """
    Replace the content of ``[compound.NAME.R2]`` (and optionally
    ``[compound.NAME.R2_err]``) in the raw TOML text.

    If the R2 block does not exist yet, it is appended.  The R2_err
    block is appended directly after R2 when uncertainties are available.
    """
    header     = f"[compound.{compound_name}.R2]"
    err_header = f"[compound.{compound_name}.R2_err]"
    r2_content, err_content = _r2_block_content(r2_dict, r2_err_dict)

    # Build the combined replacement: R2 block + optional R2_err block
    replacement_body = "\n" + r2_content + "\n"

    if header not in text:
        text = _append_r2_block(text, compound_name, r2_content)
    else:
        # Replace R2 block body using section-split approach
        parts = re.split(r"(\n\[[^\n]+\])", text)
        result = [parts[0]]
        i = 1
        while i < len(parts):
            sec_header = parts[i]
            sec_body   = parts[i + 1] if i + 1 < len(parts) else ""
            stripped   = sec_header.lstrip("\n")
            if stripped == header:
                result.append(sec_header)
                result.append(replacement_body)
            elif stripped == err_header:
                pass   # remove old R2_err block; re-written below
            else:
                result.append(sec_header)
                result.append(sec_body)
            i += 2
        text = "".join(result)

    # Append the R2_err block immediately after the R2 block
    if err_content is not None:
        err_block = (
            f"\n[compound.{compound_name}.R2_err]\n"
            "# 1σ uncertainties from lmfit covariance (s⁻¹)\n"
            f"{err_content}\n"
        )
        insert_after = text.index(header) + len(header)
        # Move past the R2 block body to the next section
        next_sec = re.search(r"\n\[", text[insert_after:])
        if next_sec:
            insert_pos = insert_after + next_sec.start()
            text = text[:insert_pos] + err_block + text[insert_pos:]
        else:
            text = text.rstrip() + err_block

    return text


def _append_r2_block(text: str, compound_name: str, content: str) -> str:
    """Append an R2 block right after the ``[compound.NAME]`` header."""
    compound_header = f"[compound.{compound_name}]"
    if compound_header not in text:
        raise ValueError(
            f"Cannot find '{compound_header}' in the config to append R2 block."
        )
    insert_after = text.index(compound_header) + len(compound_header)
    r2_block = f"\n\n[compound.{compound_name}.R2]\n{content}\n"
    return text[:insert_after] + r2_block + text[insert_after:]


def _write_reporter_signal(
    text: str,
    compound_name: str,
    r2_dict: Dict[float, list],
) -> str:
    """
    For the reporter workflow, write ``reporter_signal_plus`` as a key under
    ``[compound.NAME]`` rather than in the R2 sub-table.
    """
    # r2_dict = {0.0: [signal_value]}
    signal = list(r2_dict.values())[0][0]
    key    = "reporter_signal_plus"
    header = f"[compound.{compound_name}]"

    # Replace existing key or insert after the compound header
    kv_pattern = re.compile(
        r"(^" + re.escape(key) + r"\s*=\s*)[^\n]+", re.MULTILINE
    )
    compound_start = text.find(header)
    if compound_start == -1:
        raise ValueError(f"Compound '{compound_name}' not found in config.")

    # Scope the search to this compound's section only
    next_header = re.search(r"\n\[", text[compound_start + len(header):])
    section_end = (
        compound_start + len(header) + next_header.start()
        if next_header else len(text)
    )
    section = text[compound_start:section_end]

    if kv_pattern.search(section):
        section = kv_pattern.sub(
            lambda m: m.group(1) + f"{signal:.6g}", section
        )
    else:
        # Append before the R2 sub-block or at end of section
        r2_header = f"[compound.{compound_name}.R2]"
        if r2_header in section:
            idx = section.index(r2_header)
            section = section[:idx] + f"{key} = {signal:.6g}\n\n" + section[idx:]
        else:
            section = section.rstrip() + f"\n{key} = {signal:.6g}\n"

    return text[:compound_start] + section + text[section_end:]


def _relaxation_data_content(raw_dict: Dict[float, list]) -> str:
    """
    Render the body of a ``[compound.NAME.relaxation_data]`` block: one
    inline table per field, holding the raw delay/intensity arrays that
    ``fit_experiment`` returns for the ``fastcsar`` workflow.

    Unlike ``csar``/``titration``, ``fastcsar`` doesn't fit an R2 at all
    — it hands back ``[delays, I_free, I_protein]`` for the analytical
    point-ratio formula downstream (Eq. 11, Rüdisser 2020 —
    ``workflow_FastCSAR``) to consume directly. There,
    ``delays = [delays_free, delays_protein]`` (one array per condition)
    and ``I_free``/``I_protein`` are each a single-element list wrapping
    that condition's intensity array. This flattens that into four named
    arrays so the config stays self-describing.
    """
    def _arr(values) -> str:
        return "[" + ", ".join(f"{v:.6e}" for v in values) + "]"

    lines = []
    for B in sorted(raw_dict.keys()):
        delays, I_free, I_protein = raw_dict[B]
        delays_free       = delays[0]    if len(delays) > 0 else []
        delays_protein    = delays[1]    if len(delays) > 1 else []
        intensity_free    = I_free[0]    if I_free    else []
        intensity_protein = I_protein[0] if I_protein else []
        inline = (
            f"delays_free = {_arr(delays_free)}, "
            f"delays_protein = {_arr(delays_protein)}, "
            f"intensity_free = {_arr(intensity_free)}, "
            f"intensity_protein = {_arr(intensity_protein)}"
        )
        lines.append(f'"{B:.4g}" = {{ {inline} }}')
    return "\n".join(lines)


def _write_relaxation_data_block(
    text: str,
    compound_name: str,
    raw_dict: Dict[float, list],
) -> str:
    """
    Insert or replace ``[compound.NAME.relaxation_data]`` — the raw
    delay/intensity arrays for the ``fastcsar`` workflow (no R2 is
    fitted here; see ``FittedR2.raw`` and the point-ratio formula
    downstream, e.g. ``workflow_FastCSAR``, which reads this data
    directly rather than a pre-fitted R2).

    Note this is a *new* subtable, distinct from ``[compound.NAME.R2]``
    — a compound run through ``fastcsar`` will not have an R2 block at
    all, only this one.
    """
    header  = f"[compound.{compound_name}.relaxation_data]"
    content = _relaxation_data_content(raw_dict)

    if header not in text:
        compound_header = f"[compound.{compound_name}]"
        if compound_header not in text:
            raise ValueError(
                f"Cannot find '{compound_header}' in the config to append "
                "relaxation_data block."
            )
        # Insert after everything already in this compound's section
        # (including any other subtables) — inserting right after the
        # opening header would break any scalar keys, e.g. delta_sigma,
        # that follow it (they'd be misparsed as belonging to this new
        # subtable instead of [compound.NAME]).
        insert_at = _compound_section_end(text, compound_name)
        block = f"\n{header}\n{content}\n"
        text = text[:insert_at] + block + text[insert_at:]
    else:
        parts  = re.split(r"(\n\[[^\n]+\])", text)
        result = [parts[0]]
        i = 1
        while i < len(parts):
            sec_header = parts[i]
            sec_body   = parts[i + 1] if i + 1 < len(parts) else ""
            stripped   = sec_header.lstrip("\n")
            if stripped == header:
                result.append(sec_header)
                result.append("\n" + content + "\n")
            else:
                result.append(sec_header)
                result.append(sec_body)
            i += 2
        text = "".join(result)

    return text


def _remove_compound_text(text: str, name: str) -> str:
    """
    Remove all ``[compound.NAME]`` and ``[compound.NAME.*]`` blocks from
    the raw TOML text.

    Splits on TOML section headers (lines of the form ``\n[...]``) and
    discards any section whose header matches the compound name.
    """
    compound_re = re.compile(
        r"\[compound\." + re.escape(name) + r"(?:[\].])?"
    )
    parts = re.split(r"(\n\[[^\n]+\])", text)
    result = [parts[0]]
    i = 1
    while i < len(parts):
        sec_header = parts[i]
        sec_body   = parts[i + 1] if i + 1 < len(parts) else ""
        if compound_re.search(sec_header):
            pass   # discard
        else:
            result.append(sec_header)
            result.append(sec_body)
        i += 2
    return "".join(result)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def _read_workflow_name(config_path: str) -> Optional[str]:
    if not Path(config_path).exists():
        return None
    try:
        with open(config_path, "rb") as fh:
            cfg = tomllib.load(fh)
        return cfg.get("workflow", {}).get("name")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    if not Path(config_path).exists():
        sys.exit(f"Config file not found: {config_path}")

    # (A) ── workflow selection ───────────────────────────────────────────────
    workflow = select_workflow(config_path)

    # Persist the workflow name into [workflow] right away. Every other
    # [workflow] key (B / B_high / B_low, ligand_concentration,
    # protein_concentrations, fit_KD, fastcsar_output) gets written
    # on-demand as it's collected further down, via _write_toml_scalar
    # calls that *append into* an existing [workflow] section — but none
    # of them ever created the section's ``name`` key. If the config
    # didn't already have a [workflow] section (e.g. a fresh
    # compounds-only file), the section _write_toml_scalar creates would
    # end up missing ``name`` entirely. csar_workflows.py's
    # run_from_config reads ``wf_cfg["name"]`` with no default and
    # raises KeyError if it's absent, so without this the file
    # populate_config produces could not be fed to it directly. Writing
    # it here, first, guarantees [workflow] always has it, and every
    # later _write_toml_scalar call for this workflow just appends
    # alongside it instead of creating a name-less section.
    _write_toml_scalar(config_path, "[workflow]", "name", _toml_str(workflow))

    # (B) ── parse config ─────────────────────────────────────────────────────
    cfg, compound_names = parse_config(config_path)

    # (C) ── collect experiment paths ─────────────────────────────────────────
    experiments, skipped = collect_paths(compound_names, workflow, cfg, config_path)

    if not experiments:
        print("No experiment paths provided for any compound — nothing to do.")
        return

    # (D) ── load + fit ───────────────────────────────────
    print(f"── (D) Fitting {len(experiments)} compound(s) ─────────────────────────────")
    fitted, fastcsar_mode = run_fitting(experiments, workflow, cfg, config_path)

    if not fitted:
        print("No compounds were successfully fitted — config not modified.")
        return

    # (E) ── write results ────────────────────────────────────────────────────
    print(f"── (E) Updating config ─────────────────────────────────────────────")
    update_config(config_path, fitted, skipped, workflow, fastcsar_mode=fastcsar_mode)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {sys.argv[0]} experiment.toml")
    main(sys.argv[1])
