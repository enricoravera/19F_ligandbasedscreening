"""
add_compound.py
===============
Create or update compound entries in a CSAR/FastCSAR TOML configuration file.

Two entry modes
---------------
**ORCA mode** (``--orca FILE``)
    Reads an ORCA NMR output file, extracts the ¹⁹F chemical shielding
    tensor(s), applies Haeberlen decomposition to obtain δσ and η, and
    handles CF₃ rotational averaging via the Waugh treatment
    (Mehring et al. 1971, as used in Rüdisser et al. 2020).

**Manual mode** (``--manual`` or explicit ``--delta-sigma`` / ``--eta`` flags)
    Accepts CSA parameters on the command line or through interactive prompts.

Usage examples
--------------
Add a compound from an ORCA output (auto-detect CF/CF₃)::

    python add_compound.py experiment.toml \\
        --orca path/to/orca.out --name compound_6

Add a specific F nucleus by ORCA index::

    python add_compound.py experiment.toml \\
        --orca path/to/orca.out --name compound_6 --nucleus-index 12

Override auto-detected fluorine type::

    python add_compound.py experiment.toml \\
        --orca path/to/orca.out --name compound_6 --fluorine-type CF

Add a compound manually (non-interactive)::

    python add_compound.py experiment.toml \\
        --name compound_3 --fluorine-type CF3 --delta-sigma 40.0 --eta 0.0

Add a compound interactively::

    python add_compound.py experiment.toml --manual

List existing compounds::

    python add_compound.py experiment.toml --list

The script appends to the existing config file without disturbing its content,
comments, or formatting.  If the compound name already exists, it will ask for
confirmation before overwriting.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tensor math: Haeberlen decomposition and CF₃ rotational averaging
# ---------------------------------------------------------------------------

def symmetrise(sigma: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a 3×3 shielding tensor."""
    return 0.5 * (sigma + sigma.T)


def haeberlen(sigma: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Decompose a 3×3 shielding tensor into Haeberlen parameters.

    The tensor is first symmetrised, then diagonalised.  Principal components
    are assigned following the Haeberlen convention::

        |σ_zz − σ_iso| ≥ |σ_yy − σ_iso| ≥ |σ_xx − σ_iso|

    Parameters
    ----------
    sigma : (3, 3) ndarray
        Chemical shielding tensor in ppm (any units, consistently).

    Returns
    -------
    sigma_iso : float
        Isotropic shielding  σ_iso = Tr(σ)/3.
    delta_sigma : float
        Reduced anisotropy  δσ = σ_zz − σ_iso   (ppm).
    eta : float
        Axial asymmetry  η = (σ_yy − σ_xx) / δσ.
    sigma_xx, sigma_yy, sigma_zz : float
        Principal components in the Haeberlen ordering  (ppm).
    """
    sigma_sym = symmetrise(np.asarray(sigma, dtype=float))
    eigenvalues, _ = np.linalg.eigh(sigma_sym)   # already sorted ascending

    sigma_iso = eigenvalues.mean()
    deviations = np.abs(eigenvalues - sigma_iso)

    # Assign by Haeberlen convention: sort by |σ_i − σ_iso| descending
    order = np.argsort(deviations)[::-1]          # [zz, yy, xx] indices
    sigma_zz = eigenvalues[order[0]]
    sigma_yy = eigenvalues[order[1]]
    sigma_xx = eigenvalues[order[2]]

    delta_sigma = sigma_zz - sigma_iso
    eta = (sigma_yy - sigma_xx) / delta_sigma if abs(delta_sigma) > 1e-6 else 0.0

    return sigma_iso, delta_sigma, eta, sigma_xx, sigma_yy, sigma_zz


def average_cf3_tensors(tensors: List[np.ndarray]) -> np.ndarray:
    """
    Rotationally averaged CF₃ shielding tensor (Waugh treatment).

    As noted in Rüdisser 2020 (Discussion): for unhindered CF₃ rotation,
    averaging the three individual F tensors in Cartesian coordinates and
    diagonalising gives the same result as the full Waugh formula
    (Mehring et al. 1971).  The resulting tensor is axially symmetric
    (η = 0) and its reduced anisotropy δσ is roughly one-third that of
    a single CF tensor.

    Parameters
    ----------
    tensors : list of (3, 3) ndarray
        The three individual fluorine shielding tensors in ppm, all in the
        same molecular Cartesian frame.

    Returns
    -------
    (3, 3) ndarray
        Rotationally averaged tensor.
    """
    if len(tensors) != 3:
        raise ValueError(f"CF₃ averaging requires exactly 3 tensors, got {len(tensors)}.")
    return np.mean(tensors, axis=0)


# ---------------------------------------------------------------------------
# ORCA result parsing helpers
# ---------------------------------------------------------------------------

def _load_orca(orca_file: str) -> dict:
    """Parse an ORCA NMR output and return the result dict."""
    try:
        from orca_nmr_parser import parse_orca_output, result_to_dict
    except ImportError:
        sys.exit("Error: orca_nmr_parser is not installed or not on PYTHONPATH.")
    return result_to_dict(parse_orca_output(orca_file))


def _cf3_groups(coordinates: list, f_indices: List[int]) -> List[List[int]]:
    """
    Group F atom indices into CF₃ sets by proximity to a common C atom.

    Uses a C–F bond distance threshold of 1.6 Å (generous to handle
    varied geometries).  Returns a list of lists; single-F groups are
    returned as one-element lists (CF).

    Parameters
    ----------
    coordinates : list of dict
        As returned by orca_nmr_parser: each dict has keys
        ``index``, ``element``, ``x``, ``y``, ``z``  (coordinates in Å).
    f_indices : list of int
        Nucleus indices (matching ``coordinates[i]["index"]``) of the F atoms
        to be grouped.
    """
    BOND_THRESHOLD = 1.6   # Å

    # Build index → coordinate map
    coord_map = {c["index"]: np.array([c["x"], c["y"], c["z"]]) for c in coordinates}
    c_coords   = {c["index"]: np.array([c["x"], c["y"], c["z"]])
                  for c in coordinates if c["element"] == "C"}

    # For each F, find the nearest C atom
    f_to_c: Dict[int, int] = {}
    for fi in f_indices:
        fpos = coord_map[fi]
        nearest_c, nearest_d = None, np.inf
        for ci, cpos in c_coords.items():
            d = np.linalg.norm(fpos - cpos)
            if d < nearest_d:
                nearest_d = d
                nearest_c = ci
        if nearest_d > BOND_THRESHOLD:
            warnings.warn(
                f"F atom {fi}: nearest C is {nearest_d:.2f} Å away — "
                "possible geometry issue."
            )
        f_to_c[fi] = nearest_c

    # Invert: C → list of bonded F atoms
    c_to_f: Dict[int, List[int]] = {}
    for fi, ci in f_to_c.items():
        c_to_f.setdefault(ci, []).append(fi)

    return list(c_to_f.values())


def extract_from_orca(
    orca_file: str,
    nucleus_index: Optional[int] = None,
    fluorine_type: Optional[str] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Extract CSA parameters for all (or one selected) F nucleus/group from an
    ORCA NMR output file.

    Parameters
    ----------
    orca_file : str
        Path to the ORCA output file.
    nucleus_index : int, optional
        If given, restrict extraction to the F nucleus with this ORCA index.
        For CF₃, give the index of any one of the three F atoms.
    fluorine_type : str, optional
        ``'CF3'`` or ``'CF'``.  Auto-detected from molecular connectivity
        if not provided.
    verbose : bool
        Print extracted parameters.

    Returns
    -------
    list of dict
        One entry per F group, each with keys:
        ``nucleus_indices``, ``fluorine_type``, ``delta_sigma``, ``eta``,
        ``sigma_iso``, ``sigma_xx``, ``sigma_yy``, ``sigma_zz``.
    """
    results  = _load_orca(orca_file)
    coords   = results["coordinates"]
    block    = results["shielding_blocks"][-1]

    # Collect all F nuclei and their tensors
    f_nuclei = {
        n["nucleus_index"]: np.array(n["total_tensor"]["rows"], dtype=float)
        for n in block["nuclei"]
        if n["element"] == "F"
    }

    if not f_nuclei:
        sys.exit("No fluorine nuclei found in ORCA output.")

    if nucleus_index is not None and nucleus_index not in f_nuclei:
        available = sorted(f_nuclei.keys())
        sys.exit(
            f"Nucleus index {nucleus_index} not found among F nuclei: {available}"
        )

    # Group into CF / CF₃ families based on connectivity
    all_f_indices = sorted(f_nuclei.keys())
    groups = _cf3_groups(coords, all_f_indices)

    # If a specific nucleus was requested, keep only its group
    if nucleus_index is not None:
        groups = [g for g in groups if nucleus_index in g]
        if not groups:
            sys.exit(f"Could not find a CF/CF₃ group containing nucleus {nucleus_index}.")

    entries = []
    for group in groups:
        group_sorted = sorted(group)
        n_f = len(group_sorted)

        # Determine / validate fluorine type
        if fluorine_type is not None:
            ftype = fluorine_type.upper()
        elif n_f == 3:
            ftype = "CF3"
        elif n_f == 1:
            ftype = "CF"
        else:
            warnings.warn(
                f"Unexpected group size {n_f} for indices {group_sorted} — "
                "treating as CF."
            )
            ftype = "CF"

        # Compute averaged tensor
        if ftype == "CF3":
            if len(group_sorted) != 3:
                warnings.warn(
                    f"CF3 averaging requested but found {len(group_sorted)} F atom(s) "
                    f"in group {group_sorted}.  Using available tensors."
                )
            tensors = [f_nuclei[i] for i in group_sorted if i in f_nuclei]
            # Pad to 3 if needed (degenerate geometry)
            while len(tensors) < 3:
                tensors.append(tensors[-1])
            sigma_avg = average_cf3_tensors(tensors[:3])
        else:
            sigma_avg = f_nuclei[group_sorted[0]]

        sigma_iso, delta_sigma, eta, sxx, syy, szz = haeberlen(sigma_avg)

        entry = {
            "nucleus_indices": group_sorted,
            "fluorine_type":   ftype,
            "delta_sigma":     round(delta_sigma, 4),
            "eta":             round(eta, 4),
            "sigma_iso":       round(sigma_iso, 4),
            "sigma_xx":        round(sxx, 4),
            "sigma_yy":        round(syy, 4),
            "sigma_zz":        round(szz, 4),
        }
        entries.append(entry)

        if verbose:
            print(f"\n  Nucleus/group : {group_sorted}  [{ftype}]")
            print(f"  σ_iso         = {sigma_iso:>10.3f} ppm")
            print(f"  σ_xx          = {sxx:>10.3f} ppm")
            print(f"  σ_yy          = {syy:>10.3f} ppm")
            print(f"  σ_zz          = {szz:>10.3f} ppm")
            print(f"  δσ            = {delta_sigma:>10.3f} ppm")
            print(f"  η             = {eta:>10.4f}")

    return entries


# ---------------------------------------------------------------------------
# TOML config read / write helpers
# ---------------------------------------------------------------------------

def _read_existing_names(config_path: str) -> List[str]:
    """Return list of compound names already in the config."""
    if not Path(config_path).exists():
        return []
    try:
        with open(config_path, "rb") as fh:
            cfg = tomllib.load(fh)
        return list(cfg.get("compound", {}).keys())
    except Exception:
        return []


def _compound_toml_block(
    name: str,
    fluorine_type: str,
    delta_sigma: float,
    eta: float,
    comment: str = "",
) -> str:
    """
    Render a TOML block for one compound (without R2 entries — those are
    added later by the spectral population script).

    Parameters
    ----------
    name : str
        Compound identifier used as the TOML table key.
    fluorine_type : str
        ``'CF3'`` or ``'CF'``.
    delta_sigma : float
        Reduced anisotropy δσ in ppm.
    eta : float
        Axial asymmetry η.
    comment : str
        Optional single-line comment appended after the header.

    Returns
    -------
    str
        TOML text (does **not** include a trailing newline after the R2 table).
    """
    lines = []
    if comment:
        lines.append(f"# {comment}")
    lines.append(f"[compound.{name}]")
    lines.append(f'fluorine_type = "{fluorine_type}"')
    lines.append(f"delta_sigma   = {delta_sigma:.4f}")
    lines.append(f"eta           = {eta:.4f}")
    lines.append("")
    lines.append(f"[compound.{name}.R2]")
    lines.append("# Populated by the spectral processing script.")
    lines.append('# Format:  "B_Tesla" = [R2_free, R2_protein]  (s^-1)')
    return "\n".join(lines)


def append_compound(
    config_path: str,
    name: str,
    fluorine_type: str,
    delta_sigma: float,
    eta: float,
    comment: str = "",
    overwrite: bool = False,
) -> None:
    """
    Append a compound block to an existing (or new) TOML config file.

    If ``config_path`` does not exist, a minimal file containing only the
    compound entry is created (the ``[workflow]`` section must be added
    separately or generated via ``csar_workflows.write_example_config``).

    Parameters
    ----------
    config_path : str
        Path to the TOML file.
    name : str
        Compound name (TOML table key under ``compound``).
    fluorine_type, delta_sigma, eta : …
        CSA parameters.
    comment : str
        Optional comment prepended to the block.
    overwrite : bool
        If True, replace an existing entry with the same name.
        If False and the name already exists, raise ``ValueError``.
    """
    existing = _read_existing_names(config_path)

    if name in existing:
        if not overwrite:
            raise ValueError(
                f"Compound '{name}' already exists in {config_path}. "
                "Pass overwrite=True to replace it."
            )
        _remove_compound(config_path, name)

    block = _compound_toml_block(name, fluorine_type, delta_sigma, eta, comment)

    path = Path(config_path)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        # Ensure there is a blank separator before the new block
        separator = "\n\n" if text.rstrip() else ""
        new_text = text.rstrip() + separator + "\n" + block + "\n"
    else:
        new_text = block + "\n"

    path.write_text(new_text, encoding="utf-8")
    print(f"[config] {'Updated' if name in existing else 'Added'} compound '{name}' "
          f"in {config_path}")


def _remove_compound(config_path: str, name: str) -> None:
    """
    Remove all TOML blocks belonging to compound ``name`` from the file.
    This covers both ``[compound.NAME]`` and ``[compound.NAME.R2]``.
    """
    text = Path(config_path).read_text(encoding="utf-8")
    # Match any header of the form [compound.NAME] or [compound.NAME.*]
    # and everything that follows until the next top-level [section]
    pattern = re.compile(
        r"\n?#.*\n"                                  # optional comment line
        r"(?=\[compound\." + re.escape(name) + r")"  # look-ahead: our header
        r".*?"                                        # content
        r"(?=\n\[|\Z)",                              # until next header or EOF
        re.DOTALL,
    )
    # Simpler, more robust: split on section headers and rebuild
    blocks = re.split(r"(\n\[)", text)
    rebuilt = []
    skip = False
    for i, part in enumerate(blocks):
        # parts alternate: content, delimiter
        if part == "\n[":
            # Peek at what follows in the next content part
            next_content = blocks[i + 1] if i + 1 < len(blocks) else ""
            if re.match(
                r"compound\." + re.escape(name) + r"[\].]", next_content
            ):
                skip = True
                continue
            else:
                skip = False
                rebuilt.append(part)
        else:
            if not skip:
                rebuilt.append(part)
    Path(config_path).write_text("".join(rebuilt), encoding="utf-8")


# ---------------------------------------------------------------------------
# Interactive / manual entry
# ---------------------------------------------------------------------------

def _prompt(prompt_text: str, default=None, cast=str):
    """Prompt the user for a value, with an optional default."""
    default_str = f"  [{default}]" if default is not None else ""
    raw = input(f"{prompt_text}{default_str}: ").strip()
    if raw == "" and default is not None:
        return cast(default)
    try:
        return cast(raw)
    except (ValueError, TypeError):
        print(f"  Invalid input — expected {cast.__name__}. Try again.")
        return _prompt(prompt_text, default, cast)


def manual_entry_interactive() -> Tuple[str, str, float, float]:
    """
    Collect compound parameters interactively.

    Returns
    -------
    name, fluorine_type, delta_sigma, eta
    """
    print("\n── Manual compound entry ──────────────────────────")
    name = _prompt("Compound name")
    ftype = ""
    while ftype not in ("CF", "CF3"):
        ftype = _prompt("Fluorine type (CF or CF3)", default="CF3").upper()
    delta_sigma = _prompt("Reduced anisotropy δσ (ppm)", cast=float)
    default_eta = 0.0 if ftype == "CF3" else None
    eta = _prompt("Axial asymmetry η", default=default_eta, cast=float)
    print("───────────────────────────────────────────────────\n")
    return name, ftype, delta_sigma, eta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="add_compound",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "config",
        metavar="CONFIG.toml",
        help="Path to the CSAR TOML config file (created if absent).",
    )

    # ---- mode flags ----
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--orca",
        metavar="FILE",
        help="ORCA NMR output file to extract shielding tensors from.",
    )
    mode.add_argument(
        "--manual",
        action="store_true",
        help="Prompt interactively for compound parameters.",
    )
    mode.add_argument(
        "--list",
        action="store_true",
        dest="list_compounds",
        help="List compound names already present in the config and exit.",
    )

    # ---- ORCA options ----
    orca = p.add_argument_group("ORCA options")
    orca.add_argument(
        "--nucleus-index",
        type=int,
        metavar="N",
        help="ORCA nucleus index of the F atom (or any F in the CF₃ group) "
             "to use.  Defaults to all F groups found.",
    )

    # ---- shared compound options ----
    comp = p.add_argument_group("Compound parameters")
    comp.add_argument(
        "--name",
        metavar="NAME",
        help="Compound name (required unless --manual or --list).",
    )
    comp.add_argument(
        "--fluorine-type",
        choices=["CF", "CF3"],
        metavar="TYPE",
        help="Fluorine type: CF or CF3.  Auto-detected in ORCA mode.",
    )
    comp.add_argument(
        "--delta-sigma",
        type=float,
        metavar="PPM",
        help="Reduced anisotropy δσ in ppm (manual mode).",
    )
    comp.add_argument(
        "--eta",
        type=float,
        metavar="ETA",
        help="Axial asymmetry η (manual mode; default 0 for CF3).",
    )

    # ---- behaviour flags ----
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing entry with the same compound name.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose tensor output.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ list
    if args.list_compounds:
        names = _read_existing_names(args.config)
        if not names:
            print(f"No compounds found in {args.config}.")
        else:
            print(f"Compounds in {args.config}:")
            for n in names:
                print(f"  {n}")
        return

    # ------------------------------------------------------------------ ORCA
    if args.orca:
        if not Path(args.orca).exists():
            sys.exit(f"ORCA file not found: {args.orca}")
        print(f"\nExtracting shielding tensors from: {args.orca}")
        entries = extract_from_orca(
            args.orca,
            nucleus_index=args.nucleus_index,
            fluorine_type=args.fluorine_type,
            verbose=not args.quiet,
        )
        if not entries:
            sys.exit("No F nuclei extracted — check the ORCA file.")

        for i, entry in enumerate(entries):
            if args.name:
                # Single requested compound: use provided name
                name = args.name if len(entries) == 1 else f"{args.name}_{i+1}"
            else:
                # Auto-name from nucleus indices
                idx_str = "_".join(str(x) for x in entry["nucleus_indices"])
                name = f"F{idx_str}"
                print(f"\n  → Using auto-generated name '{name}'. "
                      "Pass --name to override.")

            comment = (
                f"ORCA file: {Path(args.orca).name}  |  "
                f"nucleus indices: {entry['nucleus_indices']}"
            )
            append_compound(
                args.config,
                name=name,
                fluorine_type=entry["fluorine_type"],
                delta_sigma=entry["delta_sigma"],
                eta=entry["eta"],
                comment=comment,
                overwrite=args.overwrite,
            )
        return

    # ----------------------------------------------------------------- manual
    if args.manual:
        name, ftype, delta_sigma, eta = manual_entry_interactive()
    else:
        # Non-interactive manual: all params on CLI
        missing = [f for f in ("name", "delta_sigma") if getattr(args, f.replace("-", "_")) is None]
        if missing:
            parser.error(
                f"Missing required arguments for manual entry: "
                + ", ".join(f"--{m.replace('_','-')}" for m in missing)
                + ".  Use --orca, --manual, or provide all flags explicitly."
            )
        name        = args.name
        ftype       = (args.fluorine_type or "CF3").upper()
        delta_sigma = args.delta_sigma
        eta         = args.eta if args.eta is not None else (0.0 if ftype == "CF3" else 0.0)

    append_compound(
        args.config,
        name=name,
        fluorine_type=ftype,
        delta_sigma=delta_sigma,
        eta=eta,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
