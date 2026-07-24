"""
sum_inv_r6_from_structure.py
=============================
Compute Sigma_i 1/r_i^6 (Å⁻⁶) around a specific atom (e.g. a ligand Cl or
F) using REAL, solvent-exposed hydrogen atoms from an actual protonated
PDB structure -- protein sidechain/backbone H's plus (optionally) the
ligand's own H's -- rather than a synthetic packed-H-shell or flat-plane
model.

This is the direct structural analogue of integral_I(R_X) in
compare_R2DD.py and of the packed-surface Sigma(1/r^6) in
surface_packing.py, but computed from one specific real crystal structure
instead of an idealized geometry.

Pipeline
--------
1. Parse + protonate the structure (protein via per-residue templates,
   ligand via generic valence rules) using protonate_pdb.py.
2. Run the same unified all-atom Shrake-Rupley SASA test as
   sigma0_from_pdb.py to find which H atoms are solvent-exposed (SASA >
   0), since only exposed protons are physically available to relax the
   observed nucleus via the intermolecular dipolar mechanism (buried H's
   are shielded by the protein itself).
3. For the target atom (by residue name + atom name, e.g. OIT/CL20), sum
   1/r^6 over every exposed H (optionally excluding H's that belong to
   the target's own residue, since intramolecular ligand H's relax by a
   different, typically-averaged-out mechanism than the intermolecular
   protein-contact mechanism this whole project is modeling).

Usage
-----
    python sum_inv_r6_from_structure.py 8AWW.pdb --chains A --target OIT CL20
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from compare_R2DD import integral_I, r_eff_from_integral, VDW_RADIUS
from surface_packing import vdw_radius
from protonate_pdb import parse_pdb, protonate_structure, Atom
from sigma0_from_pdb import compute_sasa


@dataclass
class TargetResult:
    target_name: str
    target_resname: str
    target_coord: np.ndarray
    n_exposed_H_total: int
    n_exposed_H_used: int    # after excluding same-residue H's, if requested
    distances: np.ndarray
    sum_inv_r6: float        # Å⁻⁶
    r_eff: float              # Å


def find_exposed_hydrogens(
    pdb_path: str,
    chains: list[str] | None = None,
    include_ligand: bool = True,
    sasa_probe: float = 1.4,
    points_per_atom: int = 200,
) -> tuple[list[Atom], list[Atom], np.ndarray]:
    """
    Protonate the structure and classify every H atom as exposed or
    buried via the unified all-atom SASA test.

    Returns
    -------
    (heavy_atoms, h_atoms, h_sasa)
    """
    heavy = parse_pdb(pdb_path, chains=chains)
    if not include_ligand:
        heavy = [a for a in heavy if not a.is_het]

    full = protonate_structure(heavy)
    h_atoms = [a for a in full if a.elem == "H"]

    heavy_coords = np.array([a.coord for a in heavy])
    heavy_elems = [a.elem for a in heavy]
    all_coords = np.vstack([heavy_coords, np.array([a.coord for a in h_atoms])]) \
        if h_atoms else heavy_coords
    all_elems = heavy_elems + ["H"] * len(h_atoms)

    _, per_atom_sasa = compute_sasa(all_coords, all_elems, sasa_probe, points_per_atom)
    h_sasa = per_atom_sasa[len(heavy):]

    return heavy, h_atoms, h_sasa


def sum_inv_r6_around_target(
    heavy: list[Atom],
    h_atoms: list[Atom],
    h_sasa: np.ndarray,
    target_resname: str,
    target_atom_name: str,
    exclude_same_residue: bool = True,
) -> TargetResult:
    """
    Sum 1/r^6 (Å⁻⁶) from a named target heavy atom (e.g. a ligand Cl) to
    every solvent-exposed hydrogen in the structure.

    Parameters
    ----------
    target_resname, target_atom_name : str
        Identify the target heavy atom, e.g. ("OIT", "CL20").
    exclude_same_residue : bool
        If True (default), hydrogens belonging to the SAME residue/ligand
        instance as the target are excluded from the sum -- these are
        intramolecular protons (e.g. the ligand's own ring/amine H's),
        which relax the target nucleus by a different, typically
        conformer-averaged intramolecular mechanism than the
        intermolecular protein-contact mechanism this calculation is
        estimating. Set False to include all exposed H's regardless of
        which residue they belong to.
    """
    target_atom = next(
        (a for a in heavy if a.resname == target_resname and a.name == target_atom_name),
        None,
    )
    if target_atom is None:
        raise ValueError(f"Target atom {target_resname}/{target_atom_name} not found.")

    exposed_idx = np.where(h_sasa > 0.0)[0]
    exposed_h = [h_atoms[k] for k in exposed_idx]

    if exclude_same_residue:
        exposed_h = [
            h for h in exposed_h
            if not (h.resname == target_atom.resname
                    and h.resseq == target_atom.resseq
                    and h.chain == target_atom.chain)
        ]

    coords = np.array([h.coord for h in exposed_h]) if exposed_h else np.zeros((0, 3))
    d = np.linalg.norm(coords - target_atom.coord, axis=1) if len(coords) else np.zeros(0)

    sum_inv_r6 = float(np.sum(d ** -6)) if len(d) else 0.0
    r_eff = sum_inv_r6 ** (-1.0 / 6.0) if sum_inv_r6 > 0 else float("nan")

    return TargetResult(
        target_name=f"{target_atom.resname}{target_atom.resseq}/{target_atom.name}",
        target_resname=target_atom.resname,
        target_coord=target_atom.coord,
        n_exposed_H_total=len(exposed_idx),
        n_exposed_H_used=len(exposed_h),
        distances=d,
        sum_inv_r6=sum_inv_r6,
        r_eff=r_eff,
    )


def print_report(results: list[TargetResult], R_ref_AA: float = VDW_RADIUS["F"]) -> None:
    I_flat = integral_I(R_ref_AA)
    r_eff_flat = r_eff_from_integral(R_ref_AA)
    header = "─" * 70
    print(f"\n{header}")
    print("  Sigma(1/r^6) from real, solvent-exposed structural H atoms")
    print(header)
    print(f"  Flat-surface model reference (compare_R2DD.py, R_X={R_ref_AA} Å):")
    print(f"    I(flat)     = {I_flat:.4e} Å⁻⁶")
    print(f"    r_eff(flat) = {r_eff_flat:.3f} Å")
    print(header)
    for res in results:
        print(f"\n  Target: {res.target_name}  at {tuple(np.round(res.target_coord, 3))}")
        print(f"    Exposed H's in structure (total)       : {res.n_exposed_H_total}")
        print(f"    Exposed H's used (same-residue excluded): {res.n_exposed_H_used}")
        if res.n_exposed_H_used > 0:
            print(f"    Nearest exposed H distance             : {res.distances.min():.3f} Å")
            print(f"    Mean exposed H distance                : {res.distances.mean():.3f} Å")
        print(f"    Σᵢ 1/rᵢ⁶ (structural, exposed H only)  : {res.sum_inv_r6:.4e} Å⁻⁶")
        print(f"    Equivalent single-H distance r_eff     : {res.r_eff:.3f} Å")
        if res.sum_inv_r6 > 0:
            print(f"    Ratio  Σ1/r⁶ (structural) / I (flat)   : {res.sum_inv_r6 / I_flat:.3f}")
    print(f"\n{header}\n")


def plot_distance_histograms(results: list[TargetResult], save_path: str | None = None) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), squeeze=False)
    axes = axes[0]
    for ax, res in zip(axes, results):
        if res.n_exposed_H_used == 0:
            ax.set_title(f"{res.target_name}  (no exposed H found)")
            continue
        ax.hist(res.distances, bins=30, color="#EA580C", alpha=0.8, edgecolor="white")
        r_mean = res.distances.mean()
        ax.axvline(r_mean, color="#16A34A", lw=2, ls="--", label=f"r_mean={r_mean:.2f} Å")
        ax.axvline(res.r_eff, color="#DC2626", lw=2, ls="-.", label=f"r_eff={res.r_eff:.2f} Å")
        ax.set_xlabel("Distance to exposed H (Å)")
        ax.set_ylabel("Exposed H count")
        ax.set_title(f"{res.target_name}  (N={res.n_exposed_H_used} exposed H)")
        ax.legend(fontsize=8)
    fig.suptitle("Structural Σ1/r⁶: distance to real solvent-exposed H atoms",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sum_inv_r6_from_structure.py structure.pdb "
              "[--chains A] [--target RESNAME ATOMNAME ...] [--include-same-residue]")
        sys.exit(1)

    pdb_path = sys.argv[1]
    args = sys.argv[2:]

    chains = None
    if "--chains" in args:
        i = args.index("--chains")
        chains = []
        for tok in args[i + 1:]:
            if tok.startswith("--"):
                break
            chains.append(tok)

    exclude_same_residue = "--include-same-residue" not in args

    targets = []
    if "--target" in args:
        i = args.index("--target")
        rest = args[i + 1:]
        pairs = []
        for tok in rest:
            if tok.startswith("--"):
                break
            pairs.append(tok)
        for k in range(0, len(pairs) - 1, 2):
            targets.append((pairs[k], pairs[k + 1]))
    if not targets:
        targets = [("OIT", "CL20"), ("OIT", "CL23")]  # default: both ligand Cl's

    heavy, h_atoms, h_sasa = find_exposed_hydrogens(pdb_path, chains=chains)
    print(f"Protonated structure: {len(heavy)} heavy atoms, {len(h_atoms)} H atoms, "
          f"{(h_sasa > 0).sum()} solvent-exposed")

    results = [
        sum_inv_r6_around_target(heavy, h_atoms, h_sasa, resname, atomname,
                                   exclude_same_residue=exclude_same_residue)
        for resname, atomname in targets
    ]
    print_report(results)
    plot_distance_histograms(results, save_path="sum_inv_r6_structural.png")
