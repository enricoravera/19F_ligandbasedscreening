"""
sigma0_from_pdb.py
===================
Estimate SIGMA0 -- the areal density (Å⁻²) of solvent-exposed protein
hydrogen atoms -- directly from a real PDB structure, as a data-driven
replacement/check for the mean-field value (1e-2 Å⁻²) hard-coded in
compare_R2DD.py.

This is the "exposed proton surface density" calculation: unlike
surface_packing.py / protein_sigma0.py (which ask how many EXTERNAL probe
H atoms could be packed onto the surface at maximum hard-sphere density),
this module asks how many of the protein's/ligand's OWN real hydrogens
are solvent-exposed, which is the structural quantity SIGMA0 is actually
meant to represent.

Method
------
1. Parse heavy atoms from the PDB (protein ATOM records; optionally
   ligand/cofactor HETATM records too).
2. Add real hydrogens using protonate_pdb.py's per-residue template
   geometry (backbone + all 20 canonical sidechains) and generic
   valence-based ligand protonation -- the same protonation used
   elsewhere in this project, validated to give bond lengths in the
   correct 0.96-1.09 Å range with zero atomic clashes (see
   protonate_pdb.py docstring).
3. Build a Shrake-Rupley solvent-accessible-surface (SASA) point cloud on
   every heavy atom using a water probe (1.4 Å by default), keeping only
   points not buried by neighboring atoms. Sum point-patch areas -> total
   heavy-atom SASA (Å²). This area is the "substrate" that SIGMA0's
   density is defined per unit of.
4. Classify each placed H as "exposed" using the same rolling-probe test,
   evaluated at the H atom's own position (excluding its own bonded
   parent and the parent's other bonded neighbors, which sit within
   normal bond distance and are not physical burial).
5. SIGMA0,estimated = N_exposed_H / SASA_heavy_atoms.

Usage
-----
    python sigma0_from_pdb.py structure.pdb                  # protein only
    python sigma0_from_pdb.py structure.pdb --chains A B      # specific chains
    python sigma0_from_pdb.py structure.pdb --include-ligand  # + HETATM ligands
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from compare_R2DD import SIGMA0
from surface_packing import fibonacci_sphere, vdw_radius
from protonate_pdb import parse_pdb, protonate_structure, Atom


# ============================================================================
# Shrake-Rupley SASA (heavy atoms), KDTree-accelerated
# ============================================================================

def compute_sasa(
    coords: np.ndarray,
    elems: list[str],
    probe_radius: float = 1.4,
    points_per_atom: int = 200,
) -> tuple[float, np.ndarray]:
    """
    Shrake-Rupley SASA over an arbitrary atom point set (works for
    heavy-atom-only or all-atom-including-H point sets alike; each atom's
    own van der Waals radius is used, so bonded neighbors do not need to
    be manually excluded -- normal bond geometry keeps them from
    spuriously "burying" each other once each atom's own radius is
    correctly accounted for).

    Returns
    -------
    (total_SASA, per_atom_SASA)  in Å²
    """
    n = len(elems)
    radii = np.array([vdw_radius(e) for e in elems])
    shell_radii = radii + probe_radius

    tree = cKDTree(coords)
    unit_sphere = fibonacci_sphere(points_per_atom)

    per_atom_sasa = np.zeros(n)
    max_shell = shell_radii.max()

    for i in range(n):
        candidates = coords[i] + shell_radii[i] * unit_sphere  # (P,3)
        neighbor_idx = tree.query_ball_point(coords[i], shell_radii[i] + max_shell)
        neighbor_idx = [j for j in neighbor_idx if j != i]

        keep = np.ones(points_per_atom, dtype=bool)
        for j in neighbor_idx:
            d = np.linalg.norm(candidates - coords[j], axis=1)
            keep &= d >= (shell_radii[j] - 1e-6)

        n_exposed = np.count_nonzero(keep)
        patch_area = 4 * np.pi * shell_radii[i] ** 2 / points_per_atom
        per_atom_sasa[i] = n_exposed * patch_area

    return float(per_atom_sasa.sum()), per_atom_sasa





# ============================================================================
# Main driver
# ============================================================================

@dataclass
class ExposedProtonResult:
    n_heavy_atoms: int
    total_sasa_AA2: float
    n_h_placed: int
    n_h_exposed: int
    sigma0_estimate: float   # Å⁻²
    sigma0_literature: float


def estimate_sigma0(
    pdb_path: str,
    chains: list[str] | None = None,
    include_ligand: bool = False,
    sasa_probe: float = 1.4,
    points_per_atom: int = 200,
) -> ExposedProtonResult:
    """
    Estimate the exposed-proton areal density SIGMA0 directly from a PDB
    structure's real (protonated) geometry.

    Parameters
    ----------
    chains : list[str], optional
        Restrict to specific chain IDs (None = all chains).
    include_ligand : bool
        If True, HETATM ligand/cofactor atoms are protonated and included
        in both the SASA substrate and the exposed-H count (waters are
        always excluded). If False (default), only the protein
        polypeptide (ATOM records, standard amino acids) is used, matching
        the "bare protein surface" that SIGMA0 is meant to model.
    """
    heavy = parse_pdb(pdb_path, chains=chains)
    if not include_ligand:
        heavy = [a for a in heavy if not a.is_het]
    print(f"Parsed {len(heavy)} heavy atoms from {pdb_path}"
          f"  (chains={chains or 'all'}, ligand={'included' if include_ligand else 'excluded'})")

    full = protonate_structure(heavy)
    h_atoms = [a for a in full if a.elem == "H"]
    print(f"Added {len(h_atoms)} hydrogen atoms via protonate_pdb")

    heavy_coords = np.array([a.coord for a in heavy])
    heavy_elems = [a.elem for a in heavy]
    total_sasa, _ = compute_sasa(heavy_coords, heavy_elems, sasa_probe, points_per_atom)
    print(f"Total heavy-atom SASA (probe={sasa_probe} Å): {total_sasa:.1f} Å²")

    # Run ONE unified Shrake-Rupley pass over ALL atoms (heavy + H)
    # together, so each H atom's own van der Waals radius and its real
    # bonded geometry naturally keep it from spuriously "burying" or being
    # buried by its own bonded neighbors -- no manual bonded-neighbor
    # exclusion list is needed, unlike a point-in-shell test on a bare H
    # center (which is not itself a physical SASA calculation and was
    # producing false negatives for genuinely solvent-facing H atoms,
    # e.g. terminal Lys -NH3+ groups, whenever a heavy atom 3-4 bonds up
    # the same flexible sidechain happened to swing within probe range).
    all_coords = np.vstack([heavy_coords, np.array([a.coord for a in h_atoms])]) \
        if h_atoms else heavy_coords
    all_elems = heavy_elems + ["H"] * len(h_atoms)
    _, per_atom_sasa_all = compute_sasa(all_coords, all_elems, sasa_probe, points_per_atom)

    h_sasa = per_atom_sasa_all[len(heavy):]
    exposed_mask = h_sasa > 0.0
    n_exposed = int(exposed_mask.sum())
    print(f"Solvent-exposed H atoms: {n_exposed} / {len(h_atoms)} "
          f"({100 * n_exposed / max(len(h_atoms), 1):.1f}%)")

    sigma0_est = n_exposed / total_sasa
    print(f"\nEstimated SIGMA0 = N_exposed_H / SASA = {sigma0_est:.4e} Å⁻²")
    print(f"compare_R2DD.py SIGMA0 (mean-field)    = {SIGMA0:.4e} Å⁻²")
    print(f"Ratio (this estimate / mean-field)     = {sigma0_est / SIGMA0:.3f}×")

    return ExposedProtonResult(
        n_heavy_atoms=len(heavy),
        total_sasa_AA2=total_sasa,
        n_h_placed=len(h_atoms),
        n_h_exposed=n_exposed,
        sigma0_estimate=sigma0_est,
        sigma0_literature=SIGMA0,
    )


def plot_sigma0_comparison(result: ExposedProtonResult, label: str,
                            save_path: str | None = None) -> None:
    """Bar chart: exposed-proton SIGMA0 from real structure vs the
    mean-field SIGMA0 used in compare_R2DD.py."""
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = [f"Exposed-H\n({label})", "Mean-field\n(compare_R2DD.py)"]
    values = [result.sigma0_estimate, result.sigma0_literature]
    colors = ["#EA580C", "#2563EB"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"$\sigma_0$ (Å$^{-2}$)")
    ratio = result.sigma0_estimate / result.sigma0_literature
    ax.set_title(f"Exposed-proton surface density: {ratio:.2f}× "
                 f"the mean-field value", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sigma0_from_pdb.py structure.pdb [--chains A B] [--include-ligand]")
        sys.exit(1)

    pdb_path = sys.argv[1]
    chains = None
    include_ligand = False
    args = sys.argv[2:]
    if "--include-ligand" in args:
        include_ligand = True
        args.remove("--include-ligand")
    if "--chains" in args:
        i = args.index("--chains")
        chains = []
        for tok in args[i + 1:]:
            if tok.startswith("--"):
                break
            chains.append(tok)

    result = estimate_sigma0(pdb_path, chains=chains, include_ligand=include_ligand)
    plot_sigma0_comparison(result, label=chains[0] if chains and len(chains) == 1 else "structure",
                            save_path="sigma0_exposed_H.png")
