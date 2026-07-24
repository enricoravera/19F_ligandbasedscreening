"""
surface_packing.py
===================
Geometry-aware replacement for the infinite-flat-surface integral I(R_X) in
compare_R2DD.py, using an *actual* small-molecule structure (XYZ file).

Physical idea
-------------
`compare_R2DD.py` replaces the unknown Σᵢ 1/rᵢ⁶ (sum over surrounding
protein ¹H's) with the integral of a uniform H density σ₀ over an infinite
flat plane in vdW contact with the observed nucleus X:

    I = π σ₀ / (2 z⁴),     z = R_H + R_X

That is a mean-field idealization: real surfaces are curved, and real H
atoms cannot overlap. This module computes the *actual* geometric analogue
for a given molecule:

  1. Parse atomic coordinates from an XYZ file.
  2. Build the solvent(H)-accessible surface of the molecule: the locus of
     points where a probe H atom (radius R_H) would sit in van der Waals
     contact with the molecule, with self-occluded points removed (a
     standard "rolling ball" / Shrake–Rupley-style construction).
  3. Greedily pack that accessible surface with non-overlapping H atoms at
     the maximum density geometry allows (closest-first packing subject to
     an H–H hard-sphere exclusion of 2·R_H).
  4. For every fluorine atom in the molecule, compute the F···H distance to
     every packed H atom and report:
       - the full distance distribution (for a histogram),
       - the simple mean distance r_mean,
       - the physically-relevant packed sum  Σᵢ 1/rᵢ⁶  (Å⁻⁶), which plugs
         in directly wherever `integral_I(R_X)` is used in
         compare_R2DD.py,
       - the equivalent single-proton distance r_eff = (Σ 1/r⁶)^(-1/6),
         consistent with `r_eff_from_integral` in compare_R2DD.py.

This gives a maximum-density, finite-molecule estimate to compare against
the infinite-flat-surface mean-field estimate.

Usage
-----
    python surface_packing.py molecule.xyz
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

# Reuse the same vdW radii, H-surface parameters, and R2,DD machinery as
# compare_R2DD.py so all three models are directly comparable.
from compare_R2DD import (
    VDW_RADIUS, R_H_SURF, SIGMA0,
    integral_I, r_eff_from_integral,
    R2_DD_from_sum_inv_r6, R2_DD_integral, R2_DD_rulethumh,
    tau_c_from_MW, GAMMA,
)


# ============================================================================
# XYZ parsing
# ============================================================================

@dataclass
class Molecule:
    symbols: list[str]
    coords: np.ndarray  # (N, 3) Å
    comment: str = ""


def read_xyz(path: str) -> Molecule:
    """
    Read a standard XYZ file:

        N
        comment line
        Sym  x  y  z
        ...

    Returns
    -------
    Molecule
    """
    with open(path, "r") as fh:
        lines = fh.readlines()

    n_atoms = int(lines[0].strip())
    comment = lines[1].rstrip("\n") if len(lines) > 1 else ""

    symbols: list[str] = []
    coords = np.zeros((n_atoms, 3), dtype=float)

    for i in range(n_atoms):
        parts = lines[2 + i].split()
        sym = parts[0]
        # Normalize element capitalization, e.g. "F", "Cl", "cl" -> "Cl"
        sym = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()
        symbols.append(sym)
        coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

    return Molecule(symbols=symbols, coords=coords, comment=comment)


def vdw_radius(symbol: str) -> float:
    """
    Look up the van der Waals radius (Å) for an element symbol, falling
    back to a generic value if the element is not in VDW_RADIUS.
    """
    if symbol in VDW_RADIUS:
        return VDW_RADIUS[symbol]
    # Reasonable fallback (Bondi 1964 generic heavy-atom radius)
    _fallback = {
        "P": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
        "Fe": 2.05, "Zn": 1.39, "Na": 2.27, "Mg": 1.73,
    }
    return _fallback.get(symbol, 1.70)


# ============================================================================
# Fibonacci sphere sampling
# ============================================================================

def fibonacci_sphere(n_points: int) -> np.ndarray:
    """
    Generate n_points quasi-uniformly distributed on the unit sphere using
    the Fibonacci-lattice construction.

    Returns
    -------
    (n_points, 3) array of unit vectors.
    """
    i = np.arange(0, n_points, dtype=float)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - (2.0 * i + 1.0) / n_points
    r_xy = np.sqrt(np.clip(1.0 - z**2, 0.0, None))
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    return np.column_stack([x, y, z])


# ============================================================================
# Solvent(H)-accessible surface construction
# ============================================================================

def build_accessible_surface(
    mol: Molecule,
    probe_radius: float = R_H_SURF,
    points_per_atom: int = 400,
) -> np.ndarray:
    """
    Build the H-accessible surface of the molecule: the set of points where
    the CENTER of a probe H atom (radius `probe_radius`) sits in van der
    Waals contact with the nearest molecular atom, with points buried
    inside any other atom's contact shell removed.

    This is the standard rolling-probe (Shrake–Rupley-style) construction,
    evaluated on a per-atom Fibonacci-sphere lattice rather than a
    continuous surface, which is sufficient for the packing step below.

    Parameters
    ----------
    mol : Molecule
    probe_radius : float
        Radius of the probe H atom (Å). Defaults to R_H_SURF from
        compare_R2DD.py so the two models share the same H size.
    points_per_atom : int
        Sampling density of the per-atom Fibonacci lattice. Higher =
        smoother surface but slower; 400 is enough for small molecules.

    Returns
    -------
    (M, 3) array of accessible surface points (candidate H-atom centers).
    """
    n_atoms = len(mol.symbols)
    radii = np.array([vdw_radius(s) for s in mol.symbols])
    shell_radii = radii + probe_radius  # locus radius per atom

    unit_sphere = fibonacci_sphere(points_per_atom)

    accepted_points = []
    for i in range(n_atoms):
        candidates = mol.coords[i] + shell_radii[i] * unit_sphere  # (P, 3)

        # A candidate point is accessible if it is NOT inside any other
        # atom's contact shell (distance to that atom's center >= its own
        # shell radius). Use a small numerical tolerance.
        keep_mask = np.ones(points_per_atom, dtype=bool)
        for j in range(n_atoms):
            if j == i:
                continue
            d = np.linalg.norm(candidates - mol.coords[j], axis=1)
            buried = d < (shell_radii[j] - 1e-6)
            keep_mask &= ~buried

        if np.any(keep_mask):
            accepted_points.append(candidates[keep_mask])

    if not accepted_points:
        return np.zeros((0, 3))

    return np.vstack(accepted_points)


# ============================================================================
# Densest-packing of H atoms on the accessible surface
# ============================================================================

def spacing_for_density(sigma: float) -> float:
    """
    Nearest-neighbor spacing `s` (Å) for a 2D hexagonal close-packed lattice
    of areal density `sigma` (Å⁻²):

        sigma = 2 / (√3 · s²)   =>   s = sqrt(2 / (√3 · sigma))

    Used to translate a target H areal density (e.g. SIGMA0 from
    compare_R2DD.py) into an equivalent minimum-separation constraint for
    pack_hydrogens.
    """
    return np.sqrt(2.0 / (np.sqrt(3.0) * sigma))


def pack_hydrogens(
    surface_points: np.ndarray,
    seed_point: np.ndarray | None = None,
    probe_radius: float = R_H_SURF,
    target_density: float | None = None,
) -> np.ndarray:
    """
    Greedily pack non-overlapping H atoms onto `surface_points`.

    By default (target_density=None) this uses the maximum density the
    geometry allows: candidate H centers are accepted one at a time
    (nearest-to-seed first) as long as they are at least 2*probe_radius
    from every already-accepted H center (hard-sphere H-H vdW exclusion).

    If `target_density` (Å⁻²) is given, the minimum H-H separation is
    instead set to whichever is LARGER of (a) the hard-sphere contact
    distance 2*probe_radius, or (b) the hex-lattice spacing that reproduces
    `target_density` (via spacing_for_density). This lets you cap the
    packed-surface model at the same areal H density used by the
    flat-surface model (SIGMA0 in compare_R2DD.py, 10⁻² Å⁻²) for a
    density-matched, apples-to-apples comparison — isolating the effect of
    molecular curvature/finite size from the effect of packing density.

    Parameters
    ----------
    surface_points : (M, 3) array
        Candidate H-atom centers on the accessible surface (from
        build_accessible_surface).
    seed_point : (3,) array, optional
        Point to sort candidates by distance to (typically the F atom of
        interest), so packing grows outward from the region that matters.
        If None, sorts from the surface centroid.
    probe_radius : float
        H vdW radius (Å); sets the hard-sphere floor on H-H separation.
    target_density : float, optional
        Areal H density (Å⁻²) to cap the packing at. None (default) means
        uncapped maximum-density packing.

    Returns
    -------
    (K, 3) array of accepted (packed) H-atom center coordinates.
    """
    if len(surface_points) == 0:
        return np.zeros((0, 3))

    if seed_point is None:
        seed_point = surface_points.mean(axis=0)

    order = np.argsort(np.linalg.norm(surface_points - seed_point, axis=1))
    candidates = surface_points[order]

    min_sep = 2.0 * probe_radius
    if target_density is not None:
        min_sep = max(min_sep, spacing_for_density(target_density))

    accepted: list[np.ndarray] = []
    accepted_arr = np.zeros((0, 3))

    for pt in candidates:
        if accepted_arr.shape[0] > 0:
            d = np.linalg.norm(accepted_arr - pt, axis=1)
            if np.any(d < min_sep):
                continue
        accepted.append(pt)
        accepted_arr = np.vstack(accepted) if accepted else np.zeros((0, 3))

    return accepted_arr


# ============================================================================
# Per-fluorine analysis
# ============================================================================

def estimate_packed_density(packed_points: np.ndarray) -> float:
    """
    Estimate the empirical areal H density (Å⁻²) of a packed point set from
    its mean nearest-neighbor spacing, assuming local 2D hexagonal packing:

        sigma ≈ 2 / (√3 · s_nn²)

    Returns NaN if fewer than 2 points are given.
    """
    if len(packed_points) < 2:
        return float("nan")
    from scipy.spatial import cKDTree
    tree = cKDTree(packed_points)
    nn_dist, _ = tree.query(packed_points, k=2)
    s_nn = float(np.mean(nn_dist[:, 1]))
    return 2.0 / (np.sqrt(3.0) * s_nn ** 2)


@dataclass
class FluorineResult:
    index: int
    coord: np.ndarray
    distances: np.ndarray          # (K,) Å, F to each packed H
    r_mean: float                  # Å, simple arithmetic mean
    sum_inv_r6: float               # Å⁻⁶, Σ 1/r⁶  (drop-in for integral_I)
    r_eff: float                   # Å, (Σ 1/r⁶)^(-1/6)
    n_packed_H: int
    R2_DD_packed: float = float("nan")   # s⁻¹, filled in if B0/tau_c given
    packing_density: float | None = None  # Å⁻², requested cap (None = uncapped)
    empirical_density: float = float("nan")  # Å⁻², actual density achieved


def analyze_fluorines(
    mol: Molecule,
    probe_radius: float = R_H_SURF,
    points_per_atom: int = 400,
    B0: float | None = None,
    tau_c: float | None = None,
    nuc_obs: str = "19F",
    nuc_partner: str = "1H",
    target_density: float | None = None,
) -> list[FluorineResult]:
    """
    Build the accessible surface once, then for every F atom in the
    molecule pack H atoms (seeded from that F, so packing density is
    highest near the nucleus of interest) and compute the distance
    statistics needed for the R2,DD calculation.

    Parameters
    ----------
    B0, tau_c : float, optional
        If both are given, also compute R2_DD_packed (s⁻¹) for each F atom
        via R2_DD_from_sum_inv_r6, using the packed-surface Σ 1/r⁶ in place
        of the flat-surface integral I(R_X). If either is None, R2_DD_packed
        is left as NaN.
    target_density : float, optional
        Areal H density (Å⁻²) to cap the packing at (see pack_hydrogens).
        Pass compare_R2DD.SIGMA0 to match the flat-surface model's density
        exactly, isolating the curvature/finite-size effect from the
        packing-density effect. None (default) = uncapped max density.

    Returns
    -------
    List of FluorineResult, one per F atom in the molecule (in file order).
    """
    surface = build_accessible_surface(mol, probe_radius, points_per_atom)
    if len(surface) == 0:
        raise ValueError(
            "No accessible surface points found — check the input "
            "geometry (overlapping atoms?) or increase points_per_atom."
        )

    results = []
    for i, sym in enumerate(mol.symbols):
        if sym != "F":
            continue
        f_coord = mol.coords[i]
        packed = pack_hydrogens(
            surface, seed_point=f_coord, probe_radius=probe_radius,
            target_density=target_density,
        )

        d = np.linalg.norm(packed - f_coord, axis=1)
        sum_inv_r6 = float(np.sum(d ** -6)) if len(d) else 0.0
        r_eff = sum_inv_r6 ** (-1.0 / 6.0) if sum_inv_r6 > 0 else np.nan
        r_mean = float(np.mean(d)) if len(d) else np.nan

        r2_packed = float("nan")
        if B0 is not None and tau_c is not None and sum_inv_r6 > 0:
            r2_packed = R2_DD_from_sum_inv_r6(
                B0, tau_c, sum_inv_r6, nuc_obs=nuc_obs, nuc_partner=nuc_partner
            )

        results.append(FluorineResult(
            index=i,
            coord=f_coord,
            distances=d,
            r_mean=r_mean,
            sum_inv_r6=sum_inv_r6,
            r_eff=r_eff,
            n_packed_H=len(d),
            R2_DD_packed=r2_packed,
            packing_density=target_density,
            empirical_density=estimate_packed_density(packed),
        ))

    if not results:
        raise ValueError("No fluorine (F) atoms found in the input structure.")

    return results


# ============================================================================
# Reporting / plotting
# ============================================================================

def print_report(
    mol: Molecule,
    results: list[FluorineResult],
    MW_kDa: float | None = None,
    B0: float | None = None,
    structural_sigma0: float | None = None,
) -> None:
    header = "─" * 66
    print(f"\n{header}")
    print(f"  Packed-H surface model  │  {mol.comment or '(no comment line)'}")
    print(f"  {len(mol.symbols)} atoms,  {len(results)} fluorine atom(s)")
    print(header)

    R_F = vdw_radius("F")
    I_flat = integral_I(R_F)
    r_eff_flat = r_eff_from_integral(R_F)
    print(f"  Flat-surface model (compare_R2DD.py), for reference:")
    print(f"    I(flat), σ₀=0.01 Å⁻²   = {I_flat:.4e} Å⁻⁶")
    print(f"    r_eff(flat)             = {r_eff_flat:.3f} Å")
    if structural_sigma0 is not None:
        I_struct = integral_I(R_F, sigma0=structural_sigma0)
        r_eff_struct = r_eff_from_integral(R_F, sigma0=structural_sigma0)
        print(f"    I(flat), σ₀={structural_sigma0:.4f} Å⁻² = {I_struct:.4e} Å⁻⁶  (structural)")
        print(f"    r_eff(flat, structural) = {r_eff_struct:.3f} Å")
    if B0 is not None and MW_kDa is not None:
        tau_c = tau_c_from_MW(MW_kDa)
        r2_flat = R2_DD_integral(B0, tau_c, R_F)
        r2_rule = R2_DD_rulethumh(MW_kDa)
        print(f"    R2,DD (flat surface)  = {r2_flat:.3f} s⁻¹   "
              f"(B0={B0:.1f} T, MW={MW_kDa:.0f} kDa, τc={tau_c*1e9:.2f} ns)")
        print(f"    R2,DD (rule-of-thumb) = {r2_rule:.3f} s⁻¹")
        if structural_sigma0 is not None:
            r2_struct = R2_DD_integral(B0, tau_c, R_F, sigma0=structural_sigma0)
            print(f"    R2,DD (flat surface, structural σ₀) = {r2_struct:.3f} s⁻¹")
    print(header)

    for res in results:
        print(f"\n  F atom #{res.index}  at ({res.coord[0]:.3f}, "
              f"{res.coord[1]:.3f}, {res.coord[2]:.3f})")
        print(f"    Packed H atoms on accessible surface : {res.n_packed_H}")
        cap_str = f"capped at {res.packing_density:.4f} Å⁻²" if res.packing_density else "uncapped (max density)"
        print(f"    Packing mode                          : {cap_str}")
        if np.isnan(res.empirical_density):
            print(f"    Empirical packed density              : n/a (fewer than 2 packed H atoms)")
        else:
            print(f"    Empirical packed density              : {res.empirical_density:.4f} Å⁻²  "
                  f"({res.empirical_density / SIGMA0:.2f}× flat-model σ₀)")
        print(f"    Mean F···H distance   r_mean          : {res.r_mean:.3f} Å")
        print(f"    Σᵢ 1/rᵢ⁶ (packed)                     : {res.sum_inv_r6:.4e} Å⁻⁶")
        print(f"    Equivalent single-H distance r_eff    : {res.r_eff:.3f} Å")
        print(f"    Ratio  Σ1/r⁶ (packed) / I (flat model): {res.sum_inv_r6 / I_flat:.3f}")
        if not np.isnan(res.R2_DD_packed):
            print(f"    R2,DD (packed surface)                : {res.R2_DD_packed:.3f} s⁻¹")
    print(f"\n{header}\n")


def plot_model_comparison(
    results: list[FluorineResult],
    MW_kDa: float,
    B0_ref: float,
    B0_range: tuple[float, float] = (7.0, 28.2),
    nuc_obs: str = "19F",
    nuc_partner: str = "1H",
    results_capped: list[FluorineResult] | None = None,
    structural_sigma0: float | None = None,
    results_capped_structural: list[FluorineResult] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Compare R2,DD models side by side:
      (A) Rule-of-thumb            R2,DD ≈ MW [kDa] s⁻¹
      (B) Flat-surface integral    (compare_R2DD.py, mean-field σ0=0.01 Å⁻²
          on an infinite plane)
      (C) Packed molecular surface, max density (this module; mean over all
          F atoms in the molecule, shaded band = min–max across F atoms)
      (D) Packed molecular surface, capped at σ0 — optional, pass
          `results_capped` (analyze_fluorines run with target_density=SIGMA0)
          to isolate the curvature/finite-size effect from the
          packing-density effect.
      (E) Flat-surface integral evaluated at a STRUCTURALLY-DERIVED σ0 —
          optional, pass `structural_sigma0` (e.g. the exposed-proton
          areal density measured directly from a real PDB structure via
          sigma0_from_pdb.py, ≈0.062 Å⁻² for 8AWW) instead of the
          mean-field 0.01 Å⁻² used in model (B). This isolates the effect
          of the density value itself, on the same flat-plane geometry as
          (B), from the curvature/packing effects isolated by (C)/(D).
      (F) Packed molecular surface, capped at the STRUCTURAL σ0 — optional,
          pass `results_capped_structural` (analyze_fluorines run with
          target_density=structural_sigma0) to isolate the curvature/
          finite-size effect at the density actually measured from a real
          structure, rather than the mean-field density used by (D).

    When the molecule has several F atoms, (C), (D), and (F) are
    summarized as a mean curve/bar with a shaded band / error bar spanning
    the per-atom min–max, rather than one series per atom, so the figure
    stays readable regardless of how many F atoms are present.

    Two panels:
      Left  : R2,DD vs B0, all models overlaid.
      Right : bar chart of R2,DD at B0_ref for all models.

    Parameters
    ----------
    results : list[FluorineResult]
        Output of analyze_fluorines with uncapped (max-density) packing.
    results_capped : list[FluorineResult], optional
        Output of analyze_fluorines with target_density=SIGMA0 (or another
        cap), same F atoms/order as `results`. If given, adds model (D).
    structural_sigma0 : float, optional
        Areal H density (Å⁻²) measured from a real structure. If given,
        adds model (E): the flat-surface integral re-evaluated at this
        density instead of the mean-field SIGMA0.
    results_capped_structural : list[FluorineResult], optional
        Output of analyze_fluorines with target_density=structural_sigma0,
        same F atoms/order as `results`. If given, adds model (F).
    MW_kDa : float
        Protein MW used for the rule-of-thumb model and for tau_c (via
        tau_c_from_MW), so all models describe the same complex.
    B0_ref : float
        Reference field (T) for the bar-chart panel.
    B0_range : (float, float)
        Field range (T) for the sweep panel.
    """
    R_F = vdw_radius("F")
    tau_c = tau_c_from_MW(MW_kDa)
    B0_vals = np.linspace(*B0_range, 200)

    colors = {"thumb": "#DC2626", "flat": "#2563EB", "packed": "#16A34A",
              "capped": "#7C3AED", "structural": "#EA580C", "capped_structural": "#0891B2"}

    def _r2_curves(res_list, B_vals):
        """(n_F, n_B) array of R2,DD(B) for every F atom with sum_inv_r6>0."""
        rows = []
        for res in res_list:
            if res.sum_inv_r6 <= 0:
                continue
            rows.append([
                R2_DD_from_sum_inv_r6(B, tau_c, res.sum_inv_r6, nuc_obs, nuc_partner)
                for B in B_vals
            ])
        return np.array(rows) if rows else np.zeros((0, len(B_vals)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: R2,DD vs B0 ────────────────────────────────────────────────
    ax = axes[0]
    R2_rule = R2_DD_rulethumh(MW_kDa)
    R2_flat = np.array([R2_DD_integral(B, tau_c, R_F, nuc_obs, nuc_partner) for B in B0_vals])
    nu_vals = B0_vals * GAMMA[nuc_partner] / (2 * np.pi * 1e6)

    ax.axhline(R2_rule, color=colors["thumb"], lw=2, ls="--",
               label=f"(A) Rule-of-thumb  ({R2_rule:.1f} s⁻¹)")
    ax.plot(nu_vals, R2_flat, color=colors["flat"], lw=2,
            label="(B) Flat surface, σ₀=0.01 Å⁻² (mean-field)")

    packed_curves = _r2_curves(results, B0_vals)
    if len(packed_curves):
        mean_c, min_c, max_c = packed_curves.mean(0), packed_curves.min(0), packed_curves.max(0)
        n_f = len(packed_curves)
        ax.plot(nu_vals, mean_c, color=colors["packed"], lw=2, ls="-.",
                label=f"(C) Packed, max density (mean of {n_f} F atom(s))")
        if n_f > 1:
            ax.fill_between(nu_vals, min_c, max_c, color=colors["packed"], alpha=0.15)

    if results_capped is not None:
        capped_curves = _r2_curves(results_capped, B0_vals)
        if len(capped_curves):
            mean_c, min_c, max_c = capped_curves.mean(0), capped_curves.min(0), capped_curves.max(0)
            n_f = len(capped_curves)
            ax.plot(nu_vals, mean_c, color=colors["capped"], lw=2, ls=":",
                    label=f"(D) Packed, σ₀-capped (mean of {n_f} F atom(s))")
            if n_f > 1:
                ax.fill_between(nu_vals, min_c, max_c, color=colors["capped"], alpha=0.15)

    if structural_sigma0 is not None:
        R2_structural = np.array([
            R2_DD_integral(B, tau_c, R_F, nuc_obs, nuc_partner, sigma0=structural_sigma0)
            for B in B0_vals
        ])
        ax.plot(nu_vals, R2_structural, color=colors["structural"], lw=2, ls="--",
                label=f"(E) Flat surface, σ₀={structural_sigma0:.3f} Å⁻² (structural)")

    if results_capped_structural is not None:
        capped_struct_curves = _r2_curves(results_capped_structural, B0_vals)
        if len(capped_struct_curves):
            mean_c, min_c, max_c = (capped_struct_curves.mean(0), capped_struct_curves.min(0),
                                     capped_struct_curves.max(0))
            n_f = len(capped_struct_curves)
            sigma_label = f"{structural_sigma0:.3f}" if structural_sigma0 is not None else "structural"
            ax.plot(nu_vals, mean_c, color=colors["capped_structural"], lw=2, ls=(0, (3, 1, 1, 1)),
                    label=f"(F) Packed, σ₀={sigma_label}-capped (mean of {n_f} F atom(s))")
            if n_f > 1:
                ax.fill_between(nu_vals, min_c, max_c, color=colors["capped_structural"], alpha=0.15)

    ax.axvline(B0_ref * GAMMA[nuc_partner] / (2 * np.pi * 1e6), color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"$\nu_H$ (MHz)")
    ax.set_ylabel(r"$R_{2,\rm DD,b}$ (s⁻¹)")
    ax.set_title(f"R₂,DD vs B₀  (MW={MW_kDa:.0f} kDa, τc={tau_c*1e9:.2f} ns)")
    ax.legend(fontsize=7.5)

    # ── Right: bar comparison at B0_ref ─────────────────────────────────
    ax = axes[1]

    def _r2_at_ref(res_list):
        vals = np.array([
            R2_DD_from_sum_inv_r6(B0_ref, tau_c, res.sum_inv_r6, nuc_obs, nuc_partner)
            for res in res_list if res.sum_inv_r6 > 0
        ])
        return vals

    labels = ["(A) Rule-\nof-thumb", "(B) Flat\nσ₀=0.01", "(C) Packed\nmax density"]
    values = [R2_rule, R2_DD_integral(B0_ref, tau_c, R_F, nuc_obs, nuc_partner)]
    errs   = [0.0, 0.0]
    bar_colors = [colors["thumb"], colors["flat"], colors["packed"]]

    vals_c = _r2_at_ref(results)
    values.append(vals_c.mean() if len(vals_c) else 0.0)
    errs.append([[values[-1] - vals_c.min()], [vals_c.max() - values[-1]]] if len(vals_c) > 1 else 0.0)

    if results_capped is not None:
        labels.append("(D) Packed\nσ₀-capped")
        bar_colors.append(colors["capped"])
        vals_d = _r2_at_ref(results_capped)
        values.append(vals_d.mean() if len(vals_d) else 0.0)
        errs.append([[values[-1] - vals_d.min()], [vals_d.max() - values[-1]]] if len(vals_d) > 1 else 0.0)

    if structural_sigma0 is not None:
        labels.append(f"(E) Flat\nσ₀={structural_sigma0:.3f}")
        bar_colors.append(colors["structural"])
        values.append(R2_DD_integral(B0_ref, tau_c, R_F, nuc_obs, nuc_partner,
                                       sigma0=structural_sigma0))
        errs.append(0.0)

    if results_capped_structural is not None:
        sigma_label = f"{structural_sigma0:.3f}" if structural_sigma0 is not None else "struct"
        labels.append(f"(F) Packed\nσ₀={sigma_label}-capped")
        bar_colors.append(colors["capped_structural"])
        vals_f = _r2_at_ref(results_capped_structural)
        values.append(vals_f.mean() if len(vals_f) else 0.0)
        errs.append([[values[-1] - vals_f.min()], [vals_f.max() - values[-1]]] if len(vals_f) > 1 else 0.0)

    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white")
    for i, (bar, v, e) in enumerate(zip(bars, values, errs)):
        if isinstance(e, list):
            ax.errorbar(bar.get_x() + bar.get_width() / 2, v, yerr=e,
                        color="0.2", capsize=4, lw=1.2)
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)

    nu_ref = B0_ref * GAMMA[nuc_partner] / (2 * np.pi * 1e6)
    ax.set_ylabel(r"$R_{2,\rm DD,b}$ (s⁻¹)")
    n_f_total = len(results)
    ax.set_title(f"Model comparison at {nu_ref:.0f} MHz {nuc_partner}, "
                 f"{B0_ref:.2f} T  (n={n_f_total} F atom(s), error bars = min–max)")
    ax.tick_params(axis="x", labelsize=7.5)

    fig.suptitle(f"R₂,DD,b: rule-of-thumb vs flat surface vs packed molecular "
                 f"surface  ({nuc_obs}–{nuc_partner})", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_distance_histograms(
    results: list[FluorineResult],
    save_path: str | None = None,
) -> None:
    """One histogram panel per F atom, showing the packed F-H distance
    distribution, with r_mean and r_eff marked."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)
    axes = axes[0]

    for ax, res in zip(axes, results):
        ax.hist(res.distances, bins=30, color="#2563EB", alpha=0.75,
                edgecolor="white")
        ax.axvline(res.r_mean, color="#16A34A", lw=2, ls="--",
                   label=f"r_mean = {res.r_mean:.2f} Å")
        ax.axvline(res.r_eff, color="#DC2626", lw=2, ls="-.",
                   label=f"r_eff (Σ1/r⁶) = {res.r_eff:.2f} Å")
        ax.set_xlabel("F···H distance (Å)")
        ax.set_ylabel("Packed H count")
        ax.set_title(f"F atom #{res.index}  (N_H = {res.n_packed_H})")
        ax.legend(fontsize=8)

    fig.suptitle("Packed-surface F···H distance distributions", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python surface_packing.py molecule.xyz [MW_kDa] [B0_tesla] [structural_sigma0]")
        sys.exit(1)

    xyz_path = sys.argv[1]
    MW_kDa = float(sys.argv[2]) if len(sys.argv) > 2 else 23.0   # default: trypsin
    B0     = float(sys.argv[3]) if len(sys.argv) > 3 else 16.4   # default: 700 MHz 1H
    # Structurally-derived exposed-proton areal density, e.g. from
    # sigma0_from_pdb.py on a real PDB structure (8AWW: ≈0.062 Å⁻²,
    # ~6.2x the mean-field SIGMA0=0.01 Å⁻² used elsewhere in this project).
    structural_sigma0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.062

    mol = read_xyz(xyz_path)
    tau_c = tau_c_from_MW(MW_kDa)

    # (C) Uncapped: maximum density the geometry allows.
    results = analyze_fluorines(
        mol, probe_radius=R_H_SURF, points_per_atom=400, B0=B0, tau_c=tau_c
    )
    print_report(mol, results, MW_kDa=MW_kDa, B0=B0, structural_sigma0=structural_sigma0)

    # (D) Capped at the same areal H density used by the flat-surface model
    # (SIGMA0), for a density-matched, apples-to-apples comparison that
    # isolates the curvature/finite-size effect from the packing-density
    # effect.
    results_capped = analyze_fluorines(
        mol, probe_radius=R_H_SURF, points_per_atom=400, B0=B0, tau_c=tau_c,
        target_density=SIGMA0,
    )
    print("  Density-capped packing (target = flat-model σ₀):")
    print_report(mol, results_capped, MW_kDa=MW_kDa, B0=B0)

    # (F) Capped at the structurally-derived exposed-proton density instead
    # of the mean-field SIGMA0, isolating the curvature/finite-size effect
    # at the density actually measured from a real structure.
    results_capped_structural = analyze_fluorines(
        mol, probe_radius=R_H_SURF, points_per_atom=400, B0=B0, tau_c=tau_c,
        target_density=structural_sigma0,
    )
    print(f"  Density-capped packing (target = structural σ₀={structural_sigma0:.4f}):")
    print_report(mol, results_capped_structural, MW_kDa=MW_kDa, B0=B0)

    plot_distance_histograms(results, save_path="packed_H_distances.png")
    plot_model_comparison(results, MW_kDa=MW_kDa, B0_ref=B0,
                           results_capped=results_capped,
                           structural_sigma0=structural_sigma0,
                           results_capped_structural=results_capped_structural,
                           save_path="R2DD_model_comparison.png")
