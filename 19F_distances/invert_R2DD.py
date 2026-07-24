"""
invert_R2DD.py
================
Invert the forward R2,DD models in compare_R2DD.py / surface_packing.py:
given a MEASURED dipolar relaxation rate R2,DD (plus known B0, tau_c),
estimate either

  1. Sigma(1/r^6)  -- exact, closed-form (R2,DD is exactly linear in it);
  2. z             -- the flat-surface distance from the observed nucleus
                       to the hydrogen plane, at an ASSUMED sigma0;
  3. sigma0 (flat) -- the flat-surface hydrogen areal density, at an
                       ASSUMED z (or R_X);
  4. sigma0(packed)-- the packed-molecular-surface hydrogen areal density
                       that reproduces the measured R2,DD on a given real
                       molecular geometry (numerical, via bisection on the
                       existing forward model in surface_packing.py).

Why this is well-posed
-----------------------
R2_DD_from_sum_inv_r6(B0, tau_c, S) is EXACTLY LINEAR in S = Sigma(1/r^6):

    R2,DD = S * [ (mu0/4pi)^2 hbar^2 gI^2 gS^2 * 1e60 / 2 ] * spectral_sum(B0, tau_c)

so step 1 is a single division, not a fit. Steps 2-3 are then closed-form
since the flat-surface model I(z) = pi*sigma0/(2 z^4) is analytically
invertible in either variable alone. Step 4 has no closed form (packing is
a discrete greedy geometric process) so it is solved numerically.

What is NOT identifiable from R2,DD alone
-------------------------------------------
A single scalar measurement (R2,DD at one field, for one nucleus) can only
constrain ONE unknown at a time. Sigma(1/r^6) is a single number; z and
sigma0 trade off against each other in the flat model (a closer, sparser
surface gives the same signal as a farther, denser one), and packed-model
density trades off against which molecule/geometry is assumed. Recovering
both z AND sigma0 independently requires either an assumed value for one
of them (as implemented here) or additional independent measurements
(e.g. R2,DD at a second field strength does NOT help, since the field
dependence factors out identically for every model -- see the field
ratio panel in compare_R2DD.make_figure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compare_R2DD import (
    GAMMA, R_H_SURF, SIGMA0,
    R2_DD_from_sum_inv_r6, tau_c_from_MW,
)


# ============================================================================
# Step 1: R2,DD -> Sigma(1/r^6)   (exact, closed form)
# ============================================================================

def sum_inv_r6_from_R2(
    R2_measured: float,
    B0: float,
    tau_c: float,
    nuc_obs: str = "19F",
    nuc_partner: str = "1H",
) -> float:
    """
    Invert R2_DD_from_sum_inv_r6 for Sigma(1/r^6) (Å⁻⁶), given a measured
    R2,DD (s⁻¹) and known B0/tau_c. Exact: R2,DD is linear in Sigma(1/r^6),
    so this is R2_measured / R2_DD_from_sum_inv_r6(B0, tau_c, 1.0, ...).

    Parameters
    ----------
    R2_measured : float
        Measured (or assumed) R2,DD,b in s⁻¹.
    B0 : float
        Magnetic field strength in Tesla.
    tau_c : float
        Rotational correlation time in seconds.
    nuc_obs, nuc_partner : str
        Nucleus labels (default ¹⁹F observed, ¹H partner).

    Returns
    -------
    float
        Sigma(1/r^6) in Å⁻⁶ consistent with the measured rate.
    """
    R2_per_unit = R2_DD_from_sum_inv_r6(B0, tau_c, 1.0, nuc_obs, nuc_partner)
    return R2_measured / R2_per_unit


# ============================================================================
# Step 2/3: flat-surface model -- invert I = pi*sigma0/(2 z^4)
# ============================================================================

def z_from_sum_inv_r6(sum_inv_r6: float, sigma0: float = SIGMA0) -> float:
    """
    Invert the flat-surface integral for the nucleus-to-plane distance z
    (Å), given Sigma(1/r^6) (Å⁻⁶) and an ASSUMED areal H density sigma0
    (Å⁻²):

        I = pi*sigma0 / (2 z^4)   =>   z = ( pi*sigma0 / (2*I) )^(1/4)

    Returns NaN if sum_inv_r6 <= 0 (no finite z reproduces a
    non-positive/zero signal in this model).
    """
    if sum_inv_r6 <= 0:
        return float("nan")
    return (np.pi * sigma0 / (2.0 * sum_inv_r6)) ** 0.25


def R_X_from_sum_inv_r6(sum_inv_r6: float, sigma0: float = SIGMA0,
                         R_H: float = R_H_SURF) -> float:
    """
    Same as z_from_sum_inv_r6, but returns the implied van der Waals
    radius of the observed nucleus, R_X = z - R_H, instead of z itself.
    """
    z = z_from_sum_inv_r6(sum_inv_r6, sigma0)
    return z - R_H if not np.isnan(z) else float("nan")


def sigma0_from_sum_inv_r6_flat(sum_inv_r6: float, z: float | None = None,
                                  R_X: float | None = None,
                                  R_H: float = R_H_SURF) -> float:
    """
    Invert the flat-surface integral for the areal H density sigma0
    (Å⁻²), given Sigma(1/r^6) (Å⁻⁶) and an ASSUMED distance -- either z
    directly, or R_X (with z = R_H + R_X):

        I = pi*sigma0 / (2 z^4)   =>   sigma0 = 2 z^4 I / pi

    Exactly one of z or R_X must be given.
    """
    if (z is None) == (R_X is None):
        raise ValueError("Provide exactly one of z or R_X.")
    if z is None:
        z = R_H + R_X
    return 2.0 * z**4 * sum_inv_r6 / np.pi


# ============================================================================
# Step 4: packed-molecular-surface model -- numerical inversion for sigma0
# ============================================================================

@dataclass
class PackedInversionResult:
    sigma0: float                  # Å⁻², solved packing density (NaN if out of range)
    sum_inv_r6_target: float       # Å⁻⁶, the value being matched
    sum_inv_r6_achieved: float     # Å⁻⁶, actually achieved at the solved sigma0
    sigma0_floor: float            # Å⁻², density at which packing saturates to a single contact
    sigma0_ceiling: float          # Å⁻², density at which packing saturates to max density
    sum_inv_r6_floor: float        # Å⁻⁶, Sigma(1/r^6) at the single-contact floor
    sum_inv_r6_ceiling: float      # Å⁻⁶, Sigma(1/r^6) at max-density packing
    status: str                     # "ok", "below_floor", "above_ceiling"


def sigma0_from_sum_inv_r6_packed(
    sum_inv_r6_target: float,
    mol,
    probe_radius: float = R_H_SURF,
    points_per_atom: int = 400,
    sigma0_bracket: tuple[float, float] = (1e-4, 0.25),
    tol: float = 1e-4,
    max_iter: int = 60,
) -> PackedInversionResult:
    """
    Numerically invert the packed-molecular-surface model for the areal H
    packing density sigma0 (Å⁻²) that reproduces a target Sigma(1/r^6)
    (Å⁻⁶), for a given molecule (mol from surface_packing.read_xyz).

    Sigma(1/r^6) is a monotonically non-decreasing function of sigma0 in
    this model (denser packing -> more/closer hydrogens -> larger sum),
    but has no closed form (packing is a discrete greedy geometric
    process), so this uses bisection on the existing forward model
    (surface_packing.analyze_fluorines) rather than an analytic formula.

    The achievable range is bounded:
      - a FLOOR at the lowest sigma0 that still places one hydrogen (the
        single nearest van der Waals contact point) -- Sigma(1/r^6) cannot
        go below this no matter how sparse the requested density;
      - a CEILING at the hard-sphere maximum packing density -- Sigma(1/r^6)
        cannot exceed this no matter how dense the requested density.
    If the target falls outside [floor, ceiling], no sigma0 reproduces it
    on this molecule and the corresponding bound is reported instead.

    Parameters
    ----------
    sum_inv_r6_target : float
        The Sigma(1/r^6) value to match (Å⁻⁶), e.g. from
        sum_inv_r6_from_R2.
    mol : surface_packing.Molecule
        Molecule to pack hydrogens onto (must contain at least one F atom;
        uses the mean over all F atoms present, matching
        plot_model_comparison's convention).
    sigma0_bracket : (float, float)
        Initial (lo, hi) bracket for bisection, Å⁻². Widen if you expect
        sigma0 outside this range.
    tol : float
        Relative tolerance on sigma0 for convergence.

    Returns
    -------
    PackedInversionResult
    """
    from surface_packing import analyze_fluorines

    def mean_sum_inv_r6(sigma0):
        results = analyze_fluorines(
            mol, probe_radius=probe_radius, points_per_atom=points_per_atom,
            target_density=sigma0,
        )
        vals = [r.sum_inv_r6 for r in results if r.sum_inv_r6 > 0]
        return float(np.mean(vals)) if vals else 0.0

    lo, hi = sigma0_bracket
    f_lo = mean_sum_inv_r6(lo)
    f_hi = mean_sum_inv_r6(hi)

    if sum_inv_r6_target <= f_lo:
        return PackedInversionResult(
            sigma0=float("nan"), sum_inv_r6_target=sum_inv_r6_target,
            sum_inv_r6_achieved=f_lo, sigma0_floor=lo, sigma0_ceiling=hi,
            sum_inv_r6_floor=f_lo, sum_inv_r6_ceiling=f_hi,
            status="below_floor",
        )
    if sum_inv_r6_target >= f_hi:
        return PackedInversionResult(
            sigma0=float("nan"), sum_inv_r6_target=sum_inv_r6_target,
            sum_inv_r6_achieved=f_hi, sigma0_floor=lo, sigma0_ceiling=hi,
            sum_inv_r6_floor=f_lo, sum_inv_r6_ceiling=f_hi,
            status="above_ceiling",
        )

    # Bisection: mean_sum_inv_r6 is monotonic non-decreasing in sigma0.
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = mean_sum_inv_r6(mid)
        if f_mid < sum_inv_r6_target:
            lo = mid
        else:
            hi = mid
        if (hi - lo) / hi < tol:
            break

    sigma0_solution = 0.5 * (lo + hi)
    achieved = mean_sum_inv_r6(sigma0_solution)

    return PackedInversionResult(
        sigma0=sigma0_solution, sum_inv_r6_target=sum_inv_r6_target,
        sum_inv_r6_achieved=achieved,
        sigma0_floor=sigma0_bracket[0], sigma0_ceiling=sigma0_bracket[1],
        sum_inv_r6_floor=f_lo, sum_inv_r6_ceiling=f_hi,
        status="ok",
    )


# ============================================================================
# Convenience: full pipeline from a measured R2,DD
# ============================================================================

@dataclass
class InversionSummary:
    R2_measured: float
    sum_inv_r6: float
    z_flat: float                        # Å, at assumed sigma0=SIGMA0
    sigma0_flat: float                   # Å⁻², at assumed R_X (Bondi F)
    packed: "PackedInversionResult | None"


def invert_R2_measurement(
    R2_measured: float,
    B0: float,
    MW_kDa: float,
    R_X_assumed: float,
    mol=None,
    nuc_obs: str = "19F",
    nuc_partner: str = "1H",
    sigma0_assumed: float = SIGMA0,
) -> InversionSummary:
    """
    Run the full inversion pipeline on one measured R2,DD value:
      1. Sigma(1/r^6) from R2 (exact).
      2. z from Sigma(1/r^6), assuming sigma0 = sigma0_assumed (flat model).
      3. sigma0 from Sigma(1/r^6), assuming R_X = R_X_assumed (flat model).
      4. If a molecule is given, sigma0 from Sigma(1/r^6) on the packed
         molecular surface (numerical).

    See the module docstring for why z and sigma0 cannot both be
    recovered independently from a single R2,DD measurement.
    """
    tau_c = tau_c_from_MW(MW_kDa)
    S = sum_inv_r6_from_R2(R2_measured, B0, tau_c, nuc_obs, nuc_partner)

    z = z_from_sum_inv_r6(S, sigma0=sigma0_assumed)
    sigma0_flat = sigma0_from_sum_inv_r6_flat(S, R_X=R_X_assumed)

    packed = None
    if mol is not None:
        packed = sigma0_from_sum_inv_r6_packed(S, mol)

    return InversionSummary(
        R2_measured=R2_measured, sum_inv_r6=S,
        z_flat=z, sigma0_flat=sigma0_flat, packed=packed,
    )


def print_inversion_report(summary: InversionSummary, R_X_assumed: float | None = None) -> None:
    header = "─" * 62
    print(f"\n{header}")
    print("  R2,DD inversion")
    print(header)
    print(f"  Measured R2,DD                          : {summary.R2_measured:.3f} s⁻¹")
    print(f"  Implied Σᵢ 1/rᵢ⁶ (exact)                 : {summary.sum_inv_r6:.4e} Å⁻⁶")
    print(header)
    print("  Flat-surface model:")
    print(f"    z  (assuming σ₀={SIGMA0:.4f} Å⁻²)         : {summary.z_flat:.3f} Å")
    if R_X_assumed is not None:
        z_contact = R_H_SURF + R_X_assumed
        if not np.isnan(summary.z_flat) and summary.z_flat < z_contact:
            print(f"      ⚠ below hard van der Waals contact (z_min = R_H+R_X = "
                  f"{z_contact:.3f} Å) -- not physically achievable at this σ₀;")
            print(f"        the assumed σ₀ is too low for this measurement, or the "
                  f"measurement/assumptions should be reconsidered.")
    print(f"    σ₀ (assuming given R_X)                 : {summary.sigma0_flat:.4e} Å⁻²")
    if summary.packed is not None:
        p = summary.packed
        print(header)
        print("  Packed-molecular-surface model:")
        if p.status == "ok":
            print(f"    σ₀ (numerical)                          : {p.sigma0:.4e} Å⁻²")
            print(f"    Σ1/r⁶ achieved at that σ₀               : {p.sum_inv_r6_achieved:.4e} Å⁻⁶")
        elif p.status == "below_floor":
            print(f"    Target Σ1/r⁶ is BELOW the achievable range on this molecule --")
            print(f"    even a single hydrogen at the nearest van der Waals contact point")
            print(f"    already produces more signal than this measurement implies.")
            print(f"    Floor Σ1/r⁶ (single contact)             : {p.sum_inv_r6_floor:.4e} Å⁻⁶")
            print(f"    Interpretation: consistent with no real intermolecular contact on")
            print(f"    this geometry, or a more distant/shielded contact than this")
            print(f"    molecule's own surface can represent.")
        else:
            print(f"    Target Σ1/r⁶ is ABOVE the achievable range on this molecule --")
            print(f"    even hard-sphere maximum-density packing cannot reproduce this")
            print(f"    measurement (Σ1/r⁶ ceiling = {p.sum_inv_r6_ceiling:.4e} Å⁻⁶).")
            print(f"    Interpretation: the true contact is likely closer/more numerous")
            print(f"    than any geometry on this molecule allows -- check R2_measured,")
            print(f"    tau_c, or whether a different/larger molecule should be used.")
    print(f"{header}\n")


if __name__ == "__main__":
    import sys
    from surface_packing import read_xyz

    if len(sys.argv) < 2:
        print("Usage: python invert_R2DD.py R2_measured [MW_kDa] [B0_tesla] [molecule.xyz]")
        sys.exit(1)

    R2_meas = float(sys.argv[1])
    MW_kDa = float(sys.argv[2]) if len(sys.argv) > 2 else 23.0
    B0 = float(sys.argv[3]) if len(sys.argv) > 3 else 16.4
    xyz_path = sys.argv[4] if len(sys.argv) > 4 else None

    from compare_R2DD import VDW_RADIUS
    mol = read_xyz(xyz_path) if xyz_path else None

    summary = invert_R2_measurement(
        R2_meas, B0, MW_kDa, R_X_assumed=VDW_RADIUS["F"], mol=mol,
    )
    print_inversion_report(summary, R_X_assumed=VDW_RADIUS["F"])
