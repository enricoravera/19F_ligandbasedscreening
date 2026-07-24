"""
recover_distance_from_NMRD.py
================================
Recover an effective distance from fitted homonuclear ¹H-¹H NMRD
relaxation dispersion data (Wang et al. 2021, JACS, as implemented in
homonuclear_R1_NMRD.py), using the SAME kind of closed-form inversion
logic as invert_R2DD.py -- but applied to the correct (homonuclear,
longitudinal) model, since invert_R2DD.py's equations were built for a
different physical model (heteronuclear ¹⁹F-¹H transverse R2 relaxation
from a SUM over many external protein protons).

Two routes are provided
-------------------------
ROUTE 1 (primary, exact): direct inversion of the paper's own bound-state
dipolar constant,

    A_B = S^2 * (1/64) * mu0^2 gammaH^4 hbar^2 / (pi^2 r_env^6)

for r_env, the effective distance between the small molecule's methyl
protons and a single representative proton of the macromolecular contact
environment. This is the distance the paper's OWN model is built around,
requires no assumptions borrowed from a different relaxation mechanism,
and is exact given A_B (i.e. given ApB and an assumed/known bound
population pB).

ROUTE 2 (bridge, approximate): re-expresses the fitted amplitude as an
equivalent Sigma(1/r^6)-like quantity and feeds it through invert_R2DD.py's
flat-surface/packed-surface distance and density inversions. This lets
you compare an NMRD-derived contact distance against the same
flat-surface-plane / packed-molecular-surface pictures used elsewhere in
this project. IMPORTANT CAVEAT: invert_R2DD.py's machinery was derived
for HETERONUCLEAR (¹⁹F observed, ¹H partner) TRANSVERSE (R2) relaxation
summed over MANY surface protons, using the gammaI^2*gammaS^2 combination
and the R2 spectral-density sum (Ernst et al. formula). Route 2 forces a
homonuclear, longitudinal, single-proton quantity through that machinery
by treating A_B*pB as if it were a Sigma(1/r^6)-equivalent measured via
the R2,DD formula instead of the paper's own R1 formula -- this is a
DIFFERENT relaxation mechanism and the distance recovered by Route 2 is
NOT expected to numerically equal r_env from Route 1. Route 2 is offered
for structural comparison only (e.g. "is this contact distance in the
same ballpark as the flat-surface picture"), not as an equally-valid
alternate calculation of the same physical quantity.

Usage
-----
    from recover_distance_from_NMRD import (
        r_env_from_AB, r_env_from_fit, bridge_to_R2DD_inversion,
    )

    # Route 1, from a two-state NMRD fit result and an assumed pB:
    fit = fit_nmrd_two_state(B0, R1, R1_err)
    result = r_env_from_fit(fit, p_B=0.01)

    # Route 2, structural comparison against the flat/packed models:
    bridged = bridge_to_R2DD_inversion(fit, p_B=0.01, mol=my_molecule)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import hbar, mu_0

from compare_R2DD import GAMMA, SIGMA0, R_H_SURF
from homonuclear_R1_NMRD import TwoStateFitResult, A_B_from_renv
from invert_R2DD import (
    z_from_sum_inv_r6, sigma0_from_sum_inv_r6_flat,
    sigma0_from_sum_inv_r6_packed, PackedInversionResult,
)


# ============================================================================
# ROUTE 1: direct inversion of the paper's own A_B for r_env (exact)
# ============================================================================

@dataclass
class REnvResult:
    r_env: float          # Å, the recovered effective distance
    A_B: float             # s⁻², de-weighted dipolar constant (ApB / pB)
    ApB: float              # s⁻², the fitted product (as returned by the fit)
    p_B_assumed: float      # the bound population used to de-weight ApB
    S2_assumed: float


def r_env_from_AB(A_B: float, S2: float = 1.0) -> float:
    """
    Invert the paper's bound-state dipolar constant (Eq. 3 prefactor) for
    the effective distance r_env (Å):

        A_B = S^2 * (1/64) * mu0^2 gammaH^4 hbar^2 / (pi^2 r_env^6)
        =>   r_env = [ S^2 * mu0^2 gammaH^4 hbar^2 / (64 pi^2 A_B) ]^(1/6)

    This is the exact algebraic inverse of homonuclear_R1_NMRD.A_B_from_renv;
    round-trips exactly (A_B_from_renv(r_env_from_AB(A_B)) == A_B).

    Parameters
    ----------
    A_B : float
        Bound-state dipolar constant, s⁻². NOT the fitted ApB product --
        that must first be divided by an assumed/known pB (see
        r_env_from_fit for the full pipeline including that step).
    S2 : float
        Order parameter assumed for the bound-state interaction. Must
        match whatever value was assumed when A_B was computed/fit.

    Returns
    -------
    float
        r_env in Å.
    """
    gH = GAMMA["1H"]
    r_env_SI = (S2 * mu_0**2 * gH**4 * hbar**2 / (64.0 * np.pi**2 * A_B)) ** (1.0 / 6.0)
    return r_env_SI * 1e10  # m -> Å


def r_env_from_fit(
    fit: TwoStateFitResult,
    p_B: float,
    S2: float = 1.0,
) -> REnvResult:
    """
    Full Route 1 pipeline: given a fitted TwoStateFitResult (from
    homonuclear_R1_NMRD.fit_nmrd_two_state, which returns the PRODUCT
    ApB = A_B*pB, since a single NMRD profile cannot separate the two)
    and an ASSUMED/independently-known bound population p_B, recover the
    effective bound-state distance r_env.

    This mirrors invert_R2DD.py's philosophy exactly: a single relaxation
    measurement (or, here, a single NMRD profile) constrains only one
    combined quantity (ApB); recovering a physical distance requires
    supplying the other unknown (here p_B, there sigma0 or R_X) as an
    assumption.

    Parameters
    ----------
    fit : TwoStateFitResult
        Output of homonuclear_R1_NMRD.fit_nmrd_two_state.
    p_B : float
        Assumed/known bound population of the small molecule (0 < p_B << 1,
        consistent with the paper's weak-binding hypothesis). This is NOT
        determined by the NMRD fit itself -- it must come from an
        independent source (concentrations, a competition/titration
        experiment, literature Kd, etc.).
    S2 : float
        Order parameter assumed for the bound-state interaction (default
        1.0, matching the paper's Figure 1 convention).

    Returns
    -------
    REnvResult
    """
    if not (0 < p_B < 1):
        raise ValueError("p_B must be in (0, 1).")
    A_B = fit.ApB / p_B
    r_env = r_env_from_AB(A_B, S2=S2)
    return REnvResult(
        r_env=r_env, A_B=A_B, ApB=fit.ApB, p_B_assumed=p_B, S2_assumed=S2,
    )


def print_r_env_report(result: REnvResult) -> None:
    header = "─" * 60
    print(f"\n{header}")
    print("  Route 1: direct r_env inversion (paper's own model)")
    print(header)
    print(f"  Fitted A_B·p_B                : {result.ApB*1e-6:.3f} × 10⁶ s⁻²")
    print(f"  Assumed p_B                   : {result.p_B_assumed:.4g}")
    print(f"  => A_B                        : {result.A_B*1e-6:.3f} × 10⁶ s⁻²")
    print(f"  Assumed S²                    : {result.S2_assumed:.3f}")
    print(f"  Recovered r_env               : {result.r_env:.3f} Å")
    print(header)
    print("  NOTE: r_env trades off exactly against p_B (r_env ∝ p_B^(-1/6)),")
    print("  so a 10x error in the assumed p_B only shifts r_env by ~1.47x.")
    print(f"{header}\n")


# ============================================================================
# ROUTE 2: bridge to invert_R2DD.py's flat/packed-surface machinery
#          (approximate, cross-mechanism -- see module docstring caveat)
# ============================================================================

@dataclass
class BridgedResult:
    sum_inv_r6_equiv: float          # Å⁻⁶, the ApB-derived "as-if-R2,DD" quantity
    z_flat: float                     # Å, flat-surface distance at sigma0=SIGMA0
    sigma0_flat: float                # Å⁻², flat-surface density at an assumed R_X
    packed: "PackedInversionResult | None"


def bridge_to_R2DD_inversion(
    fit: TwoStateFitResult,
    p_B: float,
    R_X_assumed: float = R_H_SURF,
    mol=None,
) -> BridgedResult:
    """
    ROUTE 2 (approximate, cross-mechanism -- read the module docstring
    caveat before using this). Re-expresses A_B (from the homonuclear R1
    fit, de-weighted by an assumed p_B) as an equivalent Sigma(1/r^6)-like
    quantity in the SAME units invert_R2DD.py expects, by inverting A_B
    for r_env (Route 1) and then treating r_env as if it were a single
    heteronuclear contact distance z, i.e. sum_inv_r6_equiv = r_env^-6.

    This is offered purely so an NMRD-derived contact distance can be
    placed on the same flat-surface / packed-surface plots and tables as
    the rest of this project, NOT as a rigorous re-derivation -- it
    silently assumes the homonuclear bound-state contact distance r_env
    is structurally comparable to a heteronuclear surface-integral contact
    distance, which is a modeling choice, not a physical identity.

    Parameters
    ----------
    fit, p_B : see r_env_from_fit.
    R_X_assumed : float
        Van der Waals radius of the (hypothetical) observed nucleus for
        the flat-surface sigma0 inversion, Å. Defaults to R_H_SURF (i.e.
        treating the "observed nucleus" as itself proton-sized, since
        this is a homonuclear ¹H measurement).
    mol : surface_packing.Molecule, optional
        If given, also runs the packed-molecular-surface inversion
        (sigma0_from_sum_inv_r6_packed) on this equivalent quantity.

    Returns
    -------
    BridgedResult
    """
    r1_result = r_env_from_fit(fit, p_B)
    sum_inv_r6_equiv = r1_result.r_env ** -6   # Å⁻⁶, treating r_env as a contact distance

    z_flat = z_from_sum_inv_r6(sum_inv_r6_equiv, sigma0=SIGMA0)
    sigma0_flat = sigma0_from_sum_inv_r6_flat(sum_inv_r6_equiv, R_X=R_X_assumed)

    packed = None
    if mol is not None:
        packed = sigma0_from_sum_inv_r6_packed(sum_inv_r6_equiv, mol)

    return BridgedResult(
        sum_inv_r6_equiv=sum_inv_r6_equiv, z_flat=z_flat,
        sigma0_flat=sigma0_flat, packed=packed,
    )


def print_bridged_report(bridged: BridgedResult, r_env_route1: float) -> None:
    header = "─" * 60
    print(f"\n{header}")
    print("  Route 2: bridged to invert_R2DD.py flat/packed models")
    print("  (approximate, cross-mechanism -- see module docstring)")
    print(header)
    print(f"  r_env (Route 1, for reference)      : {r_env_route1:.3f} Å")
    print(f"  Equivalent Σ(1/r⁶) = r_env⁻⁶        : {bridged.sum_inv_r6_equiv:.4e} Å⁻⁶")
    print(f"  Flat-surface z  (at σ₀={SIGMA0:.4f})   : {bridged.z_flat:.3f} Å")
    print(f"  Flat-surface σ₀ (at assumed R_X)     : {bridged.sigma0_flat:.4e} Å⁻²")
    if bridged.packed is not None:
        p = bridged.packed
        if p.status == "ok":
            print(f"  Packed-surface σ₀ (numerical)        : {p.sigma0:.4e} Å⁻²")
        else:
            bound = "below floor" if p.status == "below_floor" else "above ceiling"
            print(f"  Packed-surface σ₀: target {bound} of achievable range on this molecule")
    print(f"{header}\n")


# ============================================================================
# Convenience: full pipeline from raw NMRD data to distance
# ============================================================================

def recover_distance_from_nmrd_data(
    B0: np.ndarray,
    R1: np.ndarray,
    R1_err: np.ndarray | None,
    p_B: float,
    r_HH_AA: float = 1.78,
    S2_free: float = 0.25,
    S2_bound: float = 1.0,
    mol=None,
    R_X_assumed: float = R_H_SURF,
):
    """
    End-to-end convenience wrapper: fit raw multi-field NMRD data
    (B0, R1, R1_err) with the two-state weak-binding model, then run both
    Route 1 (direct r_env) and Route 2 (bridged flat/packed comparison).

    Returns
    -------
    (fit, r1_result, bridged_result)
    """
    from homonuclear_R1_NMRD import fit_nmrd_two_state

    fit = fit_nmrd_two_state(B0, R1, R1_err, r_HH_AA=r_HH_AA, S2=S2_free)
    r1_result = r_env_from_fit(fit, p_B=p_B, S2=S2_bound)
    bridged = bridge_to_R2DD_inversion(fit, p_B=p_B, R_X_assumed=R_X_assumed, mol=mol)

    return fit, r1_result, bridged


if __name__ == "__main__":
    import sys
    from homonuclear_R1_NMRD import (
        fit_nmrd_two_state, print_two_state_report,
        R1_effective_weak_binding, A_methyl_from_rHH,
    )

    if len(sys.argv) < 2:
        print("Usage: python recover_distance_from_NMRD.py p_B [demo]")
        print("  Runs a demo using synthetic lactate-like NMRD data if")
        print("  'demo' is passed or no data-loading is wired in.")
        print("  p_B: assumed bound population (e.g. 0.01 for ~1% bound)")
        sys.exit(1)

    p_B = float(sys.argv[1])

    # Demo: synthetic NMRD data resembling the paper's lactate/donor-2 case
    print("Running on synthetic demo NMRD data (paper's lactate/donor-2-like "
          "parameters: tau_F=22.8ps, tau_B=35.2ns, ApB=15.8e6 s^-2)\n")

    A_methyl = A_methyl_from_rHH(1.78, S2=0.25)
    B0_vals = np.array([14.1, 10.0, 5.0, 2.5, 1.25, 0.625, 0.33, 0.23, 0.16,
                         0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015])
    tau_F_true, tau_B_true, ApB_true = 22.8e-12, 35.2e-9, 15.8e6
    R1_true = np.array([
        R1_effective_weak_binding(B, tau_F_true, A_methyl, tau_B_true, ApB_true)
        for B in B0_vals
    ])
    rng = np.random.default_rng(42)
    R1_meas = R1_true + rng.normal(0, 0.08, size=len(B0_vals))
    R1_err = np.full_like(R1_meas, 0.1)

    fit, r1_result, bridged = recover_distance_from_nmrd_data(
        B0_vals, R1_meas, R1_err, p_B=p_B,
    )
    print_two_state_report(fit)
    print_r_env_report(r1_result)
    print_bridged_report(bridged, r1_result.r_env)
