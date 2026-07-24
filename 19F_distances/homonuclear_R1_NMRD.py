"""
homonuclear_R1_NMRD.py
========================
Homonuclear ¹H-¹H longitudinal (R1) relaxation dispersion (NMRD) model,
following Wang, Pisano, Ghini et al., "Detection of Metabolite-Protein
Interactions in Complex Biological Samples by High-Resolution
Relaxometry", J. Am. Chem. Soc. 2021, 143, 9393-9404 (Eqs. 1-4).

This is a DIFFERENT relaxation mechanism from the rest of this project:
compare_R2DD.py / surface_packing.py model heteronuclear (¹⁹F-¹H)
TRANSVERSE (R2) dipolar relaxation of an observed nucleus surrounded by
external protein protons, forward from an assumed geometry/density. This
module models the HOMONUCLEAR (¹H-¹H) LONGITUDINAL (R1) relaxation of a
small molecule's own methyl-group protons, exchanging between a free
(unbound) and a macromolecule-bound state, and is meant to be FIT to
measured multi-field relaxation dispersion (NMRD) data -- exactly the
kind of data high-resolution relaxometry (HRR) or fast-field-cycling
(FFC) instruments produce, as in the reference paper.

Physical model (paper Eqs. 1-4)
---------------------------------
Two-site fast exchange, population-weighted average:

    R1_eff(B0) = pF * R1_F(B0) + pB * R1_B(B0),   pF + pB = 1        (1)

Free form -- intramethyl ¹H-¹H dipolar relaxation only, methyl rotating
infinitely fast (order parameter S² = 1/4 folded into A_methyl):

    R1_F(B0) = A_methyl * [ J(wH, tauF) + 4*J(2*wH, tauF) ]           (2)

    A_methyl = S² * (3/32) * mu0² gammaH⁴ hbar² / (pi² r_HH^6)

Bound form -- dipolar relaxation from the methyl protons to a single
"effective" surrounding proton representing the whole macromolecular
contact environment, with the SAME 5 spectral-density-term structure
that gives rise to R2,DD elsewhere in this project, but assembled into
R1 instead of R2 (paper Eq. 3, isotropic overall tumbling):

    R1_B(B0) = A_B * [ J(0, tauB) + 3*J(wH, tauB) + 6*J(2*wH, tauB) ]  (3)

    A_B = S² * (1/64) * mu0² gammaH⁴ hbar² / (pi² r_env^6)

Spectral density (paper Eq. 4, note the 2/5 prefactor -- THIS IS
DIFFERENT from the unprefactored J(omega, tau_c) used elsewhere in this
project for the heteronuclear R2,DD model; the numerical factors are
distributed differently between J and the dipolar prefactor in the two
papers' conventions, so the two J functions must NOT be interchanged):

    J(w, tau_c) = (2/5) * tau_c / (1 + w² tau_c²)                      (4)

Fast-exchange, weak-binding limit (pB << pF ~ 1, as adopted in the
paper): the number of free parameters reduces to three -- tauF, tauB,
and the PRODUCT ApB (A_B * pB), since pB and A_B cannot be separated
from a single-metabolite NMRD profile without independent knowledge of
either the bound population or r_env.

What you get from fitting one NMRD profile
---------------------------------------------
- tauF : free-molecule rotational correlation time (tens to hundreds of ps)
- tauB : bound-complex rotational correlation time (a few ns to ~1 us) --
  this is the size handle: comparable to the macromolecule's own tau_c,
  so it identifies (or at least narrows down) which protein the
  metabolite is binding.
- A_B * pB : amplitude of the dispersion, entangling the bound population
  with the intermolecular dipolar geometry (r_env). Cannot be split into
  A_B and pB without extra information (e.g. a known/assumed r_env, or an
  independent estimate of pB from concentrations).

Usage
-----
    from homonuclear_R1_NMRD import fit_nmrd_free_only, fit_nmrd_two_state

    # metabolite with no evidence of binding (e.g. alanine):
    fit_free = fit_nmrd_free_only(B0_array, R1_array, R1_err_array)

    # metabolite with evidence of binding (e.g. lactate, TSP):
    fit_bound = fit_nmrd_two_state(B0_array, R1_array, R1_err_array)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.constants import hbar, mu_0
from scipy.optimize import curve_fit

from compare_R2DD import GAMMA


# ============================================================================
# Paper-specific spectral density (Eq. 4) -- NOT the same J as
# compare_R2DD.J; do not interchange (see module docstring).
# ============================================================================

def J_paper(omega: float, tau_c: float) -> float:
    """
    Spectral density function from Wang et al. 2021, Eq. 4:

        J(w, tau_c) = (2/5) * tau_c / (1 + w^2 tau_c^2)

    Includes the 2/5 prefactor that the paper's dipolar constants
    (A_methyl, A_B) are normalized against. This is DIFFERENT from
    compare_R2DD.J (no prefactor, used with a differently-normalized
    dipolar constant for the heteronuclear R2,DD model elsewhere in this
    project) -- the two are not interchangeable.
    """
    return (2.0 / 5.0) * tau_c / (1.0 + (omega * tau_c) ** 2)


# ============================================================================
# Dipolar constants (Eqs. 2-3, amplitude prefactors)
# ============================================================================

def A_methyl_from_rHH(r_HH_AA: float, S2: float = 0.25) -> float:
    """
    Intramethyl dipolar constant (paper Eq. 2), in s⁻²:

        A_methyl = S² * (3/32) * mu0² gammaH⁴ hbar² / (pi² r_HH^6)

    Parameters
    ----------
    r_HH_AA : float
        H-H distance within the methyl group, in Å (paper cites PubChem,
        ref 72, for this geometric constant -- typically ~1.78 Å).
    S2 : float
        Order parameter for intramethyl dipole-dipole interactions.
        Default 0.25, matching the paper's assumption of infinitely fast
        methyl rotation (S² = 1/4 for a freely rotating methyl group).

    Returns
    -------
    float
        A_methyl in s⁻².
    """
    gH = GAMMA["1H"]
    r_HH_SI = r_HH_AA * 1e-10
    return S2 * (3.0 / 32.0) * mu_0**2 * gH**4 * hbar**2 / (np.pi**2 * r_HH_SI**6)


def A_B_from_renv(r_env_AA: float, S2: float = 1.0) -> float:
    """
    Bound-state dipolar constant (paper Eq. 3), in s⁻²:

        A_B = S² * (1/64) * mu0² gammaH⁴ hbar² / (pi² r_env^6)

    Parameters
    ----------
    r_env_AA : float
        Distance between the methyl protons and a single effective
        proton representing the surrounding contact environment, in Å.
        The paper's Figure 1 caption uses r_env = 2.29 Å (229 pm) with
        S²=1 to get A_B = 10⁹ s⁻² -- this combination is used as the
        validation point for this function.
    S2 : float
        Order parameter for the bound-state intermolecular interaction.
        Default 1.0 (no additional averaging assumed).

    Returns
    -------
    float
        A_B in s⁻².
    """
    gH = GAMMA["1H"]
    r_env_SI = r_env_AA * 1e-10
    return S2 * (1.0 / 64.0) * mu_0**2 * gH**4 * hbar**2 / (np.pi**2 * r_env_SI**6)


# ============================================================================
# Forward model: R1_eff(B0)
# ============================================================================

def R1_free(B0: float, tau_F: float, A_methyl: float) -> float:
    """
    R1 of the free (unbound) small molecule's methyl protons (Eq. 2), s⁻¹.

    Parameters
    ----------
    B0 : float
        Magnetic field, Tesla.
    tau_F : float
        Free-molecule rotational correlation time, seconds.
    A_methyl : float
        Intramethyl dipolar constant, s⁻² (see A_methyl_from_rHH).
    """
    wH = GAMMA["1H"] * B0
    return A_methyl * (J_paper(wH, tau_F) + 4.0 * J_paper(2.0 * wH, tau_F))


def R1_bound(B0: float, tau_B: float, A_B: float) -> float:
    """
    R1 of the bound (macromolecule-associated) small molecule's methyl
    protons (Eq. 3), s⁻¹.

    Parameters
    ----------
    B0 : float
        Magnetic field, Tesla.
    tau_B : float
        Bound-complex rotational correlation time, seconds (this is
        essentially the macromolecule's own tau_c).
    A_B : float
        Bound-state dipolar constant, s⁻² (see A_B_from_renv), OR the
        product A_B*pB when pB is not known separately (see
        R1_effective_weak_binding).
    """
    wH = GAMMA["1H"] * B0
    return A_B * (J_paper(0.0, tau_B) + 3.0 * J_paper(wH, tau_B)
                  + 6.0 * J_paper(2.0 * wH, tau_B))


def R1_effective(B0: float, tau_F: float, A_methyl: float,
                  tau_B: float, A_B: float, p_B: float) -> float:
    """
    Full two-state population-weighted R1 (Eq. 1), s⁻¹:

        R1_eff = pF*R1_F + pB*R1_B,   pF = 1 - pB
    """
    p_F = 1.0 - p_B
    return p_F * R1_free(B0, tau_F, A_methyl) + p_B * R1_bound(B0, tau_B, A_B)


def R1_effective_weak_binding(B0: float, tau_F: float, A_methyl: float,
                                tau_B: float, ApB: float) -> float:
    """
    Two-state R1 in the paper's weak-binding limit (pB << pF ~ 1), s⁻¹:

        R1_eff ~ R1_F(tau_F, A_methyl) + ApB * [J(0,tauB) + 3J(wH,tauB)
                                                  + 6J(2wH,tauB)]

    This is the model actually fit in the paper (3 free parameters:
    tau_F, tau_B, and the product ApB = A_B*pB), since a single-field-
    series NMRD profile of one metabolite cannot separate A_B and pB.

    Parameters
    ----------
    ApB : float
        The fitted product A_B * pB, s⁻² (this is what Tables 1-2 of the
        paper report as "ABpB").
    """
    wH = GAMMA["1H"] * B0
    R1_F = R1_free(B0, tau_F, A_methyl)
    dispersion = ApB * (J_paper(0.0, tau_B) + 3.0 * J_paper(wH, tau_B)
                         + 6.0 * J_paper(2.0 * wH, tau_B))
    return R1_F + dispersion


# ============================================================================
# Fitting: free-only model (e.g. alanine -- no evidence of binding)
# ============================================================================

@dataclass
class FreeOnlyFitResult:
    tau_F: float
    tau_F_err: float
    A_methyl_fixed: float
    r_HH_fixed_AA: float


def fit_nmrd_free_only(
    B0: np.ndarray,
    R1: np.ndarray,
    R1_err: np.ndarray | None = None,
    r_HH_AA: float = 1.78,
    S2: float = 0.25,
    tau_F_guess: float = 50e-12,
) -> FreeOnlyFitResult:
    """
    Fit an NMRD profile with NO evidence of macromolecule binding (e.g.
    alanine in the paper) to the free-form-only model (Eq. 2), with
    A_methyl fixed from an assumed r_HH (methyl group geometry, a known
    constant, not a fit parameter) and only tau_F floated.

    Parameters
    ----------
    B0 : array
        Magnetic fields, Tesla.
    R1 : array
        Measured longitudinal relaxation rates, s⁻¹.
    R1_err : array, optional
        1-sigma uncertainties on R1, s⁻¹. If given, used to weight the
        fit (standard weighted least squares).
    r_HH_AA : float
        Assumed intramethyl H-H distance, Å. Default 1.78 Å.
    S2 : float
        Order parameter, default 0.25 (free, fast-rotating methyl).
    tau_F_guess : float
        Initial guess for tau_F, seconds.

    Returns
    -------
    FreeOnlyFitResult
    """
    A_methyl = A_methyl_from_rHH(r_HH_AA, S2)

    def model(B0_arr, tau_F):
        return np.array([R1_free(B, tau_F, A_methyl) for B in B0_arr])

    sigma = R1_err if R1_err is not None else None
    popt, pcov = curve_fit(model, B0, R1, p0=[tau_F_guess], sigma=sigma,
                            absolute_sigma=sigma is not None, bounds=(0, np.inf))
    tau_F_fit = popt[0]
    tau_F_err = np.sqrt(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else float("nan")

    return FreeOnlyFitResult(
        tau_F=tau_F_fit, tau_F_err=tau_F_err,
        A_methyl_fixed=A_methyl, r_HH_fixed_AA=r_HH_AA,
    )


# ============================================================================
# Fitting: two-state weak-binding model (e.g. lactate, TSP, creatinine)
# ============================================================================

@dataclass
class TwoStateFitResult:
    tau_F: float
    tau_F_err: float
    tau_B: float
    tau_B_err: float
    ApB: float
    ApB_err: float
    A_methyl_fixed: float
    r_HH_fixed_AA: float


def fit_nmrd_two_state(
    B0: np.ndarray,
    R1: np.ndarray,
    R1_err: np.ndarray | None = None,
    r_HH_AA: float = 1.78,
    S2: float = 0.25,
    tau_F_guess: float = 50e-12,
    tau_B_guess: float = 30e-9,
    ApB_guess: float = 1e7,
) -> TwoStateFitResult:
    """
    Fit an NMRD profile WITH evidence of macromolecule binding (e.g.
    lactate, TSP, creatinine in the paper) to the weak-binding two-state
    model (R1_effective_weak_binding): tau_F, tau_B, and ApB = A_B*pB are
    floated; A_methyl is fixed from an assumed r_HH, as in the free-only
    fit.

    This is the SAME three-parameter model the paper fits (see paper
    Table 1: "ABpB", "tauF", "tauB" columns), reduced from the full
    5-parameter model (Eq. 1-3) under the weak-binding hypothesis
    pB << pF ~ 1, which the paper found sufficient by Akaike Information
    Criterion model selection.

    Internally, tau_F, tau_B, and ApB are fit in RESCALED units (ps, ns,
    and 10^6 s^-2 respectively) rather than raw SI units. tau_F (~1e-11),
    tau_B (~1e-8), and ApB (~1e7) otherwise span ~18 orders of magnitude,
    which badly conditions curve_fit's gradient-based least-squares
    solver and was empirically found to converge to systematically wrong
    (much-worse-chi^2) optima even when started exactly at the true
    parameter values. Rescaling to O(1)-O(100) units resolves this.

    Parameters
    ----------
    B0, R1, R1_err : arrays
        Same as fit_nmrd_free_only.
    r_HH_AA, S2 : float
        Same as fit_nmrd_free_only (fixed methyl geometry).
    tau_F_guess, tau_B_guess, ApB_guess : float
        Initial guesses for tau_F (s), tau_B (s), and ApB (s⁻²).

    Returns
    -------
    TwoStateFitResult
    """
    A_methyl = A_methyl_from_rHH(r_HH_AA, S2)

    # Rescaled units: tau_F in ps, tau_B in ns, ApB in 1e6 s^-2.
    def model_scaled(B0_arr, tau_F_ps, tau_B_ns, ApB_1e6):
        tau_F = tau_F_ps * 1e-12
        tau_B = tau_B_ns * 1e-9
        ApB = ApB_1e6 * 1e6
        return np.array([
            R1_effective_weak_binding(B, tau_F, A_methyl, tau_B, ApB)
            for B in B0_arr
        ])

    sigma = R1_err if R1_err is not None else None
    p0_scaled = [tau_F_guess * 1e12, tau_B_guess * 1e9, ApB_guess * 1e-6]
    popt, pcov = curve_fit(
        model_scaled, B0, R1, p0=p0_scaled, sigma=sigma,
        absolute_sigma=sigma is not None,
        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=20000,
    )
    tau_F_fit = popt[0] * 1e-12
    tau_B_fit = popt[1] * 1e-9
    ApB_fit = popt[2] * 1e6

    perr_scaled = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [float("nan")] * 3
    tau_F_err = perr_scaled[0] * 1e-12
    tau_B_err = perr_scaled[1] * 1e-9
    ApB_err = perr_scaled[2] * 1e6

    return TwoStateFitResult(
        tau_F=tau_F_fit, tau_F_err=tau_F_err,
        tau_B=tau_B_fit, tau_B_err=tau_B_err,
        ApB=ApB_fit, ApB_err=ApB_err,
        A_methyl_fixed=A_methyl, r_HH_fixed_AA=r_HH_AA,
    )


# ============================================================================
# Reporting
# ============================================================================

def print_free_only_report(result: FreeOnlyFitResult) -> None:
    header = "─" * 56
    print(f"\n{header}")
    print("  Free-form-only NMRD fit (no binding evidence)")
    print(header)
    print(f"  τ_F  = {result.tau_F*1e12:8.1f} ± {result.tau_F_err*1e12:.1f} ps")
    print(f"  (A_methyl fixed at r_HH = {result.r_HH_fixed_AA:.2f} Å "
          f"-> {result.A_methyl_fixed:.3e} s⁻²)")
    print(f"{header}\n")


def print_two_state_report(result: TwoStateFitResult) -> None:
    header = "─" * 56
    print(f"\n{header}")
    print("  Two-state weak-binding NMRD fit")
    print(header)
    print(f"  τ_F   = {result.tau_F*1e12:8.1f} ± {result.tau_F_err*1e12:.1f} ps")
    print(f"  τ_B   = {result.tau_B*1e9:8.1f} ± {result.tau_B_err*1e9:.1f} ns")
    print(f"  A_B·p_B = ({result.ApB*1e-6:.2f} ± {result.ApB_err*1e-6:.2f}) × 10⁶ s⁻²")
    print(f"  (A_methyl fixed at r_HH = {result.r_HH_fixed_AA:.2f} Å "
          f"-> {result.A_methyl_fixed:.3e} s⁻²)")
    print(f"{header}\n")


# ============================================================================
# Demo / self-check reproducing paper Figure 1
# ============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reproduce paper Figure 1: illustration of binding effect on NMRD,
    # tauF=50ps, tauB=25ns, ApB values corresponding to 0%, 1%, 5% bound
    # fraction with A_B fixed such that r_env=229pm (A_B=1e9 s^-2).
    tau_F = 50e-12
    tau_B = 25e-9
    A_B = A_B_from_renv(2.29, S2=1.0)
    print(f"A_B check (should be ~1e9 s^-2): {A_B:.3e}")

    A_methyl = A_methyl_from_rHH(1.78, S2=0.25)
    B0_vals = np.logspace(np.log10(0.01), np.log10(20), 200)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for pB, label, color in [(0.0, "No binding", "#EAB308"),
                               (0.01, "1% binding", "#16A34A"),
                               (0.05, "5% binding", "#2563EB")]:
        R1_vals = [R1_effective(B, tau_F, A_methyl, tau_B, A_B, pB) for B in B0_vals]
        ax.semilogx(B0_vals, R1_vals, color=color, lw=2, label=label)

    ax.set_xlabel("Magnetic field (T)")
    ax.set_ylabel(r"$R_1$ (s$^{-1}$)")
    ax.set_title("Reproduction of Wang et al. 2021, Figure 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig("homonuclear_R1_figure1_check.png", dpi=150, bbox_inches="tight")
    print("Saved homonuclear_R1_figure1_check.png")
    print()
    print("NOTE: this reproduces Eqs. 1-4 literally as written in the paper,")
    print("and A_B_from_renv(2.29, S2=1.0) matches the paper's stated")
    print("A_B=1e9 s^-2 at r_env=229pm (Figure 1 caption) to within rounding.")
    print("The resulting curve magnitudes were not cross-checked pixel-by-")
    print("pixel against the published Figure 1 image -- only the formulas")
    print("and the one exact (A_B, r_env) numeric pair stated in the text")
    print("were verified. If you have the source data or exact figure")
    print("values, it would be worth confirming the low-field plateau")
    print("magnitude matches before relying on this for quantitative work.")
