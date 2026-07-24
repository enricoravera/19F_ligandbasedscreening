"""
compare_R2DD.py
===============
Compare two estimates of the dipolar transverse relaxation rate R₂,DD,b
for a ¹⁹F nucleus in the bound state of a protein–ligand complex:

  (A) Rule-of-thumb (Rüdisser et al. 2020, p. 586):
          R₂,DD,b ≈ MW [kDa] s⁻¹

  (B) Surface integral model (Fiorucci & Ravera):
      Replace the unknown sum Σᵢ 1/rᵢ⁶ with the integral of 1/r⁶ over an
      infinite flat surface carrying hydrogen atoms at density σ₀ = 10⁻² Å⁻²,
      assuming van-der-Waals contact between the observed nucleus X and the
      surface:

          I = π σ₀ / (2 z⁴)     where  z = R_H + R_X   [Å]

      This I is then inserted into the standard heteronuclear dipolar R₂
      expression (Ernst et al. 1987):

          R₂,DD = (1/2) d̄² [4J(0) + J(ωF−ωH) + 3J(ωF) + 6J(ωH) + 6J(ωF+ωH)]

      with  d̄² = (μ₀/4π)² ℏ² γF² γH² × I [SI units].

The script produces four panels:
  (1) Integral I(R_X) for biologically common atom types.
  (2) R₂,DD,b vs B₀ at fixed MW (trypsin, 23 kDa): integral vs rule-of-thumb.
  (3) R₂,DD,b vs MW at fixed B₀ (16.4 T, 700 MHz ¹H): integral vs rule-of-thumb.
  (4) Field-dependence ratio R₂,DD(B)/R₂,DD(11.75 T): shows how flat the
      rule-of-thumb approximation really is.

Usage
-----
    python compare_R2DD.py
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.constants import hbar, mu_0

# ============================================================================
# Physical constants and gyromagnetic ratios
# ============================================================================

# Gyromagnetic ratios in rad s⁻¹ T⁻¹
GAMMA = {
    "1H":  2 * np.pi * 42.5774e6,
    "19F": 2 * np.pi * 40.0774e6,
    "13C": 2 * np.pi * 10.7084e6,
    "15N": 2 * np.pi * -4.3160e6,
}

# ============================================================================
# Van der Waals radii (Å) — Bondi 1964 / Alvarez 2013
# ============================================================================

VDW_RADIUS = {
    "H": 1.20, "C": 1.70, "N": 1.55,
    "O": 1.52, "F": 1.47, "S": 1.80,
}

# Surface hydrogen density and H van der Waals radius from the LaTeX document
SIGMA0   = 1e-2   # Å⁻²
R_H_SURF = 1.20   # Å  (H atoms on the protein surface)


# ============================================================================
# (B) Surface integral model
# ============================================================================

def integral_I(R_X_AA: float, sigma0: float | None = None) -> float:
    """
    Effective Σᵢ 1/rᵢ⁶ from the infinite-flat-surface integral (Å⁻⁶).

        I = π σ₀ / (2 z⁴)   where  z = R_H + R_X  [Å]

    Parameters
    ----------
    R_X_AA : float
        Van der Waals radius of the observed nucleus in Å.
    sigma0 : float, optional
        Areal H density (Å⁻²) to use in place of the module-level SIGMA0
        constant. Lets callers compare different densities (e.g. a
        structurally-derived exposed-proton density) without editing the
        global default.

    Returns
    -------
    float
        I in Å⁻⁶.
    """
    s0 = SIGMA0 if sigma0 is None else sigma0
    z = R_H_SURF + R_X_AA
    return np.pi * s0 / (2.0 * z**4)


def integral_I_SI(R_X_AA: float, sigma0: float | None = None) -> float:
    """Same as :func:`integral_I` but in m⁻⁶  (1 Å⁻⁶ = 10⁶⁰ m⁻⁶)."""
    return integral_I(R_X_AA, sigma0=sigma0) * 1e60


# ============================================================================
# Spectral density and dipolar R₂
# ============================================================================

def J(omega: float, tau_c: float) -> float:
    """Isotropic Lorentzian spectral density J(ω) = τ_c / (1 + ω²τ_c²)."""
    return tau_c / (1.0 + (omega * tau_c) ** 2)


def R2_DD_from_sum_inv_r6(
    B0: float,
    tau_c: float,
    sum_inv_r6_AA: float,
    nuc_obs:     str = "19F",
    nuc_partner: str = "1H",
) -> float:
    """
    Heteronuclear transverse dipolar relaxation rate R₂,DD,b from an
    explicit, already-computed Σᵢ 1/rᵢ⁶ (Å⁻⁶). This is the common core
    used by both the flat-surface integral model (R2_DD_integral) and any
    geometry-based estimate of Σ 1/r⁶ (e.g. a packed molecular surface).

    Formula (Ernst et al. 1987 / Cavanagh et al.):

        R₂,DD = (d̄²/2) [4J(0) + J(ωI−ωS) + 3J(ωI) + 6J(ωS) + 6J(ωI+ωS)]

    where  d̄² = (μ₀/4π)² ℏ² γI² γS² × Σ(1/r⁶)_SI

    Parameters
    ----------
    B0 : float
        Magnetic field strength in Tesla.
    tau_c : float
        Rotational correlation time of the complex in seconds.
    sum_inv_r6_AA : float
        Σᵢ 1/rᵢ⁶ in Å⁻⁶ (e.g. from integral_I, or from summing over an
        explicit set of H atom distances).
    nuc_obs, nuc_partner : str
        Nucleus labels (default ¹⁹F observed, ¹H partner).

    Returns
    -------
    float
        R₂,DD,b in s⁻¹.
    """
    gI = GAMMA[nuc_obs]
    gS = GAMMA[nuc_partner]
    omI = gI * B0
    omS = gS * B0

    sum_inv_r6_SI = sum_inv_r6_AA * 1e60   # Å⁻⁶ -> m⁻⁶
    d_sq = (mu_0 / (4 * np.pi))**2 * hbar**2 * gI**2 * gS**2 * sum_inv_r6_SI

    sd_sum = (
        4 * J(0.0,            tau_c) +
        J(abs(omI - omS),     tau_c) +
        3 * J(omI,            tau_c) +
        6 * J(omS,            tau_c) +
        6 * J(omI + omS,      tau_c)
    )
    return 0.5 * d_sq * sd_sum


def R2_DD_integral(
    B0: float,
    tau_c: float,
    R_X_AA: float,
    nuc_obs:     str = "19F",
    nuc_partner: str = "1H",
    sigma0: float | None = None,
) -> float:
    """
    Heteronuclear transverse dipolar relaxation rate R₂,DD,b using the
    infinite-flat-surface integral I(R_X) to replace the unknown Σᵢ 1/rᵢ⁶.
    Thin wrapper around R2_DD_from_sum_inv_r6.

    Parameters
    ----------
    B0 : float
        Magnetic field strength in Tesla.
    tau_c : float
        Rotational correlation time of the complex in seconds.
    R_X_AA : float
        Van der Waals radius of the observed nucleus in Å.
    nuc_obs, nuc_partner : str
        Nucleus labels (default ¹⁹F observed, ¹H partner).
    sigma0 : float, optional
        Areal H density (Å⁻²) override, passed through to integral_I.

    Returns
    -------
    float
        R₂,DD,b in s⁻¹.
    """
    I_AA = integral_I(R_X_AA, sigma0=sigma0)   # Å⁻⁶
    return R2_DD_from_sum_inv_r6(B0, tau_c, I_AA, nuc_obs, nuc_partner)


def R2_DD_single_pair(
    B0: float,
    tau_c: float,
    r_AA: float,
    nuc_obs:     str = "19F",
    nuc_partner: str = "1H",
) -> float:
    """
    Heteronuclear R₂,DD for a **single** spin pair at distance r.
    Used to cross-check the integral against an explicit distance.

    Parameters
    ----------
    r_AA : float
        Internuclear distance in Å.
    """
    gI = GAMMA[nuc_obs]
    gS = GAMMA[nuc_partner]
    omI = gI * B0
    omS = gS * B0

    r_m = r_AA * 1e-10                          # Å → m
    d_sq = (mu_0 / (4 * np.pi))**2 * hbar**2 * gI**2 * gS**2 / r_m**6

    sd_sum = (
        4 * J(0.0,            tau_c) +
        J(abs(omI - omS),     tau_c) +
        3 * J(omI,            tau_c) +
        6 * J(omS,            tau_c) +
        6 * J(omI + omS,      tau_c)
    )
    return 0.5 * d_sq * sd_sum


# ============================================================================
# (A) Rule-of-thumb  (paper p. 586)
# ============================================================================

def R2_DD_rulethumh(MW_kDa: float) -> float:
    """
    Empirical rule-of-thumb from Rüdisser et al. 2020 (p. 586):

        R₂,DD,b ≈ MW [kDa]  s⁻¹

    This is field-independent by assumption.
    """
    return np.asarray(MW_kDa, dtype=float)


# ============================================================================
# Auxiliary: τ_c from MW  (empirical Stokes-like rule for globular proteins)
# ============================================================================

def tau_c_from_MW(MW_kDa: float) -> float:
    """
    Approximate rotational correlation time (s) from protein MW.

    Uses the common empirical rule τ_c ≈ 0.6 ns per kDa,
    appropriate for globular proteins in water at ~25 °C.
    """
    return MW_kDa * 0.6e-9


# ============================================================================
# Effective single-proton distance equivalent to the surface integral
# ============================================================================

def r_eff_from_integral(R_X_AA: float, sigma0: float | None = None) -> float:
    """
    The single H–X distance r_eff that would produce the same Σ(1/r⁶)
    as the surface integral I.

        r_eff = I^(−1/6)   [Å]
    """
    return integral_I(R_X_AA, sigma0=sigma0) ** (-1.0 / 6.0)


# ============================================================================
# Plotting
# ============================================================================

def make_figure(save_path: str | None = None) -> None:
    """Generate the four-panel comparison figure."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "R₂,DD,b: paper rule-of-thumb vs surface-integral model  (¹⁹F–¹H)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    colors = {"integral": "#2563EB", "thumb": "#DC2626", "ratio": "#16A34A"}

    # ── (1) Integral I(R_X) and effective single-proton distance ─────────────
    ax = axes[0, 0]
    R_X_vals = np.linspace(1.0, 2.2, 200)
    I_vals   = integral_I(R_X_vals)
    r_eff    = r_eff_from_integral(R_X_vals)

    ax2 = ax.twinx()
    ax.plot(R_X_vals, I_vals * 1e4, color=colors["integral"], lw=2,
            label=r"$I(R_X)$ [10⁻⁴ Å⁻⁶]")
    ax2.plot(R_X_vals, r_eff, color="#7C3AED", lw=2, ls="--",
             label=r"$r_{\rm eff}$ [Å]")

    for atom, R_X in VDW_RADIUS.items():
        if atom == "H":
            continue
        ax.axvline(R_X, color="0.7", lw=0.8, ls=":")
        ax.text(R_X + 0.02, ax.get_ylim()[1] * 0.05
                if ax.get_ylim()[1] > 0 else 0.5,
                atom, fontsize=8, color="0.5", va="bottom")

    ax.set_xlabel(r"$R_X$ (van der Waals radius, Å)")
    ax.set_ylabel(r"$I$ (10⁻⁴ Å⁻⁶)", color=colors["integral"])
    ax2.set_ylabel(r"$r_{\rm eff}$ (Å)", color="#7C3AED")
    ax.set_title("Surface integral and equivalent single-proton distance")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # Mark ¹⁹F vdw radius explicitly
    R_F = VDW_RADIUS["F"]
    ax.axvline(R_F, color=colors["integral"], lw=1.2, ls="-.")
    ax.text(R_F + 0.02, max(I_vals * 1e4) * 0.85, "F", fontsize=9,
            color=colors["integral"])

    # ── (2) R₂,DD,b vs B₀ (trypsin, 23 kDa) ─────────────────────────────────
    ax = axes[0, 1]
    MW_trypsin = 23.0
    tau_c_tryp = tau_c_from_MW(MW_trypsin)
    B0_vals    = np.linspace(7.0, 28.2, 200)   # 300 – 1200 MHz ¹H

    R2_int  = np.array([R2_DD_integral(B, tau_c_tryp, VDW_RADIUS["F"]) for B in B0_vals])
    R2_rule = R2_DD_rulethumh(MW_trypsin)

    ax.plot(B0_vals * GAMMA["1H"] / (2 * np.pi * 1e6), R2_int,
            color=colors["integral"], lw=2, label="Surface integral")
    ax.axhline(R2_rule, color=colors["thumb"], lw=2, ls="--",
               label=f"Rule-of-thumb  ({R2_rule:.0f} s⁻¹)")

    # Mark the two fields used in the paper
    for B_paper, label in [(11.75, "500 MHz"), (16.4, "700 MHz")]:
        ax.axvline(B_paper * GAMMA["1H"] / (2 * np.pi * 1e6),
                   color="0.5", lw=0.8, ls=":")
        ax.text(B_paper * GAMMA["1H"] / (2 * np.pi * 1e6) + 5,
                R2_int.min() * 0.95, label, fontsize=7.5, color="0.4")

    ax.set_xlabel(r"$\nu_H$ (MHz)")
    ax.set_ylabel(r"$R_{2,\rm DD,b}$ (s⁻¹)")
    ax.set_title(f"R₂,DD vs B₀  (trypsin, {MW_trypsin:.0f} kDa, "
                 f"τ_c = {tau_c_tryp*1e9:.1f} ns)")
    ax.legend(fontsize=9)

    # ── (3) R₂,DD,b vs MW at B₀ = 16.4 T ────────────────────────────────────
    ax = axes[1, 0]
    B0_fixed   = 16.4   # T  (700 MHz ¹H)
    MW_vals    = np.linspace(5, 100, 200)

    R2_int_mw  = np.array([
        R2_DD_integral(B0_fixed, tau_c_from_MW(MW), VDW_RADIUS["F"])
        for MW in MW_vals
    ])
    R2_rule_mw = R2_DD_rulethumh(MW_vals)

    ax.plot(MW_vals, R2_int_mw,  color=colors["integral"], lw=2, label="Surface integral")
    ax.plot(MW_vals, R2_rule_mw, color=colors["thumb"],    lw=2, ls="--", label="Rule-of-thumb")
    ax.axvline(23, color="0.5", lw=0.8, ls=":")
    ax.text(24, R2_int_mw.max() * 0.95, "trypsin\n(23 kDa)", fontsize=7.5, color="0.4")

    ax.set_xlabel("MW (kDa)")
    ax.set_ylabel(r"$R_{2,\rm DD,b}$ (s⁻¹)")
    ax.set_title(f"R₂,DD vs MW  (B₀ = {B0_fixed} T, τ_c = 0.6 ns/kDa)")
    ax.legend(fontsize=9)

    # ── (4) Field-dependence: R₂,DD(B) / R₂,DD(11.75 T) ─────────────────────
    ax = axes[1, 1]
    B0_ref     = 11.75  # T  (500 MHz)
    R2_ref_23  = R2_DD_integral(B0_ref, tau_c_from_MW(23),  VDW_RADIUS["F"])
    R2_ref_50  = R2_DD_integral(B0_ref, tau_c_from_MW(50),  VDW_RADIUS["F"])
    R2_ref_100 = R2_DD_integral(B0_ref, tau_c_from_MW(100), VDW_RADIUS["F"])

    for MW, R2_ref, ls in [(23, R2_ref_23, "-"), (50, R2_ref_50, "--"), (100, R2_ref_100, ":")]:
        tau_c = tau_c_from_MW(MW)
        ratio = np.array([R2_DD_integral(B, tau_c, VDW_RADIUS["F"]) for B in B0_vals]) / R2_ref
        ax.plot(B0_vals * GAMMA["1H"] / (2 * np.pi * 1e6), ratio,
                color=colors["integral"], lw=1.8, ls=ls, label=f"{MW} kDa")

    ax.axhline(1.0, color="0.6", lw=0.8, ls="-")
    ax.axvline(B0_ref * GAMMA["1H"] / (2 * np.pi * 1e6), color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"$\nu_H$ (MHz)")
    ax.set_ylabel(r"$R_{2,\rm DD}(B_0)\ /\ R_{2,\rm DD}(500\,\text{MHz})$")
    ax.set_title("Field-dependence of R₂,DD (relative to 500 MHz)")
    ax.legend(fontsize=9, title="Protein MW")
    ax.set_ylim(bottom=0)

    # ── Print numerical summary ───────────────────────────────────────────────
    print_summary(tau_c_tryp, B0_fixed, MW_trypsin)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================================
# All-models figure (A-F): rule-of-thumb, flat surface, and packed
# molecular surface models together
# ============================================================================

def make_figure_all_models(
    xyz_path: str = "fluorobenzene.xyz",
    MW_kDa: float = 23.0,
    B0_ref: float = 16.4,
    structural_sigma0: float = 0.062,
    save_path: str | None = None,
) -> None:
    """
    Generate the compare_R2DD.py-style comparison figure extended to
    include all six R2,DD models, up through model (F):

      (A) Rule-of-thumb            R2,DD ≈ MW [kDa] s⁻¹
      (B) Flat surface, mean-field σ0 = SIGMA0 = 0.01 Å⁻²
      (C) Packed molecular surface, max density (hard-sphere H-H contact)
      (D) Packed molecular surface, capped at the mean-field σ0
      (E) Flat surface, structurally-derived σ0 (e.g. from
          sigma0_from_pdb.py on a real PDB structure)
      (F) Packed molecular surface, capped at the structural σ0

    Models (C), (D), (F) require an actual molecular geometry (an XYZ
    file with at least one F atom), since they are computed from the
    packed-surface machinery in surface_packing.py rather than the
    idealized infinite-plane integral used by (A)/(B)/(E). This function
    imports surface_packing locally to avoid a circular import (that
    module imports FROM compare_R2DD.py at module load time).

    Parameters
    ----------
    xyz_path : str
        Path to the molecule used for the packed-surface models (C/D/F).
        Defaults to "fluorobenzene.xyz"; must exist in the working
        directory or be given as an absolute/relative path.
    MW_kDa, B0_ref : float
        Protein MW and reference field, shared by all models so they
        describe the same complex/measurement condition.
    structural_sigma0 : float
        Areal H density (Å⁻²) for models (E) and (F). Defaults to 0.062,
        the exposed-proton density measured from 8AWW via
        sigma0_from_pdb.py (~6.2x the mean-field SIGMA0=0.01 Å⁻²).
    save_path : str, optional
        If given, save the figure here instead of showing it.
    """
    # Local import: surface_packing.py imports FROM this module, so this
    # import must stay inside the function body to avoid a circular
    # top-level import.
    from surface_packing import (
        read_xyz, analyze_fluorines, plot_model_comparison as _spc_plot,
    )

    mol = read_xyz(xyz_path)
    tau_c = tau_c_from_MW(MW_kDa)

    # (C) Uncapped: maximum density the geometry allows.
    results = analyze_fluorines(
        mol, B0=B0_ref, tau_c=tau_c,
    )
    # (D) Capped at the mean-field SIGMA0.
    results_capped = analyze_fluorines(
        mol, B0=B0_ref, tau_c=tau_c, target_density=SIGMA0,
    )
    # (F) Capped at the structural sigma0.
    results_capped_structural = analyze_fluorines(
        mol, B0=B0_ref, tau_c=tau_c, target_density=structural_sigma0,
    )

    print(f"All-models figure: molecule={xyz_path}, MW={MW_kDa:.0f} kDa, "
          f"B0={B0_ref:.2f} T, structural σ₀={structural_sigma0:.4f} Å⁻²")
    print(f"  (C) packed max-density F atoms found : {len(results)}")
    print(f"  (D) packed @ mean-field σ0 F atoms   : {len(results_capped)}")
    print(f"  (F) packed @ structural σ0 F atoms   : {len(results_capped_structural)}")

    # plot_model_comparison already renders (A)/(B)/(C)/(D)/(E)/(F)
    # together in a compare_R2DD-style two-panel layout (R2,DD vs B0, and
    # a bar chart at the reference field) — reuse it directly rather than
    # duplicating its plotting logic here.
    _spc_plot(
        results, MW_kDa=MW_kDa, B0_ref=B0_ref,
        results_capped=results_capped,
        structural_sigma0=structural_sigma0,
        results_capped_structural=results_capped_structural,
        save_path=save_path,
    )


def make_figure_4panel_all_models(
    xyz_path: str = "fluorobenzene.xyz",
    structural_sigma0: float = 0.062,
    save_path: str | None = None,
) -> None:
    """
    Regenerate the ORIGINAL 4-panel make_figure() layout (same panels, same
    axes, same trypsin/23 kDa default condition), but with all six models
    plotted together on every panel where that is meaningful:

      (A) Rule-of-thumb            R2,DD ≈ MW [kDa] s⁻¹
      (B) Flat surface, mean-field σ0 = SIGMA0 = 0.01 Å⁻²
      (C) Packed molecular surface, max density
      (D) Packed molecular surface, capped at the mean-field σ0
      (E) Flat surface, structurally-derived σ0
      (F) Packed molecular surface, capped at the structural σ0

    Panel-by-panel handling
    ------------------------
    Panel 1 (I(R_X) vs R_X): (C)/(D)/(F) are properties of one specific
      molecule/F-atom, not a continuous function of R_X the way (B)/(E)
      are (an infinite-plane integral evaluated at any vdW radius). They
      are shown instead as horizontal reference lines at R_X = R_F marking
      their actual Σ(1/r⁶), so they can still be compared to the (B)/(E)
      curves at the one R_X value where all six models are defined.
    Panel 2 (R2,DD vs B0, fixed MW): all six models are real curves here
      (Σ1/r⁶ is B0-independent, only the spectral-density prefactor
      changes with B0), so all six are drawn directly.
    Panel 3 (R2,DD vs MW, fixed B0): same as panel 2 — all six are real
      curves (Σ1/r⁶ does not depend on MW, only tau_c does).
    Panel 4 (field-dependence ratio R2,DD(B)/R2,DD(B_ref)): this ratio is
      mathematically independent of Σ(1/r⁶) — it cancels in the division,
      since all models share the same spectral-density functional form.
      (B)/(C)/(D)/(E)/(F) would therefore all overlap exactly with the
      existing curves; rather than draw five identical overlapping lines,
      this panel is left as in the original figure with a note added to
      the title.

    Models (C), (D), (F) require actual molecular geometry (an XYZ file
    with at least one F atom) via surface_packing.py, imported locally
    here to avoid a circular import.

    Parameters
    ----------
    xyz_path : str
        Molecule used for the packed-surface models (C/D/F). Defaults to
        "fluorobenzene.xyz".
    structural_sigma0 : float
        Areal H density (Å⁻²) for models (E) and (F). Defaults to 0.062
        (8AWW exposed-proton density via sigma0_from_pdb.py).
    save_path : str, optional
        If given, save the figure here instead of showing it.
    """
    from surface_packing import read_xyz, analyze_fluorines

    MW_trypsin = 23.0
    tau_c_tryp = tau_c_from_MW(MW_trypsin)
    B0_fixed   = 16.4    # T (700 MHz 1H), used for panel 3
    B0_vals    = np.linspace(7.0, 28.2, 200)   # 300-1200 MHz 1H, panel 2
    MW_vals    = np.linspace(5, 100, 200)      # panel 3
    R_F        = VDW_RADIUS["F"]

    # ── Molecule-derived Sigma(1/r^6) for models C/D/F, at the F atom(s)
    # found in xyz_path. Mean over all F atoms in the molecule if more
    # than one is present, matching plot_model_comparison's convention.
    mol = read_xyz(xyz_path)
    results_C = analyze_fluorines(mol)  # uncapped, max density
    results_D = analyze_fluorines(mol, target_density=SIGMA0)
    results_F = analyze_fluorines(mol, target_density=structural_sigma0)

    def _mean_sum_inv_r6(results):
        vals = [r.sum_inv_r6 for r in results if r.sum_inv_r6 > 0]
        return float(np.mean(vals)) if vals else float("nan")

    sum_inv_r6_C = _mean_sum_inv_r6(results_C)
    sum_inv_r6_D = _mean_sum_inv_r6(results_D)
    sum_inv_r6_F = _mean_sum_inv_r6(results_F)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle(
        "R₂,DD,b: rule-of-thumb vs flat-surface vs packed-surface models, "
        "(A)-(F)  (¹⁹F–¹H)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    colors = {
        "thumb": "#DC2626", "flat": "#2563EB", "packed": "#16A34A",
        "capped": "#7C3AED", "structural": "#EA580C", "capped_structural": "#0891B2",
    }

    # ── (1) Integral I(R_X) and equivalent single-proton distance ────────
    ax = axes[0, 0]
    R_X_vals = np.linspace(1.0, 2.2, 200)
    I_vals   = integral_I(R_X_vals)
    I_struct = integral_I(R_X_vals, sigma0=structural_sigma0)
    r_eff    = r_eff_from_integral(R_X_vals)

    ax2 = ax.twinx()
    ax.plot(R_X_vals, I_vals * 1e4, color=colors["flat"], lw=2,
            label=r"(B) $I(R_X)$, σ₀=0.01  [10⁻⁴ Å⁻⁶]")
    ax.plot(R_X_vals, I_struct * 1e4, color=colors["structural"], lw=2, ls="--",
            label=fr"(E) $I(R_X)$, σ₀={structural_sigma0:.3f}  [10⁻⁴ Å⁻⁶]")
    ax2.plot(R_X_vals, r_eff, color="#7C3AED", lw=1.5, ls=":",
             label=r"$r_{\rm eff}$(B) [Å]")

    # (C)/(D)/(F) are single points at R_X = R_F (fixed by the molecule's
    # F atom), shown as markers + horizontal guides for comparison against
    # the (B)/(E) curves.
    for label, val, color in [
        ("(C) Packed, max density", sum_inv_r6_C, colors["packed"]),
        ("(D) Packed, σ₀=0.01-capped", sum_inv_r6_D, colors["capped"]),
        ("(F) Packed, σ₀=struct-capped", sum_inv_r6_F, colors["capped_structural"]),
    ]:
        if np.isnan(val):
            continue
        ax.scatter([R_F], [val * 1e4], color=color, zorder=5, s=40, label=label)
        ax.axhline(val * 1e4, color=color, lw=0.8, ls=":", alpha=0.6)

    for atom, R_X in VDW_RADIUS.items():
        if atom == "H":
            continue
        ax.axvline(R_X, color="0.7", lw=0.8, ls=":")
        ax.text(R_X + 0.02, ax.get_ylim()[1] * 0.05
                if ax.get_ylim()[1] > 0 else 0.5,
                atom, fontsize=8, color="0.5", va="bottom")

    ax.set_xlabel(r"$R_X$ (van der Waals radius, Å)")
    ax.set_ylabel(r"$I$ (10⁻⁴ Å⁻⁶)")
    ax2.set_ylabel(r"$r_{\rm eff}$ (Å)", color="#7C3AED")
    ax.set_title("Surface integral, plus packed-surface Σ(1/r⁶) at R_F")
    ax.axvline(R_F, color=colors["flat"], lw=1.0, ls="-.", alpha=0.5)
    ax.text(R_F + 0.02, max(I_vals * 1e4) * 0.85, "F", fontsize=9, color=colors["flat"])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="upper right")

    # ── (2) R2,DD,b vs B0 (trypsin, 23 kDa) — all six models ─────────────
    ax = axes[0, 1]
    R2_flat = np.array([R2_DD_integral(B, tau_c_tryp, R_F) for B in B0_vals])
    R2_struct = np.array([R2_DD_integral(B, tau_c_tryp, R_F, sigma0=structural_sigma0)
                           for B in B0_vals])
    R2_rule = R2_DD_rulethumh(MW_trypsin)
    nu_vals = B0_vals * GAMMA["1H"] / (2 * np.pi * 1e6)

    ax.axhline(R2_rule, color=colors["thumb"], lw=2, ls="--",
               label=f"(A) Rule-of-thumb  ({R2_rule:.0f} s⁻¹)")
    ax.plot(nu_vals, R2_flat, color=colors["flat"], lw=2, label="(B) Flat, σ₀=0.01")
    if not np.isnan(sum_inv_r6_C):
        R2_C = np.array([R2_DD_from_sum_inv_r6(B, tau_c_tryp, sum_inv_r6_C) for B in B0_vals])
        ax.plot(nu_vals, R2_C, color=colors["packed"], lw=2, ls="-.", label="(C) Packed, max density")
    if not np.isnan(sum_inv_r6_D):
        R2_D = np.array([R2_DD_from_sum_inv_r6(B, tau_c_tryp, sum_inv_r6_D) for B in B0_vals])
        ax.plot(nu_vals, R2_D, color=colors["capped"], lw=2, ls=":", label="(D) Packed, σ₀=0.01-capped")
    ax.plot(nu_vals, R2_struct, color=colors["structural"], lw=2, ls="--",
            label=f"(E) Flat, σ₀={structural_sigma0:.3f}")
    if not np.isnan(sum_inv_r6_F):
        R2_F = np.array([R2_DD_from_sum_inv_r6(B, tau_c_tryp, sum_inv_r6_F) for B in B0_vals])
        ax.plot(nu_vals, R2_F, color=colors["capped_structural"], lw=2, ls=(0, (3, 1, 1, 1)),
                label=f"(F) Packed, σ₀=struct-capped")

    for B_paper, label in [(11.75, "500 MHz"), (16.4, "700 MHz")]:
        ax.axvline(B_paper * GAMMA["1H"] / (2 * np.pi * 1e6), color="0.5", lw=0.8, ls=":")

    ax.set_xlabel(r"$\nu_H$ (MHz)")
    ax.set_ylabel(r"$R_{2,\rm DD,b}$ (s⁻¹)")
    ax.set_title(f"R₂,DD vs B₀  (trypsin, {MW_trypsin:.0f} kDa, "
                 f"τ_c = {tau_c_tryp*1e9:.1f} ns)")
    ax.legend(fontsize=7)

    # ── (3) R2,DD,b vs MW at B0 = 16.4 T — all six models ─────────────────
    ax = axes[1, 0]
    R2_flat_mw = np.array([R2_DD_integral(B0_fixed, tau_c_from_MW(MW), R_F) for MW in MW_vals])
    R2_struct_mw = np.array([
        R2_DD_integral(B0_fixed, tau_c_from_MW(MW), R_F, sigma0=structural_sigma0)
        for MW in MW_vals
    ])
    R2_rule_mw = R2_DD_rulethumh(MW_vals)

    ax.plot(MW_vals, R2_rule_mw, color=colors["thumb"], lw=2, ls="--", label="(A) Rule-of-thumb")
    ax.plot(MW_vals, R2_flat_mw, color=colors["flat"], lw=2, label="(B) Flat, σ₀=0.01")
    if not np.isnan(sum_inv_r6_C):
        R2_C_mw = np.array([R2_DD_from_sum_inv_r6(B0_fixed, tau_c_from_MW(MW), sum_inv_r6_C)
                             for MW in MW_vals])
        ax.plot(MW_vals, R2_C_mw, color=colors["packed"], lw=2, ls="-.", label="(C) Packed, max density")
    if not np.isnan(sum_inv_r6_D):
        R2_D_mw = np.array([R2_DD_from_sum_inv_r6(B0_fixed, tau_c_from_MW(MW), sum_inv_r6_D)
                             for MW in MW_vals])
        ax.plot(MW_vals, R2_D_mw, color=colors["capped"], lw=2, ls=":", label="(D) Packed, σ₀=0.01-capped")
    ax.plot(MW_vals, R2_struct_mw, color=colors["structural"], lw=2, ls="--",
            label=f"(E) Flat, σ₀={structural_sigma0:.3f}")
    if not np.isnan(sum_inv_r6_F):
        R2_F_mw = np.array([R2_DD_from_sum_inv_r6(B0_fixed, tau_c_from_MW(MW), sum_inv_r6_F)
                             for MW in MW_vals])
        ax.plot(MW_vals, R2_F_mw, color=colors["capped_structural"], lw=2, ls=(0, (3, 1, 1, 1)),
                label="(F) Packed, σ₀=struct-capped")

    ax.axvline(23, color="0.5", lw=0.8, ls=":")
    ax.text(24, ax.get_ylim()[1] * 0.9, "trypsin\n(23 kDa)", fontsize=7.5, color="0.4")
    ax.set_xlabel("MW (kDa)")
    ax.set_ylabel(r"$R_{2,\rm DD,b}$ (s⁻¹)")
    ax.set_title(f"R₂,DD vs MW  (B₀ = {B0_fixed} T, τ_c = 0.6 ns/kDa)")
    ax.legend(fontsize=7)

    # ── (4) Field-dependence ratio — unchanged (model-independent) ───────
    ax = axes[1, 1]
    B0_ref = 11.75  # T (500 MHz)
    R2_ref_23  = R2_DD_integral(B0_ref, tau_c_from_MW(23),  R_F)
    R2_ref_50  = R2_DD_integral(B0_ref, tau_c_from_MW(50),  R_F)
    R2_ref_100 = R2_DD_integral(B0_ref, tau_c_from_MW(100), R_F)

    for MW, R2_ref, ls in [(23, R2_ref_23, "-"), (50, R2_ref_50, "--"), (100, R2_ref_100, ":")]:
        tau_c = tau_c_from_MW(MW)
        ratio = np.array([R2_DD_integral(B, tau_c, R_F) for B in B0_vals]) / R2_ref
        ax.plot(B0_vals * GAMMA["1H"] / (2 * np.pi * 1e6), ratio,
                color=colors["flat"], lw=1.8, ls=ls, label=f"{MW} kDa")

    ax.axhline(1.0, color="0.6", lw=0.8, ls="-")
    ax.axvline(B0_ref * GAMMA["1H"] / (2 * np.pi * 1e6), color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"$\nu_H$ (MHz)")
    ax.set_ylabel(r"$R_{2,\rm DD}(B_0)\ /\ R_{2,\rm DD}(500\,\text{MHz})$")
    ax.set_title("Field-dependence of R₂,DD (relative to 500 MHz)\n"
                 "[ratio is identical for all 6 models — Σ1/r⁶ cancels]",
                 fontsize=10)
    ax.legend(fontsize=9, title="Protein MW")
    ax.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def print_summary(
    tau_c:    float,
    B0:       float,
    MW_kDa:   float,
    R_X_AA:   float | None = None,
) -> None:
    """Print a compact numerical comparison for the default (¹⁹F) case."""

    if R_X_AA is None:
        R_X_AA = VDW_RADIUS["F"]

    z    = R_H_SURF + R_X_AA
    I_AA = integral_I(R_X_AA)
    I_SI = integral_I_SI(R_X_AA)
    r_eq = r_eff_from_integral(R_X_AA)
    R2i  = R2_DD_integral(B0, tau_c, R_X_AA)
    R2r  = R2_DD_rulethumh(MW_kDa)
    R2_vdw = R2_DD_single_pair(B0, tau_c, z)   # single proton at contact distance

    nu_H = B0 * GAMMA["1H"] / (2 * np.pi * 1e6)

    header = "─" * 62
    print(f"\n{header}")
    print(f"  R₂,DD comparison  │  ¹⁹F, B₀ = {B0:.2f} T ({nu_H:.0f} MHz ¹H)")
    print(f"  Protein: {MW_kDa:.0f} kDa,  τ_c = {tau_c*1e9:.2f} ns")
    print(header)
    print(f"  Surface parameters")
    print(f"    σ₀            = {SIGMA0:.2e} Å⁻²")
    print(f"    R_H (surface) = {R_H_SURF:.2f} Å")
    print(f"    R_F (obs. nuc)= {R_X_AA:.2f} Å")
    print(f"    z = R_H+R_F   = {z:.2f} Å")
    print(header)
    print(f"  Integral  I       = {I_AA:.4e} Å⁻⁶  =  {I_SI:.4e} m⁻⁶")
    print(f"  Equiv. single-H   = {r_eq:.2f} Å  "
          f"(1 H at this distance gives same Σ(1/r⁶))")
    print(header)
    print(f"  {'Method':<38}  {'R₂,DD (s⁻¹)':>12}")
    print(f"  {'─'*38}  {'─'*12}")
    print(f"  {'(A) Rule-of-thumb  (MW kDa⁻¹ s⁻¹)':<38}  {R2r:>12.2f}")
    print(f"  {'(B) Surface integral':<38}  {R2i:>12.2f}")
    print(f"  {'    Single H at vdW contact':.<28} (z={z:.2f} Å)   {R2_vdw:>8.2f}")
    print(f"  {'Ratio  (B)/(A)':<38}  {R2i/R2r:>12.3f}")
    print(header)

    # Break down by field
    print(f"\n  {'B₀ (T)':>8}  {'νH (MHz)':>9}  {'R₂,DD integral':>15}  "
          f"{'R₂,DD rule':>12}  {'Ratio':>7}")
    print(f"  {'─'*8}  {'─'*9}  {'─'*15}  {'─'*12}  {'─'*7}")
    for B in [9.4, 11.75, 14.1, 16.4, 18.8, 23.5]:
        r2i = R2_DD_integral(B, tau_c, R_X_AA)
        r2r = R2_DD_rulethumh(MW_kDa)
        nu  = B * GAMMA["1H"] / (2 * np.pi * 1e6)
        print(f"  {B:>8.2f}  {nu:>9.0f}  {r2i:>15.3f}  {r2r:>12.2f}  {r2i/r2r:>7.3f}")
    print(header)

    # Also show for other atom types
    print(f"\n  Dependence on the observed nucleus (B₀={B0:.1f} T, MW={MW_kDa:.0f} kDa)")
    print(f"  {'Atom':>6}  {'R_X (Å)':>8}  {'z (Å)':>7}  "
          f"{'I (Å⁻⁶)':>12}  {'r_eff (Å)':>10}  {'R₂,DD (s⁻¹)':>13}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*10}  {'─'*13}")
    for atom, R_X in VDW_RADIUS.items():
        if atom == "H":
            continue
        nuc = {"C": "13C", "N": "15N", "F": "19F"}.get(atom)
        I_a   = integral_I(R_X)
        r_e   = r_eff_from_integral(R_X)
        if nuc:
            r2_a = R2_DD_integral(B0, tau_c, R_X,
                                  nuc_obs=nuc, nuc_partner="1H")
            r2_str = f"{r2_a:>13.3f}"
        else:
            r2_str = f"{'N/A':>13}"
        print(f"  {atom:>6}  {R_X:>8.2f}  {R_H_SURF+R_X:>7.2f}  "
              f"{I_a:>12.4e}  {r_e:>10.2f}  {r2_str}")
    print(header + "\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import sys

    # Default case: trypsin (23 kDa), B₀ = 16.4 T (700 MHz ¹H)
    MW_kDa = 23.0
    B0     = 16.4
    tau_c  = tau_c_from_MW(MW_kDa)

    make_figure(save_path="R2DD_comparison.png")

    # If an XYZ file is given, also produce the all-models figure (A-F),
    # which additionally requires molecular geometry for the
    # packed-surface models (C)/(D)/(F). Usage:
    #   python compare_R2DD.py molecule.xyz [MW_kDa] [B0_tesla] [structural_sigma0]
    if len(sys.argv) > 1:
        xyz_path = sys.argv[1]
        mw_arg = float(sys.argv[2]) if len(sys.argv) > 2 else MW_kDa
        b0_arg = float(sys.argv[3]) if len(sys.argv) > 3 else B0
        sigma0_arg = float(sys.argv[4]) if len(sys.argv) > 4 else 0.062
        make_figure_all_models(
            xyz_path=xyz_path, MW_kDa=mw_arg, B0_ref=b0_arg,
            structural_sigma0=sigma0_arg,
            save_path="R2DD_all_models_comparison.png",
        )
