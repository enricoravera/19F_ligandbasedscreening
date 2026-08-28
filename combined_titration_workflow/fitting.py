from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lmfit
import numpy as np


@dataclass
class FitSummary:
    success: bool
    message: str
    best_values: dict[str, float]
    stderr: dict[str, float]
    guesses: dict[str, float]


_GAMMA_MHZ_PER_T = {
    "19F": 40.0774,
    "1H": 42.5774,
    "13C": 10.7084,
    "15N": -4.3160,
}


def _gamma_rad_s_t(nucleus: str = "19F") -> float:
    return _GAMMA_MHZ_PER_T[nucleus] * 2.0 * np.pi * 1e6


def _spectral_density_iso(omega: float, tau: float) -> float:
    return tau / (1.0 + (omega * tau) ** 2)


def r2_csa_bound_guess(
    delta_sigma_ppm: float,
    eta: float,
    magnetic_field_T: float,
    tau_c_s: float,
    nucleus: str = "19F",
) -> float:
    delta_sigma = delta_sigma_ppm * 1e-6
    gamma = _gamma_rad_s_t(nucleus)
    omega = gamma * magnetic_field_T
    j0 = _spectral_density_iso(0.0, tau_c_s)
    j_omega = _spectral_density_iso(omega, tau_c_s)
    prefactor = (1.0 / 20.0) * gamma**2 * magnetic_field_T**2 * delta_sigma**2 * (1.0 + eta**2 / 3.0)
    return float(prefactor * (4.0 * j0 + 3.0 * j_omega))


def _bound_fraction(P_tot_uM: np.ndarray, KD_uM: float, L_tot_uM: float, n_sites: int = 1) -> np.ndarray:
    frac = n_sites * P_tot_uM / (KD_uM + L_tot_uM)
    return np.clip(frac, 0.0, 1.0)


def _stderr_map(result: lmfit.model.ModelResult) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in result.params.items():
        out[name] = float(param.stderr) if param.stderr is not None else float("nan")
    return out


def fit_cpmg_titration(
    protein_concentrations_uM: np.ndarray,
    r2_values_s_inv: np.ndarray,
    ligand_concentration_uM: float,
    delta_sigma_ppm: float,
    eta: float,
    magnetic_field_T: float,
    tau_c_s: float,
    *,
    protein_mw_kda: float = 23.0,
    n_sites: int = 1,
    fit_kd: bool = True,
    fit_r2free: bool = False,
    fix_r2b_to_guess: bool = False,
    r2_errors_s_inv: np.ndarray | None = None,
    kd_guess_uM: float = 500.0,
) -> FitSummary:
    prot = np.asarray(protein_concentrations_uM, dtype=float)
    r2_vals = np.asarray(r2_values_s_inv, dtype=float)

    r2_free_guess = r2_vals[0] if np.isclose(prot[0], 0.0) else float(np.min(r2_vals))
    r2_dd_guess = float(protein_mw_kda)
    r2_csa_guess = r2_csa_bound_guess(delta_sigma_ppm, eta, magnetic_field_T, tau_c_s, "19F")
    r2b_guess = r2_dd_guess + r2_csa_guess

    def cpmg_model(P_tot: np.ndarray, KD: float, R2_b: float, R2_free: float) -> np.ndarray:
        frac = _bound_fraction(P_tot, KD, ligand_concentration_uM, n_sites=n_sites)
        return R2_free + frac * (R2_b - R2_free)

    model = lmfit.Model(cpmg_model, independent_vars=["P_tot"])
    params = model.make_params(KD=kd_guess_uM, R2_b=max(r2b_guess, r2_free_guess * 1.01), R2_free=r2_free_guess)

    params["KD"].set(min=1e-3, max=1e6, vary=fit_kd)

    r2_b_upper = max(r2b_guess * 5.0, float(np.max(r2_vals)) * 20.0, r2_free_guess * 1.10)
    params["R2_b"].set(min=r2_free_guess * (1.0 + 1e-9), max=r2_b_upper, vary=not fix_r2b_to_guess)
    if fix_r2b_to_guess:
        params["R2_b"].set(value=max(r2b_guess, r2_free_guess * 1.01), vary=False)

    if fit_r2free:
        params["R2_free"].set(
            min=max(r2_free_guess * 0.5, 1e-6),
            max=max(r2_free_guess * 1.5, 1e-4),
            vary=True,
        )
    else:
        params["R2_free"].set(value=r2_free_guess, vary=False)

    weights = None
    if r2_errors_s_inv is not None:
        sigma = np.asarray(r2_errors_s_inv, dtype=float)
        if sigma.shape == r2_vals.shape and np.all(np.isfinite(sigma)) and np.all(sigma > 0):
            weights = 1.0 / sigma

    result = model.fit(r2_vals, params, P_tot=prot, weights=weights, max_nfev=10000)
    return FitSummary(
        success=bool(result.success),
        message=result.message,
        best_values={k: float(v) for k, v in result.best_values.items()},
        stderr=_stderr_map(result),
        guesses={
            "R2_b_guess": float(r2b_guess),
            "R2_DD_guess": float(r2_dd_guess),
            "R2_CSA_guess": float(r2_csa_guess),
            "R2_free_guess": float(r2_free_guess),
            "KD_guess": float(kd_guess_uM),
        },
    )


def stokes_einstein_diffusion_guess(
    radius_nm: float,
    *,
    temperature_k: float = 298.15,
    viscosity_pa_s: float = 0.00089,
) -> float:
    k_b = 1.380649e-23
    radius_m = radius_nm * 1e-9
    return float(k_b * temperature_k / (6.0 * np.pi * viscosity_pa_s * radius_m))


def fit_dosy_single_decay(
    protein_concentrations_uM: np.ndarray,
    b_values_s_m2: np.ndarray,
    intensity_matrix: np.ndarray,
    ligand_concentration_uM: float,
    *,
    free_radius_nm: float,
    bound_radius_nm: float,
    temperature_k: float = 298.15,
    viscosity_pa_s: float = 0.00089,
    n_sites: int = 1,
    fit_kd: bool = True,
    fix_d_free_to_guess: bool = False,
    fix_d_bound_to_guess: bool = False,
    kd_guess_uM: float = 500.0,
) -> FitSummary:
    prot = np.asarray(protein_concentrations_uM, dtype=float)
    bvals = np.asarray(b_values_s_m2, dtype=float)
    intensities = np.asarray(intensity_matrix, dtype=float)

    if intensities.ndim != 2:
        raise ValueError("intensity_matrix must be 2D with shape (n_protein_points, n_b_values)")
    if intensities.shape[0] != prot.shape[0]:
        raise ValueError("intensity_matrix row count must match protein_concentrations_uM length")
    if intensities.shape[1] != bvals.shape[0]:
        raise ValueError("intensity_matrix column count must match b_values_s_m2 length")

    row0 = np.clip(intensities[:, [0]], 1e-12, None)
    ynorm = intensities / row0

    d_free_guess = stokes_einstein_diffusion_guess(
        free_radius_nm,
        temperature_k=temperature_k,
        viscosity_pa_s=viscosity_pa_s,
    )
    d_bound_guess = stokes_einstein_diffusion_guess(
        bound_radius_nm,
        temperature_k=temperature_k,
        viscosity_pa_s=viscosity_pa_s,
    )

    def dosy_model(b: np.ndarray, P_tot: np.ndarray, KD: float, D_free: float, D_bound: float) -> np.ndarray:
        frac = _bound_fraction(P_tot[:, None], KD, ligand_concentration_uM, n_sites=n_sites)
        d_app = D_free + frac * (D_bound - D_free)
        return np.exp(-b[None, :] * d_app)

    model = lmfit.Model(dosy_model, independent_vars=["b", "P_tot"])
    params = model.make_params(KD=kd_guess_uM, D_free=d_free_guess, D_bound=d_bound_guess)
    params["KD"].set(min=1e-3, max=1e6, vary=fit_kd)
    params["D_free"].set(min=1e-14, max=1e-7, vary=not fix_d_free_to_guess)
    params["D_bound"].set(min=1e-14, max=1e-7, vary=not fix_d_bound_to_guess)

    if fix_d_free_to_guess:
        params["D_free"].set(value=d_free_guess, vary=False)
    if fix_d_bound_to_guess:
        params["D_bound"].set(value=d_bound_guess, vary=False)

    result = model.fit(ynorm, params, b=bvals, P_tot=prot, max_nfev=10000)
    return FitSummary(
        success=bool(result.success),
        message=result.message,
        best_values={k: float(v) for k, v in result.best_values.items()},
        stderr=_stderr_map(result),
        guesses={
            "D_free_guess": float(d_free_guess),
            "D_bound_guess": float(d_bound_guess),
            "KD_guess": float(kd_guess_uM),
        },
    )


def fit_single_inversion_recovery(delays_s: np.ndarray, intensities: np.ndarray) -> tuple[float, float, bool]:
    x = np.asarray(delays_s, dtype=float)
    y = np.asarray(intensities, dtype=float)

    def ir_model(t: np.ndarray, M0: float, T1: float, C: float) -> np.ndarray:
        return M0 * (1.0 - 2.0 * np.exp(-t / T1)) + C

    model = lmfit.Model(ir_model, independent_vars=["t"])
    t1_guess = max(np.median(x), 1e-4)
    m0_guess = max(np.max(y) - np.min(y), 1e-6)
    c_guess = float(np.median(y[-max(2, len(y)//3):]))

    params = model.make_params(M0=m0_guess, T1=t1_guess, C=c_guess)
    params["M0"].set(min=1e-12)
    params["T1"].set(min=1e-6, max=max(float(np.max(x)) * 100.0, 1e-2))

    result = model.fit(y, params, t=x, max_nfev=10000)
    t1 = float(result.params["T1"].value)
    t1_err = float(result.params["T1"].stderr) if result.params["T1"].stderr is not None else float("nan")
    return t1, t1_err, bool(result.success)


def fit_t1_inversion_recovery_titration(
    protein_concentrations_uM: np.ndarray,
    delays_s: np.ndarray,
    intensity_matrix: np.ndarray,
    ligand_concentration_uM: float,
    *,
    t1_bound_guess_s: float | None = None,
    n_sites: int = 1,
    fit_kd: bool = True,
    fix_t1_free_to_guess: bool = False,
    fix_t1_bound_to_guess: bool = False,
    kd_guess_uM: float = 500.0,
) -> dict[str, Any]:
    prot = np.asarray(protein_concentrations_uM, dtype=float)
    x = np.asarray(delays_s, dtype=float)
    y = np.asarray(intensity_matrix, dtype=float)

    if y.ndim != 2:
        raise ValueError("intensity_matrix must be 2D with shape (n_protein_points, n_delays)")
    if y.shape[0] != prot.shape[0]:
        raise ValueError("intensity_matrix row count must match protein_concentrations_uM length")
    if y.shape[1] != x.shape[0]:
        raise ValueError("intensity_matrix column count must match delays_s length")

    t1_vals, t1_errs = [], []
    for row in y:
        t1_i, err_i, _ = fit_single_inversion_recovery(x, row)
        t1_vals.append(t1_i)
        t1_errs.append(err_i)

    t1_arr = np.asarray(t1_vals, dtype=float)
    t1_err_arr = np.asarray(t1_errs, dtype=float)

    t1_free_guess = t1_arr[0] if np.isclose(prot[0], 0.0) else float(np.max(t1_arr))
    t1_bound_guess = float(t1_bound_guess_s) if t1_bound_guess_s is not None else float(np.median(t1_arr))

    def t1_model(P_tot: np.ndarray, KD: float, T1_free: float, T1_bound: float) -> np.ndarray:
        frac = _bound_fraction(P_tot, KD, ligand_concentration_uM, n_sites=n_sites)
        return T1_free + frac * (T1_bound - T1_free)

    model = lmfit.Model(t1_model, independent_vars=["P_tot"])
    params = model.make_params(KD=kd_guess_uM, T1_free=t1_free_guess, T1_bound=t1_bound_guess)
    params["KD"].set(min=1e-3, max=1e6, vary=fit_kd)
    params["T1_free"].set(min=1e-6, max=max(t1_free_guess * 10.0, 1e-3), vary=not fix_t1_free_to_guess)
    params["T1_bound"].set(min=1e-6, max=max(t1_bound_guess * 10.0, 1e-3), vary=not fix_t1_bound_to_guess)

    if fix_t1_free_to_guess:
        params["T1_free"].set(value=t1_free_guess, vary=False)
    if fix_t1_bound_to_guess:
        params["T1_bound"].set(value=t1_bound_guess, vary=False)

    weights = None
    if np.all(np.isfinite(t1_err_arr)) and np.all(t1_err_arr > 0):
        weights = 1.0 / t1_err_arr

    result = model.fit(t1_arr, params, P_tot=prot, weights=weights, max_nfev=10000)

    fit_summary = FitSummary(
        success=bool(result.success),
        message=result.message,
        best_values={k: float(v) for k, v in result.best_values.items()},
        stderr=_stderr_map(result),
        guesses={
            "T1_free_guess": float(t1_free_guess),
            "T1_bound_guess": float(t1_bound_guess),
            "KD_guess": float(kd_guess_uM),
        },
    )
    return {
        "apparent_t1_values_s": t1_arr.tolist(),
        "apparent_t1_errors_s": t1_err_arr.tolist(),
        "fit": fit_summary,
    }


def run_combined_workflow(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if "cpmg" in config:
        cpmg = config["cpmg"]
        out["cpmg"] = fit_cpmg_titration(
            protein_concentrations_uM=np.asarray(cpmg["protein_concentrations_uM"], dtype=float),
            r2_values_s_inv=np.asarray(cpmg["r2_values_s_inv"], dtype=float),
            ligand_concentration_uM=float(cpmg["ligand_concentration_uM"]),
            delta_sigma_ppm=float(cpmg["delta_sigma_ppm"]),
            eta=float(cpmg["eta"]),
            magnetic_field_T=float(cpmg["magnetic_field_T"]),
            tau_c_s=float(cpmg["tau_c_s"]),
            protein_mw_kda=float(cpmg.get("protein_mw_kda", 23.0)),
            n_sites=int(cpmg.get("n_sites", 1)),
            fit_kd=bool(cpmg.get("fit_kd", True)),
            fit_r2free=bool(cpmg.get("fit_r2free", False)),
            fix_r2b_to_guess=bool(cpmg.get("fix_r2b_to_guess", False)),
            r2_errors_s_inv=(
                None
                if cpmg.get("r2_errors_s_inv") is None
                else np.asarray(cpmg["r2_errors_s_inv"], dtype=float)
            ),
            kd_guess_uM=float(cpmg.get("kd_guess_uM", 500.0)),
        )

    if "dosy" in config:
        dosy = config["dosy"]
        out["dosy"] = fit_dosy_single_decay(
            protein_concentrations_uM=np.asarray(dosy["protein_concentrations_uM"], dtype=float),
            b_values_s_m2=np.asarray(dosy["b_values_s_m2"], dtype=float),
            intensity_matrix=np.asarray(dosy["intensity_matrix"], dtype=float),
            ligand_concentration_uM=float(dosy["ligand_concentration_uM"]),
            free_radius_nm=float(dosy["free_radius_nm"]),
            bound_radius_nm=float(dosy["bound_radius_nm"]),
            temperature_k=float(dosy.get("temperature_k", 298.15)),
            viscosity_pa_s=float(dosy.get("viscosity_pa_s", 0.00089)),
            n_sites=int(dosy.get("n_sites", 1)),
            fit_kd=bool(dosy.get("fit_kd", True)),
            fix_d_free_to_guess=bool(dosy.get("fix_d_free_to_guess", False)),
            fix_d_bound_to_guess=bool(dosy.get("fix_d_bound_to_guess", False)),
            kd_guess_uM=float(dosy.get("kd_guess_uM", 500.0)),
        )

    if "t1" in config:
        t1 = config["t1"]
        out["t1"] = fit_t1_inversion_recovery_titration(
            protein_concentrations_uM=np.asarray(t1["protein_concentrations_uM"], dtype=float),
            delays_s=np.asarray(t1["delays_s"], dtype=float),
            intensity_matrix=np.asarray(t1["intensity_matrix"], dtype=float),
            ligand_concentration_uM=float(t1["ligand_concentration_uM"]),
            t1_bound_guess_s=(None if t1.get("t1_bound_guess_s") is None else float(t1["t1_bound_guess_s"])),
            n_sites=int(t1.get("n_sites", 1)),
            fit_kd=bool(t1.get("fit_kd", True)),
            fix_t1_free_to_guess=bool(t1.get("fix_t1_free_to_guess", False)),
            fix_t1_bound_to_guess=bool(t1.get("fix_t1_bound_to_guess", False)),
            kd_guess_uM=float(t1.get("kd_guess_uM", 500.0)),
        )

    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, FitSummary):
            return {
                "success": value.success,
                "message": value.message,
                "best_values": value.best_values,
                "stderr": value.stderr,
                "guesses": value.guesses,
            }
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_to_jsonable(v) for v in value]
        return value

    return _to_jsonable(out)
