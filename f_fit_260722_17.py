#! /usr/bin/env python3

from copy import deepcopy

import numpy as np
import lmfit
from scipy import stats
import klassez as kz

#fitting functions from TRAGICO

def exponential_ls(param, x, y, multi=1, result=False):
    """
    Least squares residuals for an exponential function with multiplicity from 1 to 3, 
    with the option to return the model.
    
    .. math::
    
        model = A*f(x, k) + a, 
    
    
    where :math:`f(x, k)` is the exponential function with multiplicity `multi` and parameters `k`, :math:`A` is the optimal scaling factor and :math:`a` is the optimal offset in the least squares sense.
    The model is computed by :func:`exponential_model`, which is called by this function.
    This function is taken from the `TRAGICO code`_.
    
    .. _TRAGICO code: https://github.com/letiziafiorucci/tragico  
    
    Parameters
    ----------
    param : lmfit.Parameters
        The parameters for the exponential model.
    x : array_like
        The independent variable data.
    y : array_like
        The dependent variable data.
    multi : int, optional
        The multiplicity of the exponential model (1, 2, or 3). Default is 1.
    result : bool, optional
        If True, return the model and the optimal A and a in the LS sense. Default is False.

    Returns
    -------
    array_like
        The residuals of the exponential model.
    or
    tuple
        If result is True, returns a tuple containing the model, optimal A, and optimal a.
    """
    
    model = exponential_model(param, x, multi)
    #print(model)
    den = np.mean(model**2)-np.mean(model)**2
    a = (np.mean(model**2)*np.mean(y)-np.mean(model*y)*np.mean(model))/den
    A = (np.mean(model*y)-(np.mean(model)*np.mean(y)))/den
  
    model *= A
    model += a  

    if any(np.isnan(model)):
        print(f'Nan in {multi}exp model')
    if result==False:

        return y-model
    else:
        return model, A, a
    
def exponential_model(param, x, multi=1, A=1, a=0):
    """
    Exponential model with multiplicity from 1 to 3.
    
    .. math::
    
        model = A*f(x, k) + a, 
    
    
    where :math:`f(x, k)` is the exponential function with multiplicity `multi` and parameters `k`, :math:`A` is the optimal scaling factor and :math:`a` is the optimal offset in the least squares sense.
    This function is taken from the `TRAGICO code`_.
    
    .. _TRAGICO code: https://github.com/letiziafiorucci/tragico  
    
    Parameters
    ----------
    param : lmfit.Parameters
        The parameters for the exponential model.
    x : array_like
        The independent variable data.
    multi : int, optional
        The multiplicity of the exponential model (1, 2, or 3). Default is 1.
    A : float, optional
        The scaling factor. Default is 1.
    a : float, optional
        The offset. Default is 0.

    Returns
    -------
    array_like
        The exponential model.
    """
    
    par=param.valuesdict()
    if multi==1:
        k = np.exp(par['k'])
        model = np.exp(-x*k)
    elif multi==2:
        k1 = np.exp(par['k1'])
        k2 = np.exp(par['k2'])
        f1 = 1 / (1 + np.exp(-par['f1']))  # Sigmoid transformation: maps (-∞, ∞) to (0, 1)
        model = f1*np.exp(-x*k1)+(1-f1)*np.exp(-x*k2)
    elif multi==3:
        k1 = np.exp(par['k1'])
        k2 = np.exp(par['k2'])
        k3 = np.exp(par['k3'])
        f1 = 1 / (1 + np.exp(-par['f1']))  # Sigmoid transformation: maps (-∞, ∞) to (0, 1)
        f2 = 1 / (1 + np.exp(-par['f2']))  # Sigmoid transformation: maps (-∞, ∞) to (0, 1)
        f2 = f2 * (1 - f1)                  # Maps (0, 1) to (0, 1-f1)
        model = f1*np.exp(-x*k1)+f2*np.exp(-x*k2)+(1-f1-f2)*np.exp(-x*k3)

    model *= A
    model += a

    return model

def exponential_ls_Jmod(param, x, y, multi=1, result=False):
    """
    Least squares residuals for an exponential function with multiplicity from 1 to 3, with J-coupling modulation.
    
    .. math::
    
        model = A*f(x, k, J) + a, 
    
    
    where :math:`f(x, k, J)` is the exponential function with multiplicity `multi` with a cosine modulation and parameters `k`, `J`, :math:`A` is the optimal scaling factor and :math:`a` is the optimal offset in the least squares sense.
    The model is computed by :func:`expontential_model_Jmod`, which is called by this function.

    Parameters
    ----------
    param : lmfit.Parameters
        The parameters for the cosine-modulated exponential model.
    x : array_like
        The independent variable data.
    y : array_like
        The dependent variable data.
    multi : int, optional
        The multiplicity of the exponential model (1, 2, or 3). Default is 1.
    result : bool, optional
        If True, return the model and the optimal A and a in the LS sense. Default is False.

    Returns
    -------
    array_like
        The residuals of the cosine-modulated exponential model.
    or
    tuple
        If result is True, returns a tuple containing the model, optimal A, and optimal a.
    """
    
    model = expontential_model_Jmod(param, x, multi)
    den = np.mean(model**2)-np.mean(model)**2
    a = (np.mean(model**2)*np.mean(y)-np.mean(model*y)*np.mean(model))/den
    A = (np.mean(model*y)-(np.mean(model)*np.mean(y)))/den
  
    model *= A
    model += a  

    if any(np.isnan(model)):
        print(f'Nan in {multi}exp Jmod model')
    if result==False:

        return y-model
    else:
        return model, A, a

def expontential_model_Jmod(param, x, multi=1, A=1, a=0):
    """
    Exponential model with multiplicity from 1 to 3, with J-coupling modulation.
    
    .. math::
    
        model = A*f(x, k, J) + a, 
    
    
    where :math:`f(x, k, J)` is the exponential function with multiplicity `multi` with a cosine modulation and parameters `k`, `J`, :math:`A` is the optimal scaling factor and :math:`a` is the optimal offset in the least squares sense.
    Wraps around :func:`exponential_model` to compute the exponential part of the model and adds a cosine modulation with frequency `J`.

    Parameters
    ----------
    param : lmfit.Parameters
        The parameters for the cosine-modulated exponential model.
    x : array_like
        The independent variable data.
    multi : int, optional
        The multiplicity of the exponential model (1, 2, or 3). Default is 1.
    A : float, optional
        The scaling factor. Default is 1.
    a : float, optional
        The offset. Default is 0.

    Returns
    -------
    array_like
        The cosine-modulated exponential model.
    """

    model = exponential_model(param, x, multi, A, a)
    model *= np.cos(2*np.pi*param['J'].value*x)
    return model
    
    
def fit_exponential(x, y, multi=1):
    """
    Fit an exponential function with multiplicity from 1 to 3 to the data (x, y) using least squares optimization.
    Calls :func:`exponential_ls` to compute the residuals and :func:`exponential_model` to compute the model.

    Parameters
    ----------
    x : array_like
        The independent variable data.
    y : array_like
        The dependent variable data.
    multi : int, optional
        The multiplicity of the exponential model (1, 2, or 3). Default is 1.

    Returns
    -------
    lmfit.MinimizerResult
        The result of the least squares optimization.
    """
    
    param = lmfit.Parameters()
    if multi==1:
        param.add('k', value=0, min=-7, max=7)
    if multi==2:
        param.add('k1', value=0, min=-7, max=7)
        param.add('k2', value=0, min=-7, max=7)
        param.add('f1', value=0, min=-5, max=5)  # f1 will be transformed to (0, 1) in the fitting function
    if multi==3:
        param.add('k1', value=0, min=-7, max=7)
        param.add('k2', value=0, min=-7, max=7)
        param.add('k3', value=0, min=-7, max=7)
        param.add('f1', value=0, min=-5, max=5)  # f1 will be transformed to (0, 1) in the fitting function
        param.add('f2', value=0, min=-5, max=5)  # f2 will be transformed to (0, 1-f1) in the fitting function
    minner = lmfit.Minimizer(exponential_ls, param, fcn_args=(x, y), fcn_kws={'multi': multi})
    result = minner.minimize()
    return result

def fit_exponential_Jmod(x, y, multi=1):
    """
    Fit an exponential function with multiplicity from 1 to 3 and J-coupling modulation to the data (x, y) using least squares optimization.
    Calls :func:`exponential_ls_Jmod` to compute the residuals and :func:`expontential_model_Jmod` to compute the model.

    Parameters
    ----------
    x : array_like
        The independent variable data.
    y : array_like
        The dependent variable data.
    multi : int, optional
        The multiplicity of the exponential model (1, 2, or 3). Default is 1.

    Returns
    -------
    lmfit.MinimizerResult
        The result of the least squares optimization.
    """
    
    param = lmfit.Parameters()
    if multi==1:
        param.add('k', value=0, min=-7, max=7)

    if multi==2:
        param.add('k1', value=0, min=-7, max=7)
        param.add('k2', value=0, min=-7, max=7)
        param.add('f1', value=0, min=-5, max=5)  # f1 will be transformed to (0, 1) in the fitting function

    if multi==3:
        param.add('k1', value=0, min=-7, max=7)
        param.add('k2', value=0, min=-7, max=7)
        param.add('k3', value=0, min=-7, max=7)
        param.add('f1', value=0, min=-5, max=5)  # f1 will be transformed to (0, 1) in the fitting function
        param.add('f2', value=0, min=-5, max=5)  # f2 will be transformed to (0, 1-f1) in the fitting function
    param.add('J', value=20, min=0, max=300)
    minner = lmfit.Minimizer(exponential_ls_Jmod, param, fcn_args=(x, y), fcn_kws={'multi': multi})
    result = minner.minimize()
    return result

#Functions for fitting spectra envelopes

def fit_skewnormal(x,y):
    r"""
    Fits the NH region of the 1D spectrum to a skew normal distribution using least squares optimization. The function returns the result of the optimization, which contains the fitted parameters of the skew normal distribution. The skew normal distribution is defined as:
    
    .. math:: 
    
    f(x) = A \cdot \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \left(1 + \text{erf}\left(\alpha \frac{x-\mu}{\sigma \sqrt{2}}\right)\right) + a
    
    
    Parameters
    ----------
    x : array_like
        The independent variable data (ppm values).
    y : array_like
        The dependent variable data (intensity values).
        
    Returns
    -------
    lmfit.MinimizerResult
        The result of the least squares optimization, which contains the fitted parameters of the skew normal distribution
    """
    
    param = lmfit.Parameters()
    param.add('a', value=4)
    param.add('u', value=8.25)
    param.add('s', value=0.6)
    minner = lmfit.Minimizer(skgaussian_ls, param, fcn_args=(x, y))
    result = minner.minimize()
    return result

def skgaussian_ls(param, x, y, result=False):
    """
    Computes the skew Gaussian model and optionally returns the model along with the scaling parameters.

    Parameters
    ----------
    param : lmfit.Parameters
        The parameters for the skew Gaussian model.
    x : array_like
        The independent variable data (ppm values).
    y : array_like
        The dependent variable data (intensity values).
    result : bool, optional
        If True, returns the model along with the scaling parameters. Default is False.

    Returns
    -------
    array_like
        The residuals (y - model) if result is False.
    tuple
        The model, A, and a if result is True.
    """
    
    model = kz.sim.f_skgaussian(x, param['u'].value, param['s'].value, 1, param['a'].value)
    #print(model)
    den = np.mean(model**2)-np.mean(model)**2
    a = (np.mean(model**2)*np.mean(y)-np.mean(model*y)*np.mean(model))/den
    A = (np.mean(model*y)-(np.mean(model)*np.mean(y)))/den
  
    model *= A
    model += a  

    if any(np.isnan(model)):
        print(f'Nan in skgaussian model')
    if result==False:

        return y-model
    else:
        return model, A, a

# ---------------------------------------------------------------------------
# Uncertainty extraction from lmfit results
# ---------------------------------------------------------------------------

def extract_R2_uncertainty(result, multi=1):
    """
    Extract R2 value(s) and propagated fitting uncertainty from an lmfit
    result produced by :func:`fit_exponential` or :func:`fit_exponential_Jmod`.

    Because the rate is fitted in log-space (``R2 = exp(k)``), the standard
    error of R2 follows by first-order error propagation::

        σ_R2 = |dR2/dk| × σ_k = exp(k) × σ_k = R2 × σ_k

    For multi-exponential models the amplitude-weighted average rate is
    returned together with its propagated uncertainty (approximate, assuming
    the fractions are perfectly known).

    Parameters
    ----------
    result : lmfit.MinimizerResult
        Output of :func:`fit_exponential` or :func:`fit_exponential_Jmod`.
    multi : int
        Multiplicity of the exponential model (1, 2, or 3).

    Returns
    -------
    R2 : float
        Fitted transverse relaxation rate (s⁻¹).
    sigma_R2 : float
        Propagated 1σ uncertainty of R2 (s⁻¹).
        ``nan`` if lmfit could not estimate the standard error (e.g. the
        covariance matrix was singular or the fit did not converge).

    Notes
    -----
    The linear parameters A (amplitude) and a (offset) are determined
    analytically inside the residual functions and therefore do not appear
    in the lmfit parameter set.  Their contribution to the uncertainty in R2
    is negligible when the signal-to-noise ratio is adequate.
    """
    import numpy as np

    par = result.params

    def _safe_stderr(p):
        """Return stderr or nan if unavailable."""
        s = par[p].stderr
        return s if (s is not None and np.isfinite(s)) else np.nan

    if multi == 1:
        k       = par['k'].value
        sigma_k = _safe_stderr('k')
        R2      = np.exp(k)
        sigma   = R2 * sigma_k          # σ_R2 = R2 × σ_k

    elif multi == 2:
        k1 = np.exp(par['k1'].value)
        k2 = np.exp(par['k2'].value)
        f1 = 1.0 / (1.0 + np.exp(-par['f1'].value))   # sigmoid → (0,1)
        f2 = 1.0 - f1
        R2 = f1 * k1 + f2 * k2                         # amplitude-weighted mean

        # Propagation: σ²_R2 = (f1·k1·σ_k1)² + (f2·k2·σ_k2)² + ((k1-k2)·σ_f)²
        # df/df1_raw = f1*(1-f1)  [derivative of sigmoid]
        sigma_k1  = k1 * _safe_stderr('k1')
        sigma_k2  = k2 * _safe_stderr('k2')
        df_df1raw = f1 * (1.0 - f1)
        sigma_f   = df_df1raw * _safe_stderr('f1')
        sigma     = np.sqrt(
            (f1 * sigma_k1) ** 2 +
            (f2 * sigma_k2) ** 2 +
            ((k1 - k2) * sigma_f) ** 2
        )

    elif multi == 3:
        k1 = np.exp(par['k1'].value)
        k2 = np.exp(par['k2'].value)
        k3 = np.exp(par['k3'].value)
        f1 = 1.0 / (1.0 + np.exp(-par['f1'].value))
        f2_raw = 1.0 / (1.0 + np.exp(-par['f2'].value))
        f2 = f2_raw * (1.0 - f1)
        f3 = 1.0 - f1 - f2
        R2 = f1 * k1 + f2 * k2 + f3 * k3

        sigma_k1   = k1 * _safe_stderr('k1')
        sigma_k2   = k2 * _safe_stderr('k2')
        sigma_k3   = k3 * _safe_stderr('k3')
        df1_draw   = f1 * (1.0 - f1)
        df2_draw   = f2_raw * (1.0 - f2_raw) * (1.0 - f1)
        sigma_f1   = df1_draw  * _safe_stderr('f1')
        sigma_f2   = df2_draw  * _safe_stderr('f2')
        sigma      = np.sqrt(
            (f1 * sigma_k1) ** 2 +
            (f2 * sigma_k2) ** 2 +
            (f3 * sigma_k3) ** 2 +
            ((k1 - k3) * sigma_f1) ** 2 +
            ((k2 - k3) * sigma_f2) ** 2
        )

    else:
        raise ValueError(f"multi must be 1, 2, or 3; got {multi}")

    return float(R2), float(sigma)

def plot_exponential_fit(a, x, y, result, multi=1, experiment_type='s', mask=None):
    """
    Plot the data and the fitted exponential model in the subplot that is passed to the function. The fitted model is computed using the parameters from the lmfit result.

    Parameters
    ----------
    a : matplotlib.axes.Axes
        The axes object to plot on.
    x : array_like
        The independent variable data.
    y : array_like
        The dependent variable data.
    result : lmfit.MinimizerResult
        The result of the least squares optimization.
    multi : int, optional
        The multiplicity of the exponential model (1, 2, or 3). Default is 1.
    experiment_type : str, optional
        ``'s'`` for selective (plain exponential) or ``'n'`` for non-selective
        (J-modulated exponential). Default is ``'s'``.
    mask : array_like of bool, optional
        Boolean mask of the same length as ``x`` and ``y``.  If provided,
        points where ``mask`` is ``True`` are plotted as filled blue circles
        ("inliers") and points where it is ``False`` are plotted as open red
        circles ("outliers excluded from fit").  If ``None`` (default), all
        points are plotted uniformly.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Generate model data using the fitted parameters
    if experiment_type == 's':
        model = exponential_ls(result.params, x, y, multi=multi, result=True)[0]
    elif experiment_type == 'n':
        model = exponential_ls_Jmod(result.params, x, y, multi=multi, result=True)[0]

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        a.plot(x[mask],  y[mask],  'bo', label='Data')
        if np.any(~mask):
            a.plot(x[~mask], y[~mask], 'ro', mfc='none', label='Outliers')
    else:
        a.plot(x, y, 'bo', label='Data')
    a.plot(x, model, 'r-', label='Fitted Model')
    a.set_xlabel('delay (s)')
    a.set_ylabel('intensity (a.u.)')

# ---------------------------------------------------------------------------
# Outlier-robust exponential fitting
# ---------------------------------------------------------------------------

def filter_outliers_fit_R2(x, y, multi=1, zthresh=2.0, max_iter=5, use_Jmod=False):
    """
    Outlier-robust exponential fit using iterative sigma-clipping (z-score
    thresholding).

    Fits an exponential model to ``(x, y)``, computes residuals, masks
    points whose z-score exceeds ``zthresh``, re-fits on the remaining
    points, and repeats until the mask stabilises or ``max_iter`` is
    reached.  Adapted from the TRAGICO ``filter_outliers_fit_T1`` routine.

    Parameters
    ----------
    x : 1-D array_like
        Independent variable (e.g. relaxation delays in seconds).
    y : 1-D array_like
        Experimental data (e.g. signal intensities).
    multi : int, optional
        Multiplicity of the exponential model (1, 2, or 3). Default is 1.
    zthresh : float, optional
        Z-score threshold above which a point is classified as an outlier.
        Default is 2.0.
    max_iter : int, optional
        Maximum number of sigma-clipping iterations. Default is 5.
    use_Jmod : bool, optional
        If ``True``, use the J-modulated exponential model
        (:func:`fit_exponential_Jmod` / :func:`exponential_ls_Jmod`);
        otherwise use the plain exponential (:func:`fit_exponential` /
        :func:`exponential_ls`). Default is ``False``.

    Returns
    -------
    mask : ndarray of bool
        Boolean mask of the same length as ``x``/``y``.  ``True`` = inlier
        (point was retained in the final fit); ``False`` = outlier (excluded
        by sigma-clipping).
    result : lmfit.MinimizerResult
        lmfit result object from the final iteration fit on the masked data.
    A : float
        Optimal amplitude scaling factor (linear parameter in the LS sense,
        computed inside the residual function).
    a : float
        Optimal offset (linear parameter, same source).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ls_func    = exponential_ls_Jmod if use_Jmod else exponential_ls
    model_func = expontential_model_Jmod if use_Jmod else exponential_model
    fit_func   = fit_exponential_Jmod   if use_Jmod else fit_exponential

    def _do_fit(x_m, y_m):
        return fit_func(x_m, y_m, multi=multi)

    # Start with all points included
    mask = np.ones(len(y), dtype=bool)

    result = _do_fit(x[mask], y[mask])

    for _ in range(max_iter):
        result = _do_fit(x[mask], y[mask])
        # Linear parameters (A, a) evaluated on the masked subset
        _, A, a = ls_func(result.params, x[mask], y[mask], multi=multi, result=True)
        # Model evaluated at ALL x (including masked-out points) for residuals
        y_fit = model_func(result.params, x, multi=multi, A=A, a=a)
        residual = y - y_fit
        # Set previously-masked points to NaN so they don't enter z-score
        residual[~mask] = np.nan
        zscore = np.abs(stats.zscore(residual, nan_policy='omit'))
        new_mask = zscore < zthresh
        # Stop if too few points remain (need at least n_params + 2)
        n_params = len(result.params.valuesdict())
        if np.sum(new_mask.astype(int)) < n_params + 2:
            break
        # Stop if the mask did not change
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    # Re-extract A and a from the final mask for the caller
    _, A, a = ls_func(result.params, x[mask], y[mask], multi=multi, result=True)
    return mask, result, A, a
