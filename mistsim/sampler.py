import numpy as np
from scipy.stats import uniform
import pocomc as pc
import pymc as pm
from . import utils
from .lstbin import LSTBin


def _log_likelihood_1bin(params, lst_bin):
    """
    The log likelihood for one LST bin. This is a vectorized call so
    params is expected to be a 2D array with shape (nwalkers, ndim).

    Parameters
    ----------
    params : np.ndarray
        The parameters of the mock 21-cm signal. Shape (nwalkers, 3), where
        the columns are the amplitude, width, and center of the Gaussian.
    lst_bin : LSTBin

    Returns
    -------
    lnL : np.ndarray
        The log likelihood for each walker.
    """
    a, w, nu21 = params.T
    t21_model = utils.gauss(lst_bin.freq, a, w, nu21)
    dstar = lst_bin.bin_fg_mle(t21_model)[1]
    # the first axis of dstar is the walker axis, don't want to transpose that
    ds = np.expand_dims(dstar, axis=-1)
    dsT = np.expand_dims(dstar, axis=-2)
    Cinv = np.expand_dims(lst_bin.C_total_inv, axis=0)
    lnL = -1 / 2 * dsT @ Cinv @ ds
    return np.squeeze(lnL)


def log_likelihood(params, lst_bins):
    lnL = 0
    for lst_bin in lst_bins:
        lnL += _log_likelihood_1bin(params, lst_bin)
    return lnL


def uniform_prior(bounds):
    """
    A uniform prior distribution to be used with the pocomc sampler.

    Parameters
    ----------
    bounds : np.ndarray
        The bounds of the parameter space. Shape (ndim, 2).

    Returns
    -------
    prior : pc.Prior
        A uniform prior distribution.
    """
    loc = bounds.T[0]
    scale = bounds.T[1] - bounds.T[0]
    return pc.Prior([uniform(loc=lo, scale=s) for lo, s in zip(loc, scale)])


def run_sampler(sampler, prior_bounds, lst_bin, **kwargs):
    """
    Run the sampler.

    Parameters
    ----------
    sampler : str
        pymc or pocomc
    prior_bounds : array-like
        Bounds of uniform prior distribution for each parameter.
    lst_bin : LSTBin or list of LSTBin
    progress : bool
        Display a progress bar.
    kwargs : dict
        Additional keyword arguments passed to pm.sample or pc.Sampler.

    Returns
    -------
    results : arviz.InferenceData or dict
        See arviz documentation (if sampler is pymc) or the documentation in
        the ``_run_poco''-method (is sampler is pocomc) for details.

    """
    sampler = sampler.lower()
    if sampler == "pymc":
        return _run_pymc(prior_bounds, lst_bin, **kwargs)
    elif sampler == "pocomc":
        prior = uniform_prior(prior_bounds)
        progress = kwargs.pop("progress", True)
        return _run_poco(prior, lst_bin, progress=progress, **kwargs)
    else:
        raise ValueError("sampler must be 'pymc' or 'pocomc'")


def _run_pymc(bounds, lst_bin, **kwargs):
    names = ["a21", "w21", "nu21"]
    model = pm.Model()
    with model:
        a, w, nu21 = [pm.Uniform(n, *b) for n, b in zip(names, bounds)]
        t21_model = utils.tensor_gauss(lst_bin.freq, a, w, nu21)
        theta = lst_bin.bin_fg_mle(t21_model)[0]
        mu = pm.Deterministic("mu", lst_bin.A @ theta + t21_model)
        Y_obs = pm.MvNormal(
            "Y_obs", mu=mu, tau=lst_bin.C_total_inv, observed=lst_bin.spec
        )
        idata = pm.sample(**kwargs)
    return idata


def _run_poco(prior, lst_bin, progress=True, **kwargs):
    """
    Run the sampler.

    Parameters
    ----------
    prior : callable
        The prior distribution.
    lst_bin : LSTBin or list of LSTBin
    progress : bool
        Display a progress bar.
    kwargs : dict
        Additional keyword arguments passed to pc.Sampler.

    Returns
    -------
    results : dict
        Summary statistics of the sampler. Contains the following keys:
        - samples : ndarray
            The samples from the posterior distribution.
        - theta_map : ndarray
            The maximum a posteriori estimate of the parameters.
        - bic : float
            The Bayesian information criterion evaluated at the MAP estimate.

    """
    if isinstance(lst_bin, LSTBin):
        lst_bin = [lst_bin]
    sampler = pc.Sampler(
        prior, log_likelihood, likelihood_args=[lst_bin], **kwargs
    )
    sampler.run(progress=progress)

    results = {}
    samples = sampler.posterior(resample=True)[0]
    results["samples"] = samples
    theta_map = np.mean(samples, axis=0)
    lnL = log_likelihood(theta_map, lst_bin)
    # 3 parameters for the Gaussian + 1 for each foreground component
    nparams = 3 + np.sum([spec.nfg for spec in lst_bin])
    nfreq = lst_bin[0].freq.size
    bic = -2 * lnL + nparams * np.log(nfreq)
    results["theta_map"] = theta_map
    results["bic"] = bic
    return results
