import numpy as np
from scipy.stats import uniform
import pocomc as pc
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


def run_sampler(bounds, lst_bin, vectorize=True, pool=None, **kwargs):
    """
    Run the sampler.

    Parameters
    ----------
    bounds : ndarray
        The bounds of the parameter space for the prior distribution.
    lst_bin : LSTBin or list of LSTBin
    vectorize : bool
        Make a vectorized call to the likelihood function.
    pool : Pool
        A pool of workers for parallel evaluation of the likelihood. Does not
        seen to have an effect if vectorize is True.
    **kwargs : dict
        Passed to `pc.Sampler.run`.

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
    loc = bounds.T[0]
    scale = bounds.T[1] - bounds.T[0]
    prior = pc.Prior([uniform(loc=lo, scale=s) for lo, s in zip(loc, scale)])
    ndim = len(bounds)
    args = (prior, log_likelihood, ndim)
    sampler = pc.Sampler(
        *args, likelihood_args=[lst_bin], vectorize=vectorize, pool=pool
    )
    sampler.run(**kwargs)

    results = {}
    samples = sampler.posterior(resample=True)[0]
    results["samples"] = samples
    theta_map = np.mean(samples, axis=0)
    lnL = log_likelihood(theta_map, lst_bin)
    nparams = ndim + np.sum([spec.nfg for spec in lst_bin])
    nfreq = lst_bin[0].freq.size
    bic = -2 * lnL + nparams * np.log(nfreq)
    results["theta_map"] = theta_map
    results["bic"] = bic
    return results
