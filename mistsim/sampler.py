import numpy as np
import pocomc as pc
from . import utils
from .lstbin import LSTBin


def log_prior(params, bounds):
    """
    Uniform priors on all parameters.
    """
    if np.any(params < bounds.T[0]) or np.any(params > bounds.T[1]):
        return -np.inf
    else:
        return 0.0


def _log_likelihood_1spec(params, lst_bin):
    """
    The log likelihood for one spectrum.
    """
    t21_model = utils.gauss(lst_bin.freq, *params)
    dstar = lst_bin.bin_fg_mle(t21_model)[1]
    return -1 / 2 * dstar.T @ lst_bin.C_total_inv @ dstar


def log_likelihood(params, lst_bins):
    lnL = 0
    for lst_bin in lst_bins:
        lnL += _log_likelihood_1spec(params, lst_bin)
    return lnL


class Sampler:
    def __init__(self, n_particles, n_dim, bounds, seed=0):
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.bounds = bounds
        rng = np.random.default_rng(seed)
        self.prior_samples = rng.uniform(
            size=(n_particles, n_dim), low=bounds.T[0], high=bounds.T[1]
        )

    def run_sampler(self, lst_bin, add_samples=0, **kwargs):
        """
        Run the sampler.

        Parameters
        ----------
        lst_bin : LSTBin or list of LSTBin
        add_samples : int
            Number of additional samples to add to the sampler.
        **kwargs : dict
            Passed to `pc.Sampler.run`.

        Returns
        -------
        results : dict
            The dictionary returned by `pc.Sampler.run` with the following
            additional keys:
            - theta_map : ndarray
                The maximum a posteriori estimate of the parameters.
            - bic : float
                The Bayesian information criterion.

        """
        if isinstance(lst_bin, LSTBin):
            lst_bin = [lst_bin]
        args = (self.n_particles, self.n_dim, log_likelihood, log_prior)
        init_kwargs = {
            "bounds": self.bounds,
            "log_likelihood_args": [lst_bin],
            "log_prior_args": [self.bounds],
            "diagonal": False,
        }
        sampler = pc.Sampler(*args, **init_kwargs)
        sampler.run(self.prior_samples, **kwargs)
        if add_samples > 0:
            sampler.add_samples(add_samples)

        results = sampler.results.copy()
        theta_map = np.mean(results["samples"], axis=0)
        lnL = log_likelihood(theta_map, lst_bin)
        nparams = self.n_dim + np.sum([spec.nfg for spec in lst_bin])
        nfreq = lst_bin[0].freq.size
        bic = -2 * lnL + nparams * np.log(nfreq)
        results["theta_map"] = theta_map
        results["bic"] = bic
        return results
