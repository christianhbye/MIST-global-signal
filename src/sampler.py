from multiprocessing import Pool
import numpy as np
import pocomc as pc
import utils


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
    def __init__(self, n_particles, n_dim, bounds, rng, n_cpus=1):
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.bounds = bounds
        self.n_cpus = n_cpus
        self.prior_samples = rng.uniform(
            size=(n_particles, n_dim), low=bounds.T[0], high=bounds.T[1]
        )

    def run_sampler(self, lst_bin):
        args = (self.n_particles, self.n_dim, log_likelihood, log_prior)
        kwargs = {
            "bounds": self.bounds,
            "log_likelihood_args": [[lst_bin]],
            "log_prior_args": [self.bounds],
            "diagonal": False,
        }
        if self.n_cpus > 1:
            with Pool(self.n_cpus) as pool:
                kwargs["pool"] = pool
                sampler = pc.Sampler(*args, **kwargs)
                sampler.run(self.prior_samples)
                logz_bs, logz_bs_error = sampler.bridge_sampling()
        else:
            sampler = pc.Sampler(*args, **kwargs)
            sampler.run(self.prior_samples)
            logz_bs, logz_bs_error = sampler.bridge_sampling()

        results = sampler.results.copy()
        results["logz_bs"] = logz_bs
        results["logz_bs_error"] = logz_bs_error
        return results
