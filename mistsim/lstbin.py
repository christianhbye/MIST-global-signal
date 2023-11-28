import numpy as np
from .utils import design_mat


class LSTBin:
    def __init__(self, freq, spec, noise_cov_inv, injected_t21, nfg):
        """
        Class for a LST bin, holding a spectrum, noise, design matrix and
        necessary covariance matrices.

        Parameters
        -----------
        freq : np.ndarray
            The frequency array corresponding to the spectrum.
        spec : np.ndarray
            The spectrum at the given LST bin with noise added.
        noise_cov_inv : np.ndarray
            Inverse noise covariance matrix.
        injected_t21 : np.ndarray
            The injected 21cm signal. An array of temperature values, same
            length as ``freq''.
        nfg : int
            Number of foreground parameters to fit.

        """
        self.freq = freq
        self.spec = spec.copy()
        self.sigma_inv = noise_cov_inv
        self.injected_t21 = injected_t21
        self.nfg = nfg
        self.A = design_mat(self.freq, nfg=self.nfg)
        self.compute_covs()

    def compute_covs(self):
        """Compute necessary covariance matrices"""
        Cinv = self.A.T @ self.sigma_inv @ self.A
        self.C = np.linalg.inv(Cinv)
        self.sigma_fg = self.A @ self.C @ self.A.T
        V = np.linalg.inv(np.linalg.inv(self.sigma_fg) - self.sigma_inv)
        self.C_total_inv = np.linalg.inv(self.sigma + V)

    def bin_fg_mle(self, model_t21):
        return fg_mle(
            self.spec, self.A, self.sigma_inv, self.injected_t21, model_t21
        )


def fg_mle(spec, A, sigma_inv, true_t21, model_t21):
    """
    Compute MLE foreground parameters given a foreground spectrum, a design
    matrix, an inverse noise covariance matrix, an injected (true) 21cm signal,
    and a model 21cm signal.

    Parameters
    ----------
    spec : np.ndarray
        The foreground spectrum.
    A : np.ndarray
        The design matrix of the foreground model.
    sigma_inv : np.ndarray
        The inverse noise covariance matrix.
    true_t21 : np.ndarray
        The injected 21cm signal.
    model_t21 : np.ndarray
        The assumed model of the 21cm signal.

    Returns
    -------
    theta_hat : np.ndarray
        The MLE foreground parameters.
    dstar : np.ndarray
        The residual spectrum after subtracting the 21cm model and best-fit
        foregrounds from the total time_bin + injected 21cm spectrum.

    """
    d = spec + true_t21
    r = d - model_t21
    C = np.linalg.inv(A.T @ sigma_inv @ A)
    theta_hat = C @ A.T @ sigma_inv @ r
    dstar = r - A @ theta_hat  # eq 8 in Monsalve 2018
    return theta_hat, dstar
