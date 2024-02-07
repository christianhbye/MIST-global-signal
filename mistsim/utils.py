import h5py
import numpy as np


def read_hdf5_convolution(path_file, print_key=False):
    with h5py.File(path_file, "r") as hf:
        if print_key:
            print([key for key in hf.keys()])

        hfX = hf.get("lst")
        lst = np.array(hfX)

        hfX = hf.get("freq")
        freq = np.array(hfX)

        hfX = hf.get("ant_temp")
        ant_temp = np.array(hfX)

    return lst, freq, ant_temp


def gen_noise(spec, ref_temp, ref_noise=3e-3, tint_ratio=1, seed=0):
    """
    Scale noise from a reference temperature and frequency to a range of
    system temperatures and frequencies according to the radiometer equation.
    This assumes equal bandwith.

    Parameters
    ----------
    spec : np.ndarray
        Spectrum to add noise to.
    ref_temp : floa
        Reference temperature in Kelvin.
    ref_noise : float
        Standard deviation of the noise added to the reference temperature.
    tint_ratio : float
        Ratio of integration times between the spectrum and the reference
        spectrum.
    seed : int
        Random seed.

    Returns
    -------
    noise : np.ndarray
        Noise realization.
    noise_cov_inv : np.ndarray
        Inverse noise covariance matrix.

    """
    rng = np.random.default_rng(seed)
    noise_std = ref_noise * spec / ref_temp / np.sqrt(tint_ratio)
    noise = rng.normal(scale=noise_std)
    noise_cov_inv = 1 / noise_std**2
    return noise, noise_cov_inv


def design_mat(freq, nfg=5, beta=-2.5, nu_fg=75):
    """
    Generate a matrix of shape (Nfreq, Nfg) that evaluates the linlog model
    given a vector of foreground parameters.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies to evaluate the model at.
    nfg : int
        Number of foreground terms.
    beta : float
        Spectral index of power law.
    nu_fg : float
        Normalization frequency in same units as freq.

    Returns
    -------
    A : np.ndarray
        The design matrix, i.e., a matrix with the linlog basis functions as
        the columns.

    """
    f_ratio = freq[:, None] / nu_fg  # dimensionless frequency ratio
    powers = np.arange(nfg)[None]
    A = f_ratio**beta * np.log(f_ratio) ** powers
    return A


def gauss(f, a=-0.2, w=20, nu21=80):
    """
    Generate a Gaussian signal. Default parameters are the ones used in
    Monsalve et al. 2023b.

    Parameters
    ----------
    f : np.ndarray
        Frequencies to evaluate the model at.
    a : float
        Amplitude of the Gaussian.
    w : float
        FWHM of the Gaussian.
    nu21 : float
        Mean of the Gaussian.

    Returns
    -------
    g : np.ndarray
        The Gaussian signal evaluated at the input frequencies.

    """
    return a * np.exp(-1 / 2 * ((f - nu21) / w) ** 2 * 8 * np.log(2))
