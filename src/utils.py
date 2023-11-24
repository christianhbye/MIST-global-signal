import h5py
import numpy as np


def read_hdf5_convolution(path_file, print_key=False):
    # path_file = home_folder + '/path/file.hdf5'
    # Show keys (array names inside HDF5 file)
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
