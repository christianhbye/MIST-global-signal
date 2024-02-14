import os
from multiprocessing import Pool
import numpy as np
from mistsim import LSTBin, run_sampler, utils

N_CPUS = 4
if N_CPUS > 1:
    os.environ["OMP_NUM_THREADS"] = "1"
rng = np.random.default_rng(seed=1913)
TRUE_PARAMS = {"a": -0.2, "w": 20, "nu21": 80}
# Parameter bounds (a, w, nu21)
BOUNDS = np.array([[-1.0, 1.0], [1.0, 60.0], [45.0, 105.0]])
NDIM = len(BOUNDS)
NBINS = 4  # XXX 24
NFG = np.arange(4, 6)  # XXX np.arange(4, 9)

lst, freq, temp = utils.read_hdf5_convolution(
    "simulations/CSA/CSA_beam_nominal_gsm_no_az_no_tilts_no_mountains.hdf5",
)
indx = (freq >= 45) * (freq <= 105)
freq = freq[indx]
temp = temp[:, indx]
nspec, nfreq = temp.shape


# injected 21cm signal
true_t21 = utils.gauss(freq, **TRUE_PARAMS)


# binning
cut = nspec % NBINS
total_temp = temp + true_t21

if cut == 0:
    binned = total_temp.copy()
else:
    binned = total_temp[:-cut]
binned = binned.reshape(nspec // NBINS, NBINS, nfreq).mean(axis=0)

# noise
tint_ratio = (nspec // NBINS) / nspec
noise_75 = 3e-3
t75 = total_temp.mean(axis=0)[freq == 75]
noise, sigma_inv = utils.gen_noise(
    binned, t75, ref_noise=noise_75, tint_ratio=tint_ratio
)


def _loop(i):
    d = {}
    for n in NFG:
        print(f"{i=}: {n}/{NFG[-1]}")
        lst_bin = LSTBin(freq, binned[i] + noise[i], np.diag(sigma_inv[i]), n)
        d[n] = run_sampler(BOUNDS, lst_bin)
    return d


with Pool(N_CPUS) as pool:
    results = {i: r for i, r in enumerate(pool.map(_loop, range(NBINS)))}

# save the results
np.savez(
    f"results_feb13_{NBINS}bins.npz",
    results=results,
    true_params=TRUE_PARAMS,
    noise=noise,
    noise_cov_inv=sigma_inv,
)
