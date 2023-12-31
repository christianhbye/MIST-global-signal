import os
import numpy as np
from mistsim import LSTBin, Sampler, utils

N_CPUS = 4
if N_CPUS > 1:
    os.environ["OMP_NUM_THREADS"] = "1"
rng = np.random.default_rng(seed=1913)
TRUE_PARAMS = {"a": -0.2, "w": 20, "nu21": 80}
# Parameter bounds (a, w, nu21)
BOUNDS = np.array([[-1.0, 1.0], [1.0, 60.0], [45.0, 105.0]])
NDIM = len(BOUNDS)
N_PARTICLES = 1000
NBINS = 24
NFG = np.arange(4, 9)

lst, freq, temp = utils.read_hdf5_convolution(
    "simulations/CSA/CSA_beam_nominal_gsm_no_az_no_tilts_no_mountains.hdf5",
)
indx = (freq >= 45) * (freq <= 105)
freq = freq[indx]
temp = temp[:, indx]
nspec, nfreq = temp.shape
fg_mean = temp.mean(axis=0)  # avg spectrum, fg only
# this discards the last 6 min integration
fg_bin = temp[: -(nspec % NBINS)].reshape(nspec // NBINS, NBINS, nfreq)
tint_ratio = fg_bin.shape[0] / nspec
fg_bin = fg_bin.mean(axis=0)

# noise
noise_75 = 3e-3
t75 = fg_mean[freq == 75]
noise, noise_cov_inv = utils.gen_noise(
    fg_bin, t75, ref_noise=noise_75, tint_ratio=tint_ratio
)

# injected 21cm signal
true_t21 = utils.gauss(freq, **TRUE_PARAMS)

# initialize the sampler and run the simulations
sampler = Sampler(N_PARTICLES, NDIM, BOUNDS, n_cpus=N_CPUS)
results = {}
for i in range(NBINS):
    results[i] = {}
    for n in NFG:
        print(f"Bin {i+1}/{NBINS}, NFG = {n}/{NFG[-1]}")
        lst_bin = LSTBin(
            freq, fg_bin[i]+noise[i], np.diag(noise_cov_inv[i]), true_t21, n
        )
        results[i][n] = sampler.run_sampler(lst_bin)

# save the results
np.savez(
    "results_dec13_24bins.npz",
    results=results,
    true_params=TRUE_PARAMS,
    noise=noise,
    noise_cov_inv=noise_cov_inv,
)
