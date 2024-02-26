from functools import partial
import os
from multiprocessing import Pool
import numpy as np
from mistsim import LSTBin, run_sampler, utils

# sampler
SAMPLER = "pymc"  # "pymc" or "pocomc"
# case
PATH = "CSA/CSA_beam_nominal_gsm_no_az_no_tilts_no_mountains.hdf5"
# chromaticity correction
CHROM = "pc"
assert CHROM in ["nc", "pc"]
print(PATH, CHROM)
NBINS = 24
SAVE_DIR = f"results/{CHROM}"
assert os.path.exists(SAVE_DIR)
SAVE_PATH = os.path.join(SAVE_DIR, f"results_{NBINS}bins_sweep_{SAMPLER}.npz")
assert not os.path.exists(SAVE_PATH)
print(SAVE_PATH)

VECTORIZE_LIKE = True  # vectorize the likelihood, not using parallelization
N_CPUS = 4
if not VECTORIZE_LIKE and N_CPUS > 1:
    os.environ["OMP_NUM_THREADS"] = "1"
TRUE_PARAMS = {"a": -0.2, "w": 20, "nu21": 80}
# Parameter bounds (a, w, nu21)
BOUNDS = np.array([[-1.0, 1.0], [1.0, 60.0], [45.0, 105.0]])
NDIM = len(BOUNDS)
NFG = np.arange(4, 9)

lst, freq, temp = utils.read_hdf5_convolution(f"simulations/{PATH}")
indx = (freq >= 45) * (freq <= 105)
freq = freq[indx]
temp = temp[:, indx]
if CHROM == "nc":
    BF = np.ones_like(temp)  # no chromaticity correction
elif CHROM == "pc":
    ac_path = PATH.replace(".hdf5", "_achromatic_75MHz.hdf5")
    ac_temp = utils.read_hdf5_convolution(f"simulations/{ac_path}")[-1]
    ac_temp = ac_temp[:, indx]
    BF = temp / ac_temp
nspec, nfreq = temp.shape


# injected 21cm signal
true_t21 = utils.gauss(freq, **TRUE_PARAMS)


# binning
cut = nspec % NBINS
total_temp = (temp + true_t21) / BF

if cut == 0:
    binned = total_temp.copy()
else:
    binned = total_temp[:-cut]
    BF_mean = BF[:-cut]
binned = binned.reshape(nspec // NBINS, NBINS, nfreq).mean(axis=0)
BF_mean = BF_mean.reshape(nspec // NBINS, NBINS, nfreq).mean(axis=0)

# noise
tint_ratio = (nspec // NBINS) / nspec
noise_75 = 3e-3
t75 = total_temp.mean(axis=0)[freq == 75]
noise, sigma_inv = utils.gen_noise(
    binned, t75, ref_noise=noise_75, tint_ratio=tint_ratio
)


def _loop(i, sampler, **kwargs):
    if sampler == "pocomc":
        vec = kwargs.pop("vec", VECTORIZE_LIKE)
        p = kwargs.pop("p", None)
        run = partial(run_sampler, vectorize=vec, pool=p, progress=False)
    elif sampler == "pymc":
        run = run_sampler
    else:
        raise ValueError("Unknown sampler")

    d = {}
    for n in NFG:
        print(f"{i=}: {n}/{NFG[-1]}")
        lst_bin = LSTBin(
            freq,
            binned[i] + noise[i],
            np.diag(sigma_inv[i]),
            n,
            chrom=BF_mean[i],
        )
        d[n] = run(sampler, BOUNDS, lst_bin)
    return d


if SAMPLER == "pymc":
    print("Running pymc")
    with Pool(N_CPUS) as pool:
        loop = partial(_loop, sampler="pymc")
        results = {i: r for i, r in enumerate(map(loop, range(NBINS)))}
elif SAMPLER == "pocomc":
    if VECTORIZE_LIKE or N_CPUS == 1:
        loop = partial(_loop, sampler="pocomc", vec=VECTORIZE_LIKE, p=None)
        results = {i: r for i, r in enumerate(map(loop, range(NBINS)))}
    else:
        with Pool(N_CPUS) as pool:
            loop = partial(_loop, sampler="pocomc", vec=False, p=pool)
            results = {i: r for i, r in enumerate(map(loop, range(NBINS)))}
else:
    raise ValueError("Unknown sampler")

# save the results
np.savez(
    SAVE_PATH,
    results=results,
    true_params=TRUE_PARAMS,
    noise=noise,
    noise_cov_inv=sigma_inv,
)
