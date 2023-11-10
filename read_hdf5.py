import h5py
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    l, f, a = read_hdf5_convolution("sim.hdf5", print_key="yes")
    print(l.shape, f.shape, a.shape)
    plt.figure()
    #    plt.plot(f, a[0])
    plt.imshow(a, aspect="auto", extent=[f[0], f[-1], l[-1], l[0]])
    plt.colorbar()
    plt.show()
