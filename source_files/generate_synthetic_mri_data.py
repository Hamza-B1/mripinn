import numpy as np


def deresolution_and_add_noise(path_to_cfd_csv_data: str,
                               n: int, mean: float, std: float,
                               output_path: str):
    data = np.genfromtxt(path_to_cfd_csv_data, dtype=float, delimiter=',', names=True)

    # downsample by removing every nth element
    mask = np.ones(data.size, dtype=bool)
    mask[::n] = False
    reduced_res = data[mask]

    # add Gaussian noise
    for i in range(reduced_res.size):
        data = reduced_res[i]
        reduced_res[i][6] += np.random.normal(reduced_res[i][6], std)
        reduced_res[i][7] += np.random.normal(reduced_res[i][7], std)
        reduced_res[i][8] += np.random.normal(reduced_res[i][8], std)

    np.savetxt("../data/fdanozzle/nozzlecfd_with_noise.csv", reduced_res, delimiter=',')


deresolution_and_add_noise("../data/fdanozzle/nozzlecfd.csv",
                           4, 0, 0.005,
                           "..data/fdanozzle/nozzlecfd_with_noise.csv")
