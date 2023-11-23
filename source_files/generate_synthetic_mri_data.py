import numpy as np


# imported CSV Format
# Point ID | p | x | y | z | p magnitude | ux | uy | uz | u_magnitude

def generate_synthetic_mri_data(path_to_cfd_csv_data: str,
                                n: int, mean: float, std: float,
                                output_path: str):
    data = np.genfromtxt(path_to_cfd_csv_data, dtype=float, delimiter=',', names=True)

    # downsample by removing every nth element
    mask = np.ones(data.size, dtype=bool)
    mask[::n] = False
    reduced_res = data[mask]

    # add Gaussian noise
    for i in reduced_res:
        noise = np.zeros(i.size)
        i[5:8] += [np.random.normal(i[5], std),
                   np.random.normal(i[6], std),
                   np.random.normal(i[7], std)
                   ]

    np.savetxt(output_path, reduced_res, delimiter=',')


generate_synthetic_mri_data("../data/fdanozzle/nozzlecfd.csv",
                            4, 0, 0.1,
                            "..data/fdanozzle/nozzlecfd_with_noise.csv")
