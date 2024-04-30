import numpy as np


def generate_mri_data(x, y, z):
    phi = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return np.exp(j)


def interpolate_vector_field(x, y, z, u, v, w, voxel_size):
    grid_shape = tuple(int(np.ceil((max(coord) - min(coord)) / voxel_size)) + 1 for coord in [x, y, z])

    grid_u = np.zeros(grid_shape)
    grid_v = np.zeros(grid_shape)
    grid_w = np.zeros(grid_shape)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                # Determine position in vector field data
                position = np.array([i * voxel_size, j * voxel_size, k * voxel_size])

                idx = np.floor(position / voxel_size).astype(int)

                dx = (position[0] - idx[0] * voxel_size) / voxel_size
                dy = (position[1] - idx[1] * voxel_size) / voxel_size
                dz = (position[2] - idx[2] * voxel_size) / voxel_size

                grid_u[i, j, k] = (1 - dx) * (1 - dy) * (1 - dz) * u[idx[0], idx[1], idx[2]] + \
                                  dx * (1 - dy) * (1 - dz) * u[idx[0] + 1, idx[1], idx[2]] + \
                                  (1 - dx) * dy * (1 - dz) * u[idx[0], idx[1] + 1, idx[2]] + \
                                  dx * dy * (1 - dz) * u[idx[0] + 1, idx[1] + 1, idx[2]] + \
                                  (1 - dx) * (1 - dy) * dz * u[idx[0], idx[1], idx[2] + 1] + \
                                  dx * (1 - dy) * dz * u[idx[0] + 1, idx[1], idx[2] + 1] + \
                                  (1 - dx) * dy * dz * u[idx[0], idx[1] + 1, idx[2] + 1] + \
                                  dx * dy * dz * u[idx[0] + 1, idx[1] + 1, idx[2] + 1]

                grid_v[i, j, k] = (1 - dx) * (1 - dy) * (1 - dz) * v[idx[0], idx[1], idx[2]] + \
                                  dx * (1 - dy) * (1 - dz) * v[idx[0] + 1, idx[1], idx[2]] + \
                                  (1 - dx) * dy * (1 - dz) * v[idx[0], idx[1] + 1, idx[2]] + \
                                  dx * dy * (1 - dz) * v[idx[0] + 1, idx[1] + 1, idx[2]] + \
                                  (1 - dx) * (1 - dy) * dz * v[idx[0], idx[1], idx[2] + 1] + \
                                  dx * (1 - dy) * dz * v[idx[0] + 1, idx[1], idx[2] + 1] + \
                                  (1 - dx) * dy * dz * v[idx[0], idx[1] + 1, idx[2] + 1] + \
                                  dx * dy * dz * v[idx[0] + 1, idx[1] + 1, idx[2] + 1]

                grid_w[i, j, k] = (1 - dx) * (1 - dy) * (1 - dz) * w[idx[0], idx[1], idx[2]] + \
                                  dx * (1 - dy) * (1 - dz) * w[idx[0] + 1, idx[1], idx[2]] + \
                                  (1 - dx) * dy * (1 - dz) * w[idx[0], idx[1] + 1, idx[2]] + \
                                  dx * dy * (1 - dz) * w[idx[0] + 1, idx[1] + 1, idx[2]] + \
                                  (1 - dx) * (1 - dy) * dz * w[idx[0], idx[1], idx[2] + 1] + \
                                  dx * (1 - dy) * dz * w[idx[0] + 1, idx[1], idx[2] + 1] + \
                                  (1 - dx) * dy * dz * w[idx[0], idx[1] + 1, idx[2] + 1] + \
                                  dx * dy * dz * w[idx[0] + 1, idx[1] + 1, idx[2] + 1]

    return grid_u, grid_v, grid_w
