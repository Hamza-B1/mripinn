import torch
import pandas as pd


def create_data_tensors_from_csv(path):
    data = pd.read_csv(path)
    x = data["Points_0"].values
    y = data["Points_1"].values
    z = data["Points_2"].values

    u = data["U_0"].values
    v = data["U_1"].values
    w = data["U_2"].values
    p = data["p"].values

    return x, y, z, u, v, w, p
