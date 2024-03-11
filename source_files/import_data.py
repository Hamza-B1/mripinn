import math
import pandas as pd


def create_data_from_csv(path):
    data = pd.read_csv(path)
    x = data["Points_0"].values
    y = data["Points_1"].values
    z = data["Points_2"].values

    u = data["U_0"].values
    v = data["U_1"].values
    w = data["U_2"].values
    p = data["p"].values

    for i in x, y, z, u, v, w, p:
        max_val = max(i)
        min_val = min(i)
        i = [(v - min_val) / (max_val - min_val) for v in i]

    return x, y, z, u, v, w, p
