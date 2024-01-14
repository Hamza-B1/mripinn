import torch
import pandas as pd


def create_data_tensors_from_csv(path):
    data = pd.read_csv(path)
    x = torch.tensor(data["Points_0"].values, requires_grad=True)
    y = torch.tensor(data["Points_1"].values, requires_grad=True)
    z = torch.tensor(data["Points_2"].values, requires_grad=True)

    u = torch.tensor(data["U_0"].values, requires_grad=True)
    v = torch.tensor(data["U_1"].values, requires_grad=True)
    w = torch.tensor(data["U_2"].values, requires_grad=True)
    p = torch.tensor(data["p"].values, requires_grad=True)

    inputs = torch.stack((x, y, z), dim=1)
    labels = torch.stack((u, v, w, p), dim=1)

    inputs.requires_grad_(True)
    labels.requires_grad_(True)
    return inputs, labels


create_data_tensors_from_csv(f"../data/CFD_vtm/csv_data/r001.csv")
