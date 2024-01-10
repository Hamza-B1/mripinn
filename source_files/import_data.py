import pandas as pd
import torch
data = pd.read_csv("../data/CFD_vtm/csv_data/r001.csv")

x, y, z, u, v, w, p = (
    torch.tensor(data["Points_0"].values, requires_grad=True),
    torch.tensor(data["Points_1"].values, requires_grad=True),
    torch.tensor(data["Points_2"].values, requires_grad=True),
    torch.tensor(data["U_0"].values, requires_grad=True),
    torch.tensor(data["U_1"].values, requires_grad=True),
    torch.tensor(data["U_2"].values, requires_grad=True),
    torch.tensor(data["p"].values, requires_grad=True),
)

print(x.size())
print(x.t().size())
