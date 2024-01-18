import torch
import torch.nn as nn
from import_data import create_data_tensors_from_csv

activation_function = nn.ReLU()

dataset_1 = create_data_tensors_from_csv("../data/CFD_vtm/csv_data/r001.csv")
dataset_6 = create_data_tensors_from_csv("../data/CFD_vtm/csv_data/r006.csv")
dataset_11 = create_data_tensors_from_csv("../data/CFD_vtm/csv_data/r0011.csv")

# for dataset in dataset_1, dataset_6, dataset_11:
#     dataset[0] =

class SteadyNavierStokes(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 20), activation_function,
            nn.Linear(20, 20), activation_function,
            nn.Linear(20, 20), activation_function,
            nn.Linear(20, 20), activation_function,
            nn.Linear(20, 20), activation_function,
            nn.Linear(20, 20), activation_function,
            nn.Linear(20, 3)
        )

    def forward(self, x, y, z):
        inputs = torch.stack((x, y, z), dim=1)
        return self.model(inputs)


net = SteadyNavierStokes()
mse = nn.MSELoss()
optimiser = torch.optim.Adam(net.parameters())


def call_network(x, y, z, network):
    results = network(x, y, z)

    u, v, w = results[:, 0], results[:, 1], results[:, 2]

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    v_z = torch.autograd.grad(v.sum(), z, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z.sum(), z, create_graph=True)[0]

    w_x = torch.autograd.grad(w.sum(), x, create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x.sum(), x, create_graph=True)[0]
    w_y = torch.autograd.grad(w.sum(), y, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y.sum(), y, create_graph=True)[0]
    w_z = torch.autograd.grad(w.sum(), z, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z.sum(), z, create_graph=True)[0]

    return u, v, w


epochs = 200

outputs = []
for epoch in range(epochs):
    for dataset in dataset_1, dataset_6, dataset_11:
        optimiser.zero_grad()

        x = torch.tensor(dataset[0], requires_grad=True, dtype=torch.float32)
        y = torch.tensor(dataset[1], requires_grad=True, dtype=torch.float32)
        z = torch.tensor(dataset[2], requires_grad=True, dtype=torch.float32)
        u_train = torch.tensor(dataset[3], requires_grad=True, dtype=torch.float32)
        v_train = torch.tensor(dataset[4], requires_grad=True, dtype=torch.float32)
        w_train = torch.tensor(dataset[5], requires_grad=True, dtype=torch.float32)

        u, v, w = call_network(x, y, z, net)

        outputs.append((u,v,w))

        u_mse = mse(u, u_train)
        v_mse = mse(v, v_train)
        w_mse = mse(w, w_train)

        loss = u_mse + v_mse + w_mse
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(loss.item())
