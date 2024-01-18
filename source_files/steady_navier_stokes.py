import torch
import torch.nn as nn
from import_data import create_data_tensors_from_csv

activation_function = nn.ReLU()
epochs = 200

dataset_1 = create_data_tensors_from_csv("../data/CFD_vtm/csv_data/r001.csv")
dataset_6 = create_data_tensors_from_csv("../data/CFD_vtm/csv_data/r006.csv")
dataset_11 = create_data_tensors_from_csv("../data/CFD_vtm/csv_data/r0011.csv")

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
            nn.Linear(20, 1)
        )

    def forward(self, x, y, z):
        inputs = torch.stack((x, y, z), dim=1)
        return self.model(inputs)


net = SteadyNavierStokes()
mse = nn.MSELoss()
optimiser = torch.optim.Adam(net.parameters(), lr=0.01)


def call_network(x, y, z, network):
    results = network(x, y, z)

    return results  # u, v, w




for epoch in range(epochs):
    for dataset in dataset_1, dataset_6, dataset_11:
        x = torch.tensor(dataset[0], requires_grad=True, dtype=torch.float32)
        y = torch.tensor(dataset[1], requires_grad=True, dtype=torch.float32)
        z = torch.tensor(dataset[2], requires_grad=True, dtype=torch.float32)
        u_train = torch.tensor(dataset[3], requires_grad=True, dtype=torch.float32)
        # v_train = torch.tensor(dataset[4], requires_grad=True, dtype=torch.float32)
        # w_train = torch.tensor(dataset[5], requires_grad=True, dtype=torch.float32)

        # u, v, w = call_network(x, y, z, net)

        u = call_network(x, y, z, net)
        u_mse = mse(torch.transpose(x, -1, 0), u_train)
        # v_mse = mse(v, v_train)
        # w_mse = mse(w, w_train)

        loss = u_mse  # + v_mse + w_mse
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(f"Loss: {loss.item()}")
        print(f"U value: {u[0]}")
