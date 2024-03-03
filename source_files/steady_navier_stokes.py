import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
from import_data import create_data_from_csv
import numpy as np
from scipy.interpolate import griddata

device = ("cuda" if torch.cuda.is_available()
          else "mps"
if torch.backends.mps.is_available()
else "cpu"
          )

print(f"Using {device} device")

torch.set_default_dtype(torch.float64)

dataset_1 = create_data_from_csv("../data/CFD_vtm/csv_data/r001.csv")
dataset_6 = create_data_from_csv("../data/CFD_vtm/csv_data/r006.csv")
dataset_11 = create_data_from_csv("../data/CFD_vtm/csv_data/r0011.csv")
dataset_2 = create_data_from_csv("../data/CFD_vtm/csv_data/r002.csv")

# Hyperparameters
lr = 0.0009
activation_function = nn.LeakyReLU()
epochs = 20

# fluid parameters
nu = 0.1
rho = 1050


# In: x,y,z, Out: u,v,w,p
class SteadyNavierStokes(nn.Module):
    def __init__(self):
        super(SteadyNavierStokes, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 20), activation_function,
            nn.Linear(20, 20), activation_function,
            nn.Linear(20, 20), activation_function,
            nn.Linear(20, 20), activation_function,
            nn.Linear(20, 20), activation_function,
            nn.Linear(20, 20), activation_function,
            nn.Linear(20, 4)
        )

    def forward(self, x, y, z):
        inputs = torch.stack((x, y, z), dim=1)
        return self.model(inputs)


model = SteadyNavierStokes()
mse_loss = nn.MSELoss()
optimiser = torch.optim.Adam(lr=lr, params=model.parameters())

start_time = time.time()
losses = []
for epoch in range(epochs):
    for dataset in dataset_1, dataset_6, dataset_11:
        x = torch.tensor(dataset[0], requires_grad=True)
        y = torch.tensor(dataset[1], requires_grad=True)
        z = torch.tensor(dataset[2], requires_grad=True)
        u_train = torch.tensor(dataset[3], requires_grad=True)
        v_train = torch.tensor(dataset[4], requires_grad=True)
        w_train = torch.tensor(dataset[5], requires_grad=True)
        p_train = torch.tensor(dataset[6], requires_grad=True)
        predictions = model(x, y, z)

        u = predictions[:, 0]
        v = predictions[:, 1]
        w = predictions[:, 2]
        p = predictions[:, 3]

        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

        v_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        v_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        v_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
        v_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

        w_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        w_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        w_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        w_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        w_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
        w_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        p_z = torch.autograd.grad(p.sum(), z, create_graph=True)[0]

        u_loss = mse_loss(u, u_train)
        v_loss = mse_loss(v, v_train)
        w_loss = mse_loss(w, w_train)
        p_loss = mse_loss(p, p_train)

        div = u_x + v_y + w_z

        loss = u_loss + p_loss + v_loss + w_loss + torch.pow(div.sum(), 2)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(f"Loss: {loss.item()}")
        losses.append(loss.item())


elapsed = time.time() - start_time
print(f"Training time: {elapsed}")
plt.plot(losses)
plt.title("Loss over epochs")
plt.show()

x = torch.tensor(dataset_2[0], requires_grad=True)
y = torch.tensor(dataset_2[1], requires_grad=True)
z = torch.tensor(dataset_2[2], requires_grad=True)
true_w = dataset_2[5]

test = model(x, y, z)

u = test[:, 0]
v = test[:, 1]
w = test[:, 2]
p = test[:, 3]

filtered_y = []
filtered_z = []
filtered_w = []
filtered_true_w = []

for i in range(len(u)):
    if x[i] == 0:
        filtered_w.append(w[i])
        filtered_z.append(y[i])
        filtered_y.append(z[i])
        filtered_true_w.append(true_w[i])

xi = np.linspace(min(z.detach()), max(z.detach()), 100)
yi = np.linspace(min(y.detach()), max(y.detach()), 100)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((z.detach(), y.detach()), w.detach(), (xi, yi), method='cubic')
zi_true = griddata((z.detach(), y.detach()), w.detach(), (xi, yi), method='cubic')
plt.contourf(xi, yi, zi_true, levels=20)

plt.title("True Axial Velocity (m/s)")
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.colorbar()

plt.show()


