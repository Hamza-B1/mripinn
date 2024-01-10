import torch
import torch.nn as nn
import pandas as pd

device = ("cuda" if torch.cuda.is_available()
          else "mps"
if torch.backends.mps.is_available()
else "cpu"
          )
print(f"Using {device} device")

# Hyperparameters
lr = 0.01
activation_function = nn.Sigmoid()
epochs = 1
# Fluid quantities
nu = 1
rho = 1050


# In: x,y,z, Out: u,v,w,p
class SteadyNavierStokes(nn.Module):
    def __init__(self):
        super(SteadyNavierStokes, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 10), activation_function,
            nn.Linear(10, 10), activation_function,
            nn.Linear(10, 10), activation_function,
            nn.Linear(10, 10), activation_function,
            nn.Linear(10, 10), activation_function,
            nn.Linear(10, 10), activation_function,
            nn.Linear(10, 10), activation_function,
            nn.Linear(10, 4)
        )

    def forward(self, x):
        return self.model(x)


model = SteadyNavierStokes().to(device)
mse_loss = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=lr)


def physics_loss(input_tensor, num_points, output_tensor):
    x, y, z = torch.split(input_tensor, num_points, dim=1)
    u, v, w, p = torch.split(output_tensor, num_points, dim=1)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]

    incompressibility_error = u_x + v_y + w_z

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), create_graph=True)[0]

    momentum_conservation_error_x = u * u_x + v * u_y + w * u_z - (-1 / rho * p_x + nu * (u_xx + u_yy + u_zz))
    momentum_conservation_error_y = u * v_x + v * v_y + w * v_z - (-1 / rho * p_y + nu * (v_xx + v_yy + v_zz))
    momentum_conservation_error_z = u * u_x + v * u_y + w * u_z - (-1 / rho * p_z + nu * (w_xx + w_yy + w_zz))

    return incompressibility_error + momentum_conservation_error_x + momentum_conservation_error_y + momentum_conservation_error_z


data = pd.read_csv("../data/CFD_vtm/csv_data/r001.csv")

x_in, y_in, z_in, v_xin, v_yin, v_zin, p_in = (
    torch.tensor(data["Points_0"].values, requires_grad=True).to(device),
    torch.tensor(data["Points_1"].values, requires_grad=True).to(device),
    torch.tensor(data["Points_2"].values, requires_grad=True).to(device),
    torch.tensor(data["U_0"].values, requires_grad=True).to(device),
    torch.tensor(data["U_1"].values, requires_grad=True).to(device),
    torch.tensor(data["U_2"].values, requires_grad=True).to(device),
    torch.tensor(data["p"].values, requires_grad=True).to(device),
)

input_data = torch.stack((x_in, y_in, z_in), 1).to(device)
label = torch.stack((v_xin, v_yin, v_zin, p_in), 1).to(device)

for epoch in range(epochs):
    num_of_points = x_in.size

    loss = mse_loss(input_data)
    optimiser.zero_grad()
