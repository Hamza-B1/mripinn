import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
import import_data

torch.set_default_dtype(torch.float64)

# Hyperparameters
lr = 0.003
activation_function = nn.LeakyReLU()
epochs = 200


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

    def forward(self, data):
        return self.model(data)


model = SteadyNavierStokes()
mse_loss = nn.HuberLoss()
optimiser = torch.optim.Adam(lr=lr, params=model.parameters())

inputs = []
labels = []

for i in (1, 6, 11):
    x, y = import_data.create_data_tensors_from_csv(f"../data/CFD_vtm/csv_data/r00{i}.csv")
    inputs.append(x)
    labels.append(y)

start_time = time.time()
losses = []

for epoch in range(epochs):
    for i in range(len(inputs)):

        my_input = inputs[i]
        label = labels[i]

        predictions = model(my_input)
        loss = mse_loss(predictions, label)

        u = predictions[:, 0]
        x = my_input[:, 0]

#        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(f"Loss: {loss}")
        losses.append(loss.item())

elapsed = time.time() - start_time
print(f"Training time: {elapsed}")
plt.plot(losses)
plt.title("Loss over epochs")
plt.show()
