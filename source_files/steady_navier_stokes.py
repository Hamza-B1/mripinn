import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
import import_data

torch.set_default_dtype(torch.float64)

# Hyperparameters
lr = 0.01
activation_function = nn.ReLU()
epochs = 40


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

    def forward(self, data):
        return self.model(data)


model = SteadyNavierStokes()
mse_loss = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=lr)

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
        num_of_points = my_input.size()
        predictions = model(my_input)
        loss = mse_loss(predictions, label)
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
