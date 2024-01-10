import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0


# model prediction
def forward(x):
    return w * x


# loss
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()


# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y)
def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

learning_rate = 0.01
iters = 10

for epoch in range(iters):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    dw = gradient(X, Y, y_pred)

    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'Epoch: {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'After training: {forward(5):.3f}')
