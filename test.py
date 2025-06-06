import numpy as np
from module import Linear
from loss import MSELoss

X = np.array([[1.0, 2.0], [3.0, 4.0]])
Y = np.array([[1.0], [0.0]])

linear = Linear(input_dim=2, output_dim=1)
loss_fn = MSELoss()

Y_hat = linear.forward(X)
loss = loss_fn.forward(Y, Y_hat)
print(f"Loss: {loss}")

delta = loss_fn.backward(Y, Y_hat)
linear.backward_update_gradient(X, delta)
grad_input = linear.backward_delta(X, delta)

linear.update_parameters(gradient_step=0.01)
linear.zero_grad()

print(f"Updated parameters: {linear._parameters}")
loss2 = loss_fn.forward(Y, linear.forward(X))
print(f"Loss after update: {loss2}")
