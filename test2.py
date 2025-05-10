import numpy as np
import matplotlib.pyplot as plt
from module import Linear, TanH, Sigmoide
from sklearn.model_selection import train_test_split

def generate_gaussian_data(n_samples=100, seed=0):
    np.random.seed(seed)
    mean0 = [-1, -1]
    cov0 = [[0.3, 0], [0, 0.3]]
    X0 = np.random.multivariate_normal(mean0, cov0, n_samples)
    Y0 = np.zeros((n_samples, 1))

    mean1 = [1, 1]
    cov1 = [[0.3, 0], [0, 0.3]]
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    Y1 = np.ones((n_samples, 1))

    X = np.vstack((X0, X1))
    Y = np.vstack((Y0, Y1))
    indices = np.arange(2 * n_samples)
    np.random.shuffle(indices)
    return X[indices], Y[indices]

X, Y = generate_gaussian_data(n_samples=100)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

net = [
    Linear(2, 5),
    TanH(),
    Linear(5, 1),
    Sigmoide()
]

epochs = 1000
lr = 0.1
loss_history = []
acc_history = []
test_loss_history = []
test_acc_history = []

def forward_without_state(X, net):
    out = X
    for layer in net:
        if isinstance(layer, Linear):
            out = out @ layer._parameters + layer._bias
        elif isinstance(layer, TanH):
            out = np.tanh(out)
        elif isinstance(layer, Sigmoide):
            out = 1 / (1 + np.exp(-out))
    return out

for epoch in range(epochs):
    output = X_train
    for layer in net:
        output = layer.forward(output)
    yhat = output

    loss = np.mean((Y_train - yhat) ** 2)
    loss_history.append(loss)

    y_pred = (yhat >= 0.5).astype(int)
    accuracy = np.mean(y_pred == Y_train)
    acc_history.append(accuracy)

    test_yhat = forward_without_state(X_test, net)
    test_loss = np.mean((Y_test - test_yhat) ** 2)
    test_y_pred = (test_yhat >= 0.5).astype(int)
    test_accuracy = np.mean(test_y_pred == Y_test)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_accuracy)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: "
              f"Train Loss = {loss:.4f} | Train Acc = {accuracy:.2%} || "
              f"Test Loss = {test_loss:.4f} | Test Acc = {test_accuracy:.2%}")

    delta = -2 * (Y_train - yhat)
    for layer in reversed(net):
        layer.backward_update_gradient(getattr(layer, "_input", None), delta)
        delta = layer.backward_delta(getattr(layer, "_input", None), delta)

    for layer in net:
        layer.update_parameters(lr)
        layer.zero_grad()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss', linestyle='--')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(acc_history, label='Train Acc')
plt.plot(test_acc_history, label='Test Acc', linestyle='--')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(X_test[test_y_pred[:, 0] == 0][:, 0], X_test[test_y_pred[:, 0] == 0][:, 1],
            label="Predicted 0", alpha=0.6)
plt.scatter(X_test[test_y_pred[:, 0] == 1][:, 0], X_test[test_y_pred[:, 0] == 1][:, 1],
            label="Predicted 1", alpha=0.6)
plt.title("Test Classification Result")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
