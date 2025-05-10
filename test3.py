import numpy as np
import matplotlib.pyplot as plt
from module import Linear, TanH, Sigmoide
from sequentiel import Sequentiel
from optim import Optim
from sgd import SGD
from loss import CrossEntropyLoss
from sklearn.model_selection import train_test_split

def generate_xor_data(n=100):
    X00 = np.random.multivariate_normal([-1,-1], np.eye(2)*0.1, n)
    X01 = np.random.multivariate_normal([1,1], np.eye(2)*0.1, n)
    X10 = np.random.multivariate_normal([1,-1], np.eye(2)*0.1, n)
    X11 = np.random.multivariate_normal([-1,1], np.eye(2)*0.1, n)

    X0 = np.vstack((X00, X01))
    X1 = np.vstack((X10, X11))

    X = np.vstack((X0, X1))
    Y = np.vstack((np.zeros((2*n,1)), np.ones((2*n,1))))
    idx = np.random.permutation(len(X))
    return X[idx], Y[idx]

X, Y = generate_xor_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

net = Sequentiel(
    Linear(2, 10),
    TanH(),
    Linear(10, 1),
    Sigmoide()
)

loss_fn = CrossEntropyLoss()
optim = Optim(net, loss_fn, eps=0.1)

history = SGD(optim, X_train, Y_train, batch_size=16, epochs=1000)

y_test_pred = (net.forward(X_test) >= 0.5).astype(int)
acc = np.mean(y_test_pred == Y_test)
print(f"\nTest Accuracy: {acc:.2%}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history["loss"], label="Loss")
plt.plot(history["accuracy"], label="Accuracy")
plt.legend()
plt.title("Training loss and accuracy")

plt.subplot(1, 2, 2)
xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = net.forward(grid)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=["blue", "red"])
plt.scatter(X_test[:,0], X_test[:,1], c=Y_test[:,0], cmap='bwr', edgecolors='k')
plt.title("Decision Boundary (Test Set)")
plt.tight_layout()
plt.show()
