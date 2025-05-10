import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from module import Linear, TanH, Sigmoide
from sequentiel import Sequentiel
from optim import Optim
from sgd import SGD
from loss import MSELoss

digits = load_digits()
X = digits.data / 16.0
Y = X.copy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

latent_dim = 10
net = Sequentiel(
    Linear(64, 32),
    TanH(),
    Linear(32, latent_dim),
    TanH(),
    Linear(latent_dim, 32),
    TanH(),
    Linear(32, 64),
    Sigmoide()
)

loss_fn = MSELoss()
optim = Optim(net, loss_fn, eps=0.1)
history = SGD(optim, X_train, Y_train, batch_size=32, epochs=500)

X_test_encoded_decoded = net.forward(X_test)

n = 10
plt.figure(figsize=(10, 2))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(X_test_encoded_decoded[i].reshape(8, 8), cmap="gray")
    plt.title("Rebuilt")
    plt.axis("off")
plt.suptitle("Compression to 10 dimensions")
plt.tight_layout()
plt.show()
