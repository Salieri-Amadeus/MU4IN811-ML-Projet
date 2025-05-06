from module import Linear, TanH, Sigmoide
from sequentiel import Sequentiel
from optim import Optim
from sgd import SGD
from loss import MSELoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 数据生成
def generate_gaussian_data(n=100):
    X0 = np.random.multivariate_normal([-1,-1], np.eye(2)*0.3, n)
    X1 = np.random.multivariate_normal([1,1], np.eye(2)*0.3, n)
    X = np.vstack((X0, X1))
    Y = np.vstack((np.zeros((n,1)), np.ones((n,1))))
    idx = np.random.permutation(2*n)
    return X[idx], Y[idx]

X, Y = generate_gaussian_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# 模型构建
net = Sequentiel(
    Linear(2, 5),
    TanH(),
    Linear(5, 1),
    Sigmoide()
)

loss_fn = MSELoss()
optim = Optim(net, loss_fn, eps=0.1)

# 训练
hist = SGD(optim, X_train, Y_train, batch_size=16, epochs=1000)

# 可视化
plt.plot(hist["loss"], label="loss")
plt.plot(hist["accuracy"], label="accuracy")
plt.legend()
plt.title("Training Loss and Accuracy")
plt.show()
