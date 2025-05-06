import numpy as np
import matplotlib.pyplot as plt
from module import Linear, TanH, Softmax  # 你需要添加 Softmax 类到 module.py 中
from sequentiel import Sequentiel
from optim import Optim
from sgd import SGD
from loss import CrossEntropyLoss
from sklearn.model_selection import train_test_split

# ====== 生成三类二维高斯分布数据 ======
def generate_multiclass_data(n_per_class=100, seed=0):
    np.random.seed(seed)
    means = [[-1, -1], [2, 0], [0, 2]]
    X = []
    Y = []
    for i, mean in enumerate(means):
        Xi = np.random.multivariate_normal(mean, np.eye(2)*0.2, n_per_class)
        yi = np.zeros((n_per_class, len(means)))
        yi[:, i] = 1  # one-hot 编码
        X.append(Xi)
        Y.append(yi)
    X = np.vstack(X)
    Y = np.vstack(Y)
    indices = np.random.permutation(len(X))
    return X[indices], Y[indices]

# ====== 数据准备 ======
X, Y = generate_multiclass_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# ====== 模型构建 ======
net = Sequentiel(
    Linear(2, 10),
    TanH(),
    Linear(10, 3),
    Softmax()
)

# ====== 损失与优化器 ======
loss_fn = CrossEntropyLoss()
optim = Optim(net, loss_fn, eps=0.1)

# ====== 训练 ======
history = SGD(optim, X_train, Y_train, batch_size=16, epochs=1000)

# ====== 测试集预测 ======
y_pred_proba = net.forward(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(Y_test, axis=1)
acc = np.mean(y_pred == y_true)

print(f"\nFinal test accuracy: {acc:.2%}")

# ====== 可视化训练曲线 ======
plt.plot(history["loss"], label="Loss")
plt.plot(history["accuracy"], label="Accuracy")
plt.title("Training Loss & Accuracy (Multiclass)")
plt.legend()
plt.show()

# ====== 可视化测试集分类结果（颜色区分） ======
colors = ['red', 'blue', 'green']
plt.title("Test set classification result")
for i in range(3):
    plt.scatter(X_test[y_pred == i][:, 0], X_test[y_pred == i][:, 1], label=f"Pred {i}", alpha=0.6, c=colors[i])
plt.legend()
plt.grid(True)
plt.show()
