import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.sum((y - yhat) ** 2, axis=1)  # 返回向量，而不是平均数

    def backward(self, y, yhat):
        return -2 * (y - yhat) / y.shape[0]

class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        # 避免 log(0)
        eps = 1e-12
        yhat = np.clip(yhat, eps, 1 - eps)
        return -np.sum(y * np.log(yhat), axis=1)  # 每个样本的损失向量

    def backward(self, y, yhat):
        # yhat 是 softmax 输出，y 是 one-hot
        return yhat - y  # 注意：用于最后一层（softmax输出）的 delta
