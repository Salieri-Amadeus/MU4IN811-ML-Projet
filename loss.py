import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.sum((y - yhat) ** 2, axis=1)

    def backward(self, y, yhat):
        return -2 * (y - yhat) / y.shape[0]

class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        #eviter log(0)
        eps = 1e-12
        yhat = np.clip(yhat, eps, 1 - eps)
        return -np.sum(y * np.log(yhat), axis=1)

    def backward(self, y, yhat):
        return yhat - y
