import numpy as np

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        if self._gradient is not None:
            self._gradient.fill(0)
        if self._bias_gradient is not None:
            self._bias_gradient.fill(0)


    def forward(self, X):
        raise NotImplementedError

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        raise NotImplementedError

    def backward_delta(self, input, delta):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._parameters = np.random.randn(input_dim, output_dim) * 0.01
        self._bias = np.zeros((1, output_dim))
        self._gradient = np.zeros_like(self._parameters)
        self._bias_gradient = np.zeros_like(self._bias)

    def forward(self, X):
        self._input = X
        return X @ self._parameters + self._bias

    def backward_update_gradient(self, input, delta):
        self._gradient += input.T @ delta
        self._bias_gradient += np.sum(delta, axis=0, keepdims=True)

    def backward_delta(self, input, delta):
        return delta @ self._parameters.T

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient
        self._bias -= gradient_step * self._bias_gradient

    def zero_grad(self):
        super().zero_grad()
        self._bias_gradient.fill(0)

class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self._output = np.tanh(X)
        return self._output

    def backward_update_gradient(self, input, delta):
        # 没有参数，无需计算梯度
        pass

    def backward_delta(self, input, delta):
        # tanh'(x) = 1 - tanh(x)^2
        return delta * (1 - self._output ** 2)

    def update_parameters(self, gradient_step=1e-3):
        # 没有参数可更新
        pass

    def zero_grad(self):
        pass

class Sigmoide(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self._output = 1 / (1 + np.exp(-X))
        return self._output

    def backward_update_gradient(self, input, delta):
        # 没有参数
        pass

    def backward_delta(self, input, delta):
        # sigmoid'(x) = sigmoid(x)(1 - sigmoid(x))
        return delta * self._output * (1 - self._output)

    def update_parameters(self, gradient_step=1e-3):
        pass

    def zero_grad(self):
        pass

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self._input = X
        # 数值稳定性：减去每行最大值
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self._output = exp / np.sum(exp, axis=1, keepdims=True)
        return self._output

    def backward_update_gradient(self, input, delta):
        pass  # 无可训练参数

    def backward_delta(self, input, delta):
        # 使用 softmax 的雅可比矩阵乘以 delta，或直接使用误差
        return delta  # 对于交叉熵损失的输出层，delta = yhat - y，已是正确梯度

    def update_parameters(self, gradient_step=1e-3):
        pass

    def zero_grad(self):
        pass
