import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError

class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.mean((y - yhat) ** 2)

    def backward(self, y, yhat):
        return -2 * (y - yhat) / y.shape[0]

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        if self._gradient is not None:
            self._gradient.fill(0)

    def forward(self, X):
        ## Calcule la passe forward
        raise NotImplementedError

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        raise NotImplementedError

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        raise NotImplementedError

class Linear(Module):
    """ 线性层模块：y = XW + b """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._parameters = np.random.randn(input_dim, output_dim) * 0.01  # 初始化权重
        self._bias = np.zeros((1, output_dim))  # 初始化偏置项
        self._gradient = np.zeros_like(self._parameters)
        self._bias_gradient = np.zeros_like(self._bias)

    def forward(self, X):
        """ 计算前向传播 """
        self._input = X  # 存储输入以备反向传播使用
        return X @ self._parameters + self._bias  # 计算线性变换

    def backward_update_gradient(self, input, delta):
        """ 计算梯度并累积到 _gradient 变量 """
        self._gradient += input.T @ delta  # 计算权重的梯度
        self._bias_gradient += np.sum(delta, axis=0, keepdims=True)  # 计算偏置的梯度

    def backward_delta(self, input, delta):
        """ 计算损失对输入的梯度，供前一层反向传播使用 """
        return delta @ self._parameters.T  # 反向传播 delta

    def update_parameters(self, gradient_step=1e-3):
        """ 使用累积的梯度更新参数 """
        self._parameters -= gradient_step * self._gradient
        self._bias -= gradient_step * self._bias_gradient

X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 输入数据 (batch=2, dim=2)
Y = np.array([[1.0], [0.0]])  # 真实输出 (batch=2, output=1)

# 创建线性层和损失函数
linear = Linear(input_dim=2, output_dim=1)
loss_fn = MSELoss()

# 前向传播
Y_hat = linear.forward(X)
loss = loss_fn.forward(Y, Y_hat)
print(f"Loss: {loss}")

# 反向传播
delta = loss_fn.backward(Y, Y_hat)
linear.backward_update_gradient(X, delta)
grad_input = linear.backward_delta(X, delta)

# 更新参数
linear.update_parameters(gradient_step=0.01)
print(f"Updated parameters: {linear._parameters}")
loss2 = loss_fn.forward(Y, linear.forward(X))
print(f"Loss after update: {loss2}")
