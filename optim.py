class Optim:
    def __init__(self, net, loss, eps=1e-3):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        # Forward
        yhat = self.net.forward(batch_x)
        # Loss (scalar)
        loss_value = self.loss.forward(batch_y, yhat).mean()
        # Backward
        delta = self.loss.backward(batch_y, yhat)
        self.net.zero_grad()
        self.net.backward(delta)
        self.net.update_parameters(self.eps)
        return loss_value
