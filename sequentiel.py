class Sequentiel:
    def __init__(self, *modules):
        self.modules = list(modules)

    def forward(self, X):
        out = X
        for module in self.modules:
            out = module.forward(out)
        return out

    def backward(self, delta):
        for module in reversed(self.modules):
            module.backward_update_gradient(getattr(module, "_input", None), delta)
            delta = module.backward_delta(getattr(module, "_input", None), delta)

    def update_parameters(self, eps):
        for module in self.modules:
            module.update_parameters(eps)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()
