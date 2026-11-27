from larp_larp.tensor import Tensor

class SGD:

    def __init__(self, params: list[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = [0.0]* (len(param.data))

    def step(self) -> None:
        for param in self.params:
            for i in range(len(param.data)): 
                param.data[i] -= self.lr * param.grad[i]

