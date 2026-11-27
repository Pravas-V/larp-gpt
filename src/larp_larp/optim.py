from larp_larp.tensor import Tensor

class SGD:

    def __init__(self, params: list[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = [0.0]* (len(param.data))

    def step(self) -> None:
        #print(f'self.params is {self.params}')
        for param in self.params:
            #print(f'param is {param}')
            for i in range(len(param.data)): 
                #print(f'param.data is {param.data}')
                #print(f'param.grad is {param.grad}')   
                param.data[i] -= self.lr * param.grad[i]

