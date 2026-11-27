import random
from larp_larp.tensor import Tensor

class Bigram:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

        weights = []
        for _ in range(self.vocab_size):
            row = [random.gauss(0, 1) for n in range(self.vocab_size)]
            weights.append(row)

        self.W = Tensor.from_list(weights)

    def __call__(self, xs: list[int]) -> Tensor:
        x_enc = []
        for ix in xs:
            one_hot = [0] * (self.vocab_size); one_hot[ix] = 1
            x_enc.append(one_hot)

        x_enc = Tensor.from_list(x_enc)

        return x_enc @ self.W

    def parameters(self) -> list[Tensor]:
        return [self.W]
        

        