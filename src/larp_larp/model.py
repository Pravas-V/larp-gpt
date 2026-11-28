import json
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

    def save(self, path : str) -> None:
        d = {
            "vocab_size" : self.vocab_size,
            "model_weights" : self.W.data,
            "dimensions" : self.W.shape
        }

        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path : str) -> "Bigram":
        with open(path, 'r') as f:
            d = json.load(f)

        model = Bigram(
            vocab_size=d["vocab_size"],
        )
        model.W = Tensor(data=d["model_weights"], shape=d["dimensions"])

        return model


###############################################################################################################
from larp_larp.nn import *


class GPT:

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding = Embedding(vocab_size, block_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = Linear(n_embd, vocab_size)


    def __call__(self, ids: list[int]):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


    def parameters(self) -> list[Tensor]:
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.ln_f.parameters())
        params.extend(self.lm_head.parameters())
        for b in self.blocks:
            params.extend(b.parameters())

        return params    







        

        