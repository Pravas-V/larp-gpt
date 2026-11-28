import random
from larp_larp.tensor import Tensor

class Embedding:

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        '''
        vocab_size : number of unique tokens
        block_size : length of context window
        n_embd : length of each token id's vector
        '''
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        token_data = []
        for _ in range(vocab_size):
            t_d = [random.gauss(0, 1) for n in range(n_embd)]
            token_data.extend(t_d)

        self.tok_emb = Tensor(token_data, (vocab_size, n_embd))

        position_data = []
        for _ in range(block_size):
            p_d = [random.gauss(0, 1) for n in range(n_embd)]
            position_data.extend(p_d)

        self.pos_emb = Tensor(position_data, (block_size, n_embd))


    def __call__(self, ids: list[int]) -> Tensor:
        T = len(ids)
        tok = self.tok_emb.rows(ids)
        pos = self.pos_emb.rows(list(range(T)))
        return tok + pos
    

    def parameters(self) -> list[Tensor]:
        return [self.tok_emb, self.pos_emb]
    



class Linear:

    def __init__(self, in_features: int, out_features: int, bias: bool = True):

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        weight_data = []
        for _ in range(in_features):
            w_d = [random.gauss(0, 1) for n in range(out_features)]
            weight_data.extend(w_d)
    
        self.W = Tensor(weight_data, (in_features, out_features))
        self.b = Tensor([0.0]*out_features, (1, out_features)) if bias else None


    def __call__(self, x: Tensor) -> Tensor:
        out = x @ self.W + self.b if self.bias else x @ self.W
        return out

    def parameters(self) -> list[Tensor]:
        return [self.W, self.b] if self.bias else [self.W]




class LayerNorm:

    def __init__(self, n_embd: int, eps: float = 1e-5):
        self.n_embd = n_embd
        self.eps = eps
        self.gamma = Tensor([1.0]*n_embd, (1, n_embd))
        self.beta = Tensor([0.0]*n_embd, (1, n_embd))


    def __call__(self, x: Tensor) -> Tensor:
        '''take rows of x and normalize them (mean = 0, std = 1)'''
        mean = x.mean_rows()
        std = (((x - x.mean_rows())**2).mean_rows() + self.eps)**0.5
        x = (x - mean) / std

        
    def parameters(self):
        pass




