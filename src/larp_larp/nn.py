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
            token_data.append(t_d)

        self.tok_emb = Tensor(token_data, (vocab_size, n_embd))

        position_data = []
        for _ in range(block_size):
            p_d = [random.gauss(0, 1) for n in range(n_embd)]
            position_data.append(t_d)

        self.pos_emb = Tensor(position_data, (block_size, n_embd))


    def __call__(self, ids: list[int]) -> Tensor:
        T = len(ids)
        tok = self.tok_emb.row(ids)
        pos = self.pos_emb.row(list(range(T)))
        return tok + pos
    

    def parameters(self) -> list[Tensor]:
        return [self.tok_emb, self.pos_emb]
    
        