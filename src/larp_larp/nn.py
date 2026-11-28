import random
from larp_larp.tensor import Tensor
from larp_larp.functional import softmax

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
        self.tok_emb = Tensor.randn((vocab_size, n_embd))
        self.pos_emb = Tensor.randn((block_size, n_embd))


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
        self.W = Tensor.randn((in_features, out_features))
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
        std = (((x - mean)**2).mean_rows() + self.eps)**0.5
        x = (x - mean) / std
        return x * self.gamma + self.beta

        
    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]




class Head:

    def __init__(self, n_embd, head_size):
        self.head_size = head_size
        self.query = Linear(n_embd, head_size, bias=False)
        self.key = Linear(n_embd, head_size, bias=False)
        self.value = Linear(n_embd, head_size, bias=False)


    def __call__(self, x: Tensor) -> Tensor:
        '''
        x : Tensor (T, n_embd)
        self.query, key, value : Linear layers (n_embd, head_size)
        '''

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # scaled dot product attention
        wei = q @ k.T
        wei *= (self.head_size**-0.5)
    

        # mask
        T = wei.shape[0]
        mask_data = [0]*(T**2)
        for i in range(T):
            for j in range(i + 1, T):
                mask_data[j + i * T] = float('-inf')
        
        mask = Tensor(mask_data, wei.shape)

        # convert to attention fractions
        wei = wei + mask
        att = softmax(wei)

        # weighted value
        out = att @ v
        return out
        

    def parameters(self) -> list[Tensor]:
        return self.query.parameters() + self.key.parameters() + self.value.parameters()
    



class MultiHead:

    def __init__(self, n_embd, n_heads):
        self.n_heads = n_heads
        self.heads = [Head(n_embd, n_embd//n_heads) for _ in range(n_heads)]
        self.proj = Linear(n_embd, n_embd)


    def __call__(self, x: Tensor) -> Tensor:
        outs = [h(x) for h in self.heads]
        out = Tensor.concat(outs)
        out = self.proj(out)
        return out        


    def parameters(self):
        params = []
        for h in self.heads:
            params.extend(h.parameters())

        return params + self.proj.parameters()



# the feed forward network
class MLP:

    def __init__(self, n_embd):
        self.fc = Linear(n_embd, 4*n_embd)
        self.proj = Linear(4*n_embd, n_embd)


    def __call__(self, x: Tensor) -> Tensor:
        h = self.fc(x)
        h = h.relu()
        out = self.proj(h)
        return out


    def parameters(self):
        return self.fc.parameters() + self.proj.parameters()




class Block:

    def __init__(self, n_embd, n_head):
        self.n_head = n_head
        self.ln1 = LayerNorm(n_embd)
        self.sa = MultiHead(n_embd, n_head)
        self.ln2 = LayerNorm(n_embd)
        self.ffwd = MLP(n_embd)

    def __call__(self, x : Tensor) -> Tensor:
        a = self.ln1(x)
        a = self.sa(a)
        x = x + a
        m = self.ln2(x)
        m = self.ffwd(m)
        x = x + m
        return x

    def parameters(self):
        return self.ln1.parameters() + self.sa.parameters() + self.ln2.parameters() + self.ffwd.parameters()
    

    
        
