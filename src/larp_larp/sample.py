import random
from larp_larp.tensor import Tensor
from larp_larp.model import Bigram
from larp_larp.tokenizer.char import CharTokenizer
from larp_larp.functional import softmax

def sample(model : Bigram, tok : CharTokenizer, start: str, n: int) -> str:
    ids = tok.encode(start)
    for _ in range(n):
        cur_tok = ids[-1]
        x_enc = [0.0] * (tok.vocab_size)
        x_enc[cur_tok] = 1.0
        x_enc = Tensor(x_enc, (1, tok.vocab_size))
        logits = x_enc @ model.W
        probs = softmax(logits)
        nxt_tok = random.choices(range(tok.vocab_size), weights=probs.data)[0]
        ids.append(nxt_tok)
    return tok.decode(ids)

if __name__ == "__main__":
    model = Bigram.load("checkpoints/bigram.json")
    tok = CharTokenizer.load("checkpoints/char_tok.json")
    print(sample(model, tok, start="cogitating", n=300))