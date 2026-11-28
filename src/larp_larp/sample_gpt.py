import random
from larp_larp.tensor import Tensor
from larp_larp.model import GPT
from larp_larp.tokenizer.char import CharTokenizer
from larp_larp.functional import softmax

def sample(
        model : GPT,
        tok : CharTokenizer,
        start : str,
        n : int
) -> str:

    ids = tok.encode(start)
    for _ in range(n):
        context = ids[-model.block_size:]
        logits = model(context)
        probs = softmax(logits).rows([len(context) - 1])
        nxt_tok = random.choices(range(tok.vocab_size), weights=probs.data)[0]
        ids.append(nxt_tok)

    return tok.decode(ids)

if __name__ == "__main__":
    model = GPT.load("checkpoints/gpt.json")
    tok = CharTokenizer.load("checkpoints/char_tok-gpt.json")
    print(sample(model, tok, start="cell", n=300))

