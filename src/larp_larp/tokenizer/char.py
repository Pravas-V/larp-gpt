import json

class CharTokenizer:

    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {s: i for i, s in enumerate(self.chars)}
        self.itos = {i: s for i, s in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        return list(self.stoi[c] for c in s)

    def decode(self, ids):
        return ''.join(list(self.itos[v] for v in ids))

    def save(self, path : str) -> None:
        d = {
            "chars" : self.chars
        }

        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path : str) -> "CharTokenizer":
        with open(path, "r") as f:
            d = json.load(f)

        tokenizer = CharTokenizer(
            d["chars"]
        )

        return tokenizer



        


    
        