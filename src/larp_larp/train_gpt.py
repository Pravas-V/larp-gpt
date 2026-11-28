from larp_larp.data import load_text, split_data, get_sequence_batch
from larp_larp.tokenizer.char import CharTokenizer
from larp_larp.model import GPT
from larp_larp.optim import SGD
from larp_larp.functional import softmax, cross_entropy

def train_gpt(
        path : str,
        steps : int,
        block_size : int,
        batch_size : int,
        n_embd : int, 
        n_head : int,
        n_layer : int,
        lr : float
):

    text = load_text(path)
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)
    train, val = split_data(ids)
    model = GPT(tokenizer.vocab_size, block_size, n_embd, n_head, n_layer)
    opt = SGD(model.parameters(), lr=lr)

    losses = []
    for step in range(steps):

        xs, ys = get_sequence_batch(train, block_size, batch_size)
        loss = 0
        for i in range(batch_size):
            logits = model(xs[i])
            probs = softmax(logits)
            loss = cross_entropy(probs, targets=ys[i]) + loss

        loss = loss * (1 / batch_size)
        losses.append(loss.data[0])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 25 == 0:
            print(f'average training loss (last 25 runs) after step {step + 1} is {sum(losses[-25: ]) / min(len(losses[-25:]), 25)}')
            print('-----------------------')
            print('\n')
            print('\n')

    return model, tokenizer



if __name__ == "__main__":
    model, tokenizer = train_gpt("data/input.txt", steps=1000, block_size=8, batch_size=4, n_embd=16, n_head=2, n_layer=1, lr=1)
    model.save("checkpoints/gpt.json")
    tokenizer.save("checkpoints/char_tok-gpt.json")
    


