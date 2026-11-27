from larp_larp.data import load_text, split_data, get_batch
from larp_larp.tokenizer.char import CharTokenizer
from larp_larp.model import Bigram
from larp_larp.optim import SGD
from larp_larp.functional import softmax, cross_entropy

def train(path, steps, batch_size, lr) -> None:
    # load data
    text = load_text(path)

    # set up vocabulary
    tokenizer = CharTokenizer(text)

    # encode text
    ids = tokenizer.encode(text)

    # data split
    train, val = split_data(ids)

    # initialize model
    model = Bigram(vocab_size=tokenizer.vocab_size)

    # initialize optimizer
    opt = SGD(model.parameters(), lr=lr)

    # training loop
    for step in range(steps):

        # select examples randomly
        xs, ys = get_batch(train, batch_size= batch_size)

        # get logits
        logits = model(xs)

        # loss
        loss = cross_entropy(logits, targets=ys)

        # flush out gradients
        opt.zero_grad()

        # back-propagate gradients
        loss.backward()

        # update params
        opt.step()

        # print training loss
        if step % 100 == 0:
            print(f'training loss after step {step + 1} is {loss}')


if __name__ == "__main__":
    train("data/input.txt", 500, 32, 0.5)
    



