from larp_larp.data import load_text, split_data, get_batch
from larp_larp.tokenizer.char import CharTokenizer
from larp_larp.model import Bigram
from larp_larp.optim import SGD
from larp_larp.functional import softmax, cross_entropy

def train(path, steps, batch_size, lr) -> None:
    # load data
    text = load_text(path)
    print(text[:5])

    # set up vocabulary
    tokenizer = CharTokenizer(text)
    print(tokenizer)

    # encode text
    ids = tokenizer.encode(text)
    print(ids[:5])

    # data split
    train, val = split_data(ids)
    print(train[:5])
    print(val[:5])

    # initialize model
    model = Bigram(vocab_size=tokenizer.vocab_size)
    print(model)

    # initialize optimizer
    opt = SGD(model.parameters(), lr=lr)
    print(opt)

    # training loop
    for step in range(steps):

        print(step)
        # select examples randomly
        xs, ys = get_batch(train, batch_size= batch_size)

        # get logits
        logits = model(xs)
        print(f'logits are {logits}')

        # loss
        loss = cross_entropy(logits, targets=ys)
        print(f'loss is {loss.data[0]}')

        # flush out gradients
        opt.zero_grad()

        # back-propagate gradients
        loss.backward()

        # update params
        opt.step()

        # print training loss
        if step % 100 == 0:
            print(f'training loss after step {step + 1} is {loss}')


# if __name__ == "__main__":
#     train("data/input.txt", 500, 32, 0.5)
    



