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
    #print(model)

    # initialize optimizer
    opt = SGD(model.parameters(), lr=lr)

    # training loop
    losses = []
    for step in range(steps):

        # select examples randomly
        xs, ys = get_batch(train, batch_size= batch_size)

        # get logits
        logits = model(xs)

        # loss
        probs = softmax(logits)
        loss = cross_entropy(probs, targets=ys)
        losses.append(loss.data[0])

        # flush out gradients
        opt.zero_grad()

        # back-propagate gradients
        loss.backward()

        # update params
        opt.step()

        # print training loss
        if (step) % 100 == 0:
            print(f'average training loss (last 25 runs) after step {step + 1} is {sum(losses[-25: ]) / min(len(losses[-25:]), 25)}')
            print(model.W.data)
            print('-----------------------')
            print('\n')
            print('\n')


if __name__ == "__main__":
    train("data/input.txt", 500, 32, 0.5)
    



