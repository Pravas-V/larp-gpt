import random

def load_text(path):
    '''Loads files from the relative path'''
    with open(path) as f:
        text = f.read()
    return text

def split_data(ids, frac=0.9):
    '''Returns train and test ids'''
    n = int(frac * len(ids))
    train = ids[:n]
    test = ids[n:]
    return train, test

def get_batch(data, batch_size):
    '''Returns cur char ids and next char ids'''
    xs, ys = [], []
    for _ in range(batch_size):
        i = random.randint(0, len(data) - 2)
        xs.append(data[i])
        ys.append(data[i + 1])
    return xs, ys

    