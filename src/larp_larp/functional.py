import math
from larp_larp.tensor import Tensor

def softmax(row: list[float]) -> list[float]:
    '''Convert logits to probabilities'''
    m = max(row, key=abs)
    row = [v - m for v in row]
    D = sum(math.exp(v) for v in row)
    probs = [math.exp(v) / D for v in row]
    return probs


def cross_entropy(logits: Tensor, targets: list[int]) -> Tensor:
    '''
    Returns average cross entropy loss
    logits : Tensor (B, V); B = Batch size, V = Vocab size
    targets : list[int]
    '''
    logits = logits.to_list()
    log_loss = 0 
    for i, row in enumerate(logits):
        t_ix = targets[i]
        p = softmax(row)[t_ix]
        log_loss += -math.log(p)

    return Tensor([log_loss / len(logits)], shape=(1, 1))

        
    

