import math
from larp_larp.tensor import Tensor

def softmax(logits: Tensor) -> Tensor:
    '''
    Returns probabilites, each row is one training example
    logtis : Tensor (B, V), B = Batch size, V = Vocab size
    '''
    log_exps = logits.exp()
    probs = log_exps / log_exps.sum_rows()
    return probs


def cross_entropy(probs: Tensor, targets: list[int]) -> Tensor:
    '''
    Returns average cross entropy loss
    logits : Tensor (B, V); B = Batch size, V = Vocab size
    targets : list[int]
    '''
    B, V = probs.shape

    t_enc = []
    for t_ix in targets:
        enc = [0.0] * (V)
        enc[t_ix] = 1.0
        t_enc.append(enc)

    t_enc = Tensor.from_list(t_enc)

    loss = -(probs * t_enc).sum_rows().log().sum() / B
    return loss

    






        
    

