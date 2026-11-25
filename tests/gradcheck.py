
def gradcheck(f, inputs, eps=1e-5, tol=1e-4):
    # flush gradients
    for tensor in inputs:
        tensor.grad = 0.0

    # run both passes
    L = f(*inputs)
    L.backward()

    # store gradients
    grad_dict = {t_idx: tensor.grad for tensor in inputs}

    # the actual checking
    for t_idx, tensor in enumerate(inputs):
        for i in range(len(tensor.data)):
            a_g = grad_dict[t_idx][i]

            tensor.data[i] += eps; up = f(*inputs)
            tensor.data[i] -= eps
            tensor.data[i] -= eps; down = f(*inputs)
            n_g = (up - down) / (2 * eps)

            if abs(a_g - n_g) > tol:
                return False

    return True