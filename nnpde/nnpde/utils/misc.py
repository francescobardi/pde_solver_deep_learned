from functools import reduce

import numpy as np
import torch


def flatten(listOfLists):
    return reduce(list.__add__, listOfLists, [])


def without(take_them, but_not_these):
    return [i for i in take_them if i not in but_not_these]


def apply_n_times(f, n):
    """Returns a new function which is f folded n times: f(f(f(f(...f(f(n))...))))

    Usage
    -----

    apply_n_times(lambda x: x**2, 3)(2) == 256
    """

    def f_folded_n_times(x):
        return reduce(lambda fx, _: f(fx), range(n), x)
    return f_folded_n_times


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def normal_distributed_tensor(size, dtype=torch.float32, requires_grad=False):
    return torch.tensor(np.random.normal(size=(size, size)).reshape((1, 1, size, size)),
                        dtype=dtype,
                        requires_grad=requires_grad)
