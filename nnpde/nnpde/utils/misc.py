from functools import reduce


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
