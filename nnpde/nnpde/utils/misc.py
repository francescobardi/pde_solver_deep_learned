from functools import reduce


def flatten(listOfLists):
    return reduce(list.__add__, listOfLists, [])


def without(take_them, but_not_these):
    return [i for i in take_them if i not in but_not_these]
