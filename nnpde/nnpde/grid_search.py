from itertools import product


def grid_search(mdl, base_parameters, grid_search_parameters, problem_instances):
    """
    Parameters
    ==========

        mdl                     Model Class,
                                expected interface: `mdl(parameters).fit(problem_instances)`
        base_parameters         dictonary of parameters which will applied for all models
        grid_search_parameters  dictonary of <parameter>: [<value for parameter key>]
        problem_instances       list of problems to train on
    """
    parameters = product(*grid_search_parameters.values())

    # the dict(zip(...)) part is necessary to ensure correct assignment
    res = [mdl(**{**base_parameters, **dict(zip(grid_search_parameters.keys(), p))}).fit(problem_instances)
           for p in parameters]
    return res
