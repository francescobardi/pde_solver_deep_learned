import time

import torch
import torch.nn.functional as F
import pandas as pd

from nnpde.helpers import compare_flops
from nnpde.problems import DirichletProblem
import nnpde.iterative_methods as im
from nnpde import metrics


def _test_model_(net, n_tests, grid_size, tol=1e-4, max_nb_iters=100000):
    """
    Parameters:
    -----------
        net : Convolutional Network
        n_tests (int): number of tests
        grid_size (int): size of the domain

    Returns:
    ----------
        A generator for the tests. Use test_results_pd
           number of iterations it takes for the jacobi iterative method to converge,
           number of iterations it takes for the trained model to converge,
           the ratio of the number of flops used in both methods.
    """

    f = torch.zeros(1, 1, grid_size, grid_size)
    u_0 = torch.ones(1, 1, grid_size, grid_size)

    for i in range(n_tests):
        # TODO This is stupid, it should not be here...
        problem_instance = DirichletProblem(N=grid_size, k=max_nb_iters * 1000)
        gtt = problem_instance.ground_truth

        # jacoby method / known solver

        start_time = time.process_time()
        u_jac = im.jacobi_method(problem_instance.B_idx, problem_instance.B, f, u_0, k = 1)
        loss_jac = F.mse_loss(gtt, u_jac)  # TODO use loss from metrics
        count_jac = 1

        nb_iters = 1
        while loss_jac >= tol and nb_iters < max_nb_iters:
            u_jac = im.jacobi_method(problem_instance.B_idx, problem_instance.B, f, u_jac,k = 1)
            loss_jac = F.mse_loss(gtt, u_jac)
            count_jac += 1
            nb_iters += 1

        jac_time = time.process_time() - start_time

        # learned solver

        start_time = time.process_time()
        u_h = im.H_method(net, problem_instance.B_idx, problem_instance.B, f, u_0 ,k = 1)
        loss_h = F.mse_loss(gtt, u_h)
        count_h = 1
        nb_iters = 1
        while loss_h >= tol and nb_iters < max_nb_iters:
            u_h = im.H_method(net, problem_instance.B_idx, problem_instance.B, f, u_h,k = 1)
            loss_h = F.mse_loss(gtt, u_h)
            count_h += 1
            nb_iters += 1

        con_time = time.process_time() - start_time

        # TODO this is stupid
        yield (count_jac, count_h, compare_flops(grid_size, count_h, count_jac), jac_time, con_time,
               jac_time / con_time, loss_jac.item(), loss_h.item())


def test_results_pd(*args, **kwargs):
    """
    Parameters:
    -----------
        net : Convolutional Network
        n_tests (int): number of tests
        grid_size (int): size of the domain

    Returns:
    ----------
        Pandas dataframe with test results
    """
    return pd.DataFrame(_test_model_(*args, **kwargs),
                        columns=['nb_iters_jac', 'nb_iters_convjac', 'flops_ratio', 'jac_cpu_time',
                                 'convjac_cpu_time', 'cpu_time_ratio', 'loss_jac', 'loss_convjac'])
