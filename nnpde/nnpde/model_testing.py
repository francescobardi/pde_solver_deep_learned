import time
import logging

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np 

from nnpde.helpers import compare_flops
from nnpde.problems import DirichletProblem
import nnpde.iterative_methods as im
from nnpde import metrics


def _test_model_(net, n_tests, grid_size, tol=1e-4, max_nb_iters=100000, convergence_tol=1e-10, max_converged_count=100):
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

        logging.debug(f'test nb {i} jac')
        start_time = time.process_time()
        u_jac = im.jacobi_method(problem_instance.B_idx, problem_instance.B, f, u_0, k = 1)
        loss_jac = F.mse_loss(gtt, u_jac)  # TODO use loss from metrics
        count_jac = 1
        converged_count = 0

        while loss_jac >= tol and count_jac < max_nb_iters:
            u_jac = im.jacobi_method(problem_instance.B_idx, problem_instance.B, f, u_jac,k = 1)
            _loss_jac = F.mse_loss(gtt, u_jac)
            if np.abs(loss_jac.item() - _loss_jac.item()) < convergence_tol:
                converged_count += 1
            else:
                converged_count = 1

            loss_jac = _loss_jac
            if converged_count > max_converged_count:
                logging.info('jac converged but did not reach required tol')
                break
            count_jac += 1

        jac_time = time.process_time() - start_time

        # learned solver

        logging.debug(f'test nb {i} convjac')
        start_time = time.process_time()
        # TODO turn off grad for net
        u_h = im.H_method(net, problem_instance.B_idx, problem_instance.B, f, u_0 ,k = 1)
        loss_h = F.mse_loss(gtt, u_h)
        count_h = 1
        converged_count = 0
        while loss_h >= tol and count_h < max_nb_iters:
            count_h += 1
            u_h = im.H_method(net, problem_instance.B_idx, problem_instance.B, f, u_h,k = 1)
            _loss_h = F.mse_loss(gtt, u_h)
            if np.abs(loss_h.item() - _loss_h.item()) < convergence_tol:
                converged_count += 1
            else:
                converged_count = 1

            loss_h = _loss_h
            if converged_count > max_converged_count:
                logging.warning('convjac converged but did not reach required tol')
                break

            if count_h % 100 == 0:
                logging.debug(f'test nb {i}, count_h {count_h}, loss {loss_h}')

        con_time = time.process_time() - start_time
        logging.debug(f'test nb {i} done')

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
