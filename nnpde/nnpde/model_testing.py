import time
import logging

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from nnpde.metrics import least_squares_loss as LSE
from nnpde.helpers import compare_flops
from nnpde.problems import DirichletProblem
import nnpde.iterative_methods as im
from nnpde import metrics


def _test_model_(model, n_tests, grid_size, tol=1e-6, max_nb_iters=50000, convergence_tol=1e-10, max_converged_count=100, domain_type='square'):
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
    net = model.net
    nb_layers = model.nb_layers

    f = torch.zeros(1, 1, grid_size, grid_size)
    initial_u = torch.ones(1, 1, grid_size, grid_size)

    for i in range(n_tests):
        # TODO This is stupid, it should not be here...
        problem_instance = DirichletProblem(
            N=grid_size, k_ground_truth=max_nb_iters, domain_type=domain_type)
        ground_truth = problem_instance.ground_truth

        # jacoby method / known solver

        logging.debug(f'test nb {i} jac')
        start_time = time.process_time()
        u_jac = im.jacobi_method(
            problem_instance.B_idx, problem_instance.B, f, initial_u, k=1)
        loss_jac = LSE(ground_truth, u_jac)
        count_jac = 1
        converged_count = 0

        while loss_jac >= tol and count_jac < max_nb_iters:
            u_jac = im.jacobi_method(
                problem_instance.B_idx, problem_instance.B, f, u_jac, k=1)
            _loss_jac = LSE(ground_truth, u_jac)
            if np.abs(loss_jac.item() - _loss_jac.item()) < convergence_tol:
                converged_count += 1
            else:
                converged_count = 1

            loss_jac = _loss_jac
            if converged_count > max_converged_count:
                logging.info(
                    'Jacobi method  converged but did not reach required tol')
                break
            count_jac += 1

        jac_time = time.process_time() - start_time

        # learned solver

        logging.debug(f'Test nb {i} H method')
        start_time = time.process_time()
        # TODO turn off grad for net
        u_H = im.H_method(net, problem_instance.B_idx,
                          problem_instance.B, f, initial_u, k=1)
        loss_H = LSE(ground_truth, u_H)
        count_H = 1
        converged_count = 0
        while loss_H >= tol and count_H < max_nb_iters:
            count_H += 1
            u_H = im.H_method(net, problem_instance.B_idx,
                              problem_instance.B, f, u_H, k=1)
            _loss_H = LSE(ground_truth, u_H)
            if np.abs(loss_H.item() - _loss_H.item()) < convergence_tol:
                converged_count += 1
            else:
                converged_count = 1

            loss_H = _loss_H
            if converged_count > max_converged_count:
                logging.warning(
                    'H method converged but did not reach required tol')
                break

            if count_H % 100 == 0:
                logging.debug(f'Test nb {i}, count_H {count_H}, loss {loss_H}')

        con_time = time.process_time() - start_time
        logging.debug(f'Test nb {i} done')

        # TODO this is stupid
        yield (count_jac, count_H, compare_flops(grid_size, count_jac, count_H, nb_layers), jac_time, con_time,
               con_time / jac_time, loss_jac.item(), loss_H.item())


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
