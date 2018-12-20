# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnpde.helpers import check_dimensions
from nnpde.utils.misc import apply_n_times

def jacobi(A,
           f,
           initial_u=None,
           boundary_index=None,
           boundary_values=None,
           max_iters=1000,
           tol=1e-3):
    """Solve the least squares equation by the method of gradient descent.


    Parameters
    ----------
    A : array-like, shape = [n, n]
        Training data.

    f : array-like, shape = [n]
        Target values.

    initial_u : array_like, shape =  [n], optional
        Initial solution. If inital_u is None it is initialized with zeros.

    max_iter : int, optional, default 1000
        Maximum number of iterations.

    tol : float, optional, default 1e-3
        Precision of the solution. Convergence is checked
        with respect to l2 norm of residuals vector.

    Returns
    -------
    u : array, shape = [n]
        Solution vector.
    res : array, shape = [n]
        Residuals vector.


    """
    # Initialization
    N = check_dimensions(A, f)

    # Disable/enable reset operator G for boundary nodes
    if boundary_values is None or boundary_index is None:
        resetBC = False
    else:
        #check_dim(boundary_values, boundary_index) TODO
        resetBC = True

    # Set initial solution vector
    if initial_u is None:
        u = np.zeros(N)
    else:
        u = initial_u

    if resetBC:
        u[boundary_index] = boundary_values

    logging.info("Jacobi method: max_iters={0}, tol={1}, resetBC={2}.".format(max_iters, tol, resetBC))

    # Pre-compute matrices
    invD = np.diag(1/np.diag(A))
    T = invD.dot(np.diag(np.diag(A))-A)

    # Solution loop
    for n_iter in range(max_iters):

        # Update solution
        u = T.dot(u) + invD.dot(f)

        # If enabled apply reset operator G
        if resetBC:
            u[boundary_index] = boundary_values

        # Compute residuals vector
        res = f - A.dot(u)

        # Set residuals to zero at boundary points
        if resetBC:
            res[boundary_index] = 0.0

        logging.debug("Norm of residuals vector at iteration {0} is {1}.\n".format(iter, np.linalg.norm(res)))

        # Check convergence
        if (np.linalg.norm(res) <= tol):
            logging.info("Jacobi method: converged in {0} iterations.\n".format(n_iter))
            return u, res


    logging.warning("Maximum number of iterations exceeded, stopping criteria not satisified. Norm of residuals vector at last iteration is {0}.\n".format(np.linalg.norm(res)))
    return u, res


def _reset_boundary_(u, boundary_index, boundary_values):
    """ Reset values at the boundary of the domain


    Parameters
    ----------
    u : tensor-like, shape = [*, *, n, n]
        variable to reset.

    boundary_index : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    boundary_values : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.

    Returns
    -------
    u : tensor-like, shape = [*, *, n, n]
        resetted values.


    """

    return u * boundary_index + boundary_values


def _jacobi_iteration_step_(u, boundary_index, boundary_values, forcing_term):
    """ Jacobi method iteration step, defined as a convolution.
    Resets the boundary.


    Parameters
    ----------
    u : tensor-like, shape = [*, *, n, n]
        variable to reset.

    boundary_index : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    boundary_values : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.

    forcing_term : tensor-like, shape = [*, *, n, n]
        matrix describing the forcing term.


    Returns
    -------
    u : tensor-like, shape = [*, *, n, n]
        resetted values.
    """

    net = nn.Conv2d(1, 1, 3, padding = 1, bias = False)

    initial_weights = torch.zeros(1,1,3,3)
    initial_weights[0,0,0,1] = 0.25
    initial_weights[0,0,2,1] = 0.25
    initial_weights[0,0,1,0] = 0.25
    initial_weights[0,0,1,2] = 0.25
    net.weight = nn.Parameter(initial_weights)

    # The final model will be defined as a convolutional network, but this step
    # is fixed.
    for param in net.parameters():
        param.requires_grad = False

    return _reset_boundary_(net(u) + forcing_term, boundary_index, boundary_values)


def jacobi_method(boundary_index, boundary_values, forcing_term, initial_u = None, k = 1000):
    """ Compute jacobi method solution by convolution


    Parameters
    ----------
    boundary_index : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    boundary_values : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.

    forcing_term : tensor-like, shape = [*, *, n, n]
        matrix describing the forcing term.

    initial_u : tensor-like, shape = [*, *, n, n]
        Initial values.

    Returns
    -------
    u : tensor-like, shape = [*, *, n, n]
        solution matrix.
    """
    N = boundary_index.size()[3]

    if initial_u is None:
        u = torch.zeros(1, 1, N, N)
    else:
        u = initial_u

    u = _reset_boundary_(u, boundary_index, boundary_values)

    def step(u_k):
        return _jacobi_iteration_step_(u_k, boundary_index, boundary_values, forcing_term)

    return apply_n_times(step, k)(u)


# TODO rename this
def H_method(net, boundary_index, boundary_values, forcing_term, initial_u=None, k=1000):
    """ Compute solution by H method

    Parameters
    ----------
    net = neural network representing H

    boundary_index : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    boundary_values : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.

    forcing_term : tensor-like, shape = [*, *, n, n]
        matrix describing the forcing term.

    initial_u : tensor-like, shape = [*, *, n, n]
        Initial values.

    Returns
    -------
    u : tensor-like, shape = [*, *, n, n]
        solution matrix.
    """

    u = _reset_boundary_(initial_u, boundary_index, boundary_values)

    def step(u_n):
        jac_it = _jacobi_iteration_step_(u_n, boundary_index, boundary_values, forcing_term)
        u_n = jac_it + net(jac_it - u_n, boundary_index)
        return _reset_boundary_(u_n, boundary_index, boundary_values)

    return apply_n_times(step, k)(u)
