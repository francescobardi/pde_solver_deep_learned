# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import check_dimensions

def jacobi(A, f, initial_u = None, b_idx = None, b = None, max_iters = 1000, tol = 1e-3):
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
    if b is None or b_idx is None:
        resetBC = False
    else:
        #check_dim(b, b_idx) TODO
        resetBC = True

    # Set initial solution vector    
    if initial_u is None:
        u = np.zeros(N)
    else:
        u = initial_u

    if resetBC:
        u[b_idx] = b

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
            u[b_idx] = b

        # Compute residuals vector
        res = f - A.dot(u)

        # Set residuals to zero at boundary points
        if resetBC:
            res[b_idx] = 0.0

        logging.debug("Norm of residuals vector at iteration {0} is {1}.\n".format(iter, np.linalg.norm(res)))
        
        # Check convergence
        if (np.linalg.norm(res) <= tol):
            logging.info("Jacobi method: converged in {0} iterations.\n".format(n_iter))
            return u, res


    logging.warning("Maximum number of iterations exceeded, stopping criteria not satisified. Norm of residuals vector at last iteration is {0}.\n".format(np.linalg.norm(res)))
    return u, res


def reset_operator(u, B_idx, B):
    """ Reset values at the boundary of the domain


    Parameters
    ----------
    u : tensor-like, shape = [*, *, n, n]
        variable to reset.

    B_idx : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    B : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.

    Returns
    -------
    u : tensor-like, shape = [*, *, n, n]
        resetted values.


    """    
    
    u = u * B_idx + B
    
    return u

def jacobi_iteration(u, B_idx, B, f):
    """ Compute one jacobi method iteration, by applying convolution


    Parameters
    ----------
    u : tensor-like, shape = [*, *, n, n]
        variable to reset.

    B_idx : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    B : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.
    
    f : tensor-like, shape = [*, *, n, n]
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
    
    for param in net.parameters():
        param.requires_grad = False
    
    
    u = net(u) + f
    u = reset_operator(u, B_idx, B)
    
    return u
    

def jacobi_method(B_idx, B, f, initial_u = None, k = 1000):
    """ Compute jacobi method solution by convolution


    Parameters
    ----------

    B_idx : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    B : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.
    
    f : tensor-like, shape = [*, *, n, n]
        matrix describing the forcing term.

    initial_u : tensor-like, shape = [*, *, n, n]
        Initial values.

    Returns
    -------
    u : tensor-like, shape = [*, *, n, n]
        solution matrix.


    """  
    # Initialization
    N = B_idx.size()[3]
    
    # Set initial solution vector
    if initial_u is None:
        u = torch.zeros(1 ,1 , N, N)
    else:
        u = initial_u    
    
    # Reset values at the boundaries
    u = reset_operator(u, B_idx, B)
    
    # Solution loop
    for n_iter in range(k):
        u = jacobi_iteration(u, B_idx, B, f)
    
    return u

def H_method(net, B_idx, B, f, initial_u = None, k = 1000):
    """ Compute solution by H method


    Parameters
    ----------
    net = neural network rappresenting H
    
    B_idx : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    B : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.
    
    f : tensor-like, shape = [*, *, n, n]
        matrix describing the forcing term.

    initial_u : tensor-like, shape = [*, *, n, n]
        Initial values.

    Returns
    -------
    u : tensor-like, shape = [*, *, n, n]
        solution matrix.


    """
    
    # Reset values at boundaries
    u = reset_operator(initial_u, B_idx, B)
    
    # Solution loop
    for n_iter in range(k):
        jac_it = jacobi_iteration(u, B_idx, B, f)
        w = jac_it - u
        u = jac_it + net(jac_it - u) * B_idx
        u = reset_operator(u, B_idx, B)
        
    
    return u


