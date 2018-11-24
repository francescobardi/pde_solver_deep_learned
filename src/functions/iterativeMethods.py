# -*- coding: utf-8 -*-
import logging

import numpy as np
from helpers import check_dimensions

def jacobi_paper(A, f, omega = 1.0,initial_u = None, max_iters = 1000, tol = 1e-3):
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

    if initial_u is None:
        u = np.zeros(N)
    else:
        u = initial_u

    #logging.debug
    print("Jacobi method: max_iters={0}, tol={1}.".format(max_iters, tol))

    # Solution loop
    for n_iter in range(max_iters):
        # Compute residuals vector
        res = f - A.dot(u)
        #print(res)
        
        # Check convergence
        if (np.linalg.norm(res) <= tol):
            #logging.debug
            print("Jacobi method: converged in {0} iterations.\n".format(n_iter))
            return u, res

        # Update solution
        u += omega * res

    #logging.debug
    print("Maximum number of iterations exceeded, stopping criteria not satisified. Norm of residuals vector at last iteration is {0}.\n".format(np.linalg.norm(res)))
    return u, res

def jacobi_real(A, f, initial_u = None, max_iters = 1000, tol = 1e-3):
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

    if initial_u is None:
        u = np.zeros(N)
    else:
        u = initial_u

    #logging.debug
    print("Jacobi method: max_iters={0}, tol={1}.".format(max_iters, tol))

    # Pre-compute matrices
    invD = np.diag(1/np.diag(A))
    T = invD.dot(np.diag(np.diag(A))-A)
    
    # Solution loop
    for n_iter in range(max_iters):
        # Compute residuals vector
        res = f - A.dot(u)
        #print(res)
        
        # Check convergence
        if (np.linalg.norm(res) <= tol):
            #logging.debug
            print("Jacobi method: converged in {0} iterations.\n".format(n_iter))
            return u, res

        # Update solution
        u = T.dot(u) + invD.dot(f)

    #logging.debug
    print("Maximum number of iterations exceeded, stopping criteria not satisified. Norm of residuals vector at last iteration is {0}.\n".format(np.linalg.norm(res)))
    return u, res

