# -*- coding: utf-8 -*-
import logging

import numpy as np
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

