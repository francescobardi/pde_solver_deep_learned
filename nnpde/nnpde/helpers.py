# -*- coding: utf-8 -*-

from functools import reduce

import numpy as np


def check_dimensions(A, f):
    """Check the dimensions of inputs"""

    Na, Ma = np.shape(A)
    Nf = np.shape(f)[0]

    if Na == Ma and Na == Nf:
        return Na
    else:
        raise ValueError("Dimensions mismatch, please check your inputs.")


def conv_net_to_matrix(net, N):
    Hs = []

    for param in net.parameters():
        ks = param.view(9)
        ks = list(map(lambda el: el.item(), ks))

        H = np.diag(np.ones(N**2)) * ks[4] + np.diag(np.ones(N**2-1), 1) * ks[5] \
            + np.diag(np.ones(N**2 - N), N) * ks[7] + np.diag(np.ones(N**2 - N - 1), N+1) * ks[8] \
            + np.diag(np.ones(N**2 - 1), -1) * ks[3] + np.diag(np.ones(N**2 - N+1), N-1) * ks[6] \
            + np.diag(np.ones(N**2 - N), -N) * ks[1] + np.diag(np.ones(N**2 - N+1), -N+1) * ks[2] \
            + np.diag(np.ones(N**2 - N - 1), -N-1) * ks[0]

        Hs.append(H)

    return reduce(lambda acc, el: np.dot(acc, el), Hs)


def build_T(N):
    """ Build Jacobi method updated matrix for Poisson problem """

    # Build diagonals
    a = np.ones(N**2-1)*0.25
    b = np.ones(N**2-N)*0.25

    # Build T
    T = np.diag(a, 1) + np.diag(a, -1) + np.diag(b, N) + np.diag(b, -N)

    return T


def build_G(B_idx):
    """ Build reset boundary matrix """

    N = B_idx.size()[2]
    M = B_idx.size()[3]

    B_idx = B_idx.view(N, M).numpy()

    a = np.zeros(N**2)
    for i in range(N):
        for j in range(N):
            a[i*N + j] = B_idx[i, j]

    G = np.diag(a)
    return G


def spectral_radius(X):
    """ Compute the spectral radius of a matrix """

    # Compute eigenvalues
    eigvals = np.linalg.eigvals(X)

    return np.max(np.absolute(eigvals))


def count_conv(grid_size, kernel_size, nb_layers=3):
    """
    Parameters
    -----------
    grid_size:
        input_shape shape of the input matrix an integer since we have a square matrix
        kernel size shape of convolution kernel
        number of layer number of convolution nb_layers

    Returns
    ----------
    number of flops (int)
    """

    nb_add_mult = nb_layers * (kernel_size**2 * grid_size**2)

    return nb_add_mult


def count_jac(grid_size: int) -> float:
    """
    Parameters
    -----------
        input_shape (int) : shape of input
    Returns
    ----------
        number of flops (float)
    """

    return 4 * grid_size**2


def compare_flops(grid_size, n_iter_jac, n_iter_conv, nb_layers):
    """
    Parameters:
    -----------

        grid_size (int): Dimension of the square matrix dim x dim.
        n_iter_conv (int): Number of iterations it took for the trained model to converge
        n_iter_jac (int): Number of iterations it took for the jacobi method to converge to the ground truth solution

    Returns:
    ----------
       ratio of the number of flops for the convolution and jacobi method

    """

    flop_conv = count_conv(grid_size, nb_layers) * n_iter_conv

    flop_jac = count_jac(grid_size) * n_iter_jac

    return flop_conv/flop_jac
