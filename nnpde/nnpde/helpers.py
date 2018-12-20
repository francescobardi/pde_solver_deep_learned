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
    """ Build Jacobi method updated matrices for Poisson problem """

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
    """ Compute the spectral_radius of a matrix """

    # Compute eigenvalues
    eigvals = np.linalg.eigvals(X)

    return np.max(np.absolute(eigvals))


def count_conv(in_shape,kernel_size,layers = 3):
    """
    Parameters
    -----------
    in_shape:
        input_shape shape of the input matrix an integer since we have a square matrix
        kernel size shape of convolution kernel
        number of layer number of convolution layers

    Returns
    ----------
    number of flops (int)
    """


    # kernel_size additions after each convolution + 2*in_shape for the forcing term for each layer
    nb_additions = layers * (kernel_size**2 * in_shape**2 + 2 * in_shape**2)

    nb_mult = layers * (kernel_size**2 * in_shape) # Kernel size multiplications for each convolution for each layer

    return layers * (nb_additions + nb_mult)


def count_jac(in_shape:int) -> float:
    """
    Parameters
    -----------
        input_shape (int) : shape of input
    Returns
    ----------
        number of flops (float)
    """
    # 4 additions for each matrix element + 1 multiplication and 1 addition for resetting boundary and 1 n^2 term for forcing term
    return 4 * in_shape**2 + 2 * in_shape**2 + in_shape**2


def compare_flops(dim, n_iter_conv, n_iter_jac):
    """
    Parameters:
    -----------

        dim (int): Dimension of the square matrix dim x dim.
        n_iter_conv (int): Number of iterations it took for the trained model to converge
        n_iter_jac (int): Number of iterations it took for the jacobi method to converge to the ground truth solution

    Returns:
    ----------
       ratio of the number of flops for the convolution and jacobi method

    """

    flop_conv = (count_conv(dim,3)) * n_iter_conv

    flop_jac = count_jac(dim) * n_iter_jac

    return flop_conv/flop_jac
