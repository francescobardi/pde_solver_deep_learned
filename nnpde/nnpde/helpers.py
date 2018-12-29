from itertools import product
from functools import reduce

import numpy as np


def check_dimensions(A, f):
    """check_dimensions for matrix computation.

    :param A: Matrix
    :param f: Another matrix
    """

    Na, Ma = np.shape(A)
    Nf = np.shape(f)[0]

    if Na == Ma and Na == Nf:
        return Na
    else:
        raise ValueError("Dimensions mismatch, please check your inputs.")


def conv_layer_to_matrix(conv_layer, dim, kernel_size=3):
    """conv_layer_to_matrix Converts a convolutional layer to a matrix.

    :param conv_layer: (PyTorch tensor) Convolutional layer to convert
    :param dim: Dimension of matrix on which the convolution is to be applied
    :returns: Matrix expression of convolutional layer
    """
    # TODO remove hardwiring to kernel size
    if kernel_size != 3:
        raise NotImplementedError("Kernel size of size different of 3 is currently not supported.")

    ks = [k.item() for k in conv_layer.view(9)]

    H = np.diag(np.ones(dim**2)) * ks[4] + np.diag(np.ones(dim**2 - 1), 1) * ks[5] \
        + np.diag(np.ones(dim**2 - dim), dim) * ks[7] + np.diag(np.ones(dim**2 - dim - 1), dim + 1) * ks[8] \
        + np.diag(np.ones(dim**2 - 1), -1) * ks[3] + np.diag(np.ones(dim**2 - dim + 1), dim - 1) * ks[6] \
        + np.diag(np.ones(dim**2 - dim), -dim) * ks[1] + np.diag(np.ones(dim**2 - dim + 1), -dim + 1) * ks[2] \
        + np.diag(np.ones(dim**2 - dim - 1), -dim - 1) * ks[0]

    return H


def conv_net_to_matrix(net, N):
    """conv_net_to_matrix Expresses a linear convolutional net of kernel size 3 as a matrix,
    to be used in a matrix multiplication.

    :param net: PyTorch linear convolutional network with an arbitrary number of layer.
    :param N: Dimension of matrix on which the convolution is to be applied.
    """
    return reduce(lambda acc, el: np.dot(acc, el), (conv_layer_to_matrix(param, N) in net.parameters()))


def build_T(N):
    """build_T Build Jacobi method updated matrix for Poisson problem.

    :param N: Size of matrix.
    """

    # Build diagonals
    a = np.ones(N**2 - 1) * 0.25
    b = np.ones(N**2 - N) * 0.25

    # Build T
    T = np.diag(a, 1) + np.diag(a, -1) + np.diag(b, N) + np.diag(b, -N)

    return T


def build_G(B_idx):
    """build_G Build reset boundary matrix.

    :param B_idx: "Boolean" Index of boundary.
    """

    N = B_idx.size()[2]
    M = B_idx.size()[3]

    B_idx = B_idx.view(N, M).numpy()

    a = np.zeros(N**2)
    for i, j in product(range(N), range(N)):
        a[i * N + j] = B_idx[i, j]

    G = np.diag(a)
    return G


def spectral_radius(X):
    """spectral_radius Computes the spectral radius of matrix X.

    :param X:
    """
    return np.max(np.abs(np.linalg.eigvals(X)))


def n_flops_conv_net_step(grid_size, kernel_size, n_layers=3):
    """n_flops_conv_net_step Estimation on the number of flops taken by the convolution.

    Counting multiply-and-add computations as one flop.

    :param grid_size:
        grid size on which the problem is solved
    :param kernel_size:
        kernel size shape of convolution kernel
    :param n_layers:
        (int) number of layer number of convolution n_layers
    :returns:
        (int) number of flops
    """

    return n_layers * (kernel_size**2 * grid_size**2)


def n_flops_jacoby_step(grid_size):
    """n_flops_jacoby_step Estimation of the number of flops taken by the Jacoby Solver.

    :param grid_size:
        (int) grid size on which the problem is solved
    :returns:
        (int) number of flops
    """

    return 4 * grid_size**2


def flops_ratio(grid_size, n_iter_jac, n_iter_conv, n_layers):
    """flops_ratio

    :param grid_size:
        (int) Dimension of the square matrix dim x dim.
    :param n_iter_jac:
        (int) Number of iterations it took for the trained model to converge
    :param n_iter_conv:
        (int) Number of iterations it took for the jacobi method to converge to the ground truth solution
    :param n_layers:
        (int) Number of layers used in the convolutional net.
    :returns:
       ratio of the number of flops for the convolution and jacobi method
    """
    flop_jac = n_flops_jacoby_step(grid_size) * n_iter_jac

    flop_conv = (flop_jac + n_flops_conv_net_step(grid_size, n_layers)) * n_iter_conv

    return flop_conv / flop_jac
