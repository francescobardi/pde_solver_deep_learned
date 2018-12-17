# -*- coding: utf-8 -*-

from functools import reduce
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


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


def plot_solution(gtt, output, N):
    Z_gtt = gtt.view(N, N).numpy()
    Z_output = output.detach().view(N, N).numpy()

    fig, axes = plt.subplots(nrows=1, ncols=2)

    fig.suptitle("Comparison")

    im_gtt = axes[0].imshow(Z_gtt)
    axes[0].set_title("Ground truth solution")

    im_output = axes[1].imshow(Z_output)
    axes[1].set_title("H method solution")

    fig.colorbar(im_gtt)
    fig.tight_layout()

    plt.show()
