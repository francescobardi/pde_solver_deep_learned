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

    # TODO I still think that this is horrible code.
    # This is not readable, and makes a shit ton of assumptions about the net.
    # TODO combine first the layers, then convert them.
    for param in net.parameters():
        ks = param.view(9)
        ks = list(map(lambda el: el.item(),ks))

        H = np.diag(np.ones(N**2)) * ks[4] + np.diag(np.ones(N**2-1),1)* ks[5] \
        + np.diag(np.ones(N**2 - N),N) * ks[7] + np.diag(np.ones(N**2 - N - 1),N+1) * ks[8] \
        + np.diag(np.ones(N**2 - 1), - 1) * ks[3] + np.diag(np.ones(N**2 - N+1),N-1) * ks[6] \
        + np.diag(np.ones(N**2 - N), -N) * ks[1] +  np.diag(np.ones(N**2 - N+1),-N+1) * ks[2] \
        + np.diag(np.ones(N**2 - N - 1),-N-1) * ks[0]

        Hs.append(H)

    return reduce(lambda acc, el: np.dot(acc, el), Hs)



def spectral_radius(T,H):
    T1 = T[0,0,:,:].double()
    H1 = torch.from_numpy(H).double()
    tmp = T1 + T1 @ H1 - H1
    eigvals = np.linalg.eigvals(tmp)
    return np.abs(np.max(np.real(eigvals)))


def get_T(N):

    b = np.ones(N**2-1)*0.25
    c = np.ones(N**2-N)*0.25

    T = np.diag(b, 1) + np.diag(b, -1) + np.diag(c, N) + np.diag(c, -N)
    
    return T

def plot_solution(gtt,output,N):
    Z_gtt = gtt.view(N,N).numpy()
    Z_output = output.detach().view(N, N).numpy()

    fig, axes = plt.subplots(nrows = 1, ncols = 2)

    fig.suptitle("Comparison")

    im_gtt = axes[0].imshow(Z_gtt)
    axes[0].set_title("Ground truth solution")

    im_output = axes[1].imshow(Z_output)
    axes[1].set_title("H method solution")

    fig.colorbar(im_gtt)
    fig.tight_layout()

    plt.show()


