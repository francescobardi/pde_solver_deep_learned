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

def count_conv(in_shape,kernel_size,layers = 3):
    
    '''
    Parameters
    -----------
    in_shape:
        input_shape shape of the input matrix an integer since we have a square matrix
        kernel size shape of convolution kernel
        number of layer number of convolution layers
    
    Returns
    ----------
    number of flops (int)
    '''
    
    
    nb_additions = layers * (kernel_size**2 * in_shape**2 + 2 * in_shape**2) # kernel_size additions after each convolution + 2*in_shape for the forcing term for each layer
    
    nb_mult = layers * (kernel_size**2 * in_shape) # Kernel size multiplications for each convolution for each layer
    
    return layers * (nb_additions + nb_mult)

def count_jac(in_shape):
    '''
    Parameters
    -----------
    in_shape:
        input_shape (int) : shape of input
    Returns
    ----------
    number of flops
    '''
    return 4 * in_shape**2 + 2 * in_shape**2 + in_shape**2 # 4 additions for each matrix element + 1 multiplication and 1 addition for resetting boundary and 1 n^2 term for forcing term



def compare_flops(n,n_iter_conv,n_iter_jac):
    """
    Parameters:
    -----------
    
        n (int): Shape of the square matrix n x n.
        n_iter_conv (int): Number of iterations it took for our model to converge
        n_iter_jac (int): Number of iterations it took for the jacobi method to converge to the ground truth solution

    Returns:
    ----------
       ratio of the number of flops for the convolution and jacobi method

    """
    
    flop_conv = (count_conv(n,3)) * n_iter_conv 

    flop_jac = count_jac(n) * n_iter_jac 

    return flop_conv/flop_jac


def test_model(net, n_tests, grid_size):
    
     """
    Parameters:
    -----------
        net : Convolutional Network
        n_tests (int): number of tests
        grid_size (int): size of the domain
    Returns:
    ----------
       number of iterations it takes for the jacobi iterative method to converge, number of iterations it takes
       for our model to converge and the ratio of the number of flops used in both methods.

    """
        
    
    losses = []
    
    loss_to_be_achieved = 1e-6
    max_nb_iters = 100000
    f = torch.zeros(1, 1, grid_size, grid_size)
    u_0 = torch.ones(1, 1, grid_size, grid_size)

    for i in range(n_tests):
        
        problem_instance = DirichletProblem(N=grid_size, k=max_nb_iters * 1000 )
        gtt = problem_instance.ground_truth
        
        # jacoby method / known solver
        
        u_jac = im.jacobi_method(problem_instance.B_idx, problem_instance.B, f, u_0, k = 1)
        loss_jac = F.mse_loss(gtt, u_jac) # TODO use loss from metrics
        count_jac = 1
        
        nb_iters = 0
        while loss_jac >= loss_to_be_achieved and nb_iters < max_nb_iters:
            u_jac = im.jacobi_method(problem_instance.B_idx, problem_instance.B, f, u_jac,k = 1)
            loss_jac = F.mse_loss(gtt, u_jac)
            count_jac += 1
            nb_iters += 1
            
        # learned solver
        
        u_h = im.H_method(net,problem_instance.B_idx, problem_instance.B, f, u_0 ,k = 1)
        loss_h = F.mse_loss(gtt, u_h)
        count_h = 1
        
        # old method 
        nb_iters = 0
        while loss_h >= loss_to_be_achieved and nb_iters < max_nb_iters:
            u_h = im.H_method(net,problem_instance.B_idx, problem_instance.B, f, u_h,k = 1)
            loss_h = F.mse_loss(gtt, u_h)
            count_h += 1
            nb_iters += 1
        
        
        yield count_jac, count_h, compare_flops(grid_size,count_h,count_jac)


