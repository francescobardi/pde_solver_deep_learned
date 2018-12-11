# -*- coding: utf-8 -*-
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt

import nnpde.functions.iterative_methods as im
import torch.nn.functional as F

def check_dimensions(A, f):
    """Check the dimensions of inputs"""

    Na, Ma = np.shape(A)
    Nf = np.shape(f)[0]

    if Na == Ma and Na == Nf:
        return Na
    else:
        raise ValueError("Dimensions mismatch, please check your inputs.")


def build_diagH_from_net(net, N):
    Hs = []
    
    for param in net.parameters():
        ks = param.view(9)
        ks = list(map(lambda el: el.item(),ks))
        
        H = np.diag(np.ones(N**2)) * ks[4] + np.diag(np.ones(N**2-1),1)* ks[5] \
        + np.diag(np.ones(N**2 - N),N) * ks[7] + np.diag(np.ones(N**2 - N - 1),N+1) * ks[8] \
        + np.diag(np.ones(N**2 - 1), - 1) * ks[3] + np.diag(np.ones(N**2 - N-1),N+1) * ks[6] \
        + np.diag(np.ones(N**2 - N), -N) * ks[1] +  np.diag(np.ones(N**2 - N-1),-N-1) * ks[2] \
        + np.diag(np.ones(N**2 - N - 1),-N-1) * ks[0]
        
        Hs.append(H)
    
    return (Hs[2].dot(Hs[1])).dot(Hs[0])

def compute_loss(net, problem_instances_list,N):
    """ Fucntion to compute the total loss given a set of problem instances"""
    
    nb_problem_instances = len(problem_instances_list)
    loss = torch.zeros(1, requires_grad = False)
    u = torch.zeros(1, 1, N, N, nb_problem_instances)
    
    for problem_instance in problem_instances_list:

        B_idx = problem_instance.B_idx
        B = problem_instance.B
        f = problem_instance.forcing_term
        initial_u = problem_instance.initial_u
        k = problem_instance.k
        ground_truth = problem_instance.ground_truth
        
        u = im.H_method(net, B_idx, B, f, initial_u, k)
        loss = loss + F.mse_loss(ground_truth, u)
        
    return loss

def calculate_spectral_radius(T,H):
    T1 = T[0,0,:,:].double()
    H1 = torch.from_numpy(H).double()
    tmp = T1 + T1 @ H1 - H1
    eigvals = np.linalg.eigvals(tmp)
    return np.abs(np.max(np.real(eigvals)))

def get_T(N):
    A = np.eye(N**2)
    
    for i in range(N, N**2-N):
        if (i%N != 0 and i%N != N-1):
            # Left and right neigh
            A[i][i-1] = -0.25 
            A[i][i+1] = -0.25
            # Up and low neigh
            A[i][i-N] = -0.25 
            A[i][i+N] = -0.25
    I = torch.zeros(1,1,N**2,N**2)
    I[0,0,:,:] = torch.eye(N**2,N**2)
    At = torch.zeros(1,1,N**2,N**2)
    At[0,0,:, :] = torch.from_numpy(A)
    T = I - At
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
    
 