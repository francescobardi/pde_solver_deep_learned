# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

def fix_layout(width:int=95):
    from IPython.core.display import display, HTML
    display(HTML('<style>.container { width:' + str(width) + '% !important; }</style>'))
    
fix_layout()

# <codecell>

import sys

import numpy as np
import matplotlib.pyplot as plt
#from logs import enable_logging, logging 
from importlib import reload
import iterative_methods as im

# <codecell>

sys.path

# <codecell>

sys.executable

# <codecell>

from nnpde

# <codecell>

sys.path

# <codecell>

!pwd

# <codecell>

from nnpde import utils

# <codecell>

import nnpde

# <codecell>

sys.path

# <codecell>

import torch
import torch.nn as nn
import torch.nn.functional as F

net = nn.Sequential(
    nn.Conv2d(1, 1, 3, padding=1, bias=False),
    nn.Conv2d(1, 1, 3, padding=1, bias=False),
    nn.Conv2d(1, 1, 3, padding=1, bias=False),
)


# Set the optimizer, you have to play with lr: if too big nan
optim = torch.optim.SGD(net.parameters(), lr = 1e-6)
#optim = torch.optim.Adam(net.parameters(), lr=1e-6)
#optim = torch.optim.ASGD(net.parameters())
# SGD seems much faster

for name, param in net.named_parameters():
    print(name, param)

# <markdowncell>

# New one based only on convolutions and pointwise tensor operations, see iterativeMethods.py

# <codecell>

losses = []

# <codecell>

some_tensor = torch.randn(1, 1, 16, 16)
some_tensor

# <codecell>

# Define train dimension
N = 16

for _ in range(20):
    net.zero_grad()

    # Sample k
    k = np.random.randint(1, 20)
    
    # Define geometry 1.0 inner points 0.0 elsewhre
    B_idx = torch.ones(1,1,N,N)
    B_idx[0,0,0,:] = torch.zeros(N)
    B_idx[0,0,N-1,:] = torch.zeros(N)
    B_idx[0,0,:,0] = torch.zeros(N)
    B_idx[0,0,:,N-1] = torch.zeros(N)
    
    # Define boundary values
    B = torch.abs(B_idx-1)*np.random.rand()*3
    
    # Initialize f: we use a zero forcing term for training
    f = torch.zeros(1, 1, N, N)

    # Initialize solution vector randomly 
    initial_u = torch.randn(1, 1, N, N, requires_grad = True)
    
    # Compute ustar = ground_truth solution torch 
    ground_truth = im.jacobi_method(B_idx, B, f, initial_u = None, k = 1000)

    # Solve the same problem, at each iteration the only thing changing are the weights, which are optimized
    for _ in range(20):
        
        # Compute the solution with the updated weights
        u = im.H_method(net, B_idx, B, f, initial_u, k)
        
        # Define the loss, CHECK if it is correct wrt paper
        loss = F.mse_loss(ground_truth, u)
        
        """ TODO 
        spectral_radius = TODO
        regularization = 1e10
        if spectral_radius > 1
           loss += regularization
        """
        
        # Exit optimization 
        tol = 1e-6
        if loss.item() <= tol:
            break
            
        # Backpropagation
        loss.backward(retain_graph =  False)
        
        # SGD step
        optim.step()
        
        # Store lossses for visualization
        losses.append(loss.item())

for name, param in net.named_parameters():
    print(name, param)

# <markdowncell>

# Plot the losses

# <codecell>

color_map = plt.get_cmap('cubehelix')
colors = color_map(np.linspace(0.1, 1, 10))

losses_fig = plt.figure()
n_iter = np.arange(np.shape(losses)[0])
plt.plot(n_iter, losses, color = colors[0], linewidth = 1, linestyle = "-", marker = "",  label='Loss')

plt.legend(bbox_to_anchor=(0., -0.3), loc=3, borderaxespad=0.)
plt.xlabel('n iteration', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss')
plt.grid(True, which = "both", linewidth = 0.5,  linestyle = "--")

print("final loss is {0}".format(losses[-1]))
#losses_fig.savefig('gridSearch.eps', bbox_inches='tight')

# <markdowncell>

# Test on a bigger grid

# <codecell>

N = 50

# Define geometry 1.0 inner points 0.0 elsewhre
B_idx = torch.ones(1,1,N,N)
B_idx[0,0,0,:] = torch.zeros(N)
B_idx[0,0,N-1,:] = torch.zeros(N)
B_idx[0,0,:,0] = torch.zeros(N)
B_idx[0,0,:,N-1] = torch.zeros(N)

# Define boundary values
B = torch.abs(B_idx-1)*4.0

# Set forcing term
f = torch.ones(1,1,N,N)*1.0

# Obtain solutions
gtt = im.jacobi_method(B_idx, B, f, torch.ones(1,1,N,N), k = 10000)
output = im.H_method(net, B_idx, B, f, torch.ones(1,1,N,N), k = 1000)

# <codecell>

print(F.mse_loss(gtt, output))

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

# <markdowncell>

# Test on L-shape domain

# <codecell>

N = 50
M = int(np.floor(N/2))

# Define geometry 1.0 inner points 0.0 elsewhre
B_idx = torch.ones(1, 1, N, N)
B_idx[0,0,0:M,0:M] = torch.zeros([M, M])
B_idx[0,0,N-1,:] = torch.zeros(N)
B_idx[0,0,:,0] = torch.zeros(N)
B_idx[0,0,:,N-1] = torch.zeros(N)

# Define boundary values
B = torch.abs(B_idx-1)*4.0

# Set forcing term
f = torch.ones(1,1,N,N)*1.0

# Obtain solutions
gtt = im.jacobi_method(B_idx, B, f, torch.ones(1,1,N,N), k = 10000)
output = im.H_method(net, B_idx, B, f, torch.ones(1,1,N,N), k = 2000)

# <codecell>

print(F.mse_loss(gtt, output))

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

# <markdowncell>

# Some way to compute the spectral radius maybe helpful. We need to build the matrix H and then compute it.
# H is built from the weights. Here we consider only the case conv1

# <codecell>

import scipy as sp
from scipy.linalg import circulant

vector_H = np.zeros(N**2)
vector_H[0] = param[0, 0, 1, 1]
vector_H[1] = param[0, 0, 1, 2]
vector_H[N**2-1] = param[0, 0, 1, 0]

vector_H[N**2-N] = param[0, 0, 0, 1]
vector_H[N**2-N-1] = param[0, 0, 0, 0]
vector_H[N**2-N+1] = param[0, 0, 0, 2]

vector_H[N] = param[0, 0, 2, 1]
vector_H[N-1] = param[0, 0, 2, 0]
vector_H[N+1] = param[0, 0, 2, 2]

p = np.zeros([3,3])
for name, param in net.named_parameters():
    for i in range(3):
        for j in range(3):
            p[i,j] = param[0,0,i,j]

H = np.diag(np.ones(N**2)*p[1,1]) + np.diag(np.ones(N**2-1)*p[1,2], 1) + np.diag(np.ones(N**2-1)*p[1,0], -1) + np.diag(np.ones(N**2-N)*p[2,1], N) + np.diag(np.ones(N**2-N)*p[0,1], -N)



#H = np.transpose(circulant(vector_H))
np.shape(H)
#print(np.real(np.linalg.eigvals(T+H.dot(T)-H)))
Heq = T.numpy()+H.dot(T.numpy())-H
reg = 0.0
#spectral_radius = (np.max(np.real(np.linalg.eigvals(Heq))))

spectral_radius = np.max(np.real(np.fft.fft(vector_T))) + np.max(np.real(np.fft.fft(vector_H)))*np.max(np.real(np.fft.fft(vector_TI)))

# <codecell>

T_I = np.reshape(T.numpy()-I.numpy(), [N**2, N**2])
Tn = np.reshape(T.numpy(), [N**2, N**2])
print(np.max(np.abs(np.real(np.linalg.eigvals(Tn-H.dot(T_I))))))
