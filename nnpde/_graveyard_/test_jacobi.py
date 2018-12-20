# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

def fix_layout(width:int=95):
    from IPython.core.display import display, HTML
    display(HTML('<style>.container { width:' + str(width) + '% !important; }</style>'))
    
fix_layout()

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
#from logs import enable_logging, logging 
from importlib import reload

import nnpde.functions.iterative_methods as im
import nnpde.functions.geometries as geo

# <codecell>

#enable_logging(lvl=100)
A = np.array([[2, 1], [5, 7]])
f = np.array([11, 13])

u, res = im.jacobi(A, f)

# <markdowncell>

# For the real Jacobi method instead the spectral radius of the updating matrix is < 1.
# For the real Jacobi methods it holds the fact that if the matrix is diagonally dominant then the spectral radius of the updating matrix is guaranteed to be < 1. (http://www.cs.unipr.it/~bagnara/Papers/PDF/SIREV95.pdf)
# This condition does not hold for the paper Jacobi method.

# <codecell>

jacobi_eig = np.linalg.eigvals(np.linalg.inv(np.diag(np.diag(A))).dot(A-np.diag(np.diag(A))))
spectral_radius = np.max(np.abs(jacobi_eig))
print(spectral_radius)

# <markdowncell>

# Build the matrix A for the 2D Poisson problem

# <codecell>

N = 3
A = np.eye(N**2)
# Domani length
L = 1.0
# Cell size
h = L/(N-1)

# set homegenous dirichlet BC value
b = 1.0

#Initilize forcing term
f = np.ones(N**2)*b

for i in range(N, N**2-N):
    if (i%N != 0 and i%N != N-1):
        # Left and right neigh
        A[i][i-1] = -0.25 
        A[i][i+1] = -0.25
        # Up and low neigh
        A[i][i-N] = -0.25 
        A[i][i+N] = -0.25 
        # set forcing term
        f[i] = 0


# <markdowncell>

# Obtain the solution with jacobi method

# <codecell>

u, res = im.jacobi(A, f, max_iters=10000,tol = 1e-3)
#u = np.linalg.inv(A).dot(f)

# <markdowncell>

# Plot the solution
# Nice reference for contour plots https://www.python-course.eu/matplotlib_contour_plot.php

# <codecell>

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

Z = np.reshape(u, [N, N])
plt.figure()
cp = plt.contourf(X, Y, Z)
plt.colorbar(cp)
plt.title('Filled Contours Plot')
plt.xlabel('x [-]')
plt.ylabel('y [-]')
plt.show()

# <markdowncell>

# Obtain the same solution with reset operator G


# <codecell>

#### Initialize A, b
N = 16
a =  np.ones(N**2)
b = -np.ones(N**2-1)*0.25
c = -np.ones(N**2-N)*0.25

A = np.diag(a) + np.diag(b, 1) + np.diag(b, -1) + np.diag(c, N) + np.diag(c, -N)
#print(A)


b_top_idx = np.arange(N)
b_bottom_idx = np.arange(N**2-N, N**2)
b_left_idx = np.linspace(N, N**2-2*N, N-2, dtype = int)
b_right_idx = np.linspace(2*N-1, N**2-N-1, N-2, dtype = int)


print(b_top_idx)
print(b_bottom_idx)
print(b_left_idx)
print(b_right_idx)

b_idx = np.append(b_top_idx, b_bottom_idx)
b_idx = np.append(b_idx, b_left_idx)
b_idx = np.append(b_idx, b_right_idx)
#print(b_idx)
b = np.ones(np.shape(b_idx))*3.0
#print(b)
f = np.ones(N**2)

u, res = im.jacobi(A, f, boundary_index = b_idx, boundary_values = b,max_iters=1000,tol = 1e-2)

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

Z = np.reshape(u, [N, N])
plt.figure()
cp = plt.contourf(X, Y, Z)
plt.colorbar(cp)
plt.title('Filled Contours Plot')
plt.xlabel('x [-]')
plt.ylabel('y [-]')
plt.show()



# <codecell>

import torch
import torch.nn as nn
import torch.nn.functional as F

# One convolution with a 3x3 kernel
#net = nn.Conv2d(1, 1, 3, padding=1, bias=False)

# Do we want to use 3 layers?
jac = nn.Conv2d(1, 1, 3, padding=1, bias=False)

initial_weights = torch.zeros(1,1,3,3)
initial_weights[0,0,0,1] = 0.25
initial_weights[0,0,2,1] = 0.25
initial_weights[0,0,1,0] = 0.25
initial_weights[0,0,1,2] = 0.25
jac.weight = nn.Parameter(initial_weights)

for param in jac.parameters():
    param.requires_grad = False
    
for name, param in jac.named_parameters():
    print(name, param)


    
net = nn.Sequential(
    nn.Conv2d(1, 1, 3, padding=1, bias=False),
    nn.Conv2d(1, 1, 3, padding=1, bias=False),
    nn.Conv2d(1, 1, 3, padding=1, bias=False),
)


# Initialize the wights s.t. H corresponds T (see 2.3.1)
#initial_weights = torch.rand(1,1,3,3)*0.01
#initial_weights[0,0,0,1] = 0.25
#initial_weights[0,0,2,1] = 0.25
#initial_weights[0,0,1,0] = 0.25
#initial_weights[0,0,1,2] = 0.25
#initial_weights[0,0,1,1] = 1.0
#net.weight = nn.Parameter(initial_weights)

# Set the optimizer, you have to play with lr: if too big nan
optim = torch.optim.SGD(net.parameters(), lr=1e-6)
#optim = torch.optim.Adam(net.parameters(), lr=1e-6)
#optim = torch.optim.ASGD(net.parameters())
# SGD is much faster

for name, param in net.named_parameters():
    print(name, param)



# <codecell>

def torch_conv_to_matrix(conv):
    t = conv.view(-1).detach().numpy()
    kernel_dim = np.int(np.sqrt(t.shape[0]))
    return t.reshape(kernel_dim, kernel_dim)


def reshape_and_cut_border(x, border_size=1):
    dim = np.int(np.sqrt(x.shape[0]))
    return x.reshape((dim, dim))[border_size:-border_size, border_size:-border_size]\
        .reshape(-1)


def convolution_as_matrix_multiplication(conv, input_dim):
    # following https://dsp.stackexchange.com/questions/35373/convolution-as-a-doubly-block-circulant-matrix-operating-on-a-vector
    # we have k * x = y, x being the input (w) and k being the kernel
    # 1. output size of y is (dim_k + dim_x - 1)^2 => pad k accordingly
    kernel_dim = conv.shape[0]
    N = input_dim
    A = np.zeros((N + kernel_dim - 1, N + kernel_dim - 1)) # N comes from the size of u
    A[-kernel_dim:, 0:kernel_dim] = conv # place the kernel in the lower left corner

    # 2. get circulant matrices for each row of A
    # remember that circulant_rows[0] should be the circulant defined by the last row
    from scipy.linalg import circulant
    circulant_of_rows = [circulant(A[row, :]) for row in reversed(range(A.shape[0]))]

    # 3. construct doubly circulant matrix
    # 
    #      X_0 X_{N-1} X_{N-2} ...
    # X =  X_1 X_0 X_{N-1} ...
    #      X_2 X_1 X_0
    #      ...
    #
    # where X_i denotes with circulant matrices from circulant_rows

    doubly_circulant_idx = circulant(range(A.shape[0]))
    X = np.hstack([np.vstack([circulant_of_rows[idx] for idx in doubly_circulant_idx[:, i]]) 
                   for i in range(doubly_circulant_idx.shape[1])])
    
    return X


def prep_y_for_conv_mat_mult(y, conv_dim):
    # 4. create corresponding b vector from u
    # how? I placed in the middle...

    # this is special for our case, don't know if this checks out in general
    u_padded_dim = np.int(np.sqrt(conv_dim))
    u_padded = np.zeros((u_padded_dim, u_padded_dim))

    dim = np.int(np.sqrt(y.shape[0]))
    u_padded[1:-1, 1:-1] = y.reshape(dim, dim)
    return u_padded



conv = np.random.rand(3, 3)
X = convolution_as_matrix_multiplication(conv, N)
t = reshape_and_cut_border(X@prep_y_for_conv_mat_mult(u, X.shape[0]).reshape(-1))


from scipy.signal import convolve2d

y_true = reshape_and_cut_border(convolve2d(conv, u.reshape(N, N)).reshape(-1))

t - y_true

# <codecell>

# Build G matrix and B to reset the boundaries
G = np.eye(N**2)
G[b_idx, b_idx] -= 1
B = np.zeros(N**2)
B[b_idx] = b

# Convert G and B to torch
Bt = torch.zeros(1,1,N**2,1)
Bt[0, 0, :,0] = torch.from_numpy(B)
Gt = torch.zeros(1,1,N**2,N**2)
Gt[0,0,:, :] = torch.from_numpy(G)

# Build T matrix for standard jacobi iteration
I = torch.zeros(1,1,N**2,N**2)
I[0,0,:,:] = torch.eye(N**2,N**2)
At = torch.zeros(1,1,N**2,N**2)
At[0,0,:, :] = torch.from_numpy(A)
T = I - At


# <codecell>

losses = []


# <markdowncell>

# The output is obtained applying k times G(Tu+net(Tu)+f+net(f)-net(u)) + B = G(Tu+HTu+f+Hf-Hu) + B.
# G is the I with zeros in the boundary nodes
# B is a vector with the boundary values in the boundary nodes and zeros elsewhere.

# <codecell>

# Function that builds matrix H from the convolutional neural net
# We will later need H in order to enforce the spectral radius constraint
def build_H_from_net(param = param):
    p = np.zeros([3,3])
    for name, param in net.named_parameters():
        for i in range(3):
            for j in range(3):
                p[i,j] = param[0,0,i,j]

    H = np.diag(np.ones(N**2)*p[1,1]) + np.diag(np.ones(N**2-1)*p[1,2], 1) + np.diag(np.ones(N**2-1)*p[1,0], -1) + np.diag(np.ones(N**2-N)*p[2,1], N) + np.diag(np.ones(N**2-N)*p[0,1], -N)
    return H

# <codecell>

kern = param[0,0,:,:]
ks = []
for i in range(kern.shape[0]):
    for j in range(kern.shape[1]):
        ks.append(kern[i][j].item())

# <codecell>

# This function builds a convolution matrix from the kernel matrix of the convolution
# Hence, it allows us to express convolution as a simple matrix multiplication

def build_diagH_from_net(param = param):
    kern = param[0,0,:,:]
    ks = []
    for i in range(kern.shape[0]):
        for j in range(kern.shape[1]):
            ks.append(kern[i][j].item())
    
    H = np.diag(np.ones(N**2)) * ks[4] + np.diag(np.ones(N**2-1),1)* ks[5] \
    + np.diag(np.ones(N**2 - N),N) * ks[7] + np.diag(np.ones(N**2 - N - 1),N+1) * ks[8] \
    + np.diag(np.ones(N**2 - 1), - 1) * ks[3] + np.diag(np.ones(N**2 - N-1),N+1) * ks[6] \
    + np.diag(np.ones(N**2 - N), -N) * ks[1] +  np.diag(np.ones(N**2 - N-1),-N-1) * ks[2] \
    + np.diag(np.ones(N**2 - N - 1),-N-1) * ks[0]
    
    return H

# <codecell>



# <codecell>

# Calculate the spectral radius of T+TH - H as the absolute value of the maximum eigenvalue

def calculate_spectral_radius(T,H):
    T1 = T[0,0,:,:].double()
    H1 = torch.from_numpy(H).double()
    tmp = T1 + T1 @ H - H
    eigvals = np.linalg.eigvals(tmp)
    return np.abs(np.max(np.real(eigvals)))

# <codecell>

import pandas as pd

# <codecell>

pd.DataFrame(build_H_from_net())

# <codecell>

H = build_diagH_from_net()

# <codecell>

H

# <codecell>

lambda_ = 0.7 # Regularization term for the spectral radius (This is not the optimal one)
k = 100 # We can sample it as before, but just for testing purposes

# <codecell>

# Define train dimension
N = 16

for _ in range(20):
    prev_loss = 0
    net.zero_grad()

    # Sample k
    #k = np.random.randint(1, 20)
    
    # Define geometry 1.0 inner points 0.0 elsewhere
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
    for _ in range(k):
        
        # Compute the soultion with the updated weights
        u = im.H_method(net, B_idx, B, f, initial_u, k)
        
        # Define the loss, CHECK if it is correct wrt paper
                
        spectral_radius = calculate_spectral_radius(T,H)
        
        print('spectral radius is')
        print(spectral_radius)
        loss = F.mse_loss(ground_truth, u) + lambda_ * max(0,spectral_radius - 1)
        
        
        # Exit optimization 
        tol = 1e-6
        if loss.item() <= tol or loss.item() - prev_loss < tol:
            break
            
        # Backpropagation
        loss.backward(retain_graph =  False)
        
        # SGD step
        optim.step()
        
        # Store lossses for visualization
        losses.append(loss.item())
        prev_loss = loss.item()

for name, param in net.named_parameters():
    print(name, param)

# <markdowncell>

# Plot the losses

# <codecell>

import matplotlib.pyplot as plt
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

# <codecell>

print(k)

# <markdowncell>

# Compare solution obtained with ground truth 

# <codecell>

# Just factorized some code to make it compatible with the new functions in iterativeMethod
k = 500
# Initialize u0
#a = np.random.rand(N**2)
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
output = im.H_method(net, B_idx, B, f, initial_u, k)
        
# Define the loss, CHECK if it is correct wrt paper

# I should look for an analitical solution

loss = F.mse_loss(ground_truth, u)

# <codecell>

print(F.mse_loss(ground_truth, output))

# <codecell>

H = build_H_from_net()
Heq = T.numpy()+H.dot(T.numpy())-H
reg = 0.0
#spectral_radius = (np.max(np.real(np.linalg.eigvals(Heq))))

spectral_radius = calculate_spectral_radius(T,H)
print(spectral_radius)
