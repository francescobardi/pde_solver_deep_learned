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
from nnpde.utils.logs import enable_logging, logging 
from importlib import reload
import nnpde.functions.iterative_methods as im
from nnpde.functions import geometries, helpers
from nnpde.problems import DirichletProblem 

# <codecell>

enable_logging(10)

# <codecell>

import torch
import torch.nn as nn
import torch.nn.functional as F

# <codecell>

# Define train dimension
N = 16

# Initialize f: we use a zero forcing term for training
f = torch.zeros(1, 1, N, N)

# For each problem instance define number of iteration to perform to obtain the solution
nb_problem_instances = 50
problem_instances = [DirichletProblem(k=k) for k in np.random.randint(1, 20, nb_problem_instances)]

# <markdowncell>

# Train using the class

# <codecell>

import nnpde.model as M 

# TODO fit would idealy take X, y, (u_0 and u_*)
model = M.JacobyWithConv(max_iters=10).fit(problem_instances)
losses = model.losses

# <codecell>

for name, param in train_net.net.named_parameters():
    print(name, param)

# <codecell>

color_map = plt.get_cmap('cubehelix')
colors = color_map(np.linspace(0.1, 1, 10))

losses_fig = plt.figure()
n_iter = np.arange(np.shape(losses)[0])
plt.plot(n_iter[:], losses[:], color = colors[0], linewidth = 1, linestyle = "-", marker = "",  label='Loss')

plt.legend(bbox_to_anchor=(0., -0.3), loc=3, borderaxespad=0.)
plt.xlabel('n iteration', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss')
plt.grid(True, which = "both", linewidth = 0.5,  linestyle = "--")

print("final loss is {0}".format(losses[-1]))
#losses_fig.savefig('gridSearch.eps', bbox_inches='tight')

# <markdowncell>

# Train without using the class

# <codecell>

# Solve the same problem, at each iteration the only thing changing are the weights, which are optimized
# TODO why though? wouldn't it make much more sense to train it more times on different problems? isn't this the same as oversampling each training sample?

for _ in range(100):
    net.zero_grad()
    loss = torch.zeros(1)
    prev_total_loss = 0

    batch_size = 1
    u_list = [None] * batch_size
    
    # Sample problem_instances
    problem_idx = np.random.choice(np.arange(nb_problem_instances), batch_size, replace = 0)

    for i in range(batch_size):

        idx = problem_idx[i]
        problem_instance = problem_instances_list[idx]
        
        B_idx = problem_instance.B_idx
        B = problem_instance.B
        f = problem_instance.forcing_term
        initial_u = problem_instance.initial_u
        k = problem_instance.k
        ground_truth = problem_instance.ground_truth
        
        # Compute the solution with the updated weights      
        u_list[i] = im.H_method(net, B_idx, B, f, initial_u, k)
        
        H = helpers.build_diagH_from_net(net,N)
        T = helpers.get_T(N)
        
        spectral_norm = helpers.calculate_spectral_radius(T,H)
        ex = 0
                
        if spectral_norm > 1:
            ex = np.nan_to_num(np.inf)

        # Define the loss, CHECK if it is correct wrt paper
        loss = loss + F.mse_loss(ground_truth, u_list[i])+ex
    


    # Backpropagation
    loss.backward(retain_graph =  False)

    # SGD step
    optim.step()
    
    total_loss = helpers.compute_loss(net, problem_instances_list,N)
    
    # Exit optimization 
    tol = 1e-2
    if total_loss.item() <= tol or total_loss.item() - prev_total_loss < tol:
        break


    # Store lossses for visualization
    losses.append(total_loss.item())
    prev_loss = total_loss.item()

for name, param in net.named_parameters():
    print(name, param)

# <markdowncell>

# Plot the losses

# <codecell>

color_map = plt.get_cmap('cubehelix')
colors = color_map(np.linspace(0.1, 1, 10))

losses_fig = plt.figure()
n_iter = np.arange(np.shape(losses)[0])
plt.plot(n_iter[:], losses[:], color = colors[0], linewidth = 1, linestyle = "-", marker = "",  label='Loss')

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
nb_iters = 2000

B_idx, B = geometries.square_geometry(N)

# Set forcing term
f = torch.ones(1,1,N,N)*1.0

# Obtain solutions
gtt = im.jacobi_method(B_idx, B, f, torch.ones(1,1,N,N), k = 10000)
output = im.H_method(net, B_idx, B, f, torch.ones(1,1,N,N), k = nb_iters)
jacoby_pure = im.jacobi_method(B_idx, B, f, torch.ones(1,1,N,N), k = nb_iters)

# <codecell>

loss_to_be_achieved = 1e-3

u_0 = torch.ones(1, 1, N, N)

# <codecell>

%%timeit

u_k_old = im.jacobi_method(B_idx, B, f, u_0, k = 1)
loss_of_old = F.mse_loss(gtt, u_k_old)
k_count_old = 1
count_old = 1
# old method 
while loss_of_old >= loss_to_be_achieved:
    u_k_old = im.jacobi_method(B_idx, B, f, u_k_old, k = 1)
    loss_of_old = F.mse_loss(gtt, u_k_old)
    k_count_old += 1

# <codecell>

%%timeit

u_k_new = im.H_method(net, B_idx, B, f, u_0, k=1)

loss_new = F.mse_loss(gtt, u_k_new)
k_count_new = 1


# new method

while loss_new >= loss_to_be_achieved:
    u_k_new = im.H_method(net, B_idx, B, f, u_k_new, k=1)
    loss_new = F.mse_loss(gtt, u_k_new)
    k_count_new += 1

# <codecell>

# This is not correct, but we have to look for a way to access the variables inside timeit

print("needed {0} iterations (compared to {1}), ratio: {2}".format(k_count_old, k_count_new, k_count_old/k_count_new))

# <codecell>

print("the loss of the new method is {0}, compared to the pure-jacoby one: {1}. computed with {2} iterations".format(F.mse_loss(gtt, output), F.mse_loss(gtt, jacoby_pure), nb_iters))

# <codecell>

helpers.plot_solution(gtt,output,N)

# <codecell>

(gtt.view(N,N) - output.view(N,N)).mean()

# <markdowncell>

# Test on L-shape domain

# <codecell>

B_idx, B = geometries.l_shaped_geometry(N)

# Set forcing term
f = torch.ones(1,1,N,N)*1.0

# Obtain solutions
gtt = im.jacobi_method(B_idx, B, f, torch.ones(1,1,N,N), k = 10000)
output = im.H_method(net, B_idx, B, f, torch.ones(1,1,N,N), k = 2000)

# <codecell>

helpers.plot_solution(gtt,output,N)
