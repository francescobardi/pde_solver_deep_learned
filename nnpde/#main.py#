# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Imports

# <codecell>

def fix_layout(width:int=95):
    from IPython.core.display import display, HTML
    display(HTML('<style>.container { width:' + str(width) + '% !important; }</style>'))
    
fix_layout()

# <codecell>

from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import nnpde.iterative_methods as im
from nnpde import geometries, helpers
from nnpde.utils.logs import enable_logging, logging 
from nnpde.problems import DirichletProblem 
from nnpde.utils import plots
import nnpde.model as M 

# <codecell>

enable_logging(10)

seed = 9 # Does not give problems
torch.manual_seed(seed)
np.random.seed(seed)

# <markdowncell>

# # Setup

# <codecell>

# Define train dimension: NxN
N = 16

# For each problem instance define number of iteration to perform to obtain the solution
nb_problem_instances = 30
problem_instances = [DirichletProblem(k=k) for k in np.random.randint(1, 20, nb_problem_instances)]

# <markdowncell>

# # Hyper-parameter search learning rate

# <codecell>

from itertools import product
import logging

def grid_search(mdl, base_parameters, grid_search_parameters, problem_instances):
    """
    Parameters
    ==========
    
        mdl                     Model Class, 
                                expected interface: `mdl(parameters).fit(problem_instances)`
        base_parameters         dictonary of parameters which will applied for all models
        grid_search_parameters  dictonary of <parameter>: [<value for parameter key>]
        problem_instances       list of problems to train on
    """
    # the `list` is necessary if you want to print the below message
    #parameters = list(product(*grid_search_parameters.values()))
    #logging.debug('testing {} models! make sure that you have the power to run this!'.format(len(parameters)))
    # `product` is equivalent to a nested for loop
    parameters = product(*grid_search_parameters.values())

    # the dict(zip(...)) part is necessary to ensure correct assignment
    res = [mdl(**{**base_parameters, **dict(zip(grid_search_parameters.keys(), p))}).fit(problem_instances) for p in parameters]
    return res

# <codecell>

def grid_search_wrapper(base_parameters, grid_search_parameters):
    return grid_search(mdl=M.JacobyWithConv,
                       base_parameters=base_parameters,
                       grid_search_parameters=grid_search_parameters,
                       problem_instances=problem_instances)

# <codecell>

# Net parameters
base_parameters = {
    "nb_layers": 3,
    "max_epochs": 100,
    "batch_size": 10,
    "stable_count": 10,
    "random_seed": 9
}

# SGD
grid_parameters = {
    "learning_rate": np.logspace(start=-6, stop=-4, num=7), #num=7 is good since it contains 1e-5
}

# <codecell>

reload(M)

# Took 3m 13s on a Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz

hyper_models = grid_search_wrapper(base_parameters, grid_parameters) \
    + grid_search_wrapper(base_parameters, {"optimizer": ["Adadelta"]})

# <codecell>

# Colors for plotting
color_map = plt.get_cmap('cubehelix')
colors = color_map(np.linspace(0.1, 1, 10))

# Initilize figure
hyper_fig = plt.figure()

# Plot SGD
i = 0
for model in hyper_models[:-1]:  
    n_epoch = np.arange(np.shape(model.losses)[0])
    plt.semilogy(n_epoch, 
                 model.losses, 
                 color=colors[i], 
                 linewidth=1, 
                 linestyle="-", 
                 marker=(i+2, 0, 0), 
                 markevery=10, 
                 label = f'$\gamma= {learning_rates[i]:.2e} $')
    i += 1

# Plot Adadelta
n_epoch = np.arange(np.shape(hyper_models[-1].losses)[0])
plt.semilogy(n_epoch, hyper_models[-1].losses, color=colors[i], linewidth=1, linestyle="-", marker=(i+2, 0, 0), markevery=10,label='Adadelta')

# Additional settings
plt.legend(bbox_to_anchor=(1.05, 0.31), loc=3, borderaxespad=0.)
plt.xlabel('n epochs', fontsize=14)
plt.ylabel('Total loss [-]', fontsize=14)
#plt.xlim([0, base_parameters['max_epochs']])
#plt.ylim([0, 200])
plt.title('Loss evolution for different learning rates, $K=3$, batchSize=10')
plt.grid(True, which = "both", linewidth = 0.5,  linestyle = "--")

#hyper_fig.savefig('../report/fig/hyper.eps', bbox_inches='tight')
plt.draw()
plt.show()

# <markdowncell>

# # Train model using K = 1,2,3,4,5

# <codecell>

base_parameters

# <codecell>

reload(M)

params = {**base_parameters, **{'max_epochs': 200, 'optimizer': 'Adadelta'}}
models = grid_search_wrapper(params, {'nb_layers': range(1, 6)})

#models = [M.JacobyWithConv(**{**params, 'nb_layers': nb_layers}).fit(problem_instances) for nb_layers in [1,2,3,4,5]]

# <codecell>

# Colors for plotting
color_map = plt.get_cmap('cubehelix')
colors = color_map(np.linspace(0.1, 1, 10))

# Initilize figure
comparison_K_fig = plt.figure()

# Plot SGD
i = 0
for model in models[:]:  
    n_epoch = np.arange(np.shape(model.losses)[0])
    plt.semilogy(n_epoch, model.losses, color=colors[i], linewidth=1, linestyle="-", marker=(i+2, 0, 0), markevery=100, label = '$K= {0} $'.format(model.nb_layers))
    print("For K={0} final loss is {1}".format(model.nb_layers, model.losses[-1]))
    i += 1

# Additional settings
plt.legend(bbox_to_anchor=(1.05, 0.31), loc=3, borderaxespad=0.)
plt.xlabel('n epochs', fontsize=14)
plt.ylabel('Total loss [-]', fontsize=14)
#plt.xlim([0, max_epochs])
#plt.ylim([0, 800])
plt.title('Loss evolution for different learning rates, $K=3$, batchSize=10')
plt.grid(True, which = "both", linewidth = 0.5,  linestyle = "--")

#hyper_fig.savefig('../report/fig/comparison_K.eps', bbox_inches='tight')
plt.draw()
plt.show()

# <codecell>

M._ConvNet_(0)

# <codecell>

models[1].max_epochs = 300
models[1].fit(problem_instances)

# <markdowncell>

# # Test on a bigger grid

# <codecell>

from nnpde.metrics import least_squares_loss as LSE

# <codecell>

# Grid size NxN
N = 64

# Use sufficiently high number of iterations to get ground truth solution
k_ground_truth = 20000

# Initialize Laplace problem on Square geometry
problem = DirichletProblem(N=N, k_ground_truth=20000)
B_idx = problem.B_idx
B = problem.B
f = problem.f

# Obtain solutions
ground_truth = problem.ground_truth

# Set initial_u equal for Jacobi method and for H method
initial_u = torch.ones(1,1,N,N)
k = 2000

# Obtain solution with Jacobi method
u_jacobi = im.jacobi_method(B_idx, B, f, initial_u, k = 2000)
print(f"Error after {k} iterations for Jacobi method: {LSE(ground_truth, u_jacobi)}")

# For each K obtain 
for model in models:
    u_H = im.H_method(model.net, B_idx, B, f, initial_u, k = 2000)
    print(f"Error after {k} iterations for H method with K={model.nb_layers}: {LSE(ground_truth, u_H)}")

# <markdowncell>

# # Error evolution with iterations

# <codecell>

tol = 1e-6
net = models[2].net

# <codecell>

u_jacobi = initial_u
err_jacobi = LSE(ground_truth, u_jacobi).item()
errs_jacobi = [err_jacobi] 
k_jacobi = 0

while err_jacobi >= tol:
    u_jacobi = im.jacobi_method(B_idx, B, f, u_jacobi, k = 1)
    err_jacobi = LSE(ground_truth, u_jacobi).item()
    errs_jacobi.append(err_jacobi)
    k_jacobi += 1
    
print(f"Jacobi method: error of {err_to_be_achieved} achieved after {k_jacobi} iterations.")

# <codecell>

errors_H = []
max_iters = 10000

for model in models:
    u_H = initial_u
    err_H = LSE(ground_truth, u_H).item()
    errs_H = [err_H] 
    k_H = 0

    while err_H >= tol:
        u_H = im.H_method(model.net, B_idx, B, f, u_H, k = 1)
        err_H = LSE(ground_truth, u_H).item()
        errs_H.append(err_H)
        k_H += 1
        if k_H > max_iters or err_H == np.inf:
            print(f"H method, K = {model.nb_layers}: convergence not reached after {max_iters}, final error is {err_H}.")
            break
    
    print(f"H method, K = {model.nb_layers}: error of {err_to_be_achieved} achieved after {k_H} iterations.")
    errors_H.append(errs_H)

# <codecell>

# Colors for plotting
color_map = plt.get_cmap('cubehelix')
colors = color_map(np.linspace(0.1, 1, 10))

# Initilize figure
error_k_fig = plt.figure()

i = 0
for error in errors_H:  
    n_iter = np.arange(np.shape(error)[0])
    plt.loglog(n_iter, error, color=colors[i], linewidth=1, linestyle="-", marker=(i+2, 0, 0), markevery=100, label = '$K= {0} $'.format(models[i].nb_layers))
    i += 1

# Plot error evolution for Jacobi 
n_iter = np.arange(np.shape(errs_jacobi)[0])
plt.loglog(n_iter, errs_jacobi, color=colors[i], linewidth=1, linestyle="-", marker=(i+2, 0, 0), markevery=1000, label = 'Jacobi')

# Additional settings
plt.legend(bbox_to_anchor=(1.05, 0.31), loc=3, borderaxespad=0.)
plt.xlabel('n iterations', fontsize=14)
plt.ylabel('Error [-]', fontsize=14)
#plt.xlim([0, max_epochs])
plt.ylim([tol, errors_H[0][0]])
plt.title('Error evolution for different $K$, $N={0}$'.format(N))
plt.grid(True, which = "both", linewidth = 0.5,  linestyle = "--")

#error_k_fig.savefig('../report/fig/error_k.eps', bbox_inches='tight')
plt.draw()
plt.show()

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

# <codecell>

compare_flops(16,k_count_new,k_count_old,3)

# <codecell>

Spectral radius. Don't remove Francesco will delete me

# <codecell>

B_idx = problem_instances[1].B_idx
net = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1, bias=False))
G = helpers.build_G(B_idx)
T = helpers.build_T(N)
H = helpers.conv_net_to_matrix(net, N)
I = np.eye(N)
helpers.spectral_radius(T+G.dot(H).dot(T)-G.dot(H))

# <markdowncell>

# # model testing

# <codecell>

import nnpde.model_testing as MT
import nnpde.problems as PDEF
reload(MT)
reload(PDEF)

# <codecell>

tol = 1e-4
base_parameters

# <codecell>

mdl = M.JacobyWithConv(**{**base_parameters, **{'max_epochs': 200, 'optimizer': 'Adadelta', 'nb_layers': 4}}).fit(problem_instances)

# <codecell>

tests_n10_g32 = MT.test_results_pd(mdl.net, 100, 32, tol=tol)
tests_n10_g32.to_pickle('./data/grid_32.pkl')

# <codecell>

# takes 7m!
test_results = MT.test_results_pd(mdl.net, 10, 64, tol=1e-4)
test_results.to_pickle('./data/grid_64.pkl')
test_results

# <codecell>

test_results2 = MT.test_results_pd(mdl.net, 100, 256, tol=tol)
test_results2.to_pickle('./data/grid_256.pkl')
test_results2

# <codecell>

test_results3 = MT.test_results_pd(mdl.net, 100, 512, tol=tol)
test_results3.to_pickle('./data/grid_512.pkl')
test_results3

# <codecell>

test_results_big_dimension = MT.test_results_pd(mdl.net, 10, 256)
4
test_results_big_dimension

# <codecell>

test_results_ada = MT.test_results_pd(mdlAda.net, 1, 64)
test_results_ada

# <codecell>

test_results_ada = MT.test_results_pd(mdlAda.net, 10, 64)
test_results_ada_big_dimension = MT.test_results_pd(mdl.netAda, 10, 256)
