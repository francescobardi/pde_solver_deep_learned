# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Setup" data-toc-modified-id="Setup-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Setup</a></span></li><li><span><a href="#Hyper-parameter-search-learning-rate" data-toc-modified-id="Hyper-parameter-search-learning-rate-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Hyper-parameter search learning rate</a></span></li><li><span><a href="#Train-model-using-K-=-1,2,3,4,5" data-toc-modified-id="Train-model-using-K-=-1,2,3,4,5-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Train model using K = 1,2,3,4,5</a></span></li><li><span><a href="#Test-on-a-bigger-grid" data-toc-modified-id="Test-on-a-bigger-grid-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Test on a bigger grid</a></span></li><li><span><a href="#Plotting-solutions-for-domain-shapes" data-toc-modified-id="Plotting-solutions-for-domain-shapes-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Plotting solutions for domain shapes</a></span><ul class="toc-item"><li><span><a href="#Plot-Square-Domain" data-toc-modified-id="Plot-Square-Domain-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Plot Square Domain</a></span></li><li><span><a href="#Plot-L-shape" data-toc-modified-id="Plot-L-shape-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Plot L shape</a></span></li></ul></li><li><span><a href="#Error-evolution-with-iterations" data-toc-modified-id="Error-evolution-with-iterations-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Error evolution with iterations</a></span></li><li><span><a href="#Model-testing" data-toc-modified-id="Model-testing-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Model testing</a></span></li></ul></div>

# <markdowncell>

# # Imports

# <codecell>

def fix_layout(width:int=95):
    from IPython.core.display import display, HTML
    display(HTML('<style>.container { width:' + str(width) + '% !important; }</style>'))
    
fix_layout()

# <codecell>

import os
from importlib import reload
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display

import nnpde.iterative_methods as im
from nnpde.metrics import least_squares_loss as LSE
from nnpde import geometries, helpers
from nnpde.utils.logs import enable_logging, logging 
from nnpde.problems import DirichletProblem 
from nnpde.utils import plots
import nnpde.model as M 
import nnpde.model_testing as MT
import nnpde.problems as PDEF
from nnpde.grid_search import grid_search

# <codecell>

enable_logging(20)

seed = 9 # Does not give problems
torch.manual_seed(seed)
np.random.seed(seed)

# <markdowncell>

# # Setup

# <codecell>

# Define train dimension: NxN
N = 16

# For each problem instance define number of iteration to perform to obtain the solution
nb_problem_instances = 50
problem_instances = [DirichletProblem(k=k) for k in np.random.randint(1, 20, nb_problem_instances)]

# Net parameters, will also be used further down.
base_parameters = {
    "nb_layers": 3,
    "max_epochs": 200,
    "batch_size": 10,
    "stable_count": 10,
    "random_seed": 9,
}

# SGD
grid_parameters = {
    "learning_rate": np.logspace(start=-6, stop=-4, num=7), #num=7 is good since it contains 1e-5
}

# <markdowncell>

# # Hyper-parameter search learning rate

# <codecell>

def grid_search_wrapper(base_parameters, grid_search_parameters):
    return grid_search(mdl=M.JacobyWithConv,
                       base_parameters=base_parameters,
                       grid_search_parameters=grid_search_parameters,
                       problem_instances=problem_instances)

# <codecell>

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
                 label = f'$\gamma= {model.learning_rate:.2e} $')
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
plt.title('Loss evolution for different learning rates $\gamma$ \n $K=3$, $|\mathcal{D}|=50$, $|\mathcal{B}|=10$, max epochs=200')
plt.grid(True, which = "both", linewidth = 0.5,  linestyle = "--")

#hyper_fig.savefig('../report/figs/hyper.eps', bbox_inches='tight')
plt.draw()
plt.show()

# <markdowncell>

# # Train model using K = 1,2,3,4,5

# <codecell>

base_parameters

# <codecell>

reload(M)

params = {**base_parameters, **{'max_epochs': 1000, 'optimizer': 'Adadelta'}}
models = grid_search_wrapper(params, {'nb_layers': range(1, 6)})

#models = [M.JacobyWithConv(**{**params, 'nb_layers': nb_layers}).fit(problem_instances) for nb_layers in [1,2,3,4,5]]
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
plt.xlim([0, 300])
#plt.ylim([0, 800])
plt.title('Loss evolution for different $K$ \n $|\mathcal{D}|=50$, $|\mathcal{B}|=10$, Adadelta, max epochs=1000')
plt.grid(True, which = "both", linewidth = 0.5,  linestyle = "--")

#comparison_K_fig.savefig('../report/figs/comparison_K.eps', bbox_inches='tight')
plt.draw()
plt.show()

# <markdowncell>

# # Test on a bigger grid

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

# # Plotting solutions for domain shapes 

# <markdowncell>

# ## Plot Square Domain

# <codecell>

problem_square = DirichletProblem(N=N, k_ground_truth=20000)
ground_truth_square = problem_square.ground_truth.view(N, N).numpy()

square_fig = plt.figure()
im = plt.imshow(ground_truth_square)
plt.title("Square domain.")
plt.colorbar(im)

#square_fig.savefig('../report/figs/square.eps', bbox_inches='tight')
plt.draw()
plt.show()

# <markdowncell>

# ## Plot L shape

# <codecell>

problem_l_shape = DirichletProblem(N=N, k_ground_truth=20000, domain_type = "l_shape")
ground_truth_l_shape = problem_l_shape.ground_truth.view(N, N).numpy()

square_fig = plt.figure()
im = plt.imshow(ground_truth_l_shape)
plt.title("L-shape domain.")
plt.colorbar(im)

#square_fig.savefig('../report/figs/l_shape.eps', bbox_inches='tight')
plt.draw()
plt.show()

# <markdowncell>

# # Error evolution with iterations

# <codecell>

tol = 1e-6
net = models[2].net

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
    
    print(f"H method, K = {model.nb_layers}: error of {tol} achieved after {k_H} iterations.")
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
plt.title('Error evolution for different $K$\n Grid size ${0}x{0}$'.format(N))
plt.grid(True, which = "both", linewidth = 0.5,  linestyle = "--")

#error_k_fig.savefig('../report/figs/error_k.eps', bbox_inches='tight')
plt.draw()
plt.show()

# <markdowncell>

# # Model testing

# <codecell>

tol = 1e-6
if base_parameters is None:
    raise ValueError("Execute cell with base parameters")

_base_data_path_ = '../report/data/'
_base_fig_path_ = '../report/figs/'

if not os.path.exists(_base_data_path_):
    os.makedirs(_base_data_path_)
    

def obtain_test_results(mdl, grid_size, nb_tests=50, domain_shape='square', nb_layers=4, force=False, plot=False):
    data_path = f'{_base_data_path_}nb_layers_{nb_layers}_grid_{grid_size}_domain_{domain_shape}.pkl'
    
    if force or not os.path.exists(data_path):
        test_results = MT.test_results_pd(mdl, nb_tests, grid_size, tol=tol, convergence_tol=1e-12)
        test_results['grid_size'] = grid_size
        test_results['shape'] = domain_shape
        test_results.to_pickle(data_path)
    else:
        test_results = pd.read_pickle(data_path)


    if plot:
        test_results['iters_ratio'] = test_results['nb_iters_convjac'] / test_results['nb_iters_jac'] 
        ax = sns.boxplot(data=test_results[['flops_ratio', 'cpu_time_ratio', 'iters_ratio']]\
                         .rename(columns={'flops_ratio': 'Ratio of FLOPS', 'cpu_time_ratio': 'Ratio of CPU time', 'iters_ratio': 'Ratio of #iterations'}), orient="h", palette="Set2")
        ax.set_title(f'Test results for grid size: {grid_size}')
        plt.savefig(f'{_base_fig_path_}grid_{grid_size}_domain_{domain_shape}.eps')
        display(ax)
    return test_results


def agg_for_layer(base_parameters, nb_layers, problem_instances, grid_sizes, nb_tests=20):
    mdl = M.JacobyWithConv(**{**base_parameters, **{'max_epochs': 1000, 'optimizer': 'Adadelta', 'nb_layers': nb_layers}})\
           .fit(problem_instances)
    
    test_results = [obtain_test_results(mdl, grid_size=grid_size, domain_shape=shape, nb_tests=nb_tests, nb_layers=nb_layers, plot=True, force=True) 
         for grid_size, shape in product(grid_sizes, ['l_shape', 'square'])]

    d = {'flops_ratio': 'FLOPS ratio', 'cpu_time_ratio': 'CPU time ratio', 'nb_iters_jac': 'nb iters existent solver', 'nb_iters_convjac': 'nb iters trained solver'}
    ts_concat = pd.concat(test_results).rename(columns=d)
    ta = ts_concat.groupby(['grid', 'shape'])[list(d.values())].mean().reset_index().rename(columns={'grid': 'grid size'})
    ta['nb_layers'] = nb_layers
    ts_concat['nb_layers'] = nb_layers
    return ta, ts_concat

# <codecell>

cols = [
 'nb_layers',
 'grid size',
 'shape',
 'FLOPS ratio',
 'CPU time ratio',
 'nb iters existent solver',
 'nb iters trained solver'
]

#nb_problem_instances = 100
#problem_instances_n16 = [DirichletProblem(k=k, N=16) for k in np.random.randint(1, 20, nb_problem_instances)]
#problem_instances_n16 = [DirichletProblem(k=k, N=16) for k in np.random.randint(1, 20, nb_problem_instances)]

ts = [agg_for_layer(base_parameters, grid_sizes=[32, 64], nb_layers=l, problem_instances=problem_instances)[0] for l in range(1, 6)]
#final_results = pd.concat([agg_for_layer(l, problem_instances=problem_instances)[0] for l in range(1, 5)])[cols]

#final_results.to_pickle('./data/final_test_results.pkl')

# <codecell>

final_results

# <codecell>


