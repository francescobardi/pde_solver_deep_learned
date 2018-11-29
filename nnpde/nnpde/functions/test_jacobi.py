# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import sys
sys.path.append('../../../pdl/pdl/utils/')

import numpy as np
import matplotlib.pyplot as plt
from logs import enable_logging, logging 
from importlib import reload
import iterativeMethods as im


# <codecell>

enable_logging(lvl=100)
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

N = 20
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

N = 20
a = np.ones(N**2)
b = -np.ones(N**2-1)*0.25
c = -np.ones(N**2-N)*0.25

A = np.diag(a) + np.diag(b, 1) + np.diag(b, -1) + np.diag(c, N) + np.diag(c, -N)
print(A)


b_top_idx = np.arange(N)
b_bottom_idx = np.arange(N**2-N, N**2)
b_left_idx = np.linspace(N, N**2-2*N, N-2, dtype = int)
b_right_idx = np.linspace(2*N-1, N**2-N, N-2, dtype = int)

print(b_top_idx)
print(b_bottom_idx)
print(b_left_idx)
print(b_right_idx)

b_idx = np.append(b_top_idx, b_bottom_idx)
b_idx = np.append(b_idx, b_left_idx)
b_idx = np.append(b_idx, b_right_idx)
print(b_idx)
b = np.ones(np.shape(b_idx))*1
print(b)
f = np.zeros(N**2)

u, res = im.jacobi(A, f, b_idx = b_idx, b = b,max_iters=200,tol = 1e-2)

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



# <codecell>


