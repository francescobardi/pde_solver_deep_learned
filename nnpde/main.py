# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

def fix_layout(width:int=95):
    from IPython.core.display import display, HTML
    display(HTML('<style>.container { width:' + str(width) + '% !important; }</style>'))
    
fix_layout()

# <markdowncell>

# # Notes
# 
# - **the model represents H**
# - H is learned for one, and only one iteration. But the same H is used for every iteration.
# - H should satisfy $H0 = 0$
# - H is a circulant matrix (in theory...) so we should be able to expand or shrink it according to the required dimensions. This is not clear to me, H can be anything... **TODO** we need to check this. Otherwise I don't know how to *expand* to the test dimensions.
# - The layers are defined as Convolutional Layers only. With a kernel size of (3, 3), a strife of (1, 1), without any bias and linear activation (equal to no activation). See below.
# 
# ```
# from keras.layers import Conv2D
# 
# Conv2D(<filters>,
#        kernel_size=(3, 3), 
#        strides=(1, 1), 
#        use_bias=False,
#        activation='linear')
# ```
# 
# - The learning objective, or loss is the mean_square_error off the model being used in the iteration.
# - X is $u^k$ (given the constrains, and for some iteration $k$)
# - y is $u^*$
# 
# T = some constant update matrix, c = some constant vector, $\psi$ an iterator
# $$u^{k + 1} = \psi(u^k) ) = T u^k + c$$
# $$w = \psi(u) - u$$
# $$\phi(u) = G(\psi(u) + H w) = G(\psi(u) + H(\psi(u) - u))$$
# 
# - But since the model tries to optimise the residuals... maybe the input of the model should be something different?

# <codecell>

import torch
import torch.nn as nn
import torch.nn.functional as F

# <codecell>

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # should be `nn.Conv2d(???, ???, 3, bias=False)` in our case
        self.conv1 = nn.Conv2d(1, 1, 3, bias=False) 
        
    def forward(self, x):
        x = self.conv1(x)
        
        return x

# <codecell>

from nnpde.functions import iterativeMethods as im
import numpy as np

# <codecell>

N = 10
a = np.ones(N**2)
b = -np.ones(N**2-1)*0.25
c = -np.ones(N**2-N)*0.25

A = np.diag(a) + np.diag(b, 1) + np.diag(b, -1) + np.diag(c, N) + np.diag(c, -N)


b_top_idx = np.arange(N)
b_bottom_idx = np.arange(N**2-N, N**2)
b_left_idx = np.linspace(N, N**2-2*N, N-2, dtype = int)
b_right_idx = np.linspace(2*N-1, N**2-N, N-2, dtype = int)

b_idx = np.append(b_top_idx, b_bottom_idx)
b_idx = np.append(b_idx, b_left_idx)
b_idx = np.append(b_idx, b_right_idx)
b = np.ones(np.shape(b_idx))*1
f = np.zeros(N**2)

u, res = im.jacobi(A, f, b_idx = b_idx, b = b, max_iters=200,tol = 1e-2)

# <codecell>

u

# <codecell>

T = torch.eye(N**2) - torch.from_numpy(A).float()
G = torch.eye(N**2) 

def reset_boundaries(X, G=G):
    # TODO 
    return X


def iterator_step(H, T=T, G=G):
    return reset_boundaries(T + H@T - H, G=G)
    
    
def iterations(u0, k, H, T=T, G=G):
    X = iterator_step(H, T=T, G=G)
    return F.reduce(lambda acc, el: el(acc), [(lambda u: X.mm(u)) for _ in range(k)], u0)


class CustLoss(nn.Module):
    def __init__(self, u_stars, k=5):
        super(CustLoss, self).__init__()
        
        self.k = k
        # should be u0 and u*
        # TODO `0` is the initial value, is this correct?
        self.u_stars = [(0, u_star) for u_star in u_stars]
        
    def forward(self, yPred, yTrue):
        # we don't care about y, as this would be y_true??? or is it x???
        
        H = torch.from_numpy(yPred).requires_grad(True)
    
        # since we doing for all u_stars we can do it probably in a better way (vectorised)
        return torch.sum[(torch.abs(iterations(u0, k=self.k, H=yPred) - u_star) for u0, u_star in self.u_stars)]

# <codecell>

loss_fn = CustLoss(u)


prediction = jacobi(A, b, H)
loss = loss_fn(prediction, ground_truth)
loss.backward()
optimizer.step()
H.zero_grad() # H is the model

# <codecell>

net = Net()

# <codecell>

net

# <codecell>

input = torch.ones(1, 1, 5, 5) # what to pluck here???
out = net(input)

# <codecell>

out

# <codecell>

out

# <codecell>


