# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

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

from sklearn.metrics import mean_squared_error

# <codecell>

def inner_jacoby(u):
    raise NotImplementedError
    pass


def reset_boundaries(u, boundaries):
    raise NotImplementedError
    pass
    
    
def jacoby_step(u, H):
    raise NotImplementedError
    w = inner_jacoby(u) - u
    return reset_boundaries(inner_jacoby(u) + np.dot(H, w))

# <codecell>

def our_loss(y_true, y_pred):
    """y_true is the u*
    y_pred is H
    """
    raise NotImplementedError
    
    # TODO how to get u?
    return mean_squared_error(y_true, jacoby_step(u, y_pred))
