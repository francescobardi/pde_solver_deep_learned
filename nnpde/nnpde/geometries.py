import numpy as np
import torch


def square_geometry(size):
    """Defines a square geometry of given size (as a single positive integer)
    The boundary-values are set randomly from a uniform distribtion in the interval [-1, 1].

    Example:

    b1 b1 b2
    b4 0  b2
    b4 b3 b3

    Returns:
        (geometry, boundary_index)
    """

    # Define geometry 1.0 inner points 0.0 elsewhre
    B_idx = torch.ones(1, 1, size ,size )
    B_idx[0, 0,  0,  :] = torch.zeros(size)
    B_idx[0, 0, -1,  :] = torch.zeros(size)
    B_idx[0, 0,  :,  0] = torch.zeros(size)
    B_idx[0, 0,  :, -1] = torch.zeros(size)

    # Define boundary values
    B = torch.zeros_like(B_idx)

    B[0, 0,  0, :] = np.random.uniform(-1, 1)
    B[0, 0,  :,-1] = np.random.uniform(-1, 1)
    B[0, 0, -1, :] = np.random.uniform(-1, 1)
    B[0, 0, 1:, 0] = np.random.uniform(-1, 1) # we don't want to overwrite the first value
    return B_idx, B


def l_shaped_geometry(size, l_cutout_size=None):
    """Defines a L-shaped geometry of given size (as a single positive integer), and
    l_cutout_size (think of creating the L-shape as cutting out a smaller square piece)

    l_cutout_size is by default size/2

    Returns:
        (geometry, boundary_index)
    """

    l_cutout_size = l_cutout_size or int(np.floor(size / 2))

    B_idx, B = square_geometry(size)

    _, cutout = square_geometry(l_cutout_size)
    B[0, 0, :l_cutout_size, :l_cutout_size] = cutout
    B[0, 0, :l_cutout_size - 1, :l_cutout_size - 1] = np.random.uniform(0, 1)

    B_idx[0, 0, :l_cutout_size, :l_cutout_size] = torch.zeros(1)

    return B_idx, B
