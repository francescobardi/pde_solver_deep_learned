import numpy as np
import torch


def square_geometry(size, boundary_value=None):
    """Defines a square geometry of given size (as a single positive integer)
    If boundary_value is None a random value will between [1, 4) will be chosen

    Returns:
        (geometry, boundary_index)
    """

    # Define geometry 1.0 inner points 0.0 elsewhre
    B_idx = torch.ones(1, 1, size ,size )
    B_idx[0,0,0,:] = torch.zeros(size)
    B_idx[0,0,size-1,:] = torch.zeros(size)
    B_idx[0,0,:,0] = torch.zeros(size)
    B_idx[0,0,:,size -1] = torch.zeros(size)

    # Define boundary values
    B = torch.abs(B_idx-1)*(boundary_value or (np.random.normal(0,1)))

    return B_idx, B


def l_shaped_geometry(size, l_cutout_size=None, boundary_value=None):
    """Defines a L-shaped geometry of given size (as a single positive integer), and
    l_cutout_size (think of creating the L-shape as cutting out a smaller square piece)
    If boundary_value is None a random value will between [1, 4) will be chosen.

    l_cutout_size is by default size/2

    Returns:
        (geometry, boundary_index)
    """
    # size = 50 (Should we initialize size to 50?)
    l_cutout_size = l_cutout_size or int(np.floor(size/2))

    # Define geometry 1.0 inner points 0.0 elsewhre
    B_idx = torch.ones(1, 1, size, size)
    B_idx[0,0,0:l_cutout_size,0:l_cutout_size] = \
            torch.zeros([l_cutout_size, l_cutout_size])
    B_idx[0,0,size-1,:] = torch.zeros(size)
    B_idx[0,0,:,0] = torch.zeros(size)
    B_idx[0,0,:,size-1] = torch.zeros(size)

    # Define boundary values
    B = torch.abs(B_idx-1)*(boundary_value or (np.random.normal(0,1)))

    return B_idx, B
