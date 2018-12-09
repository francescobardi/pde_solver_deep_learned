# -*- coding: utf-8 -*-
import numpy as np
import torch


def check_dimensions(A, f):
    """Check the dimensions of inputs"""

    Na, Ma = np.shape(A)
    Nf = np.shape(f)[0]

    if Na == Ma and Na == Nf:
        return Na
    else:
        raise ValueError("Dimensions mismatch, please check your inputs.")


def create_square_geometry(size, boundary_value=None):
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
    B = torch.abs(B_idx-1)*(boundary_value or (np.random.rand() + 1)  *3)

    return B, B_idx
