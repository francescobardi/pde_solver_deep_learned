import torch

from nnpde.functions import geometries
import nnpde.functions.iterative_methods as im

class DirichletProblem:
    """A class for setting a problem instance"""

    def __init__(self, B_idx=None, B=None, forcing_term=None, k=20, k_ground_truth = 1000, initial_u=None, domain_type="Square", N=16):

        if B_idx is None:
            self.B_idx, self.B = geometries.square_geometry(N)
        else:
            self.B_idx = B_idx
            self.B = B

        if forcing_term is None:
            self.forcing_term = torch.zeros(1, 1, N, N)
        else:
            self.forcing_term = forcing_term

        if initial_u is None:
            self.initial_u = torch.rand(1, 1, N, N, requires_grad=True)
        else:
            self.initial_u = initial_u

        self.k = k

        # Initial_u_jacobi is different, it must not require grad
        self.initial_u_jacobi = torch.zeros(1, 1, N, N)
        self.k_ground_truth = k_ground_truth
        self.ground_truth = im.jacobi_method(self.B_idx, self.B, self.forcing_term, self.initial_u_jacobi, self.k_ground_truth)
