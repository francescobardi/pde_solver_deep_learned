import torch

from nnpde.functions import geometries
import nnpde.functions.iterative_methods as im

__author__ = "Francesco Bardi, ADDyourName"
__credits__ = ["Francesco Bardi",
               "ADDyourName"]
__license__ = "GPL"
__maintainer__ = "Francesco Bardi"
__status__ = "Development"


class DirichletProblem:
    """Define a Dirichlet problem instance

    Parameters
    ----------
    B_idx : tensor-like, shape = [1, 1, N, N]
        variable to reset.

    B : tensor-like, shape = [1, 1, N, N], optional

    f : tensor-like, shape = [1, 1, N, N], optional
        variable to reset.

    k : int, optional, default 20
        Number of iterations to use for oTODO.

    k_ground_truth : int, optional, default 1000
        Number of iterations to used to obtain ground truth solution
        with Jacobi method.

    N  : int, optional, default 16
         Used to define the domain as....

    inital_u : tensor-like, shape = [1, 1, N, N], optional
               Default = torch.rand(1, 1, N, N, requires_grad=True)
               Initial solution

    inital_u_jacobi : tensor-like, shape = [1, 1, N, N], optional
               Default = torch.rand(1, 1, N, N, requires_grad=True)

    Returns
    -------
    self : object
        Returns an instance of self.
    """

    def __init__(self,
                 B_idx=None,
                 B=None,
                 f=None,
                 k=20,
                 k_ground_truth=1000,
                 initial_ground_truth=None,
                 initial_u=None,
                 domain_type="Square",
                 N=16):

        # Initialize Geometry and Boundary Conditions
        if B_idx is None:
            self.B_idx, self.B = geometries.square_geometry(N)
        else:
            self.B_idx = B_idx
            self.B = B

        # Initialize f
        if f is None:
            self.f = torch.zeros(1, 1, N, N)
        else:
            self.f = f

        # Initialize parameters to compute ground truth solution
        if initial_ground_truth is None:
            self.initial_ground_truth = torch.rand(1, 1, N, N)
        else:
            self.initial_ground_truth = initial_ground_truth

        self.k_ground_truth = k_ground_truth

        # Compute ground truth solution using Jacobi method
        self.ground_truth = im.jacobi_method(
            self.B_idx, self.B, self.f, self.initial_ground_truth, self.k_ground_truth)

        # Initialize parameters to obtain u
        if initial_u is None:
            self.initial_u = torch.rand(1, 1, N, N, requires_grad=True)
        else:
            self.initial_u = initial_u

        self.k = k

    def compute_solution(self, net):
        """Compute solution using optim method
        """
        self.u = im.H_method(net, self.B_idx, self.B,
                             self.f, self.initial_u, self.k)
        return self.u
