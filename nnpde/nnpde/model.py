import logging

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from nnpde.functions import helpers
import nnpde.functions.iterative_methods as im
from nnpde import metrics

class JacobyWithConv:
    """A class to obtain the optimal weights"""
    # TODO test with larger batch_size than 1...
    def __init__(self,
                 net=None,
                 batch_size=1,
                 learning_rate=1e-6,
                 max_iters=1000,
                 tol=1e-6,
                 k_range=[1, 20],
                 N=16):

        if net is None:
            self.net = nn.Sequential(
                nn.Conv2d(1, 1, 3, padding=1, bias=False),
                nn.Conv2d(1, 1, 3, padding=1, bias=False),
                nn.Conv2d(1, 1, 3, padding=1, bias=False),
            )
        else:
            self.net = net


        # Set the optimizer, you have to play with lr: if too big nan
        self.optim = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
        ##optim = torch.optim.Adadelta(net.parameters())
        #optim = torch.optim.Adam(net.parameters(), lr=1e-6)
        #optim = torch.optim.ASGD(net.parameters())
        # SGD seems much faster
        self.learning_rate = learning_rate

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.tol = tol
        self.k_range = k_range

        self.T = helpers.get_T(N)
        self.H = None
        self.N = N


    def _optimization_step_(self):
        self.net.zero_grad()
        loss = torch.zeros(1)

        # Sample problem_instances
        # TODO this can select the same problems multiple times! this should
        # not be the case. Split the problems into chunks. see utils.misc.chunks
        problem_idx = np.random.choice(np.arange(len(self.problem_instances)), self.batch_size, replace = 0)

        # TODO for i in batch_size??? this doesn't make sense at all...
        # for batch in batches and each batch has size batch_size
        for i in range(self.batch_size):
            logging.debug(f"training with batch {i}")
            idx = problem_idx[i]
            problem_instance = self.problem_instances[idx]

            B_idx = problem_instance.B_idx
            B = problem_instance.B
            f = problem_instance.forcing_term
            initial_u = problem_instance.initial_u
            k = problem_instance.k
            ground_truth = problem_instance.ground_truth

            # Compute the solution with the updated weights
            u = im.H_method(self.net, B_idx, B, f, initial_u, k)

            # Compute the spectral norm and set the loss to infinity if spectral norm > 1
            # TODO We do this prior to applying the gradient... this doesn't
            # seem right to me. exaclty  IT DOES DON WORK AS YOU CANNOT DIFFERENTIATE a nan
            """
            H = helpers.conv_net_to_matrix(self.net, self.N)
            if helpers.spectral_radius(self.T, H) > 1:
                ex = np.nan_to_num(np.inf)
            else:
                ex = 0
            """
            
            # Define the loss, CHECK if it is correct wrt paper
            loss = loss + F.mse_loss(ground_truth, u) #+ ex #TODO remove comment after properly enforcing constraint 

        # Backpropagation
        loss.backward(retain_graph =  False)

        # SGD step
        self.optim.step()


    def fit(self, problem_instances):
        self.problem_instances = problem_instances
        losses = []
        prev_total_loss = metrics.compute_loss(self.net,
                                              self.problem_instances,
                                              self.N).item()

        # TODO express this as a while-loop with max_iter and tolerance in the "check"
        for _ in range(self.max_iters):
            self._optimization_step_()

            total_loss = metrics.compute_loss(self.net,
                                              self.problem_instances,
                                              self.N)

            # Exit optimization
            if total_loss.item() <= self.tol or np.abs(total_loss.item() - prev_total_loss) < self.tol:
                break


            # Store lossses for visualization
            losses.append(total_loss.item())
            prev_total_loss = total_loss.item()

        self.H = helpers.conv_net_to_matrix(self.net, self.N)
        self.losses = losses

        return self
