import logging
import copy
from functools import reduce

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from nnpde.functions import helpers
import nnpde.functions.iterative_methods as im
from nnpde import metrics
from nnpde.utils.misc import chunks


class _ConvNet_(nn.Module):
    def __init__(self, nb_layers):
        super(_ConvNet_, self).__init__()

        self.convLayers = nn.ModuleList([nn.Conv2d(1, 1, 3, padding=1, bias=False)
                                         for _ in range(nb_layers)])

    def forward(self, x, boundary):
        return reduce(lambda acc, el: el(acc) * boundary, self.convLayers, x)


class JacobyWithConv:
    """A class to obtain the optimal weights"""

    def __init__(self,
                 net=None,
                 batch_size=1,
                 learning_rate=1e-6,
                 max_iters=1000,
                 nb_layers=3,
                 tol=1e-4,
                 stable_count=5,
                 k_range=[1, 20],
                 N=16,
                 optimizer='Adadelta',
                 check_spectral_radius=False):

        if net is None:
            self.net = _ConvNet_(nb_layers=nb_layers)
        else:
            self.net = net

        self.learning_rate = learning_rate
        if optimizer == 'Adadelta':
            self.optim = torch.optim.Adadelta(self.net.parameters())
        else:
            self.optim = torch.optim.SGD(
                self.net.parameters(), lr=learning_rate)

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.tol = tol
        self.stable_count = stable_count

        self.T = helpers.build_T(N)
        self.H = None
        self.N = N

    def _optimization_step_(self):

        shuffled_problem_instances = np.random.permutation(
            self.problem_instances)

        for problem_chunk in chunks(shuffled_problem_instances, self.batch_size):
            self.net.zero_grad()

            # Compute loss using only batch
            loss = metrics.compute_loss(self.net, problem_chunk)

            # Backpropagate loss function
            loss.backward(retain_graph=True)

            # Update weights
            self.optim.step()

    def fit(self, problem_instances):
        """
             Returns
             -------
             self : object
                 Returns the instance (self).
        """
        # Initialization
        self.problem_instances = problem_instances
        losses = []
        prev_total_loss = np.inf
        count = 0

        # Optimization loop
        for n_epoch in range(self.max_iters):

            # Update weights
            self._optimization_step_()

            # Compute total loss
            total_loss = metrics.compute_loss(
                self.net, self.problem_instances).item()

            # Check convergence
            if np.abs(total_loss - prev_total_loss) < self.tol:
                count += 1
                if count > self.stable_count:
                    losses.append(total_loss)
                    self.losses = losses
                    return self
            else:
                # Reset counter
                count = 0

            # Store lossses for visualization
            losses.append(total_loss)
            prev_total_loss = total_loss

            # Display information every 100 iterations
            if n_epoch % 100 == 0:
                logging.info(
                    f"iter {n_epoch} with total loss {prev_total_loss}")

        #self.H = helpers.conv_net_to_matrix(self.net, self.N)
        self.losses = losses

        return self
