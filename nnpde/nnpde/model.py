import logging
import copy
from functools import reduce

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from nnpde import helpers
import nnpde.iterative_methods as im
from nnpde import metrics
from nnpde.utils.misc import chunks, set_seed


class _ConvNet_(nn.Module):
    def __init__(self, nb_layers):
        super(_ConvNet_, self).__init__()

        self.convLayers = nn.ModuleList([nn.Conv2d(1, 1, 3, padding=1, bias=False)
                                         for _ in range(nb_layers)])
        """
        initial_weights = torch.rand(1,1,3,3)
        #initial_weights[0,0,0,1] = 0.25
        #initial_weights[0,0,2,1] = 0.25
        #initial_weights[0,0,1,0] = 0.25
        #initial_weights[0,0,1,2] = 0.25

        for name, param in self.convLayers.named_parameters():
            param = nn.Parameter(initial_weights)
            print(name, param)
        """

    def forward(self, x, boundary):
        return reduce(lambda acc, el: el(acc) * boundary, self.convLayers, x)


class JacobyWithConv:
    """A class to obtain the optimal weights"""

    def __init__(self,
                 net=None,
                 batch_size=1,
                 learning_rate=1e-6,
                 max_epochs=1000,
                 nb_layers=3,
                 tol=1e-4,
                 stable_count=5,
                 N=16,
                 optimizer='SGD',
                 check_spectral_radius=False,
                 random_seed=None):

        if random_seed is not None:
            set_seed(random_seed)

        if net is None:
            self.nb_layers = nb_layers
            self.net = _ConvNet_(nb_layers=self.nb_layers)
        else:
            self.net = net

        self.learning_rate = learning_rate
        if optimizer == 'Adadelta':
            logging.info(f"Using optimizer {optimizer}")
            self.optim = torch.optim.Adadelta(self.net.parameters())
        else:
            self.optim = torch.optim.SGD(
                self.net.parameters(), lr=learning_rate)

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tol = tol
        self.stable_count = stable_count

        self.T = helpers.build_T(N)
        self.H = None
        self.N = N

    def _optimization_step_(self, problem_instances):

        shuffled_problem_instances = np.random.permutation(problem_instances)

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
        losses = []
        prev_total_loss = metrics.compute_loss(
            self.net, problem_instances).item()
        convergence_counter = 0
        logging.info(
            f"Training with max_epochs: {self.max_epochs}, tol: {self.tol}. Initial loss is {prev_total_loss}")

        # Optimization loop
        for n_epoch in range(self.max_epochs):

            # Update weights
            self._optimization_step_(problem_instances)

            # Compute total loss
            total_loss = metrics.compute_loss(
                self.net, problem_instances).item()

            # Store lossses for visualization
            losses.append(total_loss)
            prev_total_loss = total_loss

            # Check convergence
            if np.abs(total_loss - prev_total_loss) < self.tol:
                convergence_counter += 1
                if convergence_counter > self.stable_count:
                    break
            else:
                convergence_counter = 0


            # Display information every 100 iterations
            if n_epoch % 100 == 0:
                logging.info(
                    f"iter {n_epoch} with total loss {prev_total_loss}")

        #self.H = helpers.conv_net_to_matrix(self.net, self.N)
        self.losses = losses
        logging.info(
            f"Convergence reached in {n_epoch} epochs with total loss {total_loss}")
        return self

        return self
