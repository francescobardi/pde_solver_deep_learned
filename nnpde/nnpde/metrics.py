import torch
import torch.nn.functional as F
import nnpde.functions.iterative_methods as im


def compute_loss(net, problem_instances):
    """ Fucntion to compute the total loss given a set of problem instances"""

    loss = torch.zeros(1)

    for problem_instance in problem_instances:

        ground_truth = problem_instance.ground_truth

        # Compute solution
        u = problem_instance.compute_solution(net)

        loss = loss + F.mse_loss(ground_truth, u)

    return loss
