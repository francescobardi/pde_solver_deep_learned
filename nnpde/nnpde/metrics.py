import torch
import torch.nn.functional as F
import nnpde.functions.iterative_methods as im

def compute_loss(net, problem_instances_list,N):
    """ Fucntion to compute the total loss given a set of problem instances"""

    nb_problem_instances = len(problem_instances_list)
    loss = torch.zeros(1, requires_grad = False)
    u = torch.zeros(1, 1, N, N, nb_problem_instances)

    for problem_instance in problem_instances_list:

        B_idx = problem_instance.B_idx
        B = problem_instance.B
        f = problem_instance.forcing_term
        initial_u = problem_instance.initial_u
        k = problem_instance.k
        ground_truth = problem_instance.ground_truth

        u = im.H_method(net, B_idx, B, f, initial_u, k)
        loss = loss + F.mse_loss(ground_truth, u)

    return loss
