import torch
from arguments import *


def world2grid(means3D : torch.tensor, sim_args : MPMParams):
    pos_min, pos_max = means3D.min(dim=0)[0], means3D.max(dim=0)[0]
    pos_center = (pos_min + pos_max) / 2.0
    scaling_modifier = sim_args.grid_extent / 2.0 / (pos_max - pos_min).max()

    transformed_means3D = (means3D - pos_center) * scaling_modifier + torch.ones(3).cuda() * sim_args.grid_extent / 2.0

    return transformed_means3D, pos_center, scaling_modifier


def grid2world(means3D : torch.tensor, covs : torch.tensor, scaling_modifier, pos_center, sim_args : MPMParams):
    transformed_means3D = (means3D - torch.ones(3).cuda() * sim_args.grid_extent / 2.0) / scaling_modifier + pos_center
    transformed_covs = covs / (scaling_modifier * scaling_modifier)
    return transformed_means3D, transformed_covs.view(-1, 6)