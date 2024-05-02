import torch
import math
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

def get_mat_from_upper(upper_mat):
    upper_mat = upper_mat.reshape(-1, 6)
    mat = torch.zeros((upper_mat.shape[0], 9), device="cuda")
    mat[:, :3] = upper_mat[:, :3]
    mat[:, 3] = upper_mat[:, 1]
    mat[:, 4] = upper_mat[:, 3]
    mat[:, 5] = upper_mat[:, 4]
    mat[:, 6] = upper_mat[:, 2]
    mat[:, 7] = upper_mat[:, 4]
    mat[:, 8] = upper_mat[:, 5]

    return mat.view(-1, 3, 3)

def get_upper_from_mat(mat):
    mat = mat.view(-1, 9)
    upper_mat = torch.zeros((mat.shape[0], 6), device="cuda")
    upper_mat[:, :3] = mat[:, :3]
    upper_mat[:, 3] = mat[:, 4]
    upper_mat[:, 4] = mat[:, 5]
    upper_mat[:, 5] = mat[:, 8]

    return upper_mat

def rotate_covs(upper_cov_tensor, mats):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(mats)):
        cov_tensor = torch.matmul(mats[i], torch.matmul(cov_tensor, mats[i].T))
    return get_upper_from_mat(cov_tensor)

def rotate(points, mats):
    for i in range(len(mats)):
        points = torch.mm(points, mats[i].T)
    return points

# # Generate a rotation matrix for degree (specified in units of degrees) and an axis (0, 1, or 2)
def get_rotation_matrix(degree, axis):
    cos_theta = torch.cos(degree / 180.0 * torch.pi)
    sin_theta = torch.sin(degree / 180.0 * torch.pi)
    if axis == 0:
        r = torch.tensor(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
        )
    elif axis == 1:
        r = torch.tensor(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
    else:
        r = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
    return r.cuda()

def get_rotation_matrices(degrees):
    assert len(degrees) == 3
    mats = []
    for i in range(2):
        mats.append(get_rotation_matrix(degrees[i], i))
    return mats