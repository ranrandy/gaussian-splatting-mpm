import os
import sys
from argparse import ArgumentParser
from arguments import *

import numpy
import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.utils.system_utils import searchForMaxIteration

from utils.camera_utils import get_render_poses_spiral


def load_model(model_args):
    gaussians = GaussianModel(sh_degree=3)

    if model_args.loaded_iter == -1:
        loaded_iter = searchForMaxIteration(os.path.join(model_args.model_path, "point_cloud"))
    else:
        model_args.loaded_iter
    print("Loading trained model at iteration {}".format(loaded_iter))

    gaussians.load_ply(os.path.join(model_args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
    return gaussians


def simulate(model_args, mpm_args):
    # Load the Gaussian Cloud
    gaussians = load_model(model_args)

    # Simulation parameters --> will move to mpm_args
    influenced_region_bound = [
        [-0.2, -0.2, -0.2],
        [ 0.2,  0.2,  0.2],
    ]

    # Simulate & Render
    for fid in range(mpm_args.num_frames):
        max_bounded = torch.logical_and(gaussians.get_xyz <= influenced_region_bound[1], axis=1)
        min_bounded = torch.logical_and(gaussians.get_xyz >= influenced_region_bound[0], axis=1)
        sim_gs = torch.logical_and(max_bounded, min_bounded)
        gaussians._xyz[sim_gs] += 0.3
        
        #TODO: Continue tmr

    return



if __name__ == "__main__":
    parser = ArgumentParser(description="Simulation parameters")
    model_args = ModelParams(parser)
    mpm_args = MPMParams(parser)
    args = parser.parse_args(sys.argv[1:])

    simulate(model_args.extract(args), mpm_args.extract(args))

    print("\nSimulation complete.")

