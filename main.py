import os
import sys
import json
from argparse import ArgumentParser
from arguments import *
from mpm_solver.solver import *
from internel_filling.filling import *

import math
import numpy as np
import torch

sys.path.append("gaussian_splatting")

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.utils.system_utils import searchForMaxIteration

from tqdm import tqdm
import imageio
from gaussian_splatting.utils.graphics_utils import focal2fov, getProjectionMatrix
from utils.render_utils import TinyCam, to8b
from utils.transform_utils import *

ti.init(arch=ti.cuda)


def load_model(args):
    '''
        Load the optimized Gaussian cloud 
    '''

    gaussians = GaussianModel(sh_degree=3)

    if args.loaded_iter == -1:
        loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    else:
        loaded_iter = args.loaded_iter
    print("Loading trained model at iteration {}".format(loaded_iter))

    gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
    return gaussians

def load_cameras(args):
    '''
        Load the saved camera settings
    '''

    cameras = []

    with open(os.path.join(args.model_path, "cameras.json")) as f:

        cam_infos = json.load(f)
        
        for cam_info in cam_infos:

            width, height = cam_info["width"], cam_info["height"]
            FovX, FovY = focal2fov(cam_info["fx"], width), focal2fov(cam_info["fy"], height)
            
            cam_center = np.array(cam_info["position"]).astype(np.float32)

            C2W = np.zeros((4, 4))
            C2W[:3, :3] = np.array(cam_info["rotation"])
            C2W[:3, 3] = cam_center
            C2W[3, 3] = 1.0
            view_mat = np.linalg.inv(C2W).transpose().astype(np.float32)

            proj_mat = getProjectionMatrix(znear=0.01, zfar=100, fovX=FovX, fovY=FovY).numpy().transpose().astype(np.float32)
            full_proj_mat = view_mat @ proj_mat

            cameras.append(TinyCam(width=width, height=height,
                                        FovX=FovX, FovY=FovY,
                                        cam_center=cam_center,
                                        view_mat=view_mat, 
                                        full_proj_mat=full_proj_mat))
    return cameras


def render_frame(viewpoint_camera : TinyCam, pc : GaussianModel, sim_gs_mask, sim_means3D, bg_color, args, scaling_modifier = 1.0):
    '''
        Rasterize the Gaussian cloud
    '''

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FovX * 0.5)
    tanfovy = math.tan(viewpoint_camera.FovY * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.view_mat,
        projmatrix=viewpoint_camera.full_proj_mat,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.cam_center,
        prefiltered=False,
        debug=args.debug)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    means3D = torch.cat([pc.get_xyz[~sim_gs_mask], sim_means3D], dim=0)
    opacities = torch.cat([pc.get_opacity[~sim_gs_mask], pc.get_opacity[sim_gs_mask]], dim=0)
    scales = torch.cat([pc.get_scaling[~sim_gs_mask], pc.get_scaling[sim_gs_mask]], dim=0)
    rotations = torch.cat([pc.get_rotation[~sim_gs_mask], pc.get_rotation[sim_gs_mask]], dim=0)
    shs = torch.cat([pc.get_features[~sim_gs_mask], pc.get_features[sim_gs_mask]], dim=0)

    rendered_image, _ = rasterizer(
        means3D = means3D,
        means2D = None,
        shs = shs,
        colors_precomp = None,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)
    return rendered_image.detach().cpu().numpy().transpose(1, 2, 0)

def save_frame(frame, save_path, fid, save_seq):
    save_seq.append(frame)
    imageio.imwrite(os.path.join(save_path, f"{fid:04d}.png"), to8b(frame))


def simulate(model_args : ModelParams, sim_args : MPMParams, render_args : RenderParams):

    # ------------------------------------------ Settings ------------------------------------------
    
    # Model settings
    gaussians = load_model(model_args)
    viewpoint_cams = load_cameras(model_args)

    # Simulation settings
    influenced_region_bound = torch.tensor(np.array(sim_args.sim_area)).cuda()

    max_bounded_gs_mask = (gaussians.get_xyz <= influenced_region_bound[1]).all(dim=1)
    min_bounded_gs_mask = (gaussians.get_xyz >= influenced_region_bound[0]).all(dim=1) 
    simulatable_gs_mask = torch.logical_and(max_bounded_gs_mask, min_bounded_gs_mask)
    num_sim_gs = torch.sum(simulatable_gs_mask)

    print(f"Number of simulatable Gaussians: {num_sim_gs}")

    # Render settings
    viewpoint_camera = viewpoint_cams[render_args.view_cam_idx]
    viewpoint_camera.toCuda()

    bg_color = [1, 1, 1] if render_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    save_images_folder = os.path.join(render_args.output_path, "images")
    os.makedirs(save_images_folder, exist_ok=True)
    
    rendered_img_seq = []

    # ------------------------------------------ Initialization ------------------------------------------

    sim_means3D = gaussians.get_xyz[simulatable_gs_mask].detach().clone().cpu()
    sim_scales = gaussians.get_scaling[simulatable_gs_mask].detach().clone().cpu()
    sim_rotations = gaussians.get_rotation[simulatable_gs_mask].detach().clone().cpu()

    transformed_sim_means3D, transformed_sim_covs, pos_center, scaling_modifier = world2grid(sim_means3D, sim_scales, sim_rotations, sim_args, gaussians.covariance_activation)
    sim_volumes = get_particle_volume(transformed_sim_means3D, sim_args)

    mpm_solver = MPM_Simulator(transformed_sim_means3D, transformed_sim_covs, sim_volumes, sim_args)
    mpm_solver.set_boundary_conditions(sim_args.boundary_conditions, sim_args)

    # # Test for adding a surface collider
    # point = [0.0, 0.0, -0.8]
    # normal = [0.0, 0.0, 1.0]
    # mpm_solver.add_surface_collider(point, normal)

    # ------------------------------------------ Simulate, Render, and Save ------------------------------------------

    # Render initial frame
    rendered_img = render_frame(viewpoint_camera, gaussians, simulatable_gs_mask, sim_means3D.cuda(), background, model_args)
    save_frame(rendered_img, save_images_folder, 0, rendered_img_seq)

    for fid in tqdm(range(1, render_args.num_frames + 1)):
        # MPM Steps
        for ffid in range(sim_args.steps_per_frame):
            mpm_solver.p2g2p(sim_args.substep_dt)

            sim_means3D = grid2world(mpm_solver.mpm_state.particle_xyz.to_torch().cpu(), mpm_solver.mpm_state.particle_cov.to_torch().cpu(), scaling_modifier, pos_center, sim_args)
        
            # Render current frame
            rendered_img = render_frame(viewpoint_camera, gaussians, simulatable_gs_mask, sim_means3D.cuda(), background, model_args)
            save_frame(rendered_img, save_images_folder, fid * ffid, rendered_img_seq)
            # print(mpm_solver.mpm_state.particle_xyz.to_torch().min(dim=0)[0], mpm_solver.mpm_state.particle_xyz.to_torch().max(dim=0)[0])

    os.system(f"ffmpeg -framerate 25 -i {save_images_folder}/%04d.png -c:v libx264 -s {viewpoint_camera.width}x{viewpoint_camera.height} -y -pix_fmt yuv420p {render_args.output_path}/simulated.mp4")

    print("Done.")


if __name__ == "__main__":
    # Load the config
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument('--config_path', type=str, required=True)
    config_args, remaining_argv = config_parser.parse_known_args()
    with open(config_args.config_path, 'r') as f:
        config = json.load(f)

    # Load the other argments
    parser = ArgumentParser(description="Simulation parameters")
    model_args = ModelParams(parser, config["model"])
    sim_args = MPMParams(parser, config["mpm"])
    render_args = RenderParams(parser, config["render"])
    args = parser.parse_args(remaining_argv)

    simulate(model_args.extract(args), sim_args.extract(args), render_args.extract(args))


