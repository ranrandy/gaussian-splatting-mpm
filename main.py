import os
import sys
import json
from argparse import ArgumentParser
from arguments import *

import math
import numpy as np
import torch

sys.path.append("gaussian_splatting")

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.utils.system_utils import searchForMaxIteration

import imageio
from gaussian_splatting.utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View
from utils.render_utils import TinyCam, to8b



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


def render_frame(viewpoint_camera : TinyCam, pc : GaussianModel, sim_gs_mask, delta_means3D, bg_color, args, scaling_modifier = 1.0):
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
        debug=args.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    means3D = torch.cat([pc.get_xyz[~sim_gs_mask], pc.get_xyz[sim_gs_mask] + delta_means3D], dim=0)
    opacity = torch.cat([pc.get_opacity[~sim_gs_mask], pc.get_opacity[sim_gs_mask]], dim=0)
    scales = torch.cat([pc.get_scaling[~sim_gs_mask], pc.get_scaling[sim_gs_mask]], dim=0)
    rotations = torch.cat([pc.get_rotation[~sim_gs_mask], pc.get_rotation[sim_gs_mask]], dim=0)
    shs = torch.cat([pc.get_features[~sim_gs_mask], pc.get_features[sim_gs_mask]], dim=0)

    rendered_image, _ = rasterizer(
        means3D = means3D,
        means2D = None,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)
    return rendered_image.detach().cpu().numpy().transpose(1, 2, 0)


def simulate(model_args, mpm_args):
    gaussians = load_model(model_args)
    viewpoint_cams = load_cameras(model_args)


    # Simulation settings --> will move to mpm_args
    influenced_region_bound = torch.tensor(np.array([
        [-0.2, -0.2, -0.2], # min bound
        [ 0.2,  0.2,  0.2], # max bound
    ])).cuda()

    max_bounded_gs_mask = (gaussians._xyz <= influenced_region_bound[1]).all(dim=1)
    min_bounded_gs_mask = (gaussians._xyz >= influenced_region_bound[0]).all(dim=1) 
    simulatable_gs_mask = torch.logical_and(max_bounded_gs_mask, min_bounded_gs_mask)
    num_sim_gs = torch.sum(simulatable_gs_mask)

    # Render settings
    viewpoint_camera = viewpoint_cams[model_args.view_cam_idx]
    viewpoint_camera.toCuda()

    bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    os.makedirs(os.path.join(model_args.save_path, "images"), exist_ok=True)


    # Simulate
    rendered_img_seq = []

    ### Render Initial frame
    delta_means3D = torch.tensor([0.0, 0.0, 0.0]).repeat(num_sim_gs, 1).cuda()
    rendered_img = render_frame(viewpoint_camera, gaussians, simulatable_gs_mask, delta_means3D, background, model_args)
    rendered_img_seq.append(rendered_img)
    imageio.imwrite(os.path.join(model_args.save_path, "images", f"{0:04d}.png"), to8b(rendered_img))

    for fid in range(1, mpm_args.num_frames + 1):
        ### MPM Step
        ### TODO: delta_means3D, delta_rotation, ... = MPM_step(gaussians, mpm_args, ...)

        ### Weekly Progress 1: Naive translation & rotation
        delta_means3D[:, 0] = delta_means3D[:, 0] + 0.05
        
        ### Render this frame
        rendered_img = render_frame(viewpoint_camera, gaussians, simulatable_gs_mask, delta_means3D, background, model_args)
        rendered_img_seq.append(rendered_img)
        imageio.imwrite(os.path.join(model_args.save_path, "images", f"{fid:04d}.png"), to8b(rendered_img))

    os.system(f"ffmpeg -framerate 25 -i {args.save_path}/images/%04d.png -c:v libx264 -s {viewpoint_camera.width}x{viewpoint_camera.height} -y -pix_fmt yuv420p {args.save_path}/simulated.mp4")

    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Simulation parameters")
    model_args = ModelParams(parser)
    mpm_args = MPMParams(parser)
    args = parser.parse_args(sys.argv[1:])

    simulate(model_args.extract(args), mpm_args.extract(args))


