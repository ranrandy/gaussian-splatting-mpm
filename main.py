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
from gaussian_splatting.utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2
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

    # gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))

    gaussians.load_multiple_plys([os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"), os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud2.ply")])
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

def modify_cam(viewpoint_camera : TinyCam, center_view_world_space, observant_coordinates):
    position, R = get_camera_position_and_rotation(
                    100,
                    60,
                    40.75,
                    center_view_world_space,
                    observant_coordinates,
                )
    tmp = np.zeros((4, 4))
    tmp[:3, :3] = R.tolist()
    tmp[:3, 3] = position.tolist()
    tmp[3, 3] = 1
    C2W = np.linalg.inv(tmp)
    R = C2W[:3, :3].transpose()
    T = C2W[:3, 3]

    proj_mat = getProjectionMatrix(znear=0.01, zfar=100, fovX=viewpoint_camera.FovX, fovY=viewpoint_camera.FovY).transpose(0, 1).cuda()
    viewpoint_camera.view_mat = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
    viewpoint_camera.view_mat = viewpoint_camera.view_mat.to(torch.float32)
    viewpoint_camera.cam_center = T.astype(np.float32)
    viewpoint_camera.full_proj_mat = (viewpoint_camera.view_mat.unsqueeze(0).bmm(proj_mat.unsqueeze(0))).squeeze(0)
    viewpoint_camera.full_proj_mat = viewpoint_camera.full_proj_mat.to(torch.float32)
    return viewpoint_camera

def render_frame(viewpoint_camera : TinyCam, pc : GaussianModel, sim_gs_mask, sim_means3D, sim_covs, bg_color, args, rotation_matrices, pos_center, scaling_modifier = 1.0):
    '''
        Rasterize the Gaussian cloud
    '''

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FovX * 0.5)
    tanfovy = math.tan(viewpoint_camera.FovY * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=1060,
        image_width=1888,
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
    # means3D = torch.cat([pc.get_xyz[~sim_gs_mask], sim_means3D], dim=0)
    # opacities = torch.cat([pc.get_opacity[~sim_gs_mask], pc.get_opacity[sim_gs_mask]], dim=0)
    # shs = torch.cat([pc.get_features[~sim_gs_mask], pc.get_features[sim_gs_mask]], dim=0)
    # covs = torch.cat([pc.get_covariance()[~sim_gs_mask], sim_covs], dim=0)

    means3D = sim_means3D
    opacities = pc.get_opacity[sim_gs_mask]
    shs = pc.get_features[sim_gs_mask]
    covs = sim_covs

    means3D = apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(means3D), scaling_modifier, pos_center
        ),
        rotation_matrices,
    )
    covs = covs / (scaling_modifier * scaling_modifier)
    covs = apply_inverse_cov_rotations(covs, rotation_matrices)

    rendered_image, _ = rasterizer(
        means3D = means3D,
        means2D = None,
        shs = shs,
        colors_precomp = None,
        opacities = opacities,
        scales = None,
        rotations = None,
        cov3D_precomp = covs)
    return rendered_image.detach().cpu().numpy().transpose(1, 2, 0)

def save_frame(frame, save_path, fid, save_seq):
    save_seq.append(frame)
    imageio.imwrite(os.path.join(save_path, f"{fid:04d}.png"), to8b(frame))


def simulate(model_args : ModelParams, sim_args : MPMParams, render_args : RenderParams):

    # ------------------------------------------ Settings ------------------------------------------
    
    # Model settings
    gaussians = load_model(model_args)
    viewpoint_cams = load_cameras(model_args)

    # gaussians.drop_low_opacity(0.02)
    # gaussians.drop_empty_gaussians(sim_args.mask)

    rotation_matrices = generate_rotation_matrices([torch.tensor(0.0)], [torch.tensor(0.0)])
    rotated_gaussians = apply_rotations(gaussians.get_xyz, rotation_matrices)

    # Simulation settings
    influenced_region_bound = torch.tensor(np.array(sim_args.sim_area)).cuda()

    max_bounded_gs_mask = (rotated_gaussians <= influenced_region_bound[1]).all(dim=1)
    min_bounded_gs_mask = (rotated_gaussians >= influenced_region_bound[0]).all(dim=1) 
    simulatable_gs_mask = torch.logical_and(max_bounded_gs_mask, min_bounded_gs_mask)
    num_sim_gs = torch.sum(simulatable_gs_mask)

    # simulatable_gs_mask = torch.zeros(rotated_gaussians.shape[0], dtype=torch.bool).cuda()

    # for bounds in sim_args.sim_area:
    #     bounds = torch.tensor(np.array(bounds)).cuda()
    #     max_bounded_gs_mask = (rotated_gaussians <= bounds[1]).all(dim=1)
    #     min_bounded_gs_mask = (rotated_gaussians >= bounds[0]).all(dim=1)
    #     # Update the simulatable mask to include any Gaussians within current bounds
    #     simulatable_gs_mask |= torch.logical_and(max_bounded_gs_mask, min_bounded_gs_mask)

    # num_sim_gs = torch.sum(simulatable_gs_mask)

    print(f"Number of simulatable Gaussians: {num_sim_gs}")

    bg_color = [1, 1, 1] if render_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    save_images_folder = os.path.join(render_args.output_path, "images")
    os.makedirs(save_images_folder, exist_ok=True)
    
    rendered_img_seq = []

    particle_position_tensor_to_ply(rotated_gaussians, "/home/fastblob/Documents/PhysGaussians/gaussian-splatting-mpm/output_ply/rotated_particles.ply")

    # ------------------------------------------ Initialization ------------------------------------------

    sim_means3D = rotated_gaussians[simulatable_gs_mask].detach()
    sim_covs = gaussians.get_covariance()[simulatable_gs_mask].detach()
    
    sim_covs = apply_cov_rotations(sim_covs, rotation_matrices)

    transformed_sim_means3D, pos_center, scaling_modifier = world2grid(sim_means3D, sim_args)
    transformed_sim_covs = sim_covs * (scaling_modifier * scaling_modifier)

    mpm_space_viewpoint_center = (
        torch.tensor([0.5, 1.4, -0.1]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor([0, 0, 1])
        .reshape((1, 3))
        .cuda()
    )    
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scaling_modifier,
        pos_center,
    )
    print(viewpoint_center_worldspace)
    print(observant_coordinates)

    # Render settings
    viewpoint_camera = viewpoint_cams[0]
    viewpoint_camera = modify_cam(viewpoint_camera, viewpoint_center_worldspace, observant_coordinates)
    viewpoint_camera.toCuda()
    
    sim_volumes = get_particle_volume(transformed_sim_means3D, sim_args)

    mpm_solver = MPM_Simulator(transformed_sim_means3D, transformed_sim_covs, sim_volumes, sim_args)
    mpm_solver.set_boundary_conditions(sim_args.boundary_conditions, sim_args)


    mpm_solver.add_surface_collider((0.0, 0.0, 0.67), (0.0, 0.0, 1.0))
    # mpm_solver.add_surface_collider((0.0, 0.5, 0.0,), (0.0, 1.0, 0.0))
    # mpm_solver.add_surface_collider((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    # mpm_solver.add_surface_collider((2.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    # mpm_solver.add_surface_collider((0.0, 2.0, 0.0), (0.0, -1.0, 0.0))
    # mpm_solver.add_surface_collider((0.0, 0.0, 2.0), (0.0, 0.0, -1.0))


    # # Test for adding a surface collider
    # point = [0.0, 0.0, -0.8]
    # normal = [0.0, 0.0, 1.0]
    # mpm_solver.add_surface_collider(point, normal)

    # ------------------------------------------ Simulate, Render, and Save ------------------------------------------

    # Render initial frame
    rendered_img = render_frame(viewpoint_camera, gaussians, simulatable_gs_mask, sim_means3D, sim_covs, background, model_args, rotation_matrices, pos_center)
    save_frame(rendered_img, save_images_folder, 0, rendered_img_seq)

    for fid in tqdm(range(1, render_args.num_frames + 1)):
        # MPM Steps
        for _ in range(sim_args.steps_per_frame):
            mpm_solver.p2g2p(sim_args.substep_dt)

        mpm_solver.postprocess()

        sim_means3D, sim_covs = grid2world(
            mpm_solver.mpm_state.particle_xyz.to_torch().cuda(), 
            mpm_solver.mpm_state.particle_cov.to_torch().cuda(), 
            scaling_modifier, pos_center, sim_args)
        
        # Render current frame
        rendered_img = render_frame(viewpoint_camera, gaussians, simulatable_gs_mask, sim_means3D, sim_covs, background, model_args, rotation_matrices, pos_center)
        save_frame(rendered_img, save_images_folder, fid, rendered_img_seq)

    os.system(f"ffmpeg -framerate 25 -i {save_images_folder}/%04d.png -c:v libx264 -s {viewpoint_camera.width}x{viewpoint_camera.height-1} -y -pix_fmt yuv420p {render_args.output_path}/simulated.mp4")

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


