
import os
import sys
import json
import taichi as ti
import torch
import numpy as np
import math
import random
from tqdm import tqdm

from argparse import ArgumentParser

sys.path.append("gaussian_splatting")

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.dataset_readers import getNerfppNorm
from gaussian_splatting.utils.loss_utils import *

from mpm_solver.solver import *
from internel_filling.filling import get_particle_volume

from PIL import Image
from copy import deepcopy
from gaussian_splatting.utils.graphics_utils import focal2fov
from gaussian_splatting.scene.dataset_readers import CameraInfo
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.utils.general_utils import PILtoTorch

import imageio
from utils.render_utils import to8b

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

ti.init(arch=ti.cuda, device_memory_GB=10.0)


data_root = "data_extra/mpm_synthetic"
model_root = "models_extra"

image_width = 512
image_height = 512
image_bg = np.array([1, 1, 1])
image_bg_cuda = torch.tensor(image_bg, dtype=torch.float32, device="cuda")

# SIM_AREA = np.array([
#     [ 0.0, 0.2, 0.0], # XYZ_MIN
#     [ 1.0, 1.2, 1.0], # XYZ_MAX
# ]) # All gaussians are simulatable

grid_extent = 2.0
n_grid = 50

G = np.array([0.0, -9.81, 0.0])

train_num_frames = 20
test_num_frames = 10

total_iters = 300

debug_mode = True


class SystemIndentifier:
    def __init__(self, data_path, model_path, sim_args, args):
        self.sim_args = sim_args
        self.args = args

        self.total_iters = total_iters

        self.load_data_and_cameras(data_path)
        self.load_physics_info(data_path)
        self.load_model(model_path)


    def load_data_and_cameras(self, data_path):
        # Load cameras from json
        with open(os.path.join(data_path, 'camera.json'), 'r') as cam_file:
            cameras = json.load(cam_file)

            cam_infos_all = []
            for frame_id in tqdm(range(train_num_frames+test_num_frames), desc=f"Loading cameras for each frame"):
                cam_infos = []
                for cam_id, camera in enumerate(cameras):

                    intrinsic = np.array(camera['K'])
                    c2w = deepcopy(np.array(camera['c2w']))
                    c2w[:3, 1:3] *= -1
                    w2c = np.linalg.inv(c2w)
                    R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                    T = w2c[:3, 3]

                    FovX = focal2fov(intrinsic[0][0], image_width)
                    FovY = focal2fov(intrinsic[1][1], image_height)

                    cam_name = camera['camera']
                    image_path = os.path.join(data_path, cam_name, f"{frame_id:03}.png")
                    image = Image.open(image_path)
                    im_data = np.array(image.convert("RGBA"))
                    norm_data = im_data / 255.0
                    arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + image_bg * (1 - norm_data[:, :, 3:4])
                    image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

                    cam_infos.append(
                        CameraInfo(uid=cam_id,
                                R=R,
                                T=T,
                                FovY=FovY,
                                FovX=FovX,
                                image=image,
                                image_path=image_path,
                                image_name=f"{cam_name}_{frame_id:03}.png",
                                width=image_width,
                                height=image_height))
                cam_infos_all.append(cam_infos)

        # Get scene extent
        nerf_normalization = getNerfppNorm(cam_infos_all[0])
        self.spatial_lr_scale = nerf_normalization["radius"]

        # Group camera info to gaussian_renderer usable format
        self.cameras_all = []
        for frame_id in range(train_num_frames+test_num_frames):
            cam_infos_fid = cam_infos_all[frame_id]
            camera_list = []
            for id, c in enumerate(cam_infos_fid):
                resized_image_rgb = PILtoTorch(c.image, (image_width, image_height))
                gt_image = resized_image_rgb[:3, ...]
                camera_list.append(Camera(
                            colmap_id=c.uid, R=c.R, T=c.T, 
                            FoVx=c.FovX, FoVy=c.FovY, 
                            image=gt_image, gt_alpha_mask=None,
                            image_name=c.image_name, uid=id, data_device="cuda"))
            self.cameras_all.append(camera_list)

        # Load camera capture times
        self.dt = []
        with open(os.path.join(data_path, "frame.json"), 'r') as file:
            frame_time_steps = json.load(file)
            for fid in range(1, len(frame_time_steps)):
                self.dt.append(frame_time_steps[fid][f"{fid:03d}"] - frame_time_steps[fid-1][f"{fid-1:03d}"])


    def load_physics_info(self, data_path):
        with open(os.path.join(data_path, 'physical.json'), 'r') as physical_file:
            self.physics_info = json.load(physical_file)


    def load_model(self, model_path):
        self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.load_ply(os.path.join(model_path, "static_gaussians", "point_cloud.ply"))

        self.n_particles = self.gaussians.get_xyz.shape[0]

        with open(os.path.join(model_path, "init_velocity.json"), 'r') as file:
            self.init_v = torch.tensor(json.load(file)).repeat(self.n_particles, 1)
        

    def train(self):
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(self.args.output_path)
        else:
            print("Tensorboard not available: not logging progress")

        self.training_setup()

        optimized_E, optimized_nu = None, None

        for iteration in tqdm(range(1, self.total_iters+1), desc="Training Physical Parameters"):
            
            sim_means3D = self.gaussians.get_xyz
            sim_covs = self.gaussians.get_covariance()
            init_velocities = torch.zeros(3).float().repeat(self.n_particles, 1)

            transformed_sim_means3D = self.world2grid(sim_means3D)
            transformed_sim_covs = sim_covs * (self.scaling_modifier * self.scaling_modifier)
            
            sim_volumes = get_particle_volume(transformed_sim_means3D.detach(), self.sim_args)

            sim_args.E = optimized_E if optimized_E is not None else sim_args.E
            sim_args.nu = optimized_nu if optimized_E is not None else sim_args.nu

            mpm_solver = MPM_Simulator(transformed_sim_means3D, transformed_sim_covs, sim_volumes, sim_args, init_v=init_velocities)
            mpm_solver.set_bc_ground_only()
        
            for fid in range(train_num_frames): 
                # A random camera at frame {fid}
                cam_id = random.randint(1, len(self.cameras_all[fid])) - 1
                viewpoint_cam = self.cameras_all[fid][cam_id]
                gt_image = viewpoint_cam.original_image
                
                if fid == 0: # Optimize gaussians
                    rendered_image = self.render(viewpoint_cam, self.gaussians, sim_means3D, sim_covs)
                    loss = 0.8 * l1_loss(rendered_image, gt_image) + 0.2 * ssim(rendered_image, gt_image)
                    loss.backward()
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)
                else: # Optimize physical parameters
                    # Forward Start
                    for s in range(30):
                        mpm_solver.p2g2p(0.03 / 30, s)

                    mpm_solver.postprocess_forward()
                    # Forward End

                    # Compute image-space loss
                    mpm_sim_means3D = mpm_solver.mpm_state.particle_xyz.to_torch()[30].cuda().requires_grad_(True)
                    mpm_sim_covs = mpm_solver.mpm_state.particle_cov.to_torch().cuda().requires_grad_(True)
                    sim_means3D, sim_covs = self.grid2world(mpm_sim_means3D, mpm_sim_covs, sim_args)
                    
                    rendered_image = self.render(viewpoint_cam, self.gaussians, sim_means3D, sim_covs)
                    loss = 0.8 * l1_loss(rendered_image, gt_image) + 0.2 * ssim(rendered_image, gt_image)
                    loss.backward()
                    mpm_solver.clear_grads()

                    # gi = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                    # imageio.imwrite(os.path.join("debug", f"gt_{fid:03d}_{cam_id:03d}.png"), to8b(gi))
                    
                    # Transfer torch gradients to taichi
                    mpm_solver.mpm_state.set_grads(
                        mpm_sim_means3D.grad.cpu().numpy().astype(np.float32), 
                        mpm_sim_covs.grad.cpu().numpy().astype(np.float32))

                    # Backward Start
                    mpm_solver.postprocess_backward()

                    for s in reversed(range(30)):
                        mpm_solver.p2g2p_backward(0.03 / 30, s)
                    # Backward End
                    
                    # Update physical parameters
                    mpm_solver.learn()

                    # Prepare for next frame/iteration
                    mpm_solver.mpm_state.cycle_init()
                    # compute_mu_lam_from_E_nu(self.n_particles, mpm_solver.mpm_model.logE, mpm_solver.mpm_model.y, mpm_solver.mpm_model.mu, mpm_solver.mpm_model.lam)

                    # print(mpm_solver.mpm_model.logE.to_torch().mean(), mpm_solver.mpm_model.y.to_torch().mean())
                    # print(10 ** mpm_solver.mpm_model.logE.to_torch().mean(), 0.49 / (1.0 + torch.exp(-mpm_solver.mpm_model.y.to_torch().mean())))
                    # print()

                optimized_E = 10 ** mpm_solver.mpm_model.logE.to_torch().mean().item()
                optimized_nu = 0.49 / (1.0 + torch.exp(-mpm_solver.mpm_model.y.to_torch().mean())).item()

                print(optimized_E, optimized_nu)

                if tb_writer and fid > 0:
                    tb_writer.add_scalar('loss_total', loss.item(), iteration * 19 + fid)
                    tb_writer.add_scalar('optimized_E', optimized_E, iteration * 19 + fid)
                    tb_writer.add_scalar('optimized_nu', optimized_nu, iteration * 19 + fid)


    def render(self, viewpoint_camera : Camera, pc : GaussianModel, sim_means3D : torch.tensor, sim_covs : torch.tensor):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=image_bg_cuda,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=debug_mode
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = sim_means3D
        covs = sim_covs

        means2D = screenspace_points
        opacities = pc.get_opacity
        shs = pc.get_features

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacities,
            scales = None,
            rotations = None,
            cov3D_precomp = covs)
        return rendered_image


    def training_setup(self):
        l = [
            {'params': [self.gaussians._xyz], 'lr': 0.0000016 * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self.gaussians._features_dc], 'lr': 0.0025, "name": "f_dc"},
            {'params': [self.gaussians._features_rest], 'lr': 0.0025 / 20.0, "name": "f_rest"},
            {'params': [self.gaussians._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self.gaussians._scaling], 'lr': 0.005, "name": "scaling"},
        ]
        self.gaussians.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


    def world2grid(self, means3D):
        pos_min, pos_max = means3D.min(dim=0)[0] - 0.3, means3D.max(dim=0)[0] + 0.3
        self.pos_center = ((pos_min + pos_max) / 2.0).detach()
        self.scaling_modifier = grid_extent / 2.0 / (pos_max - pos_min).max().detach()

        transformed_means3D = (means3D - self.pos_center) * self.scaling_modifier + torch.ones(3).cuda() * grid_extent / 2.0
        return transformed_means3D


    def grid2world(self, means3D : torch.tensor, covs : torch.tensor, sim_args : MPMParams):
        transformed_means3D = (means3D - torch.ones(3).cuda() * sim_args.grid_extent / 2.0) / self.scaling_modifier + self.pos_center
        transformed_covs = covs / (self.scaling_modifier * self.scaling_modifier)
        return transformed_means3D, transformed_covs.view(-1, 6)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--scene', type=str, default="torus")
    parser.add_argument('--output_path', type=str, default="outputs_extra/torus_debug")
    sim_args = MPMParams(parser)
    args = parser.parse_args()

    data_path = os.path.join(data_root, args.scene)
    model_path = os.path.join(model_root, args.scene)
    sim_args.fitting = True

    os.makedirs(args.output_path, exist_ok=True)

    si = SystemIndentifier(data_path, model_path, sim_args, args)
    si.train()

