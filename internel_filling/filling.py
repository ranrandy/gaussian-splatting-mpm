'''
    Reference: https://github.com/XPandora/PhysGaussian/blob/main/particle_filling/filling.py
'''

import torch
import taichi as ti
from arguments import *

# Initialize Particle Volume

@ti.kernel
def assign_particle_to_grid(pos: ti.template(), grid: ti.template(), grid_dx: float):
    for pi in pos:
        p = pos[pi]
        cell_index = ti.floor(p / grid_dx).cast(int)
        ti.atomic_add(grid[cell_index], 1)


@ti.kernel
def compute_particle_volume(pos: ti.template(), grid: ti.template(), particle_vol: ti.template(), grid_dx: float):
    for pi in pos:
        p = pos[pi]
        cell_index = ti.floor(p / grid_dx).cast(int)
        particle_vol[pi] = (grid_dx ** 3) / grid[cell_index]


def get_particle_volume(pos : torch.tensor, args : MPMParams, uniform: bool = False):
    num_pts = pos.shape[0]
    ti_pos = ti.Vector.field(n=3, dtype=ti.f32, shape=num_pts)
    ti_pos.from_torch(pos)

    grid = ti.field(dtype=int, shape=(args.n_grid, args.n_grid, args.n_grid))
    grid_dx = args.grid_extent / args.n_grid
    particle_vol = ti.field(dtype=ti.f32, shape=num_pts)

    assign_particle_to_grid(ti_pos, grid, grid_dx)
    compute_particle_volume(ti_pos, grid, particle_vol, grid_dx)

    if uniform: # Same volume for all particles
        return torch.mean(particle_vol.to_torch()).repeat(num_pts)
    else:
        return particle_vol.to_torch()
