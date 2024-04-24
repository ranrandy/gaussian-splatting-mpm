import torch
import os
import numpy as np
import taichi as ti
import mcubes

@ti.kernel
def assign_particle_to_grid(pos: ti.template(), grid: ti.template(), grid_dx: float):
    # for pi in range(pos.shape[0]):
    #     p = pos[pi]
    #     i = ti.floor(p[0] / grid_dx, dtype=int)
    #     j = ti.floor(p[1] / grid_dx, dtype=int)
    #     k = ti.floor(p[2] / grid_dx, dtype=int)
    #     ti.atomic_add(grid[i, j, k], 1)
    for pi in pos:
        p = pos[pi]
        cell_index = ti.floor(p / grid_dx).cast(int)
        ti.atomic_add(grid[cell_index], 1)


@ti.kernel
def compute_particle_volume(
    pos: ti.template(), grid: ti.template(), particle_vol: ti.template(), grid_dx: float
):
    # for pi in range(pos.shape[0]):
    #     p = pos[pi]
    #     i = ti.floor(p[0] / grid_dx, dtype=int)
    #     j = ti.floor(p[1] / grid_dx, dtype=int)
    #     k = ti.floor(p[2] / grid_dx, dtype=int)
    #     particle_vol[pi] = (grid_dx * grid_dx * grid_dx) / grid[i, j, k]
    for pi in pos:
        p = pos[pi]
        cell_index = ti.floor(p / grid_dx).cast(int)
        particle_vol[pi] = (grid_dx ** 3) / grid[cell_index]

def get_particle_volume(pos, grid_n: int, grid_dx: float, uniform: bool = False):
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    particle_vol = ti.field(dtype=float, shape=pos.shape[0])

    assign_particle_to_grid(ti_pos, grid, grid_dx)
    compute_particle_volume(ti_pos, grid, particle_vol, grid_dx)

    if uniform:
        vol = particle_vol.to_torch()
        vol = torch.mean(vol).repeat(pos.shape[0])
        return vol
    else:
        return particle_vol.to_torch()