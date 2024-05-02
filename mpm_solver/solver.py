import taichi as ti
import taichi.math as tm
from mpm_solver.model import *
from mpm_solver.utils import *
from mpm_solver.collider import *
from mpm_solver.boundary_conditions import *


@ti.data_oriented
class MPM_Simulator:
    def __init__(self, xyzs, covs, volumes, args):
        self.n_particles = xyzs.shape[0]
        self.mpm_model = MPM_model(self.n_particles, args)
        self.mpm_state = MPM_state(self.n_particles, xyzs, covs, volumes, args)

        self.time = 0.0

        self.collider_params = []

        self.particle_preprocess = []
        self.grid_postprocess = []

    def p2g2p(self, dt: ti.f32):
        self.mpm_state.reset_grid_state()

        # Particle operations
        for pp in self.particle_preprocess:
            if pp.isActive(self.time):
                pp.apply(self.mpm_state)
        compute_stress_from_F_trial(self.mpm_state, self.mpm_model, dt)
        
        # Particle to Grid
        p2g(self.mpm_state, self.mpm_model, dt)
        
        # Grid operations
        grid_normalization_and_gravity(self.mpm_state, self.mpm_model, dt)
        for gp in self.grid_postprocess:
            if gp.isCollide:
                gp.collide(self.time, dt, self.mpm_state, self.mpm_model)

            elif gp.isActive(self.time):
                gp.apply(self.mpm_state, self.mpm_model.dx)
        
        # Grid to Particle
        g2p(self.mpm_state, self.mpm_model, dt)
        
        self.time += dt

    def set_boundary_conditions(self, bc_args_arr, sim_args : MPMParams):
        for bc_args in bc_args_arr:
            bc_type = bc_args["type"]

            bc = boundaryConditionTypeCallBacks[bc_type](self.n_particles, bc_args, sim_args)
            
            if bc.type in preprocess_bc:
                self.particle_preprocess.append(bc)
            
            if bc.type in postprocess_bc:
                self.grid_postprocess.append(bc)

    def postprocess(self):
        compute_cov_from_F(self.mpm_state, self.mpm_model)
        compute_R_from_F(self.mpm_state, self.mpm_model)

    # a surface specified by a point and the normal vector
    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky", # For now, not used
        friction=0.0,
        start_time=0.0, # For now, not used
        end_time=999.0, # For now, not used
    ):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / tm.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = MPM_Collider(tm.vec3(point[0], point[1], point[2]), tm.vec3(normal[0], normal[1], normal[2]), friction)

        # collider_param.point = tm.vec3(point[0], point[1], point[2])
        # collider_param.normal = tm.vec3(normal[0], normal[1], normal[2])
        # collider_param.friction = friction

        self.collider_params.append(collider_param)

        collide_type = "ground"

        cl = collideTypeCallBacks[collide_type](collider_param.point, collider_param.normal, collider_param.friction)

        self.grid_postprocess.append(cl)
        # self.modify_bc.append(None)
