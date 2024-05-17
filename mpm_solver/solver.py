import taichi as ti
import taichi.math as tm
from mpm_solver.model import *
from mpm_solver.utils import *
from mpm_solver.collider import *
from mpm_solver.boundary_conditions import *


@ti.data_oriented
class MPM_Simulator:
    def __init__(self, xyzs, covs, volumes, args, init_v=None):
        self.n_particles = xyzs.shape[0]
        self.mpm_model = MPM_model(self.n_particles, args)
        if args.fitting:
            self.mpm_state = MPM_state_opt(self.n_particles, xyzs, covs, volumes, args, init_v)
        else:
            self.mpm_state = MPM_state(self.n_particles, xyzs, covs, volumes, args)

        self.time = 0.0

        self.collider_params = []

        self.particle_preprocess = []
        self.grid_postprocess = []
        self.init_particles = []

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
        
        # print(self.time)
        self.time += dt

    def p2g2p_forward(self, dt: ti.f32, s):
        self.mpm_state.reset_grid_state()

        compute_stress_from_F_opt(self.mpm_state, self.mpm_model, dt, s)
        
        # Particle to Grid
        p2g_opt(self.mpm_state, self.mpm_model, dt, s)
        
        # Grid operations
        grid_normalization_and_gravity(self.mpm_state, self.mpm_model, dt)
        self.grid_postprocess[0].apply(self.mpm_state, self.mpm_model.dx)
        
        # Grid to Particle
        g2p_opt(self.mpm_state, self.mpm_model, dt, s)
        
        self.time += dt

    def p2g2p_backward(self, dt: ti.f32, s):
        self.mpm_state.reset_grid_state()

        # Since we do not store the grid history (to save space), we redo p2g and grid op
        p2g_opt(self.mpm_state, self.mpm_model, dt, s)

        grid_normalization_and_gravity(self.mpm_state, self.mpm_model, dt)
        self.grid_postprocess[0].apply(self.mpm_state, self.mpm_model.dx)
        
        # Backward
        g2p_opt.grad(self.mpm_state, self.mpm_model, dt, s)

        grid_normalization_and_gravity.grad(self.mpm_state, self.mpm_model, dt)
        self.grid_postprocess[0].apply.grad(self.mpm_state, self.mpm_model.dx)
        
        p2g_opt.grad(self.mpm_state, self.mpm_model, dt, s)

        compute_stress_from_F_opt.grad(self.mpm_state, self.mpm_model, dt, s)

        compute_mu_lam_from_E_nu.grad(self.n_particles, self.mpm_model.logE, self.mpm_model.y, self.mpm_model.mu, self.mpm_model.lam)

    @ti.kernel
    def learn(self):
        for p in range(self.n_particles):
            # Clipping gradient for logE
            clipped_grad_logE = self.mpm_model.logE.grad[p]
            if ti.abs(self.mpm_model.logE.grad[p]) > 1.0:
                grad_sign = tm.sign(self.mpm_model.logE.grad[p])
                clipped_grad_logE = grad_sign * 1.0
            
            # Clipping gradient for y
            clipped_grad_y = self.mpm_model.y.grad[p]
            if ti.abs(self.mpm_model.y.grad[p]) > 1.0:
                grad_sign = tm.sign(self.mpm_model.y.grad[p])
                clipped_grad_y = grad_sign * 1.0
            
            self.mpm_model.logE[p] -= 0.8 * clipped_grad_logE
            self.mpm_model.y[p] -= 1.6 * clipped_grad_y

    def set_boundary_conditions(self, bc_args_arr, sim_args : MPMParams):
        for bc_args in bc_args_arr:
            bc_type = bc_args["type"]

            bc = boundaryConditionTypeCallBacks[bc_type](self.n_particles, bc_args, sim_args)
            
            if bc.type in preprocess_bc:
                self.particle_preprocess.append(bc)
            
            if bc.type in postprocess_bc:
                self.grid_postprocess.append(bc)
            
            if bc.type in init_bc:
                self.init_particles.append(bc)

        for pp in self.init_particles:
            pp.apply(self.mpm_state, self.mpm_model)
            if pp.isMaterial == False:
                compute_mu_lam_from_E_nu(self.n_particles, self.mpm_model.E, self.mpm_model.nu, self.mpm_model.mu, self.mpm_model.lam)
                pp.applymu(self.mpm_state, self.mpm_model)

    def set_bc_ground_only(self):
        bc = boundaryConditionTypeCallBacks["sticky_ground"]()
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

    def postprocess_forward(self):
        compute_cov_from_F_opt(self.mpm_state, self.mpm_model)

    def postprocess_backward(self):
        compute_cov_from_F_opt.grad(self.mpm_state, self.mpm_model)

    def clear_grads(self):
        self.mpm_model.clear_grad()
        self.mpm_state.clear_grad()
