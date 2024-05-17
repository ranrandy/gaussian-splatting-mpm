import taichi as ti
from arguments import *
from mpm_solver.utils import material_types, compute_mu_lam_from_E_nu, compute_mass_from_vol_density
import math

@ti.data_oriented
class MPM_model:
    def __init__(self, n_particles : int, args : MPMParams):
        self.args = args
        self.n_particles = n_particles
        
        # Eulerian grid settings
        self.grid_extent = args.grid_extent
        self.n_grid = args.n_grid
        self.dx = args.grid_extent / args.n_grid
        self.inv_dx = args.n_grid / args.grid_extent
        
        # Material settings
        self.init_general_params()
        self.init_elasiticity_params()  # 1. Elasticity (fixed corotated, StVk, NeoHookean)
        self.init_plasticity_params()   # 2. Followed by Plasticity (Drucker-Prager)
        self.init_other_params()        # 3. Conditioned on Yield Stress (von Mises, Herschel-Bulkley)

    def init_general_params(self):
        self.material = material_types.get(self.args.material, -1)
        if self.material != 0:
            raise TypeError("Material not supported yet")

        self.gravity = ti.Vector(self.args.gravity)

    def init_elasiticity_params(self):
        self.logE = ti.field(dtype=ti.f32, shape=self.n_particles, needs_grad=self.args.fitting)
        self.y = ti.field(dtype=ti.f32, shape=self.n_particles, needs_grad=self.args.fitting)
        self.mu = ti.field(dtype=ti.f32, shape=self.n_particles, needs_grad=self.args.fitting)
        self.lam = ti.field(dtype=ti.f32, shape=self.n_particles, needs_grad=self.args.fitting)

        self.logE.fill(math.log10(self.args.E))
        self.y.fill(-math.log(0.49 / self.args.nu - 1))
        compute_mu_lam_from_E_nu(self.n_particles, self.logE, self.y, self.mu, self.lam)

    def init_plasticity_params(self):
        pass 

    def init_other_params(self):
        pass

    def clear_grad(self):
        self.logE.grad.fill(0.0)
        self.y.grad.fill(0.0)
        self.mu.grad.fill(0.0)
        self.lam.grad.fill(0.0)


@ti.data_oriented
class MPM_state:
    def __init__(self, n_particles, xyzs, covs, volumes, args : MPMParams):
        self.n_particles = n_particles

        # Particle state declarations
        self.particle_vol = ti.field(dtype=ti.f32, shape=n_particles)                   # Volume
        self.particle_density = ti.field(dtype=ti.f32, shape=n_particles)
        self.particle_mass = ti.field(dtype=ti.f32, shape=n_particles)

        self.particle_xyz = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
        self.particle_vel = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)         # Velocity
        self.particle_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)        # Deformation gradient
        self.particle_stress = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
        self.particle_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)        # Affine momentum
        
        self.particle_init_cov = ti.field(dtype=ti.f32, shape=n_particles * 6)          # Sec 3.4. A_p
        self.particle_cov = ti.field(dtype=ti.f32, shape=n_particles * 6)               # Sec 3.4. a_p(t)

        self.particle_F_trial = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)  # Used for plasticity models
        
        # Particle state initializations
        self.particle_vol.from_torch(volumes)
        self.particle_density.fill(args.density)
        compute_mass_from_vol_density(self.n_particles, self.particle_density, self.particle_vol, self.particle_mass)

        self.particle_xyz.from_torch(xyzs)
        self.particle_vel.fill(0.0)
        self.init_identity_mat(self.particle_F)
        self.particle_stress.fill(0.0)
        self.particle_C.fill(0.0)

        self.particle_init_cov.from_torch(covs.flatten())
        self.particle_cov.copy_from(self.particle_init_cov)
                
        self.init_identity_mat(self.particle_F_trial)

        # Grid states
        grid_shape = (args.n_grid, args.n_grid, args.n_grid)
        self.grid_mass = ti.field(dtype=ti.f32, shape=grid_shape)                       # Grid mass
        self.grid_v_in = ti.Vector.field(3, dtype=ti.f32, shape=grid_shape)             # Grid momentum (before update)
        self.grid_v_out = ti.Vector.field(3, dtype=ti.f32, shape=grid_shape)            # Grid velocity (after update)
        self.reset_grid_state()

    def reset_grid_state(self):
        self.grid_mass.fill(0.0)
        self.grid_v_in.fill(0.0)
        self.grid_v_out.fill(0.0)

    @ti.kernel
    def init_identity_mat(self, F : ti.template()):
        for p in range(self.n_particles):
            F[p] = ti.Matrix.identity(ti.f32, 3)


@ti.data_oriented
class MPM_state_opt:
    def __init__(self, n_particles, xyzs, covs, volumes, args : MPMParams, init_vel):
        self.n_particles = n_particles

        # Particle state declarations
        self.particle_vol = ti.field(dtype=ti.f32, shape=n_particles)                   # Volume
        self.particle_density = ti.field(dtype=ti.f32, shape=n_particles)
        self.particle_mass = ti.field(dtype=ti.f32, shape=n_particles)

        self.particle_xyz = ti.Vector.field(3, dtype=ti.f32, shape=(31, n_particles), needs_grad=args.fitting)
        self.particle_vel = ti.Vector.field(3, dtype=ti.f32, shape=(31, n_particles), needs_grad=args.fitting)         # Velocity
        self.particle_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(31, n_particles), needs_grad=args.fitting)        # Deformation gradient
        self.particle_stress = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(31, n_particles), needs_grad=args.fitting)
        self.particle_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(31, n_particles), needs_grad=args.fitting)        # Affine momentum
        
        self.particle_init_cov = ti.field(dtype=ti.f32, shape=n_particles * 6)                                      # Sec 3.4. A_p
        self.particle_cov = ti.field(dtype=ti.f32, shape=n_particles * 6, needs_grad=args.fitting)                  # Sec 3.4. a_p(t)

        # Particle state initializations
        self.particle_vol.from_torch(volumes)
        self.particle_density.fill(args.density)
        compute_mass_from_vol_density(self.n_particles, self.particle_density, self.particle_vol, self.particle_mass)

        self.init2(xyzs, init_vel)
        self.init_identity_mat(self.particle_F)

        self.particle_init_cov.from_torch(covs.flatten())
        self.particle_cov.copy_from(self.particle_init_cov)
                
        # Grid states
        grid_shape = (args.n_grid, args.n_grid, args.n_grid)
        self.grid_mass = ti.field(dtype=ti.f32, shape=grid_shape)                       # Grid mass
        self.grid_v_in = ti.Vector.field(3, dtype=ti.f32, shape=grid_shape, needs_grad=args.fitting)             # Grid momentum (before update)
        self.grid_v_out = ti.Vector.field(3, dtype=ti.f32, shape=grid_shape, needs_grad=args.fitting)            # Grid velocity (after update)
        self.reset_grid_state()

    @ti.kernel
    def init2(self, 
              xyz_ : ti.types.ndarray(element_dim=1), 
              vel_ : ti.types.ndarray(element_dim=1)):
        for p in range(self.n_particles):
            self.particle_xyz[0, p] = xyz_[p]
            self.particle_vel[0, p] = vel_[p]
            self.particle_stress[0, p].fill(0.0)
            self.particle_C[0, p].fill(0.0)

    def reset_grid_state(self):
        self.grid_mass.fill(0.0)
        self.grid_v_in.fill(0.0)
        self.grid_v_out.fill(0.0)

    @ti.kernel
    def init_identity_mat(self, F : ti.template()):
        for p in range(self.n_particles):
            F[0, p] = ti.Matrix.identity(ti.f32, 3)
    
    @ti.kernel
    def set_grads(self, xyz_grad : ti.types.ndarray(element_dim=1), cov_grad : ti.types.ndarray()):
        for p in range(self.n_particles):
            self.particle_xyz.grad[30, p] = xyz_grad[p]

            self.particle_cov.grad[6 * p] = cov_grad[6 * p]
            self.particle_cov.grad[6 * p + 1] = cov_grad[6 * p + 1]
            self.particle_cov.grad[6 * p + 2] = cov_grad[6 * p + 2]
            self.particle_cov.grad[6 * p + 3] = cov_grad[6 * p + 3]
            self.particle_cov.grad[6 * p + 4] = cov_grad[6 * p + 4]
            self.particle_cov.grad[6 * p + 5] = cov_grad[6 * p + 5]

    def clear_grad(self):
        self.particle_xyz.grad.fill(0.0)
        self.particle_vel.grad.fill(0.0)
        self.particle_F.grad.fill(0.0)
        self.particle_stress.grad.fill(0.0)
        self.particle_C.grad.fill(0.0)
        
        self.particle_cov.grad.fill(0.0)

        self.grid_v_in.grad.fill(0.0)
        self.grid_v_out.grad.fill(0.0)

    @ti.kernel
    def cycle_init(self):
        for p in range(self.n_particles):
            self.particle_xyz[0, p] = self.particle_xyz[30, p]
            self.particle_vel[0, p] = self.particle_vel[30, p]
            self.particle_F[0, p] = self.particle_F[30, p]
            self.particle_stress[0, p] = self.particle_stress[30, p]
            self.particle_C[0, p] = self.particle_C[30, p]