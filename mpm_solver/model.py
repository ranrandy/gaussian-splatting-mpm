import taichi as ti
from arguments import *
from mpm_solver.utils import material_types, compute_mu_lam_from_E_nu, compute_mass_from_vol_density


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

        # self.grid_v_damping_scale = 1.1

    def init_elasiticity_params(self):
        self.E = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.nu = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.mu = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.lam = ti.field(dtype=ti.f32, shape=self.n_particles)

        self.E.fill(self.args.E)
        self.nu.fill(self.args.nu)
        compute_mu_lam_from_E_nu(self.n_particles, self.E, self.nu, self.mu, self.lam)

    def init_plasticity_params(self):
        # self.friction_angle = self.args.friction_angle
        # sin_phi = ti.sin(self.friction_angle / 180.0 * 3.141592653589793)
        # self.alpha = ti.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        pass 

    def init_other_params(self):
        # self.yield_stress = ti.field(dtype=ti.f32, shape=n_particles)
        pass


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
