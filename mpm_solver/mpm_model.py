import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

@ti.data_oriented
class MPM_model:
    # grid_lim: float
    # n_particles: int
    # n_grid: int
    # dx: float
    # inv_dx: float
    # grid_dim_x: int
    # grid_dim_y: int
    # grid_dim_z: int
    # mu: ti.field(dtype=ti.f32, shape=n_particles)
    # lam: ti.field(dtype=ti.f32, shape=n_particles)
    # E: ti.field(dtype=ti.f32, shape=n_particles)
    # nu: ti.field(dtype=ti.f32, shape=n_particles)
    # material: int
    # yield_stress: ti.field(dtype=ti.f32, shape=n_particles)
    # friction_angle: float
    # alpha: float
    # gravitational_acceleration = ti.types.vector(3, ti.f32)
    # hardening: float
    # xi: float
    # plastic_viscosity: float
    # softening: float
    # rpic_damping: float
    # grid_v_damping_scale: float
    # update_cov_with_F: int
    def __init__(self, n_particles, n_grid=100, grid_lim=1.0):
        self.grid_lim    = grid_lim
        self.n_particles = n_particles
        self.n_grid      = n_grid
        self.dx          = grid_lim / n_grid
        self.inv_dx      = n_grid / grid_lim
        self.grid_dim_x  = n_grid
        self.grid_dim_y  = n_grid
        self.grid_dim_z  = n_grid
        
        # Material properties and particle properties
        self.mu  = ti.field(dtype=ti.f32, shape=n_particles)
        self.lam = ti.field(dtype=ti.f32, shape=n_particles)
        self.E   = ti.field(dtype=ti.f32, shape=n_particles)
        self.nu  = ti.field(dtype=ti.f32, shape=n_particles)
        self.material = 0

        self.yield_stress = ti.field(dtype=ti.f32, shape=n_particles)
        self.friction_angle = 0.0  # degrees
        self.alpha = ti.sqrt(2.0 / 3.0) * 2.0 * ti.sin(self.friction_angle / 180.0 * tm.pi) / (3.0 - ti.sin(self.friction_angle / 180.0 * tm.pi))
        self.gravitational_acceleration = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.hardening = 0.1
        self.xi = 0.0
        self.plastic_viscosity = 0.0
        self.softening = 0.1

        self.E = ti.field(dtype=ti.f32, shape=n_particles)
        self.nu = ti.field(dtype=ti.f32, shape=n_particles)
        self.mu = ti.field(dtype=ti.f32, shape=n_particles)
        self.lam = ti.field(dtype=ti.f32, shape=n_particles)
        self.yield_stress = ti.field(dtype=ti.f32, shape=n_particles)

        self.material = 0  # Assume some default material type
        self.update_cov_with_F = False  # Default value as example
        self.friction_angle = 25.0
        sin_phi = ti.sin(self.friction_angle / 180.0 * 3.141592653589793)
        self.alpha = ti.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        self.gravitational_acceleration = ti.Vector([0.0, 0.0, -1e-5])  # Adjust as necessary #### !!!!!
        self.rpic_damping = 0.0
        self.grid_v_damping_scale = 1.1


@ti.data_oriented
class MPM_state:
    # # Particle-related properties
    # particle_x = ti.Vector.field(3, dtype=ti.f32)  # current position
    # particle_v = ti.Vector.field(3, dtype=ti.f32)  # particle velocity
    # particle_F = ti.Matrix.field(3, 3, dtype=ti.f32)  # deformation gradient
    # particle_init_cov = ti.field(dtype=ti.f32)  # initial covariance
    # particle_cov = ti.field(dtype=ti.f32)  # current covariance
    # particle_F_trial = ti.Matrix.field(3, 3, dtype=ti.f32)  # for return mapping
    # particle_R = ti.Matrix.field(3, 3, dtype=ti.f32)  # rotation matrix
    # particle_stress = ti.Matrix.field(3, 3, dtype=ti.f32)  # stress matrix
    # particle_C = ti.Matrix.field(3, 3, dtype=ti.f32)  # possibly elasticity tensor
    # particle_vol = ti.field(dtype=ti.f32)  # current volume
    # particle_mass = ti.field(dtype=ti.f32)  # mass
    # particle_density = ti.field(dtype=ti.f32)  # density
    # particle_Jp = ti.field(dtype=ti.f32)  # plastic deformation
    # particle_selection = ti.field(dtype=ti.i32)  # selection mask

    # # Grid-related properties
    # grid_m: ti.field(dtype=ti.f32)  # mass at grid nodes
    # grid_v_in: ti.Vector.field(3, dtype=ti.f32)  # input velocity
    # grid_v_out: ti.Vector.field(3, dtype=ti.f32)  # output velocity
    def __init__(self, n_particles, n_grid=100):
        self.particle_x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
        self.particle_v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
        self.particle_F = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=n_particles
        )
        self.particle_R = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=n_particles
        )
        self.particle_init_cov = ti.field(
            dtype=ti.f32, shape=n_particles * 6
        )  # Adjust as needed
        self.particle_cov = ti.field(
            dtype=ti.f32, shape=n_particles * 6
        )  # Adjust as needed
        self.particle_F_trial = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=n_particles
        )
        self.particle_stress = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=n_particles
        )
        self.particle_vol = ti.field(dtype=ti.f32, shape=n_particles)
        self.particle_mass = ti.field(dtype=ti.f32, shape=n_particles)
        self.particle_density = ti.field(dtype=ti.f32, shape=n_particles)
        self.particle_C = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=n_particles
        )
        self.particle_Jp = ti.field(dtype=ti.f32, shape=n_particles)
        self.particle_selection = ti.field(dtype=ti.i32, shape=n_particles)

        # Initialize grid fields
        grid_shape = (n_grid, n_grid, n_grid)
        self.grid_m = ti.field(dtype=ti.f32, shape=grid_shape)
        self.grid_v_in = ti.Vector.field(3, dtype=ti.f32, shape=grid_shape)
        self.grid_v_out = ti.Vector.field(3, dtype=ti.f32, shape=grid_shape)
