import taichi as ti

ti.init(arch=ti.gpu)

class MPM_model:
    def __init__(self):
        self.grid_lim = ti.field(dtype=ti.f32, shape=())
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.n_grid = ti.field(dtype=ti.i32, shape=())
        self.dx = ti.field(dtype=ti.f32, shape=())
        self.inv_dx = ti.field(dtype=ti.f32, shape=())
        self.grid_dim_x = ti.field(dtype=ti.i32, shape=())
        self.grid_dim_y = ti.field(dtype=ti.i32, shape=())
        self.grid_dim_z = ti.field(dtype=ti.i32, shape=())
        
        self.mu = ti.field(dtype=ti.f32, shape=())
        self.lam = ti.field(dtype=ti.f32, shape=())
        self.E = ti.field(dtype=ti.f32, shape=())
        self.nu = ti.field(dtype=ti.f32, shape=())
        self.material = ti.field(dtype=ti.i32, shape=())
        
        self.yield_stress = ti.field(dtype=ti.f32, shape=())
        self.friction_angle = ti.field(dtype=ti.f32, shape=())
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.gravitational_acceleration = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.hardening = ti.field(dtype=ti.f32, shape=())
        self.xi = ti.field(dtype=ti.f32, shape=())
        self.plastic_viscosity = ti.field(dtype=ti.f32, shape=())
        self.softening = ti.field(dtype=ti.f32, shape=())

        self.rpic_damping = ti.field(dtype=ti.f32, shape=())
        self.grid_v_damping_scale = ti.field(dtype=ti.f32, shape=())

        self.update_cov_with_F = ti.field(dtype=ti.i32, shape=())

class MPM_state:
    def __init__(self, num_particles, grid_size):
        self.particle_x = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)  # Current position
        self.particle_v = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)  # Velocity
        self.particle_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)  # Deformation gradient
        self.particle_init_cov = ti.field(dtype=ti.f32, shape=num_particles)  # Initial covariance matrix
        self.particle_cov = ti.field(dtype=ti.f32, shape=num_particles)  # Current covariance matrix
        self.particle_F_trial = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)  # Elastic deformation gradient for trial
        self.particle_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)  # Rotation matrix
        self.particle_stress = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)  # Stress tensor
        self.particle_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)  # Constitutive matrix?
        self.particle_vol = ti.field(dtype=ti.f32, shape=num_particles)  # Volume
        self.particle_mass = ti.field(dtype=ti.f32, shape=num_particles)  # Mass
        self.particle_density = ti.field(dtype=ti.f32, shape=num_particles)  # Density
        self.particle_Jp = ti.field(dtype=ti.f32, shape=num_particles)  # Plastic deformation
        
        self.particle_selection = ti.field(dtype=ti.i32, shape=num_particles)  # Simulation selector

        # Grid properties
        self.grid_m = ti.field(dtype=ti.f32, shape=grid_size)  # Mass at grid nodes
        self.grid_v_in = ti.Vector.field(3, dtype=ti.f32, shape=grid_size)  # Input velocity at grid nodes
        self.grid_v_out = ti.Vector.field(3, dtype=ti.f32, shape=grid_size)  # Output velocity at grid nodes



