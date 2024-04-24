import taichi as ti

ti.init(arch=ti.gpu)


@ti.struct
class MPM_model:
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    mu: ti.field(dtype=ti.f32)
    lam: ti.field(dtype=ti.f32)
    E: ti.field(dtype=ti.f32)
    nu: ti.field(dtype=ti.f32)
    material: int
    yield_stress: ti.field(dtype=ti.f32)
    friction_angle: float
    alpha: float
    gravitational_acceleration: ti.types.vector(3, ti.f32)
    hardening: float
    xi: float
    plastic_viscosity: float
    softening: float
    rpic_damping: float
    grid_v_damping_scale: float
    update_cov_with_F: int


@ti.struct
class MPM_state:
    # Particle-related properties
    particle_x: ti.Vector.field(3, dtype=ti.f32)  # current position
    particle_v: ti.Vector.field(3, dtype=ti.f32)  # particle velocity
    particle_F: ti.Matrix.field(3, 3, dtype=ti.f32)  # deformation gradient
    particle_init_cov: ti.field(dtype=ti.f32)  # initial covariance
    particle_cov: ti.field(dtype=ti.f32)  # current covariance
    particle_F_trial: ti.Matrix.field(3, 3, dtype=ti.f32)  # for return mapping
    particle_R: ti.Matrix.field(3, 3, dtype=ti.f32)  # rotation matrix
    particle_stress: ti.Matrix.field(3, 3, dtype=ti.f32)  # stress matrix
    particle_C: ti.Matrix.field(3, 3, dtype=ti.f32)  # possibly elasticity tensor
    particle_vol: ti.field(dtype=ti.f32)  # current volume
    particle_mass: ti.field(dtype=ti.f32)  # mass
    particle_density: ti.field(dtype=ti.f32)  # density
    particle_Jp: ti.field(dtype=ti.f32)  # plastic deformation
    particle_selection: ti.field(dtype=ti.i32)  # selection mask

    # Grid-related properties
    grid_m: ti.field(dtype=ti.f32)  # mass at grid nodes
    grid_v_in: ti.Vector.field(3, dtype=ti.f32)  # input velocity
    grid_v_out: ti.Vector.field(3, dtype=ti.f32)  # output velocity
