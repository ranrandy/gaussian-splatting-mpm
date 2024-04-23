import sys
import os

import numpy as np
import h5py
import taichi as ti
from mpm_model import *
from mpm_solver.utils import *

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

@ti.data_oriented
class MPM_Simulator:
    def __init__(self, n_particles, n_grid, grid_lim, device="cuda"):
        ti.init(arch=ti.cuda if device == "cuda" else ti.cpu)  # Initialize Taichi with the desired device
        self.initialize(n_particles, n_grid, grid_lim)
        self.time_profile = {}

    def initialize(self, n_particles, n_grid, grid_lim):
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.grid_lim = grid_lim

        self.mpm_model = MPM_model()
        self.mpm_model.grid_lim[None] = grid_lim
        self.mpm_model.n_grid[None] = n_grid
        self.mpm_model.grid_dim_x[None] = n_grid
        self.mpm_model.grid_dim_y[None] = n_grid
        self.mpm_model.grid_dim_z[None] = n_grid
        self.mpm_model.dx[None], self.mpm_model.inv_dx[None] = grid_lim / n_grid, float(n_grid / grid_lim)

        # Initializing arrays with zero
        self.mpm_model.E = ti.field(ti.f32, shape=n_particles)
        self.mpm_model.nu = ti.field(ti.f32, shape=n_particles)
        self.mpm_model.mu = ti.field(ti.f32, shape=n_particles)
        self.mpm_model.lam = ti.field(ti.f32, shape=n_particles)
        self.mpm_model.update_cov_with_F[None] = False
        self.mpm_model.material[None] = 0  # 0 for jelly
        self.mpm_model.plastic_viscosity[None] = 0.0
        self.mpm_model.softening[None] = 0.1
        self.mpm_model.yield_stress = ti.field(ti.f32, shape=n_particles)
        self.mpm_model.friction_angle[None] = 25.0
        sin_phi = ti.sin(self.mpm_model.friction_angle[None] / 180.0 * 3.14159265)
        self.mpm_model.alpha[None] = ti.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        self.mpm_model.gravitational_acceleration[None] = ti.Vector([0.0, 0.0, 0.0])
        self.mpm_model.rpic_damping[None] = 0.0
        self.mpm_model.grid_v_damping_scale[None] = 1.1

        self.mpm_state = MPM_state(n_particles, (n_grid, n_grid, n_grid))


        # Initializing vector and matrix fields
        for i in range(n_particles):
            self.mpm_state.particle_x[i] = ti.Vector([0.0, 0.0, 0.0])
            self.mpm_state.particle_v[i] = ti.Vector([0.0, 0.0, 0.0])
            self.mpm_state.particle_F[i] = ti.Matrix([[0.0]*3]*3)
            self.mpm_state.particle_R[i] = ti.Matrix([[0.0]*3]*3)
        
        self.mpm_state.particle_init_cov.fill(0.0)
        self.mpm_state.particle_cov.fill(0.0)
        self.mpm_state.particle_vol.fill(0.0)
        self.mpm_state.particle_mass.fill(0.0)
        self.mpm_state.particle_density.fill(0.0)
        self.mpm_state.particle_Jp.fill(0.0)
        self.mpm_state.particle_selection.fill(0)

        self.mpm_state.grid_m.fill(0.0)
        self.mpm_state.grid_v_in.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.mpm_state.grid_v_out.fill(ti.Vector([0.0, 0.0, 0.0]))

        self.time = 0.0

        # Post-processing and boundary conditions
        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []


        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []
    
    def load_from_sampling(self, sampling_h5, n_grid=100, grid_lim=1.0):
        if not os.path.exists(sampling_h5):
            print("h5 file cannot be found at ", os.path.join(os.getcwd(), sampling_h5))
            exit()

        with h5py.File(sampling_h5, "r") as h5file:
            x = np.array(h5file["x"]).transpose()  # Ensure x is a numpy array
            particle_volume = np.array(h5file["particle_volume"]).squeeze()

        self.dim, self.n_particles = x.shape[1], x.shape[0]
        self.initialize(self.n_particles, n_grid, grid_lim)

        print("Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles")
        
        # Load and set particle positions and volumes
        for i in range(self.n_particles):
            self.mpm_state.particle_x[i] = ti.Vector(x[i])
            self.mpm_state.particle_vol[i] = particle_volume[i]

        # Set initial velocity to zero
        set_vec3_to_zero(self.mpm_state.particle_v)

        # Set initial deformation gradient to identity
        set_mat33_to_identity(self.mpm_state.particle_F_trial)

        print("Particles initialized from sampling file.")
        print("Total particles: ", self.n_particles)
    
    def set_parameters(self, **kwargs):
        self.set_parameters_dict(kwargs)

    def set_parameters_dict(self, kwargs={}):
        if "material" in kwargs:
            materials = {"jelly": 0, "metal": 1, "sand": 2, "foam": 3, "snow": 4, "plasticine": 5}
            self.mpm_model.material[None] = materials.get(kwargs["material"], -1)  # Default to -1 if not found

        if "grid_lim" in kwargs:
            self.mpm_model.grid_lim[None] = kwargs["grid_lim"]
        if "n_grid" in kwargs:
            self.mpm_model.n_grid[None] = kwargs["n_grid"]
            self.mpm_model.grid_dim_x[None] = kwargs["n_grid"]
            self.mpm_model.grid_dim_y[None] = kwargs["n_grid"]
            self.mpm_model.grid_dim_z[None] = kwargs["n_grid"]
            self.mpm_model.dx[None] = self.mpm_model.grid_lim[None] / self.mpm_model.n_grid[None]
            self.mpm_model.inv_dx[None] = self.mpm_model.n_grid[None] / self.mpm_model.grid_lim[None]

        if "E" in kwargs:
            set_value_to_float_array(self.mpm_model.E, kwargs["E"])
        if "nu" in kwargs:
            set_value_to_float_array(self.mpm_model.nu, kwargs["nu"])
        if "yield_stress" in kwargs:
            set_value_to_float_array(self.mpm_model.yield_stress, kwargs["yield_stress"])
        if "density" in kwargs:
            set_value_to_float_array(self.mpm_state.particle_density, kwargs["density"])
            get_float_array_product(self.mpm_state.particle_density, self.mpm_state.particle_vol, self.mpm_state.particle_mass)

        if "rpic_damping" in kwargs:
            self.mpm_model.rpic_damping[None] = kwargs["rpic_damping"]

        if "plastic_viscosity" in kwargs:
            self.mpm_model.plastic_viscosity[None] = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            self.mpm_model.softening[None] = kwargs["softening"]
        if "grid_v_damping_scale" in kwargs:
            self.mpm_model.grid_v_damping_scale[None] = kwargs["grid_v_damping_scale"]

        #skip additional parameters

    
    def finalize_mu_lam(self):
        self.compute_mu_lam_from_E_nu()

    @ti.kernel
    def compute_mu_lam_from_E_nu(self):
        for i in range(self.n_particles):
            E = self.mpm_model.E[i]
            nu = self.mpm_model.nu[i]
            self.mpm_model.mu[i] = E / (2 * (1 + nu))
            self.mpm_model.lam[i] = E * nu / ((1 + nu) * (1 - 2 * nu))



        



        