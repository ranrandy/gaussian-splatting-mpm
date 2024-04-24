import sys
import os

import numpy as np
import h5py
import taichi as ti
import torch
from mpm_model import *
from mpm_solver.utils import *

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


@ti.data_oriented
class MPM_Simulator:
    def __init__(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda"):
        ti.init(
            arch=ti.cuda if device == "cuda" else ti.cpu
        )  # Initialize the right architecture
        self.initialize(n_particles, n_grid, grid_lim)

    def initialize(self, n_particles, n_grid=100, grid_lim=1.0):
        self.n_particles = n_particles
        self.mpm_model = MPM_model()
        self.mpm_model.grid_lim = grid_lim
        self.mpm_model.n_grid = n_grid
        self.mpm_model.grid_dim_x = n_grid
        self.mpm_model.grid_dim_y = n_grid
        self.mpm_model.grid_dim_z = n_grid
        self.mpm_model.dx = grid_lim / n_grid
        self.mpm_model.inv_dx = n_grid / grid_lim

        # Initialize fields for model properties
        self.mpm_model.E = ti.field(dtype=ti.f32, shape=n_particles)
        self.mpm_model.nu = ti.field(dtype=ti.f32, shape=n_particles)
        self.mpm_model.mu = ti.field(dtype=ti.f32, shape=n_particles)
        self.mpm_model.lam = ti.field(dtype=ti.f32, shape=n_particles)
        self.mpm_model.yield_stress = ti.field(dtype=ti.f32, shape=n_particles)

        self.mpm_model.material = 0  # Assume some default material type
        self.mpm_model.update_cov_with_F = False  # Default value as example
        self.mpm_model.friction_angle = 25.0
        sin_phi = ti.sin(self.mpm_model.friction_angle / 180.0 * 3.141592653589793)
        self.mpm_model.alpha = ti.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        self.mpm_model.gravitational_acceleration = ti.Vector(
            [0.0, 0.0, 0.0]
        )  # Adjust as necessary
        self.mpm_model.rpic_damping = 0.0
        self.mpm_model.grid_v_damping_scale = 1.1

        self.mpm_state = MPM_state()
        # Initialize particle fields
        self.mpm_state.particle_x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
        self.mpm_state.particle_v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
        self.mpm_state.particle_F = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=n_particles
        )
        self.mpm_state.particle_R = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=n_particles
        )
        self.mpm_state.particle_init_cov = ti.field(
            dtype=ti.f32, shape=n_particles * 6
        )  # Adjust as needed
        self.mpm_state.particle_cov = ti.field(
            dtype=ti.f32, shape=n_particles * 6
        )  # Adjust as needed
        self.mpm_state.particle_F_trial = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=n_particles
        )
        self.mpm_state.particle_stress = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=n_particles
        )
        self.mpm_state.particle_vol = ti.field(dtype=ti.f32, shape=n_particles)
        self.mpm_state.particle_mass = ti.field(dtype=ti.f32, shape=n_particles)
        self.mpm_state.particle_density = ti.field(dtype=ti.f32, shape=n_particles)
        self.mpm_state.particle_C = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=n_particles
        )
        self.mpm_state.particle_Jp = ti.field(dtype=ti.f32, shape=n_particles)
        self.mpm_state.particle_selection = ti.field(dtype=ti.i32, shape=n_particles)

        # Initialize grid fields
        grid_shape = (n_grid, n_grid, n_grid)
        self.mpm_state.grid_m = ti.field(dtype=ti.f32, shape=grid_shape)
        self.mpm_state.grid_v_in = ti.Vector.field(3, dtype=ti.f32, shape=grid_shape)
        self.mpm_state.grid_v_out = ti.Vector.field(3, dtype=ti.f32, shape=grid_shape)

        self.time = 0.0

        # Post-processing and boundary conditions
        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []

        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []

    def load_from_sampling(self, sampling_h5, n_grid=100, grid_lim=1.0, device="cuda"):
        if not os.path.exists(sampling_h5):
            print("H5 file cannot be found at", os.path.join(os.getcwd(), sampling_h5))
            exit()

        with h5py.File(sampling_h5, "r") as h5file:
            x = np.array(h5file["x"]).transpose()  # Assuming 'x' is stored correctly
            particle_volume = np.array(h5file["particle_volume"]).squeeze()

        self.dim, self.n_particles = x.shape[1], x.shape[0]
        self.initialize(self.n_particles, n_grid, grid_lim, device=device)

        print(
            "Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles"
        )

        # Transfer data to Taichi structures
        for i in range(self.n_particles):
            self.mpm_state.particle_x[i] = ti.Vector(x[i])

        # Initialize velocity and deformation gradient fields
        set_vec3_to_zero(self.mpm_state.particle_v)
        set_mat33_to_identity(self.mpm_state.particle_F_trial)

        # Volume data transfer
        for i in range(self.n_particles):
            self.mpm_state.particle_vol[i] = particle_volume[i]

        print("Particles initialized from sampling file.")
        print("Total particles:", self.n_particles)

    def set_parameters(self, device="cuda", **kwargs):
        self.set_parameters_dict(kwargs, device=device)

    def set_parameters_dict(self, kwargs={}, device="cuda"):
        if "material" in kwargs:
            material_types = {
                "jelly": 0,
                "metal": 1,
                "sand": 2,
                "foam": 3,
                "snow": 4,
                "plasticine": 5,
            }
            self.mpm_model.material = material_types.get(kwargs["material"], -1)
            if self.mpm_model.material == -1:
                raise TypeError("Undefined material type")

        if "grid_lim" in kwargs:
            self.mpm_model.grid_lim = kwargs["grid_lim"]
        if "n_grid" in kwargs:
            self.mpm_model.n_grid = kwargs["n_grid"]
            self.mpm_model.grid_dim_x = self.mpm_model.n_grid
            self.mpm_model.grid_dim_y = self.mpm_model.n_grid
            self.mpm_model.grid_dim_z = self.mpm_model.n_grid
            self.mpm_model.dx = self.mpm_model.grid_lim / self.mpm_model.n_grid
            self.mpm_model.inv_dx = float(
                self.mpm_model.n_grid / self.mpm_model.grid_lim
            )
            grid_shape = (
                self.mpm_model.n_grid,
                self.mpm_model.n_grid,
                self.mpm_model.n_grid,
            )
            self.mpm_state.grid_m = ti.field(dtype=ti.f32, shape=grid_shape)
            self.mpm_state.grid_v_in = ti.Vector.field(
                3, dtype=ti.f32, shape=grid_shape
            )
            self.mpm_state.grid_v_out = ti.Vector.field(
                3, dtype=ti.f32, shape=grid_shape
            )

        if "E" in kwargs:
            set_value_to_float_array(self.mpm_model.E, kwargs["E"])
        if "nu" in kwargs:
            set_value_to_float_array(self.mpm_model.nu, kwargs["nu"])
        if "yield_stress" in kwargs:
            set_value_to_float_array(
                self.mpm_model.yield_stress, kwargs["yield_stress"]
            )
        if "hardening" in kwargs:
            self.mpm_model.hardening = kwargs["hardening"]
        if "xi" in kwargs:
            self.mpm_model.xi = kwargs["xi"]
        if "friction_angle" in kwargs:
            self.mpm_model.friction_angle = kwargs["friction_angle"]
            sin_phi = ti.sin(self.mpm_model.friction_angle / 180.0 * ti.pi)
            self.mpm_model.alpha = ti.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        if "g" in kwargs:
            self.mpm_model.gravitational_acceleration = ti.Vector(
                [kwargs["g"][0], kwargs["g"][1], kwargs["g"][2]]
            )
        if "density" in kwargs:
            set_value_to_float_array(self.mpm_state.particle_density, kwargs["density"])
            get_float_array_product(
                self.mpm_state.particle_density,
                self.mpm_state.particle_vol,
                self.mpm_state.particle_mass,
            )
        if "rpic_damping" in kwargs:
            self.mpm_model.rpic_damping = kwargs["rpic_damping"]
        if "plastic_viscosity" in kwargs:
            self.mpm_model.plastic_viscosity = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            self.mpm_model.softening = kwargs["softening"]
        if "grid_v_damping_scale" in kwargs:
            self.mpm_model.grid_v_damping_scale = kwargs["grid_v_damping_scale"]
        if "additional_material_params" in kwargs:
            for params in kwargs["additional_material_params"]:
                apply_additional_params(self.mpm_state, self.mpm_model, params)
                get_float_array_product(
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                )

    def finalize_mu_lam(self):
        compute_mu_lam_from_E_nu(
            self.mpm_model.n_particles,
            self.mpm_model.E,
            self.mpm_model.nu,
            self.mpm_model.mu,
            self.mpm_model.lam,
        )

    def reset_densities_and_update_masses(self, all_particle_densities):
        # Convert PyTorch tensor to NumPy array
        numpy_densities = all_particle_densities.cpu().numpy()

        # Set values in Taichi field
        for i in range(self.n_particles):
            self.mpm_state.particle_density[i] = numpy_densities[i]

        # Compute particle mass using updated densities
        multiply_and_update_density_mass(
            self.mpm_state.particle_density, self.mpm_state.particle_vol, self.mpm_state.particle_mass
        )

    def import_tensor_to_taichi(self, tensor, taichi_field):
        # Ensure the tensor is on CPU and clone if necessary
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor_np = tensor.numpy()  # Convert to numpy array
        for i in range(taichi_field.shape[0]):
            taichi_field[i] = tensor_np[i]  # Assign values

    def import_particle_x_from_torch(self, tensor_x, clone=True):
        if tensor_x is not None:
            if clone:
                tensor_x = tensor_x.clone().detach()
            self.import_tensor_to_taichi(tensor_x, self.mpm_state.particle_x)

    def import_particle_v_from_torch(self, tensor_v, clone=True):
        if tensor_v is not None:
            if clone:
                tensor_v = tensor_v.clone().detach()
            self.import_tensor_to_taichi(tensor_v, self.mpm_state.particle_v)

    def import_matrix_tensor_to_taichi(self, tensor, taichi_field):
        if tensor.is_cuda:
            tensor = tensor.cpu()  # Convert GPU tensor to CPU
        tensor_np = tensor.numpy().reshape(-1, 3, 3)  # Reshape to match the 3x3 matrix
        for i in range(taichi_field.shape[0]):
            for j in range(3):  # For each row
                for k in range(3):  # For each column
                    taichi_field[i][j, k] = tensor_np[i, j, k]  # Assign matrix element

    def import_particle_F_from_torch(self, tensor_F, clone=True):
        if tensor_F is not None:
            if clone:
                tensor_F = tensor_F.clone().detach()
            tensor_F = tensor_F.reshape(-1, 3, 3)  # Ensure it's shaped as (-1, 3, 3)
            self.import_matrix_tensor_to_taichi(tensor_F, self.mpm_state.particle_F)

    def import_particle_C_from_torch(self, tensor_C, clone=True):
        if tensor_C is not None:
            if clone:
                tensor_C = tensor_C.clone().detach()
            tensor_C = tensor_C.reshape(-1, 3, 3)  # Ensure it's shaped as (-1, 3, 3)
            self.import_matrix_tensor_to_taichi(tensor_C, self.mpm_state.particle_C)

    def export_particle_x_to_torch(self):
        return taichi_to_torch(self.mpm_state.particle_x)

    def export_particle_v_to_torch(self):
        return taichi_to_torch(self.mpm_state.particle_v)

    def export_particle_F_to_torch(self):
        F_tensor = taichi_to_torch(self.mpm_state.particle_F)
        return F_tensor.reshape(-1, 9)

    def export_particle_C_to_torch(self):
        C_tensor = taichi_to_torch(self.mpm_state.particle_C)
        return C_tensor.reshape(-1, 9)

    def export_particle_R_to_torch(self):
        # Update rotation matrices based on current deformation gradients
        compute_R_from_F(self.mpm_state.particle_F, self.mpm_state.particle_R, self.n_particles)
        R_tensor = taichi_to_torch(self.mpm_state.particle_R)
        return R_tensor.reshape(-1, 9)

    def export_particle_cov_to_torch(self):
        if not self.mpm_model.update_cov_with_F:
            # Perform computation only if update flag is not set
            compute_cov_from_F(self.mpm_state, self.mpm_model)

        # Convert and return the Taichi field as a PyTorch tensor
        return taichi_field_to_torch(self.mpm_state.particle_cov, self.n_particles)

    # def print_time_profile(self):
    #     print("MPM Time profile:")
    #     for key, value in self.time_profile.items():
    #         print(key, sum(value))

    def load_initial_data_from_torch(self, tensor_x, tensor_volume, tensor_cov=None, n_grid=100, grid_lim=1.0):
        n_particles = tensor_x.shape[0]
        dim = tensor_x.shape[1]  # Not used directly but can be part of model configuration
        assert tensor_x.shape[0] == tensor_volume.shape[0]

        self.initialize(n_particles, n_grid, grid_lim)
        self.import_particle_x_from_torch(tensor_x)

        tensor_volume_np = tensor_volume.cpu().detach().numpy()
        for i in range(self.n_particles):
            self.mpm_state.particle_vol[i] = tensor_volume_np[i]

        if tensor_cov is not None:
            tensor_cov_np = tensor_cov.cpu().detach().numpy().reshape(-1, 6)
            for i in range(self.n_particles):
                for j in range(6):
                    self.mpm_state.particle_init_cov[6 * i + j] = tensor_cov_np[i, j]
            if self.update_cov_with_F:
                for i in range(n_particles * 6):
                    self.mpm_state.particle_cov[i] = self.mpm_state.particle_init_cov[i]

        set_vec3_to_zero(self.mpm_state.particle_v)
        set_mat33_to_identity(self.mpm_state.particle_F_trial)

        print("Particles initialized from torch data.")
        print("Total particles: ", n_particles)

    @ti.function
    def p2g2p_sanity_check(self, dt: ti.f32):
        # reset_grid_state(self.mpm_state, self.mpm_model)
        p2g_apic_with_stress(self.mpm_state, self.mpm_model, dt)
        grid_normalization_and_grativity(self.mpm_state, self.mpm_model, dt)
        g2p(self.mpm_state, self.mpm_model, dt)

