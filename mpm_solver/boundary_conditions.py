import taichi as ti
from arguments import *
from utils.transform_utils import *


@ti.data_oriented
class BasicBC:
    def __init__(self, n_particles, bc_args, sim_args : MPMParams):
        self.n_particles = n_particles
        self.substep_dt = sim_args.substep_dt

        self.id = bc_args["id"]
        self.type = bc_args["type"]
        
        self.start_time = bc_args["start_time"]
        self.end_time = bc_args["start_time"] + sim_args.substep_dt * bc_args["num_dt"]

        self.center = bc_args["center"]
        self.size = bc_args["size"]

        self.isCollide = False

    @ti.kernel
    def apply(self, state : ti.template(), dx : float):
        for grid_xyz in ti.grouped(state.grid_v_out):
            if all(ti.abs(grid_xyz * dx - self.center) < self.size):
                state.grid_v_out[grid_xyz] = ti.Vector([0.0, 0.0, 0.0])
        

    def isActive(self, time):
        return time >= self.start_time and time < self.end_time


@ti.data_oriented
class ImpulseBC(BasicBC):
    def __init__(self, n_particles, bc_args, sim_args):
        self.force = ti.Vector([bc_args["force"][0], bc_args["force"][1], bc_args["force"][2]])

        super().__init__(n_particles, bc_args, sim_args)

    @ti.kernel
    def apply(self, state : ti.template()):
        for p in range(self.n_particles):
            if all(ti.abs(state.particle_xyz[p] - self.center) < self.size): #TODO: Need to replace this with a mask
                state.particle_vel[p] = state.particle_vel[p] + self.force / state.particle_mass[p] * self.substep_dt

@ti.data_oriented
class MaterialParamsModifier(BasicBC):
    def __init__(self, n_particles, bc_args, sim_args):
        self.mu = bc_args["mu"]
        self.density = bc_args["density"]
        self.E = bc_args["E"]
        self.nu = bc_args["nu"]
        self.isMaterial = False
        super().__init__(n_particles, bc_args, sim_args)

    @ti.kernel
    def apply(self, state : ti.template(), model : ti.template()):
        for p in range(self.n_particles):
            if all(ti.abs(state.particle_xyz[p] - self.center) < self.size): #TODO: Need to replace this with a mask
                model.nu[p] = self.nu
                model.E[p] = self.E
                state.particle_density[p] = self.density

    @ti.kernel
    def applymu(self, state : ti.template(), model : ti.template()):
        for p in range(self.n_particles):
            if all(ti.abs(state.particle_xyz[p] - self.center) < self.size): #TODO: Need to replace this with a mask
                if self.mu != 1000:
                    model.mu[p] = self.mu

@ti.data_oriented
class MaterialTypeModifier(BasicBC):
    def __init__(self, n_particles, bc_args, sim_args):
        self.material = bc_args["material"]
        self.isMaterial = True


        super().__init__(n_particles, bc_args, sim_args)

    @ti.kernel
    def apply(self, state : ti.template(), model : ti.template()):
        for p in range(self.n_particles):
            if all(ti.abs(state.particle_xyz[p] - self.center) < self.size): #TODO: Need to replace this with a mask
                model.material[p] = self.material

@ti.data_oriented
class StickyGroundBC(BasicBC):
    def __init__(self):
        self.center = [1.0, 0.6, 1.0]
        self.size = [1.0, 0.1, 1.0]

    def isActive(self, time):
        return True


preprocess_bc = (
    "impulse"
)

postprocess_bc = (
    "fixed_cube",
    "sticky_ground"
)

init_bc = (
    "additional_params",
    "modify_material"
)

boundaryConditionTypeCallBacks = {
    "fixed_cube": BasicBC,
    "impulse" : ImpulseBC,
    "sticky_ground" : StickyGroundBC,
    "additional_params" : MaterialParamsModifier,
    "modify_material": MaterialTypeModifier
}
