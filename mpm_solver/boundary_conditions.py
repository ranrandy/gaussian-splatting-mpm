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
        self.end_time = bc_args["start_time"] + sim_args.frame_dt * bc_args["num_dt"]

        self.center = bc_args["center"]
        self.size = bc_args["size"]

    @ti.kernel
    def apply(self, state : ti.template()):
        for p in range(self.n_particles):
            if all(ti.abs(state.particle_xyz[p] - self.center) < self.size):
                state.particle_vel[p] = ti.Vector([0.0, 0.0, 0.0])
        

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
            if all(ti.abs(state.particle_xyz[p] - self.center) < self.size):
                state.particle_vel[p] = state.particle_vel[p] + self.force / state.particle_mass[p] * self.substep_dt


preprocess_bc = (
    "impulse"
)

postprocess_bc = (
    "fixed_cube"
)

boundaryConditionTypeCallBacks = {
    "fixed_cube": BasicBC,
    "impulse" : ImpulseBC
}