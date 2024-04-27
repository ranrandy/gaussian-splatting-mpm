import taichi as ti
import taichi.math as tm
from mpm_solver.model import *
from mpm_solver.utils import *
from mpm_solver.collider import *


@ti.data_oriented
class MPM_Simulator:
    def __init__(self, xyzs, covs, volumes, args):
        self.n_particles = xyzs.shape[0]
        self.mpm_model = MPM_model(self.n_particles, args)
        self.mpm_state = MPM_state(self.n_particles, xyzs, covs, volumes, args)

        self.time = 0.0

        # Post-processing and boundary conditions
        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []

        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []

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

        collider_param = MPM_Collider()

        collider_param.point = tm.vec3(point[0], point[1], point[2])
        collider_param.normal = tm.vec3(normal[0], normal[1], normal[2])
        collider_param.friction = friction

        self.collider_params.append(collider_param)

        @ti.kernel
        def collide(
            time: float,
            dt: float,
            state: ti.template(),
            model: ti.template(),
            param: ti.template()
        ):
            # print("TEST")
            for grid_x, grid_y, grid_z in state.grid_m:
                offset = tm.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                n = tm.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = tm.dot(offset, n)

                if dotproduct < 0.0:
                    v = state.grid_v_out[grid_x, grid_y, grid_z]
                    normal_component = tm.dot(v, n)
                    v = (
                        v - ti.min(normal_component, 0.0) * n
                    )  # Project out only inward normal component
                    if normal_component < 0.0 and tm.length(v) > 1e-20:
                        v = ti.max(
                            0.0, tm.length(v) + normal_component * param.friction
                        ) * tm.normalize(
                            v
                        )  # apply friction here
                    state.grid_v_out[grid_x, grid_y, grid_z] = tm.vec3(
                        0.0, 0.0, 0.0
                    ) # This line was in the Warp implementation but seems like a mistake. This might make the surface act sticky?

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    def p2g2p_sanity_check(self, dt: ti.f32):
        reset_grid_state(self.mpm_state, self.mpm_model)
        compute_stress_from_F_trial(self.mpm_state, self.mpm_model, dt)
        p2g_apic_with_stress(self.mpm_state, self.mpm_model, dt)
        grid_normalization_and_gravity(self.mpm_state, self.mpm_model, dt)
        for f in range (len(self.grid_postprocess)):
            self.grid_postprocess[f](self.time, dt, self.mpm_state, self.mpm_model, self.collider_params[f])
        g2p(self.mpm_state, self.mpm_model, dt)
        self.time += dt

