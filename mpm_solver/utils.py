from numpy import dtype
import taichi as ti
from mpm_solver.constitutive_models import *
import taichi.math as tm

material_types = {
    "jelly": 0,
    "metal": 1,
    "sand": 2,
    "foam": 3,
    "snow": 4,
    "plasticine": 5,
}


@ti.kernel  ##### TODO: calculate the deformation gradients.
def compute_stress_from_F_trial(state : ti.template(), model : ti.template(), dt: ti.f32):
    for p in range(model.n_particles):

        if model.material[p] == 1:    # metal
            state.particle_F[p] = von_mises_return_mapping(
                state.particle_F_trial[p], model, p
            )
        elif model.material[p] == 2:  # sand
            state.particle_F[p] = sand_return_mapping(
                state.particle_F_trial[p], model, p
            )
        elif model.material == 3:     # visplas, with StVk + VM, no thickening ??
            state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
                state.particle_F_trial[p], model, p, dt
            )
        elif model.material == 5:     # plasticine
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p
            )
        else:
            state.particle_F[p] = state.particle_F_trial[p]

        J = state.particle_F[p].determinant()
        stress = ti.Matrix.zero(ti.f32, 3, 3)
        U, S, V = ti.svd(state.particle_F[p])

        pressure = 0.0

        if model.material[p] == 0 :   
            stress = kirchoff_stress_FCR(state.particle_F[p], U, V, J, model.mu[p], model.lam[p])
        elif model.material[p] == 1:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, S, model.mu[p], model.lam[p]
            )
        elif model.material[p] == 2: # sand
            stress = kirchoff_stress_Drucker_Prager(
                state.particle_F[p], U, V, S, model.mu[p], model.lam[p]
            )
        elif model.material == 3:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, S, model.mu[p], model.lam[p]
            )


        stress = (stress + stress.transpose()) / 2.0 #TODO: We can do an ablation study for this
        state.particle_stress[p] = stress
        state.particle_pressure[p] = pressure

        # if p == 0:
        #     print(stress)
        #     print(state.particle_stress[p])


@ti.func # model: MPM_state, w: tm.mat3, dw: tm.mat3, i: ti.int32, j: ti.int32, k: ti.int32
def compute_dweight(model, w, dw, i, j, k):
    dweight = ti.Vector([
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
        ])
    return dweight * model.inv_dx


@ti.kernel
def p2g(state : ti.template(), model : ti.template(), dt: ti.f32):
    for p in range(model.n_particles):
        stress = state.particle_stress[p]
        pressure = state.particle_pressure[p]
        grid_pos = state.particle_xyz[p] * model.inv_dx
        base_pos = (grid_pos - 0.5).cast(int) # Corner of the grid cell, subtracting 0.5 to get to the bottom-left
        fx = grid_pos - base_pos.cast(float)  # Distance from the base pos

        # Equation (123) Quadratic spline kernel in [3]. 
        # Explanation: https://forum.taichi-lang.cn/t/how-the-mls-mpm-quadratic-kernel-implementation-works/1966
        wa, wb, wc = 1.5 - fx, fx - 1.0, fx - 0.5
        w = ti.Matrix([
            [wa[0] * wa[0] * 0.5, 0.75 - wb[0] * wb[0], wc[0] * wc[0] * 0.5],
            [wa[1] * wa[1] * 0.5, 0.75 - wb[1] * wb[1], wc[1] * wc[1] * 0.5],
            [wa[2] * wa[2] * 0.5, 0.75 - wb[2] * wb[2], wc[2] * wc[2] * 0.5]
        ])  # A matrix of weights for the nodes surrounding the particle
        dw = ti.Matrix([
            [fx[0] - 1.5, -2.0 * (fx[0] - 1.0), fx[0] - 0.5],
            [fx[1] - 1.5, -2.0 * (fx[1] - 1.0), fx[1] - 0.5],
            [fx[2] - 1.5, -2.0 * (fx[2] - 1.0), fx[2] - 0.5],
        ])  # Derivatives of the weight function

        for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):  
            dpos = (offset.cast(float) - fx) * model.dx  # (x_i - x_p^n), position diff from the particle to the grid node
            ix, iy, iz = base_pos + offset
            weight = w[0, offset.x] * w[1, offset.y] * w[2, offset.z]
            dweight = compute_dweight(model, w, dw, offset.x, offset.y, offset.z)
            C = state.particle_C[p]

            # m_i^n       = Σ_p w_ip^n m_p,
            # m_i^n v_i^n = Σ_p w_ip^n m_p (v_p^n + C_p^n (x_i - x_p^n))
            elastic_force = -state.particle_vol[p] * stress @ dweight
            fluid_pressure = state.particle_vol[p] * pressure * dweight
            v_in_add = (
                weight * state.particle_mass[p] * (state.particle_vel[p] + C @ dpos) # v_i^n
                + dt * elastic_force
            )
            if model.material == 6:
                v_in_add = (
                weight * state.particle_mass[p] * (state.particle_vel[p] + C @ dpos) # v_i^n
                + dt * fluid_pressure
            )
            ti.atomic_add(state.grid_v_in[ix, iy, iz], v_in_add)
            ti.atomic_add(state.grid_mass[ix, iy, iz], weight * state.particle_mass[p])


@ti.kernel
def grid_normalization_and_gravity(state: ti.template(), model: ti.template(), dt: ti.f32):
    for grid_x, grid_y, grid_z in ti.ndrange(model.n_grid, model.n_grid, model.n_grid):
        if state.grid_mass[grid_x, grid_y, grid_z] > 1e-15:
            v_out = state.grid_v_in[grid_x, grid_y, grid_z] / state.grid_mass[grid_x, grid_y, grid_z]
            v_out += dt * model.gravity
            state.grid_v_out[grid_x, grid_y, grid_z] = v_out


# @ti.kernel
# def add_damping_via_grid(state: ti.template(), scale: ti.f32):
#     for grid_xyz in ti.grouped(
#         state.grid_v_out
#     ):  ##### TODO: I am not sure whether we can use this kind of implementaion for a triple
#         state.grid_v_out[grid_xyz] *= scale


@ti.func
def update_cov(state, p, grad_v, dt):
    cov_n = ti.Matrix(
        [
            [
                state.particle_cov[p * 6],
                state.particle_cov[p * 6 + 1],
                state.particle_cov[p * 6 + 2],
            ],
            [
                state.particle_cov[p * 6 + 1],
                state.particle_cov[p * 6 + 3],
                state.particle_cov[p * 6 + 4],
            ],
            [
                state.particle_cov[p * 6 + 2],
                state.particle_cov[p * 6 + 4],
                state.particle_cov[p * 6 + 5],
            ],
        ]
    )

    cov_np1 = cov_n + dt * (grad_v @ cov_n + cov_n @ grad_v.transpose())

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


### G2P: Transfer grid data back to particles ###
@ti.kernel
def g2p(state: ti.template(), model: ti.template(), dt: ti.f32):
    for p in range(model.n_particles):
        # if state.particle_selection[p] == 0:
        grid_pos = state.particle_xyz[p] * model.inv_dx
        base_pos = (grid_pos - 0.5).cast(int)
        fx = grid_pos - base_pos.cast(float)

        wa = 1.5 - fx
        wb = fx - 1.0
        wc = fx - 0.5
        w = ti.Matrix(
            [
                [
                    wa[0] * wa[0] * 0.5,
                    0.75 - wb[0] * wb[0],
                    wc[0] * wc[0] * 0.5,
                ],
                [
                    wa[1] * wa[1] * 0.5,
                    0.75 - wb[1] * wb[1],
                    wc[1] * wc[1] * 0.5,
                ],
                [
                    wa[2] * wa[2] * 0.5,
                    0.75 - wb[2] * wb[2],
                    wc[2] * wc[2] * 0.5,
                ],
            ]
        )
        dw = ti.Matrix(
            [
                [fx[0] - 1.5, -2.0 * (fx[0] - 1.0), fx[0] - 0.5],
                [fx[1] - 1.5, -2.0 * (fx[1] - 1.0), fx[1] - 0.5],
                [fx[2] - 1.5, -2.0 * (fx[2] - 1.0), fx[2] - 0.5],
            ]
        )

        new_v = ti.Vector([0.0, 0.0, 0.0])  # Velocity increment
        new_C = ti.Matrix.zero(ti.f32, 3, 3)  # APIC C matrix increment
        new_F = ti.Matrix.zero(ti.f32, 3, 3)  # Deformation gradient increment

        for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
            dpos = offset.cast(float) - fx
            ix, iy, iz = base_pos + offset
            weight = w[0, offset.x] * w[1, offset.y] * w[2, offset.z]
            grid_v = state.grid_v_out[ix, iy, iz]
            new_v += grid_v * weight
            new_C += grid_v.outer_product(dpos) * (weight * model.inv_dx * 4.0)
            dweight = compute_dweight(model, w, dw, offset.x, offset.y, offset.z)
            new_F += grid_v.outer_product(dweight)

        state.particle_vel[p] = new_v  # v_p^(n+1) = sum_i(v_i^(n+1) * w_ip^n)
        state.particle_xyz[p] += (
            dt * new_v
        )  # x_p^(n+1) = x_p^n + (Delta_t * v_p^(n+1))
        state.particle_C[p] = (
            new_C
        )  # C_p^(n+1) = (1 / (Delta_x^2 * (b + 1))) * sum_i(w_ip^n * v_i^(n+1) * (x_i^n - x_p^n)^T)
        I = ti.Matrix.identity(ti.f32, 3)
        F_tmp = (I + new_F * dt) @ state.particle_F[
            p
        ]  # F_E_tr_p = (I + nabla_v_p^(n+1)) * F_E_n_p
        state.particle_F_trial[p] = F_tmp  # F_E_n+1_p = Z(F_E_tr_p)

        update_cov(state, p, new_F, dt)


# sequence:
# zero_grid/reset_grid_state, (!pre_p2g_operations, !particle_velocity_modifiers), compute_stress_from_F_trial
# p2g_apic_with_stress, grid_normalization_and_gravity, !add_damping_via_grid, (!grid_postprocess), g2p

@ti.kernel
def compute_mu_lam_from_E_nu(
    n_particles: ti.int32,
    E: ti.template(),
    nu: ti.template(),
    mu: ti.template(),
    lam: ti.template(),
):
    for p in range(n_particles):
        mu[p] = E[p] / (2.0 * (1.0 + nu[p]))
        lam[p] = E[p] * nu[p] / ((1.0 + nu[p]) * (1.0 - 2.0 * nu[p]))


@ti.kernel
def compute_mass_from_vol_density(
    n_particles: ti.int32,
    density: ti.template(), 
    vol: ti.template(), 
    mass: ti.template()
):
    for i in range(n_particles):
        mass[i] = density[i] * vol[i]


@ti.kernel
def compute_R_from_F(state: ti.template(), model: ti.template()):
    for p in range(model.n_particles):
        # F = state.particle_F_trial[p]
        F = state.particle_F_trial[p]
        # U = ti.Matrix.zero(ti.f32, 3, 3)
        # V = ti.Matrix.zero(ti.f32, 3, 3)
        # sig = ti.Vector.zero(ti.f32, 3)
        # wp.svd3(F, U, sig, V)
        U, sig, V = ti.svd(F)

        if U.determinant() < 0:
            U[0, 2] = -U[0, 2]
            U[1, 2] = -U[1, 2]
            U[2, 2] = -U[2, 2]

        if V.determinant() < 0:
            V[0, 2] = -V[0, 2]
            V[1, 2] = -V[1, 2]
            V[2, 2] = -V[2, 2]

        R = U @ V.transpose()
        state.particle_R[p] = R.transpose()


@ti.kernel
def compute_cov_from_F(state: ti.template(), model: ti.template()):
    for p in range(model.n_particles):
        F = state.particle_F_trial[p]

        init_cov = ti.Matrix(
            [
                [
                    state.particle_init_cov[6 * p],
                    state.particle_init_cov[6 * p + 1],
                    state.particle_init_cov[6 * p + 2],
                ],
                [
                    state.particle_init_cov[6 * p + 1],
                    state.particle_init_cov[6 * p + 3],
                    state.particle_init_cov[6 * p + 4],
                ],
                [
                    state.particle_init_cov[6 * p + 2],
                    state.particle_init_cov[6 * p + 4],
                    state.particle_init_cov[6 * p + 5],
                ],
            ]
        )

        cov = F @ init_cov @ F.transpose()

        state.particle_cov[6 * p] = cov[0, 0]
        state.particle_cov[6 * p + 1] = cov[0, 1]
        state.particle_cov[6 * p + 2] = cov[0, 2]
        state.particle_cov[6 * p + 3] = cov[1, 1]
        state.particle_cov[6 * p + 4] = cov[1, 2]
        state.particle_cov[6 * p + 5] = cov[2, 2]

