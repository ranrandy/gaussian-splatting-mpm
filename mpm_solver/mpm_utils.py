import taichi as ti
import taichi.math as tm
import torch
import numpy as np
from mpm_solver.mpm_model import *

ti.init(arch=ti.cuda)


# Functions used to compute stress from the deformation gradient F
# F: ti.Matrix, U: ti.Matrix, V: ti.Matrix, J: ti.f32, mu: ti.f32, lam: ti.f32
@ti.func  # τ = 2μ(F^E - R)F^(E^T) + λ(J - 1)I_c
def kirchoff_stress_FCR(F: tm.mat3, U: tm.mat3, V: tm.mat3, J: ti.f32, mu: ti.f32, lam: ti.f32):
    # Compute Kirchoff stress for Fixed Corotated model (tau = P F^T) (B.1.)
    R = U @ V.transpose()  # Compute rotation matrix R
    identity_matrix = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return 2.0 * mu * (F - R) @ F.transpose() + identity_matrix * lam * J * (J - 1.0)


@ti.func  # τ = U(2με + λsum(ε)1)V^T
def kirchoff_stress_StVK(F: tm.mat3, U: tm.mat3, V: tm.mat3, sig: tm.vec3, mu: ti.f32, lam: ti.f32):
    # Compute Kirchoff stress for StVK model (tau = 2με + λsum(ε)1) (B.2.)
    sig = ti.Vector([ti.max(sig[0], 0.01), ti.max(sig[1], 0.01), ti.max(sig[2], 0.01)])
    epsilon = ti.Vector([ti.log(sig[0]), ti.log(sig[1]), ti.log(sig[2])])
    log_sig_sum = ti.log(sig[0]) + ti.log(sig[1]) + ti.log(sig[2])
    ONE = ti.Vector([1.0, 1.0, 1.0])
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    tau_mat = ti.Matrix([[tau[0], 0.0, 0.0], [0.0, tau[1], 0.0], [0.0, 0.0, tau[2]]])
    return U @ tau_mat @ V.transpose() @ F.transpose()


@ti.func  # This function is never used in calculating the Kirchhoff stress in original PhysGaussian
def kirchoff_stress_NeoHookean(F: tm.mat3, U: tm.mat3, V: tm.mat3, J: ti.f32, sig: tm.vec3, mu: ti.f32, lam: ti.f32):
    # Compute Kirchoff stress for Neo-Hookean model (tau = P F^T) (B.3.)
    b = ti.Vector([sig[0] ** 2, sig[1] ** 2, sig[2] ** 2])
    mean_b = (b[0] + b[1] + b[2]) / 3.0
    b_hat = b - ti.Vector([mean_b, mean_b, mean_b])
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J**2 - 1.0) * ti.Vector(
        [1.0, 1.0, 1.0]
    )
    tau_matrix = ti.Matrix([[tau[0], 0.0, 0.0], [0.0, tau[1], 0.0], [0.0, 0.0, tau[2]]])
    return U @ tau_matrix @ V.transpose() @ F.transpose()  # τ = μ(F F^T - I) + log(J) I
    # I = ti.Matrix.identity(ti.f32, 3)
    # b = F @ F.transpose()                     # Left Cauchy-Green deformation
    # tau = mu * (b - I) + lam * ti.log(J) * I  # Kirchhoff stress
    # return tau


@ti.func
def kirchoff_stress_Drucker_Prager(
    F: tm.mat3, U: tm.mat3, V: tm.mat3, sig: tm.vec3, mu: ti.f32, lam: ti.f32
):
    # Compute Kirchoff stress for Drucker-Prager model (B.4.)
    log_sig_sum = ti.log(sig[0]) + ti.log(sig[1]) + ti.log(sig[2])
    center00 = 2.0 * mu * ti.log(sig[0]) / sig[0] + lam * log_sig_sum / sig[0]
    center11 = 2.0 * mu * ti.log(sig[1]) / sig[1] + lam * log_sig_sum / sig[1]
    center22 = 2.0 * mu * ti.log(sig[2]) / sig[2] + lam * log_sig_sum / sig[2]
    center = ti.Matrix(
        [[center00, 0.0, 0.0], [0.0, center11, 0.0], [0.0, 0.0, center22]]
    )
    return U @ center @ V.transpose() @ F.transpose()


### P2G: Transfer particle data to grid ###
@ti.kernel
def reset_grid_state(state: ti.template(), model: ti.template()):
    for x, y, z in ti.ndrange(model.grid_dim_x, model.grid_dim_y, model.grid_dim_z):
        state.grid_m[x, y, z] = 0.0
        state.grid_v_in[x, y, z] = ti.Vector([0.0, 0.0, 0.0])
        state.grid_v_out[x, y, z] = ti.Vector([0.0, 0.0, 0.0])


@ti.func
def von_mises_return_mapping(F_trial, model, p):
    # U = ti.Matrix.zero(ti.f32, 3, 3)
    # V = ti.Matrix.zero(ti.f32, 3, 3)
    # sig_old = ti.Vector.zero(ti.f32, 3)
    U, sig_mat, V = ti.svd(F_trial)
    sig_old = tm.vec3(sig_mat[0,0], sig_mat[1,1], sig_mat[2,2])

    sig = ti.Vector(
        [ti.max(sig_old[0], 0.01), ti.max(sig_old[1], 0.01), ti.max(sig_old[2], 0.01)]
    )  # Clamp to prevent NaN cases
    epsilon = ti.Vector([ti.log(sig[0]), ti.log(sig[1]), ti.log(sig[2])])
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    mu = model.mu[p]
    lam = model.lam[p]
    tau = 2.0 * mu * epsilon + lam * (epsilon[0] + epsilon[1] + epsilon[2]) * ti.Vector(
        [1.0, 1.0, 1.0]
    )
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = tau - sum_tau / 3.0

    F = tm.mat3(0.0)
    if cond.norm() > model.yield_stress[p]:
        epsilon_hat = epsilon - temp
        epsilon_hat_norm = epsilon_hat.norm() + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * mu)
        epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
        # sig_elastic = ti.Matrix.diag(ti.Vector([ti.exp(epsilon[0]), ti.exp(epsilon[1]), ti.exp(epsilon[2])]))
        sig_elastic = ti.Matrix(
            [
                [ti.exp(epsilon[0]), 0, 0],
                [0, ti.exp(epsilon[1]), 0],
                [0, 0, ti.exp(epsilon[2])],
            ]
        )
        F_elastic = U @ sig_elastic @ V.transpose()

        if model.hardening == 1:
            model.yield_stress[p] += 2.0 * mu * model.xi * delta_gamma

        F = F_elastic
        # return F_elastic
    else:
        # return F_trial
        F = F_trial
    return F
    
@ti.func
def sand_return_mapping(F_trial, state, model, p):
    # U = tm.mat3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # V = tm.mat3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # sig = ti.vec3(0.0)
    # wp.svd3(F_trial, U, sig, V)
    U, sig_mat, V = ti.svd(F_trial)
    sig = tm.vec3(sig_mat[0,0], sig_mat[1,1], sig_mat[2,2])

    epsilon = tm.vec3(
        ti.log(ti.max(ti.abs(sig[0]), 1e-14)),
        ti.log(ti.max(ti.abs(sig[1]), 1e-14)),
        ti.log(ti.max(ti.abs(sig[2]), 1e-14)),
    )
    sigma_out = tm.mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - tm.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = tm.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[p] + 2.0 * model.mu[p])
        / (2.0 * model.mu[p])
        * tr
        * model.alpha
    )

    F_elastic = tm.mat3(0.0)
    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * V.transpose()

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        # s_new = tm.vec3(tm.exp(H[0]), tm.exp(H[1]), tm.exp(H[2]))
        s_new = tm.exp(H)

        F_elastic = U * tm.mat3(s_new[0], 0.0, 0.0, 0.0, s_new[1], 0.0, 0.0, 0.0, s_new[2]) * V.transpose()
    return F_elastic

# for toothpaste
@ti.func
def viscoplasticity_return_mapping_with_StVK(F_trial, model, p, dt):
    # U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # sig_old = wp.vec3(0.0)
    # wp.svd3(F_trial, U, sig_old, V)
    U, sig_mat, V = ti.svd(F_trial)
    sig_old = tm.vec3(sig_mat[0,0], sig_mat[1,1], sig_mat[2,2])

    # sig = tm.vec3(
    #     ti.max(sig_old[0], 0.01), ti.max(sig_old[1], 0.01), ti.max(sig_old[2], 0.01)
    # )  # add this to prevent NaN in extrem cases
    sig = ti.max(sig_old, 0.01)

    b_trial = tm.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    # epsilon = tm.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    epsilon = ti.log(sig)
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    # epsilon_hat = epsilon - wp.vec3(
    #     trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    # )
    epsilon_hat = epsilon - tm.vec3(trace_epsilon / 3.0)
    s_trial = 2.0 * model.mu[p] * epsilon_hat
    s_trial_norm = tm.length(s_trial)
    y = s_trial_norm - ti.sqrt(2.0 / 3.0) * model.yield_stress[p]
    F = tm.mat3(0.0)
    if y > 0:
        mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + model.plastic_viscosity / (2.0 * mu_hat * dt)
        )
        s_new = (s_new_norm / s_trial_norm) * s_trial
        epsilon_new = 1.0 / (2.0 * model.mu[p]) * s_new + tm.vec3(trace_epsilon / 3.0)
        sig_elastic = tm.mat3(
            tm.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            tm.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            tm.exp(epsilon_new[2]),
        )
        F = U * sig_elastic * V.transpose()
    else:
        F = F_trial
    return F
    
@ti.func
def von_mises_return_mapping_with_damage(F_trial, model, p):
    # U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # sig_old = wp.vec3(0.0)
    # wp.svd3(F_trial, U, sig_old, V)
    U, sig_mat, V = ti.svd(F_trial)
    sig_old = tm.vec3(sig_mat[0,0], sig_mat[1,1], sig_mat[2,2])

    # sig = wp.vec3(
    #     wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    # )  # add this to prevent NaN in extrem cases
    sig = ti.max(sig_old, 0.01)
    # epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    epsilon = ti.log(sig)
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * tm.vec3(1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = tm.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    F = tm.mat3(0.0)
    if tm.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            F = F_trial
        else:
            epsilon_hat = epsilon - tm.vec3(temp)
            epsilon_hat_norm = tm.length(epsilon_hat) + 1e-6
            delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
            epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
            model.yield_stress[p] = model.yield_stress[p] - model.softening * tm.length(
                (delta_gamma / epsilon_hat_norm) * epsilon_hat
            )
            if model.yield_stress[p] <= 0:
                model.mu[p] = 0.0
                model.lam[p] = 0.0
            sig_elastic = tm.mat3(
                tm.exp(epsilon[0]),
                0.0,
                0.0,
                0.0,
                tm.exp(epsilon[1]),
                0.0,
                0.0,
                0.0,
                tm.exp(epsilon[2]),
            )
            F = U * sig_elastic * V.transpose()
            if model.hardening == 1:
                model.yield_stress[p] = (
                    model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
                )
    else:
        F = F_trial
    return F

@ti.kernel  ##### TODO: calculate the deformation gradients.
def compute_stress_from_F_trial(state: ti.template(), model: ti.template(), dt: ti.f32):
    for p in range(model.n_particles):
        # if state.particle_selection[p] == 0: # What is particle selection here?
        if model.material == 1:  # metal ??
            state.particle_F[p] = von_mises_return_mapping(
                state.particle_F_trial[p], model, p
            )
        elif model.material == 2:  # sand  ??
            state.particle_F[p] = sand_return_mapping(
                state.particle_F_trial[p], state, model, p
            )
        elif model.material == 3:  # visplas, with StVk + VM, no thickening ??
            state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
                state.particle_F_trial[p], model, p, dt
            )
        elif model.material == 5:
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p
            )
        else:  # elastic ??
            state.particle_F[p] = state.particle_F_trial[p]

        J = (state.particle_F[p]).determinant()
        # U = ti.Matrix.zero(ti.f32, 3, 3)
        # V = ti.Matrix.zero(ti.f32, 3, 3)
        # sig = ti.Vector.zero(ti.f32, 3)
        stress = ti.Matrix.zero(ti.f32, 3, 3)
        U, sig_mat, V = ti.svd(state.particle_F[p])
        sig = tm.vec3(sig_mat[0,0], sig_mat[1,1], sig_mat[2,2])

        if model.material == 0:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
        elif model.material == 1:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        elif model.material == 2:
            stress = kirchoff_stress_Drucker_Prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        elif model.material == 3:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )

        stress = (stress + stress.transpose()) / 2.0
        state.particle_stress[p] = stress


@ti.func # model: MPM_state, w: tm.mat3, dw: tm.mat3, i: ti.int32, j: ti.int32, k: ti.int32
def compute_dweight(model, w, dw, i, j, k):
    dweight = ti.Vector(
        [
            dw[0, i] * w[1, j] * w[2, k],
            w[0, i] * dw[1, j] * w[2, k],
            w[0, i] * w[1, j] * dw[2, k],
        ]
    )
    # dweight = tm.vec3(dw[0, i] * w[1, j] * w[2, k],
    #                   w[0, i] * dw[1, j] * w[2, k],
    #                   w[0, i] * w[1, j] * dw[2, k])
    return dweight * model.inv_dx


@ti.kernel
def p2g_apic_with_stress(state: ti.template(), model: ti.template(), dt: ti.f32):
    for p in range(model.n_particles):
        # if state.particle_selection[p] == 0:
            stress = state.particle_stress[p]
            grid_pos = state.particle_x[p] * model.inv_dx
            base_pos = (grid_pos - 0.5).cast(
                int
            )  # Corner of the grid cell, subtracting 0.5 to get to the bottom-left
            fx = grid_pos - base_pos.cast(float)  # Distance from the base pos

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
            )  # A matrix of weights for the nodes surrounding the particle
            dw = ti.Matrix(
                [
                    [fx[0] - 1.5, -2.0 * (fx[0] - 1.0), fx[0] - 0.5],
                    [fx[1] - 1.5, -2.0 * (fx[1] - 1.0), fx[1] - 0.5],
                    [fx[2] - 1.5, -2.0 * (fx[2] - 1.0), fx[2] - 0.5],
                ]
            )  # Derivatives of the weight function

            for offset in ti.static(
                ti.grouped(ti.ndrange(3, 3, 3))
            ):  ##### TODO: I am not sure whether we can use this kind of form to accelerate a triple layer for loop
                dpos = (
                    offset.cast(float) - fx
                ) * model.dx  # (x_i - x_p^n), position diff from the particle to the grid node
                ix, iy, iz = base_pos + offset
                weight = w[0, offset.x] * w[1, offset.y] * w[2, offset.z]
                dweight = compute_dweight(model, w, dw, offset.x, offset.y, offset.z)
                C = state.particle_C[p]
                C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                    C - C.transpose()
                )  # Update C_p^n matrix based on a damping factor

                if model.rpic_damping < -0.001:      # !!
                    C = ti.Matrix.zero(ti.f32, 3, 3) # !!
                # C = ti.Matrix.zero(ti.f32, 3, 3)

                # m_i^n       = Σ_p w_ip^n m_p,
                # m_i^n v_i^n = Σ_p w_ip^n m_p (v_p^n + C_p^n (x_i - x_p^n))
                # print(state.particle_vol[p])
                # print(stress)
                # print(dweight)
                elastic_force = -state.particle_vol[p] * stress @ dweight # -state.particle_vol[p] * stress * dweight
                v_in_add = (
                    weight * state.particle_mass[p] * (state.particle_v[p] + C @ dpos)
                    + dt * elastic_force             # !!
                )
                ti.atomic_add(state.grid_v_in[ix, iy, iz], v_in_add)
                ti.atomic_add(state.grid_m[ix, iy, iz], weight * state.particle_mass[p])


### Grid Operations ###
@ti.kernel
def grid_normalization_and_gravity(
    state: ti.template(), model: ti.template(), dt: ti.f32
):
    for grid_x, grid_y, grid_z in ti.ndrange(
        model.grid_dim_x, model.grid_dim_y, model.grid_dim_z
    ):
        if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
            v_out = (
                state.grid_v_in[grid_x, grid_y, grid_z]
                / state.grid_m[grid_x, grid_y, grid_z]
            )
            v_out += dt * model.gravitational_acceleration
            state.grid_v_out[grid_x, grid_y, grid_z] = v_out
            # state.grid_v_out[grid_x, grid_y, grid_z] = tm.vec3(0.0, 0.0, -9.8)


@ti.kernel
def add_damping_via_grid(state: ti.template(), scale: ti.f32):
    for grid_xyz in ti.grouped(
        state.grid_v_out
    ):  ##### TODO: I am not sure whether we can use this kind of implementaion for a triple
        state.grid_v_out[grid_xyz] *= scale


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
            grid_pos = state.particle_x[p] * model.inv_dx
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

            state.particle_v[p] = new_v  # v_p^(n+1) = sum_i(v_i^(n+1) * w_ip^n)
            state.particle_x[p] += (
                dt * new_v
            )  # x_p^(n+1) = x_p^n + (Delta_t * v_p^(n+1))
            state.particle_C[p] = (
                new_C
            )  # C_p^(n+1) = (1 / (Delta_x^2 * (b + 1))) * sum_i(w_ip^n * v_i^(n+1) * (x_i^n - x_p^n)^T)
            I = ti.Matrix.identity(ti.f32, 3)
            F_tmp = (I + new_F * dt) * state.particle_F[
                p
            ]  # F_E_tr_p = (I + nabla_v_p^(n+1)) * F_E_n_p
            state.particle_F_trial[p] = F_tmp  # F_E_n+1_p = Z(F_E_tr_p)

            if model.update_cov_with_F:
                update_cov(state, p, new_F, dt)


# sequence:
# zero_grid/reset_grid_state, (!pre_p2g_operations, !particle_velocity_modifiers), compute_stress_from_F_trial
# p2g_apic_with_stress, grid_normalization_and_gravity, !add_damping_via_grid, (!grid_postprocess), g2p


# @ti.func
def set_vec3_to_zero(target_array):
    for i in range(target_array.shape[0]):
        target_array[i] = ti.Vector([0.0, 0.0, 0.0])


# @ti.func
def set_mat33_to_identity(target_array):
    for i in range(target_array.shape[0]):
        target_array[i] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


@ti.kernel
def set_value_to_float_array(target_array: ti.template(), value: ti.f32):
    for i in target_array:
        target_array[i] = value


@ti.kernel
def get_float_array_product(arrayA: ti.template(), arrayB: ti.template(), arrayC: ti.template()):
    for i in arrayA:
        arrayC[i] = arrayA[i] * arrayB[i]


@ti.kernel
def apply_additional_params(state: ti.template(), model: ti.template(), params_modifier: ti.template()):
    for i in range(state.particle_x.shape[0]):
        pos = state.particle_x[i]
        if (
            pos[0] > params_modifier["point"][0] - params_modifier["size"][0]
            and pos[0] < params_modifier["point"][0] + params_modifier["size"][0]
            and pos[1] > params_modifier["point"][1] - params_modifier["size"][1]
            and pos[1] < params_modifier["point"][1] + params_modifier["size"][1]
            and pos[2] > params_modifier["point"][2] - params_modifier["size"][2]
            and pos[2] < params_modifier["point"][2] + params_modifier["size"][2]
        ):
            model.E[i] = params_modifier["E"]
            model.nu[i] = params_modifier["nu"]
            state.particle_density[i] = params_modifier["density"]


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


@ti.func
def multiply_and_update_density_mass(particle_density, particle_vol, particle_mass):
    for i in range(particle_density.shape[0]):
        particle_mass[i] = particle_density[i] * particle_vol[i]


@ti.kernel
def compute_R_from_F(state: ti.template(), model: ti.template()):
    for p in range(model.n_particles):
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


def taichi_to_torch(ti_field):
    field_np = ti_field.to_numpy()
    return torch.tensor(field_np)


def taichi_field_to_torch(ti_field, n_particles):
    # Create a NumPy array to hold the data
    np_array = np.zeros((n_particles, 6), dtype=np.float32)

    # Transfer data from Taichi to NumPy
    for i in range(n_particles):
        for j in range(6):
            np_array[i, j] = ti_field[6 * i + j]

    return torch.from_numpy(np_array)
