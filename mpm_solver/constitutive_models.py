import taichi as ti
import taichi.math as tm


'''
    Appendix B-1: 
        Fixed Corotated Elasticity
        tau = 2\mu(F^E - R)(F^E)^T + \lambda(J - 1)J
'''
@ti.func  
def kirchoff_stress_FCR(
    F: ti.template(), 
    U: ti.template(), 
    V: ti.template(), 
    J: ti.f32, 
    mu: ti.f32, 
    lam: ti.f32
):
    R = U @ V.transpose()
    return 2.0 * mu * (F - R) @ F.transpose() + ti.Matrix.identity(ti.f32, 3) * lam * J * (J - 1.0)


@ti.func  # τ = U(2με + λsum(ε)1)V^T
def kirchoff_stress_StVK(
    F: ti.template(), 
    U: ti.template(), 
    V: ti.template(),  
    sig: ti.template(),
    mu: ti.f32, 
    lam: ti.f32):
    # Compute Kirchoff stress for StVK model (tau = 2με + λsum(ε)1) (B.2.)
    sig_vec = ti.Vector([ti.max(sig[0, 0], 0.01), ti.max(sig[1, 1], 0.01), ti.max(sig[2, 2], 0.01)])
    epsilon = ti.Vector([ti.log(sig_vec[0]), ti.log(sig_vec[1]), ti.log(sig_vec[2])])
    log_sig_sum = ti.log(sig_vec[0]) + ti.log(sig_vec[1]) + ti.log(sig_vec[2])
    ONE = ti.Vector([1.0, 1.0, 1.0])
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    tau_mat = ti.Matrix([[tau[0], 0.0, 0.0], [0.0, tau[1], 0.0], [0.0, 0.0, tau[2]]])
    return U @ tau_mat @ V.transpose() @ F.transpose()


# @ti.func  # This function is never used in calculating the Kirchhoff stress in original PhysGaussian
# def kirchoff_stress_NeoHookean(F: tm.mat3, U: tm.mat3, V: tm.mat3, J: ti.f32, sig: tm.vec3, mu: ti.f32, lam: ti.f32):
#     # Compute Kirchoff stress for Neo-Hookean model (tau = P F^T) (B.3.)
#     b = ti.Vector([sig[0] ** 2, sig[1] ** 2, sig[2] ** 2])
#     mean_b = (b[0] + b[1] + b[2]) / 3.0
#     b_hat = b - ti.Vector([mean_b, mean_b, mean_b])
#     tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J**2 - 1.0) * ti.Vector(
#         [1.0, 1.0, 1.0]
#     )
#     tau_matrix = ti.Matrix([[tau[0], 0.0, 0.0], [0.0, tau[1], 0.0], [0.0, 0.0, tau[2]]])
#     return U @ tau_matrix @ V.transpose() @ F.transpose()  # τ = μ(F F^T - I) + log(J) I
#     # I = ti.Matrix.identity(ti.f32, 3)
#     # b = F @ F.transpose()                     # Left Cauchy-Green deformation
#     # tau = mu * (b - I) + lam * ti.log(J) * I  # Kirchhoff stress
#     # return tau


@ti.func
def kirchoff_stress_Drucker_Prager(
    F: ti.template(), 
    U: ti.template(), 
    V: ti.template(),  
    sig: ti.template(), 
    mu: ti.f32, 
    lam: ti.f32
):
    # Compute Kirchoff stress for Drucker-Prager model (B.4.)
    log_sig_sum = ti.log(sig[0, 0]) + ti.log(sig[1, 1]) + ti.log(sig[2, 2])
    center00 = 2.0 * mu * ti.log(sig[0, 0]) / sig[0, 0] + lam * log_sig_sum / sig[0, 0]
    center11 = 2.0 * mu * ti.log(sig[1, 1]) / sig[1, 1] + lam * log_sig_sum / sig[1, 1]
    center22 = 2.0 * mu * ti.log(sig[2, 2]) / sig[2, 2] + lam * log_sig_sum / sig[2, 2]
    center = ti.Matrix(
        [[center00, 0.0, 0.0], [0.0, center11, 0.0], [0.0, 0.0, center22]]
    )
    return U @ center @ V.transpose() @ F.transpose()


### P2G: Transfer particle data to grid ###
@ti.func
def von_mises_return_mapping(F_trial, model, p):
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
    else:
        F = F_trial
    return F
    
@ti.func
def sand_return_mapping(F_trial, model, p):
    U, sig_mat, V = ti.svd(F_trial)
    sig = tm.vec3(sig_mat[0,0], sig_mat[1,1], sig_mat[2,2])

    epsilon = tm.vec3(
        ti.log(ti.max(ti.abs(sig[0]), 1e-14)),
        ti.log(ti.max(ti.abs(sig[1]), 1e-14)),
        ti.log(ti.max(ti.abs(sig[2]), 1e-14)),
    )

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
        F_elastic = U @ V.transpose()

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        # s_new = tm.vec3(tm.exp(H[0]), tm.exp(H[1]), tm.exp(H[2]))
        s_new = tm.exp(H)

        F_elastic = U @ tm.mat3(s_new[0], 0.0, 0.0, 0.0, s_new[1], 0.0, 0.0, 0.0, s_new[2]) @ V.transpose()
    return F_elastic

@ti.func
def fluid_return_mapping(F_trial, model, p, dt):
    """
    Perform return mapping for a cohesive fluid-like material to improve continuity between particles.
    
    Args:
        F_trial (ti.Matrix): Trial deformation gradient matrix.
        state: Object holding state variables of the particles or elements.
        model: Object containing material model parameters.
        p (int): Index identifying the current particle/element.
        dt (float): Time step size.
    
    Returns:
        F_cohesive (ti.Matrix): Modified deformation gradient matrix representing cohesive fluid-like behavior.
    """
    # Perform SVD decomposition
    U, sig_mat, V = ti.svd(F_trial)
    sig = tm.vec3(sig_mat[0, 0], sig_mat[1, 1], sig_mat[2, 2])

    # Logarithmic strains (soft yielding)
    epsilon = tm.vec3(
        ti.log(ti.max(ti.abs(sig[0]), 0.01)),
        ti.log(ti.max(ti.abs(sig[1]), 0.01)),
        ti.log(ti.max(ti.abs(sig[2]), 0.01))
    )
    tr = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - tm.vec3(tr / 3.0)

    # Trial deviatoric stress
    s_trial = 2.0 * model.mu[p] * epsilon_hat
    s_trial_norm = tm.length(s_trial)

    # Gradual yield threshold (smoother transition)
    yield_stress = model.yield_stress[p]
    yield_value = s_trial_norm - ti.sqrt(2.0 / 3.0) * yield_stress

    # Initialize the cohesive return matrix
    F_cohesive = tm.mat3(0.0)

    # Check for yielding
    if yield_value > 0:
        # Effective shear modulus with increased viscosity
        mu_hat = model.mu[p] * (sig[0] ** 2 + sig[1] ** 2 + sig[2] ** 2) / 3.0
        plastic_factor = 1.0 + model.plastic_viscosity / (2.0 * mu_hat * dt)

        # Reduced deviatoric stress with smoother yielding
        s_new_norm = s_trial_norm - yield_value / plastic_factor
        s_new = (s_new_norm / s_trial_norm) * s_trial

        # Recompute new elastic strains considering plastic flow
        epsilon_new = (1.0 / (2.0 * model.mu[p])) * s_new + tm.vec3(tr / 3.0)

        # Construct elastic stretch matrix
        sig_elastic = tm.mat3(
            tm.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            tm.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            tm.exp(epsilon_new[2])
        )

        # Final deformation gradient with cohesive yield adjustments
        F_cohesive = U @ sig_elastic @ V.transpose()
    else:
        # Elastic behavior (more fluid-like)
        F_cohesive = F_trial

    return F_cohesive

# for toothpaste
@ti.func
def viscoplasticity_return_mapping_with_StVK(F_trial, model, p, dt):
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
    y = s_trial_norm - 0.8 * ti.sqrt(2.0 / 3.0) * model.yield_stress[p]
    F = tm.mat3(0.0)
    if y > 0:
        mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + model.plastic_viscosity * 2 / (2.0 * mu_hat * dt)
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