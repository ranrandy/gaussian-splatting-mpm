import taichi as ti

@ti.func
def set_vec3_to_zero(target_array: ti.types.ndarray()):
    for i in target_array:
        target_array[i] = ti.Vector([0.0, 0.0, 0.0])

@ti.func
def set_mat33_to_identity(target_array: ti.types.ndarray()):
    for i in target_array:
        target_array[i] = ti.Matrix.identity(ti.f32, 3)


@ti.func
def set_value_to_float_array(target_array, value):
    for i in target_array:
        target_array[i] = value

@ti.func
def get_float_array_product(arrayA, arrayB, arrayC):
    for i in arrayA:
        arrayC[i] = arrayA[i] * arrayB[i]

@ti.func
def apply_additional_params(state, model, params_modifier):
    for i in state.particle_x:
        pos = state.particle_x[i]
        if (pos[0] > params_modifier.point[0] - params_modifier.size[0] and
            pos[0] < params_modifier.point[0] + params_modifier.size[0] and
            pos[1] > params_modifier.point[1] - params_modifier.size[1] and
            pos[1] < params_modifier.point[1] + params_modifier.size[1] and
            pos[2] > params_modifier.point[2] - params_modifier.size[2] and
            pos[2] < params_modifier.point[2] + params_modifier.size[2]):
            model.E[i] = params_modifier.E
            model.nu[i] = params_modifier.nu
            state.particle_density[i] = params_modifier.density

