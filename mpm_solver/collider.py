import taichi as ti
import taichi.math as tm

@ti.data_oriented
class MPM_Collider:
    def __init__(self, point, normal, friction):
        self.point = point
        self.normal = normal
        self.friction = friction

        self.isCollide = True

    @ti.kernel
    def collide(
        self,
        time: float,
        dt: float,
        state: ti.template(),
        model: ti.template()
    ):
        # print("TEST")
        for grid_x, grid_y, grid_z in state.grid_v_out:
            offset = tm.vec3(
                float(grid_x) * model.dx - self.point[0],
                float(grid_y) * model.dx - self.point[1],
                float(grid_z) * model.dx - self.point[2],
            )
            n = tm.vec3(self.normal[0], self.normal[1], self.normal[2])
            dotproduct = tm.dot(offset, n)

            if dotproduct < 0.0:
                v = state.grid_v_out[grid_x, grid_y, grid_z]
                normal_component = tm.dot(v, n)
                v = (
                    v - ti.min(normal_component, 0.0) * n
                )  # Project out only inward normal component
                if normal_component < 0.0 and tm.length(v) > 1e-20:
                    v = ti.max(
                        0.0, tm.length(v) + normal_component * self.friction
                    ) * tm.normalize(
                        v
                    )  # apply friction here
                state.grid_v_out[grid_x, grid_y, grid_z] = v * 0.99


collideTypeCallBacks = {
    "ground": MPM_Collider
}