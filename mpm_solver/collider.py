import taichi as ti
import taichi.math as tm

@ti.data_oriented
class MPM_Collider:
    def __init__(self):
        self.point = tm.vec3(0.0)
        self.normal = tm.vec3(0.0)
        self.friction = 0.0