import taichi.math as tm

class MPM_Collider:
    def __init__(self, point, normal):
        self.point : tm.vec3
        self.normal : tm.vec3
        self.friction : float