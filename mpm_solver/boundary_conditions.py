import taichi as ti


@ti.data_oriented
class ImpulseParams:
    def __init__(self, start_time, end_time, force):
        self.start_time = start_time
        self.end_time = end_time
        self.force = ti.Vector([force[0], force[1], force[2]])
