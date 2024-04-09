import numpy as np
import torch

class TinyCam:
    def __init__(self, width, height, FovX, FovY, cam_center, view_mat, full_proj_mat):
        self.width = width
        self.height = height
        self.FovX = FovX
        self.FovY = FovY
        self.cam_center = cam_center
        self.view_mat = view_mat
        self.full_proj_mat = full_proj_mat

    def toCuda(self):
        self.cam_center = torch.tensor(self.cam_center).cuda()
        self.view_mat = torch.tensor(self.view_mat).cuda()
        self.full_proj_mat = torch.tensor(self.full_proj_mat).cuda()


def to8b(x):
    return (255*np.clip(x, 0, 1)).astype(np.uint8)
