from argparse import ArgumentParser


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, json_params=None):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            if json_params and key in json_params:
                value = json_params[key]
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, json_params=None):
        self.model_path = ""
        self.loaded_iter = -1
        
        self.debug = False
        
        super().__init__(parser, "Loading Parameters", json_params)


class MPMParams(ParamGroup):
    def __init__(self, parser, json_params=None):
        self.sim_area = []

        self.E = 2000
        self.nu = 0.2
        self.material = "metal"
        self.friction_angle = 35

        self.g = [0.0, 0.0, -9.8]
        self.density = 200.0

        self.n_grid = 100
        self.grid_extent = 2.0

        super().__init__(parser, "MPM Parameters", json_params)


class RenderParams(ParamGroup):
    def __init__(self, parser, json_params=None):
        self.output_path = ""
        self.white_background = False

        self.view_cam_idx = 0
        
        self.num_frames = 60

        super().__init__(parser, "MPM Parameters", json_params)
