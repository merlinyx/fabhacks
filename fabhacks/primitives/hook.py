from .mate_connector_frame import HookMCF, HookFlexMCF
from .primitive import Primitive
from solid import *
import igl
import subprocess
import os

class Hook(Primitive):
    def __init__(self, params, parent_part=None):
        self.arc_radius = None
        self.arc_angle = None
        self.thickness = None
        self.is_open = True # whether the hook has an open end for simulation
        self.is_open_for_connect = True # whether the hook has an open end for connecting
        super().__init__(parent_part)
        self.tid = 1
        self.ctype_str = "Hook"
        self.init_params(params)

    def connector_critical_dim(self):
        return self.thickness

    def generalized_radius(self):
        return self.thickness

    def init_params(self, params):
        self.arc_radius = params["arc_radius"]
        self.arc_angle = params["arc_angle"]
        self.thickness = params["thickness"]
        if "ranges" in params:
            self.ranges = params["ranges"]
        else:
            self.ranges = [0, self.arc_angle]
        if "phi_ranges" in params:
            self.phi_ranges = params["phi_ranges"]
        else:
            self.phi_ranges = [45, 135]
        if "is_open" in params:
            self.is_open = params["is_open"]
        if "is_open_for_connect" in params:
            self.is_open_for_connect = params["is_open_for_connect"]
        elif "is_open" in params:
            self.is_open_for_connect = params["is_open"]
        self.is_concave = True
        self.single_conn = False
        self.program = "{{\"arc_radius\": {}, \"arc_angle\": {}, \"thickness\": {}}}".format(self.arc_radius, self.arc_angle, self.thickness)
        if "can_flex" in params and params["can_flex"]:
            self.ctype_str += "Flex"
            self.mcf_factory = HookFlexMCF(self.id, self.arc_radius, self.arc_angle, self.thickness, self.ranges, self.phi_ranges)
        else:
            self.mcf_factory = HookMCF(self.id, self.arc_radius, self.arc_angle, self.thickness, self.ranges)

    # initialize V and F
    def init_mesh(self):
        scadfile = import_scad(Primitive.modules)
        b = scadfile.hook(self.arc_radius * (1. + Primitive.epsilon), self.arc_angle, self.thickness * (1. + Primitive.epsilon))
        os.makedirs(Primitive.temp_dir, exist_ok=True)
        scadpath = os.path.join(".", Primitive.temp_dir, "hook.scad")
        offpath = os.path.join(".", Primitive.temp_dir, "hook_{}_{}_{}.off".format(self.arc_radius * (1. + Primitive.epsilon), self.arc_angle, self.thickness * (1. + Primitive.epsilon)))
        scad_render_to_file(b, scadpath)
        run_args = [Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
        subprocess.run(run_args)
        self.V, self.F = igl.read_triangle_mesh(offpath)
