from .mate_connector_frame import HoleMCF, HoleCtlMCF, HoleFlexMCF
from .primitive import Primitive
from solid import *
import igl
import subprocess
import os

class Ring(Primitive):
    def __init__(self, params, parent_part=None):
        self.arc_radius = None
        self.thickness = None
        super().__init__(parent_part)
        self.tid = 2
        self.ctype_str = "Ring"
        self.init_params(params)

    def critical_dim(self):
        return self.arc_radius

    def connector_critical_dim(self):
        return self.thickness

    def generalized_radius(self):
        return self.thickness

    def init_params(self, params):
        self.arc_radius = params["arc_radius"]
        self.thickness = params["thickness"]
        self.is_concave = True
        self.single_conn = False
        self.program = "{{\"arc_radius\": {}, \"thickness\": {}}}".format(self.arc_radius, self.thickness)
        if "centerline" in params and params["centerline"]:
            self.mcf_factory = HoleCtlMCF(self.id, self.arc_radius, self.thickness)
            self.ctype_str += "Ctl"
        else:
            if "can_flex" in params and params["can_flex"]:
                self.ctype_str += "Flex"
                self.mcf_factory = HoleFlexMCF(self.id, self.arc_radius, self.thickness)
            else:
                self.mcf_factory = HoleMCF(self.id, self.arc_radius, self.thickness)

    # initialize V and F
    def init_mesh(self):
        scadfile = import_scad(Primitive.modules)
        b = scadfile.hook(self.arc_radius * (1. + Primitive.epsilon), 360, self.thickness * (1. + Primitive.epsilon))
        os.makedirs(Primitive.temp_dir, exist_ok=True)
        scadpath = os.path.join(".", Primitive.temp_dir, "ring.scad")
        offpath = os.path.join(".", Primitive.temp_dir, "ring_{}_{}.off".format(self.arc_radius * (1. + Primitive.epsilon), self.thickness * (1. + Primitive.epsilon)))
        scad_render_to_file(b, scadpath)
        run_args = [Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
        subprocess.run(run_args)
        self.V, self.F = igl.read_triangle_mesh(offpath)
