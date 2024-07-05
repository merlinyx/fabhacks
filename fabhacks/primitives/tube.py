from .primitive import Primitive
from .mate_connector_frame import TubeMCF, TubeCtlMCF
from solid import *
import igl
import subprocess
import os

class Tube(Primitive):
    def __init__(self, params, parent_part=None):
        self.inner_radius = None
        self.thickness = None
        self.length = None
        self.is_open = True # whether the rod has open ends
        super().__init__(parent_part)
        self.tid = 4
        self.ctype_str = "Tube"
        if params is None:
            params = self.default_params()
        self.init_params(params)

    def critical_dim(self):
        return self.length

    def connector_critical_dim(self):
        return self.length / 2.

    def generalized_radius(self):
        return self.inner_radius + self.thickness

    def default_params(self):
        return {"inner_radius": 0.5, "thickness": 0.1, "length": 10.}

    def init_params(self, params):
        self.inner_radius = params["inner_radius"]
        self.thickness = params["thickness"]
        self.length = params["length"]
        if "is_open" in params:
            self.is_open = params["is_open"]
        self.is_concave = True
        self.single_conn = False
        self.program = "{{\"inner_radius\": {}, \"thickness\": {}, \"length\": {}}}".format(self.inner_radius, self.thickness, self.length)
        if "centerline" in params and params["centerline"]:
            self.mcf_factory = TubeCtlMCF(self.id, self.inner_radius, self.thickness, self.length)
            self.ctype_str += "Ctl"
        else:
            self.mcf_factory = TubeMCF(self.id, self.inner_radius, self.thickness, self.length)

    # initialize V and F
    def init_mesh(self):
        scadfile = import_scad(Primitive.modules)
        b = scadfile.tube(self.inner_radius, self.thickness * (1. + Primitive.epsilon), self.length)
        os.makedirs(Primitive.temp_dir, exist_ok=True)
        scadpath = os.path.join(".", Primitive.temp_dir, "tube.scad")
        offpath = os.path.join(".", Primitive.temp_dir, "tube_{}_{}_{}.off".format(self.inner_radius, self.thickness * (1. + Primitive.epsilon), self.length))
        scad_render_to_file(b, scadpath)
        run_args = [Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
        subprocess.run(run_args)
        self.V, self.F = igl.read_triangle_mesh(offpath)
