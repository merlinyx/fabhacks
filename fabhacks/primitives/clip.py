from .mate_connector_frame import ClipMCF
from .primitive import Primitive
from solid import *
import igl
import subprocess
import os

class Clip(Primitive):
    def __init__(self, params, parent_part=None):
        self.width = None
        self.height = None
        self.thickness = None
        self.dist = None
        self.open_gap = None
        super().__init__(parent_part)
        self.tid = 3
        self.ctype_str = "Clip"
        self.init_params(params)

    def connector_critical_dim(self):
        return self.width / 2.

    def init_params(self, params):
        self.width = params["width"]
        self.height = params["height"]
        self.thickness = params["thickness"]
        self.dist = params["dist"]
        self.open_gap = params["open_gap"]
        self.is_concave = True
        self.program = "{{\"width\": {}, \"height\": {}, \"thickness\": {}, \"dist\": {}, \"open_gap\": {}}}".format(self.width, self.height, self.thickness, self.dist, self.open_gap)
        self.mcf_factory = ClipMCF(self.id, self.height)

    # initialize V and F
    def init_mesh(self):
        scadfile = import_scad(Primitive.modules)
        b = scadfile.clip(self.width * (1. + Primitive.epsilon), self.height * (1. + Primitive.epsilon), self.thickness * (1. + Primitive.epsilon), self.dist, self.open_gap)
        os.makedirs(Primitive.temp_dir, exist_ok=True)
        scadpath = os.path.join(".", Primitive.temp_dir, "clip.scad")
        offpath = os.path.join(".", Primitive.temp_dir, "clip_{}_{}_{}_{}_{}.off".format(self.width * (1. + Primitive.epsilon), self.height * (1. + Primitive.epsilon), self.thickness * (1. + Primitive.epsilon), self.dist, self.open_gap))
        scad_render_to_file(b, scadpath)
        run_args = [Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
        subprocess.run(run_args)
        self.V, self.F = igl.read_triangle_mesh(offpath)
