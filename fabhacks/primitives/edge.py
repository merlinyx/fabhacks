from .mate_connector_frame import EdgeMCF
from .primitive import Primitive
from solid import *
import igl
import subprocess
import os

class Edge(Primitive):
    def __init__(self, params, parent_part=None):
        self.width = None
        self.length = None
        self.height = None
        super().__init__(parent_part)
        self.tid = 5
        self.ctype_str = "Edge"
        if params is None:
            params = self.default_params()
        self.init_params(params)

    def critical_dim(self):
        return self.length

    def default_params(self):
        return {"width": 10, "length": 10, "height": 10}

    def init_params(self, params):
        self.width = params["width"]
        self.length = params["length"]
        self.height = params["height"]
        self.single_conn = False
        self.program = "{{\"width\": {}, \"length\": {}, \"height\": {}}}".format(self.width, self.length, self.height)
        self.mcf_factory = EdgeMCF(self.id, self.width, self.length, self.height)

    # initialize V and F
    def init_mesh(self):
        scadfile = import_scad(Primitive.modules)
        b = scadfile.edge(self.width, self.length, self.height * (1. + Primitive.epsilon))
        os.makedirs(Primitive.temp_dir, exist_ok=True)
        scadpath = os.path.join(".", Primitive.temp_dir, "edge.scad")
        offpath = os.path.join(".", Primitive.temp_dir, "edge_{}_{}_{}.off".format(self.width, self.length, self.height * (1. + Primitive.epsilon)))
        scad_render_to_file(b, scadpath)
        run_args = [Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
        subprocess.run(run_args)
        self.V, self.F = igl.read_triangle_mesh(offpath)
