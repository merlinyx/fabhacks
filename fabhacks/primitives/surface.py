from .mate_connector_frame import SurfaceMCF
from .primitive import Primitive
from solid import *
import igl
import subprocess
import os

class Surface(Primitive):
    def __init__(self, params, parent_part=None):
        self.width = None
        self.length = None
        self.is_open = False # whether the surface has no boundary
        super().__init__(parent_part)
        self.tid = 6
        self.ctype_str = "Surface"
        if params is None:
            params = self.default_params()
        self.init_params(params)

    def critical_dim(self):
        return [self.width, self.length]

    def connector_critical_dim(self):
        return [self.width / 2., self.length / 2.]

    def default_params(self):
        return {"width": 100, "length": 100}

    def init_params(self, params):
        self.width = params["width"]
        self.length = params["length"]
        if "is_open" in params:
            self.is_open = params["is_open"]
        self.single_conn = False
        self.program = "{{\"width\": {}, \"length\": {}}}".format(self.width, self.length)
        self.mcf_factory = SurfaceMCF(self.id, self.width, self.length)

    # initialize V and F
    def init_mesh(self):
        scadfile = import_scad(Primitive.modules)
        b = scadfile.surface(self.width, self.length, 0.1 * (1 + Primitive.epsilon))
        os.makedirs(Primitive.temp_dir, exist_ok=True)
        scadpath = os.path.join(".", Primitive.temp_dir, "surface.scad")
        offpath = os.path.join(".", Primitive.temp_dir, "surface_{}_{}.off".format(self.width, self.length))
        scad_render_to_file(b, scadpath)
        run_args = [Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
        subprocess.run(run_args)
        self.V, self.F = igl.read_triangle_mesh(offpath)
