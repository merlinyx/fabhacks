from .primitive import Primitive
from .mate_connector_frame import HemisphereMCF
from solid import *
import igl
import subprocess
import os

class Hemisphere(Primitive):
    def __init__(self, params, parent_part=None):
        self.radius = None
        super().__init__(parent_part)
        self.tid = 7
        self.ctype_str = "Hemisphere"
        self.init_params(params)

    def connector_critical_dim(self):
        return [self.radius, self.radius]

    def generalized_radius(self):
        return self.radius

    def init_params(self, params):
        self.radius = params["radius"]
        self.program = "{{\"radius\": {}}}".format(params["radius"])
        self.mcf_factory = HemisphereMCF(self.id, self.radius)

    # initialize V and F
    def init_mesh(self):
        scadfile = import_scad(Primitive.modules)
        b = scadfile.hemisphere(self.radius * (1. + Primitive.epsilon))
        os.makedirs(Primitive.temp_dir, exist_ok=True)
        scadpath = os.path.join(".", Primitive.temp_dir, "hemisphere.scad")
        offpath = os.path.join(".", Primitive.temp_dir, "hemisphere_{}.off".format(self.radius * (1. + Primitive.epsilon)))
        scad_render_to_file(b, scadpath)
        run_args = [Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
        subprocess.run(run_args)
        self.V, self.F = igl.read_triangle_mesh(offpath)
