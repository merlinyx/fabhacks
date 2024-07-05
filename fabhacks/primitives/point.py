from .mate_connector_frame import MateConnectorFrame
from .primitive import Primitive
from solid import *
import igl
import subprocess
import os

class Point(Primitive):
    def __init__(self, params, parent_part=None):
        self.minpos = None
        self.maxpos = None
        super().__init__(parent_part)
        self.init_params(params)
        self.ctype_str = "Point"

    def get_program(self):
        return self.program

    def contains(self, point):
        b1, b2, b3 = True, True, True
        if self.minpos is not None:
            b1 = point[0] >= self.minpos[0]
            b2 = point[1] >= self.minpos[1]
            b3 = point[2] >= self.minpos[2]
        if self.maxpos is not None:
            b1 = b1 and point[0] <= self.maxpos[0]
            b2 = b2 and point[1] <= self.maxpos[1]
            b3 = b3 and point[2] <= self.maxpos[2]
        return b1 and b2 and b3

    def generalized_radius(self):
        return 0.1

    def init_params(self, params):
        self.minpos = params["minpos"]
        self.maxpos = params["maxpos"]
        self.program = "{{\"minpos\": [{},{},{}], \"maxpos\": [{},{},{}]}}".format(
            self.minpos[0], self.minpos[1], self.minpos[2], self.maxpos[0], self.maxpos[1], self.maxpos[2])
        self.mcf_factory = MateConnectorFrame(self.id)

    # initialize V and F
    def init_mesh(self):
        scadfile = import_scad(Primitive.modules)
        b = scadfile.point()
        os.makedirs(Primitive.temp_dir, exist_ok=True)
        scadpath = os.path.join(".", Primitive.temp_dir, "point.scad")
        offpath = os.path.join(".", Primitive.temp_dir, "point.off")
        scad_render_to_file(b, scadpath)
        run_args = [Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
        subprocess.run(run_args)
        self.V, self.F = igl.read_triangle_mesh(offpath)
