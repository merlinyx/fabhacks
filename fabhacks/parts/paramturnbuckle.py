from .parampart import ParamPart
from .part import Part, PartType, PartPrimitiveType
from ..primitives.primitive import Primitive
from ..assembly.frame import Frame
from ..primitives.hook import Hook
from solid import *
import igl
import subprocess
import os
import dill as pickle
from importlib_resources import files

class ParamTurnbuckle(ParamPart):
    default_part_params = {"extended_length": 0}
    default_scafpath = str(files("fabhacks").joinpath("parts", "paramscad", "M4.scad"))

    def __init__(self, part_params, scadfpath):
        super().__init__(part_params, scadfpath)
        # process part_params, the default is the unextended turnbuckle
        self.extended_length = self.part_params["extended_length"] if "extended_length" in self.part_params else 0
        # usual initialization
        self.tid = PartType.ParamTurnbuckle
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.primitive_types = ["Hook"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 0]
        # parametric initialization
        self.param_init()
        self.init_mesh_metrics()

    def param_init(self):
        self.params_str = ""
        for value in self.part_params.values():
            self.params_str = self.params_str + "_" + str(value)
        self.extended_length = self.part_params["extended_length"] if "extended_length" in self.part_params else 0
        self.part_obj_path = os.path.join(".", Part.temp_dir, "ParamTurnbuckle{}.obj".format(self.params_str))
        # params based on part_params and the openscad file
        self.length = 55
        self.hook_d = 10
        self.hook_thickness = 1.5
        self.arc_d = self.hook_d + self.hook_thickness
        self.bolt_length = 6
        self.hook_angle = 220
        hook_y = self.bolt_length + self.hook_d + self.length/2 + self.extended_length/2
        self.cutout_width = 4
        self.hook1 = Hook({"arc_radius": self.arc_d/2. , "arc_angle": self.hook_angle, "thickness": self.hook_thickness}, parent_part=self)
        self.hook1.set_frame(Frame([self.cutout_width/2,hook_y,0], [self.hook_angle-45,90,0]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfParamTurnbuckle)
        self.hook2 = Hook({"arc_radius": self.arc_d/2., "arc_angle": self.hook_angle, "thickness": self.hook_thickness, "can_flex": True}, parent_part=self)
        self.hook2.set_frame(Frame([-self.cutout_width/2,-hook_y,0], [self.hook_angle-180-45,90,0]))
        self.hook2.child_id = 1
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfParamTurnbuckle)
        self.children = [self.hook1, self.hook2]
        self.children_varnames = ["hook1", "hook2"]
        self.precompute_for_part()

    def init_mesh_metrics(self, reload=False):
        os.makedirs(Part.temp_dir, exist_ok=True)
        offpath = os.path.join(".", Part.temp_dir, "m4_{}.off".format(self.params_str))
        if (not os.path.exists(offpath)) or reload:
            scadfile = import_scad(self.openscad_fpath)
            b = scadfile.M4turnbuckle(self.extended_length)
            scadpath = os.path.join(".", Part.temp_dir, "M4.scad")
            scad_render_to_file(b, scadpath)
            run_args = [Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
            subprocess.run(run_args)
        V, F = igl.read_triangle_mesh(offpath)
        igl.write_obj(self.part_obj_path, V, F)

        mesh_metrics_path = str(files('fabhacks').joinpath("parts", "meshes", "paramturnbuckle_{}.metrics".format(self.params_str)))
        metrics = {}
        if os.path.exists(mesh_metrics_path) and not reload:
            with open(mesh_metrics_path, 'rb') as f:
                metrics = pickle.load(f)
            self.com_offset = metrics["com_offset"]
            self.attach_length = metrics["attach_length"]
            self.mass = metrics["mass"]
            self.bbox = metrics["bbox"]
        else:
            self.compute_com(V, F)
            metrics["com_offset"] = self.com_offset
            self.attach_length = .5 * igl.bounding_box_diagonal(V)
            metrics["attach_length"] = self.attach_length
            self.compute_mass(V, F)
            metrics["mass"] = self.mass
            self.bbox, _ = igl.bounding_box(V) # returns BV (2^3 by 3 numpy array), BF
            metrics["bbox"] = self.bbox
            with open(mesh_metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
