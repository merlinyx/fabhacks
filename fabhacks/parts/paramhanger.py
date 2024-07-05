from .parampart import ParamPart
from .part import Part, PartType, PartPrimitiveType
from ..primitives.primitive import Primitive
from ..assembly.frame import Frame
from ..primitives.hook import Hook
from ..primitives.rod import Rod
from solid import *
import igl
import subprocess
import os
import dill as pickle
from importlib_resources import files

class ParamHanger(ParamPart):
    default_part_params = {"length": 435, "height": 115, "bar_d": 6.35, "bend_d": 25, "hook_d": 40, "hook_h": 105}
    default_scafpath = str(files("fabhacks").joinpath("parts", "paramscad", "hanger.scad"))

    def __init__(self, part_params, scadfpath):
        super().__init__(part_params, scadfpath)
        # process part_params, the default is basically DoubleHook
        self.length = self.part_params["length"] if "length" in self.part_params else 435
        self.height = self.part_params["height"] if "height" in self.part_params else 115
        self.bar_d = self.part_params["bar_d"] if "bar_d" in self.part_params else 6.35
        self.bend_d = self.part_params["bend_d"] if "bend_d" in self.part_params else 25
        self.hook_d = self.part_params["hook_d"] if "hook_d" in self.part_params else 40
        self.hook_h = self.part_params["hook_h"] if "hook_h" in self.part_params else 105
        # usual initialization
        self.tid = PartType.ParamHanger
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.params_str = ""
        for value in self.part_params.values():
            self.params_str = self.params_str + "_" + str(value)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "ParamHanger{}.obj".format(self.params_str))
        # params based on part_params
        bar_r = self.bar_d / 2
        bend_r = self.bend_d / 2
        # thickness = self.bar_d
        pin_d = self.bar_d
        pin_buffer = 5
        plate_h = pin_d * 2 + pin_buffer + 4
        # plate_w = pin_d * 2
        # plate_t = thickness / 2
        # crossbeam_h = pin_d
        bottom_l = self.length / 2 - bend_r - plate_h / 2
        rod_l = bottom_l * 2
        if rod_l > 20:
            rod_l -= 10
            bottom_l -= 5
        self.hook1 = Hook({"arc_radius": self.hook_d / 2., "arc_angle": 180, "thickness": bar_r}, parent_part=self)
        self.hook1.set_frame(Frame([0,0,self.hook_h - self.hook_d / 2. - bar_r], [-45,0,0]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfParamHanger)
        self.rod = Rod({"length": rod_l, "radius": bar_r, "is_open": False}, parent_part=self)
        self.rod.set_frame(Frame([0,0,-110], [-90,180,0]))
        self.rod.child_id = 1
        self.rod.set_ohe(PartPrimitiveType.RodOfParamHanger)
        self.hook2 = Hook({"arc_radius": bend_r, "arc_angle": 160, "thickness": bar_r, "is_open": False}, parent_part=self)
        self.hook2.set_frame(Frame([0,bottom_l,-97.5], [160,0,0]))
        self.hook2.child_id = 2
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfParamHanger)
        self.hook3 = Hook({"arc_radius": bend_r, "arc_angle": 160, "thickness": bar_r, "is_open": False}, parent_part=self)
        self.hook3.set_frame(Frame([0,-bottom_l,-97.5], [0,0,0]))
        self.hook3.child_id = 3
        self.hook3.set_ohe(PartPrimitiveType.Hook3OfParamHanger)
        self.children = [self.hook1, self.rod, self.hook2, self.hook3]
        self.children_varnames = ["hook1", "rod", "hook2", "hook3"]
        self.precompute_for_part() # because the parameters could change, the ranges need to be re-computed
        self.primitive_types = ["Hook", "Rod"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 1, 2, 2]
        self.init_mesh_metrics()

    def init_mesh_metrics(self):
        os.makedirs(Part.temp_dir, exist_ok=True)
        offpath = os.path.join(".", Part.temp_dir, "hanger_{}.off".format(self.params_str))
        if not os.path.exists(offpath):
            scadfile = import_scad(self.openscad_fpath)
            b = scadfile.hanger(self.length, self.height, self.bar_d, self.bend_d, self.hook_d, self.hook_h)
            scadpath = os.path.join(".", Part.temp_dir, "hanger.scad")
            scad_render_to_file(b, scadpath)
            run_args = [Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
            subprocess.run(run_args)
        V, F = igl.read_triangle_mesh(offpath)
        igl.write_obj(self.part_obj_path, V, F)

        mesh_metrics_path = str(files('fabhacks').joinpath("parts", "meshes", "paramhanger_{}.metrics".format(self.params_str)))
        metrics = {}
        if os.path.exists(mesh_metrics_path):
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
