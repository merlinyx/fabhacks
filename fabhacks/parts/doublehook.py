from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
import os

# TODO: can_flex is sort of needed for the readingnook (for now) but it messes up the other examples
class DoubleHook(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.DoubleHook
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "DoubleHook.obj")
        # params based on S-Hook.scad
        self.hook1 = Hook({"arc_radius": 11.6, "arc_angle": 270, "thickness": 2, "can_flex": True}, parent_part=self)
        # self.hook1 = Hook({"arc_radius": 11.6, "arc_angle": 270, "thickness": 2}, parent_part=self)
        self.hook1.set_frame(Frame([0,0,11.6], [0,0,0]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfDoubleHook)
        self.hook2 = Hook({"arc_radius": 11.6, "arc_angle": 270, "thickness": 2, "can_flex": True}, parent_part=self)
        # self.hook2 = Hook({"arc_radius": 11.6, "arc_angle": 270, "thickness": 2}, parent_part=self)
        self.hook2.set_frame(Frame([0,0,-11.6], [90,0,180]))
        self.hook2.child_id = 1
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfDoubleHook)
        self.children = [self.hook1, self.hook2]
        self.children_varnames = ["hook1", "hook2"]
        self.children_pairwise_ranges = {(33, 34): (7.071687953888538, 42.355160969382545)}
        self.primitive_types = ["Hook"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 0]
        self.init_mesh_metrics()
