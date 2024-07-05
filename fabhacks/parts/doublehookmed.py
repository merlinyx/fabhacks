from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
import os

class DoubleHookMedium(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.DoubleHookMedium
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "DoubleHookMedium.obj")
        # params based on S-Hook.scad (2)
        self.hook1 = Hook({"arc_radius": 16, "arc_angle": 270, "thickness": 2.9}, parent_part=self)
        self.hook1.set_frame(Frame([0,0,16], [0,0,0]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfDoubleHookMedium)
        self.hook2 = Hook({"arc_radius": 16, "arc_angle": 270, "thickness": 2.9}, parent_part=self)
        self.hook2.set_frame(Frame([0,0,-16], [90,0,180]))
        self.hook2.child_id = 1
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfDoubleHookMedium)
        self.children = [self.hook1, self.hook2]
        self.children_varnames = ["hook1", "hook2"]
        self.children_pairwise_ranges = {(35, 36): (5.955231738607253, 58.198257112029566)}
        self.primitive_types = ["Hook"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 0]
        self.init_mesh_metrics()
