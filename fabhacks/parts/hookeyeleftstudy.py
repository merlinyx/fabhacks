from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
from ..primitives.ring import Ring
import os

class HookEyeLeftS(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.HookEyeLeftS
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "HookEyeLeft.obj")
        self.hook = Hook({"arc_radius": 19, "arc_angle": 210, "thickness": 5, "is_open": False, "is_open_for_connect": True}, parent_part=self)
        self.hook.set_frame(Frame([0,0,-54.2], [90,0,180]))
        self.hook.child_id = 0
        self.hook.set_ohe(PartPrimitiveType.HookOfHookEyeLeftS)
        self.ring = Ring({"arc_radius": 14.82, "thickness": 4.5, "can_flex": True}, parent_part=self)
        self.ring.set_frame(Frame([0,0,0], [0,0,90]))
        self.ring.child_id = 1
        self.ring.set_ohe(PartPrimitiveType.RingOfHookEyeLeftS)
        self.children = [self.hook, self.ring]
        self.children_varnames = ["hook", "ring"]
        self.children_pairwise_ranges = {(77, 78): (40.156552460328754, 78.36661246625516)}
        self.primitive_types = ["Hook", "Ring"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 1]
        self.init_mesh_metrics()
