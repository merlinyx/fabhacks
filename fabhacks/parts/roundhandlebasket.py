from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
import os

class RoundHandleBasket(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.RoundHandleBasket
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "RoundHandleBasket.obj")
        self.hook = Hook({"arc_radius": 85, "arc_angle": 210, "thickness": 10, "is_open": False, "can_flex": True}, parent_part=self)
        self.hook.set_frame(Frame([0,0,120], [-75,0,0]))
        self.hook.child_id = 0
        self.hook.set_ohe(PartPrimitiveType.HookOfRoundHandleBasket)
        self.children = [self.hook]
        self.children_varnames = ["hook"]
        self.primitive_types = ["Hook"]
        self.is_concave = True
        self.child_symmetry_groups = [0]
        self.init_mesh_metrics()
