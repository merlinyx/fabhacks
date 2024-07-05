from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
import os

class CurtainRing(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.CurtainRing
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "CurtainRing.obj")
        self.hook = Hook({"arc_radius": 20.8, "arc_angle": 342, "thickness": 2.2, "is_open": False, "is_open_for_connect": True}, parent_part=self)
        self.hook.set_frame(Frame([0,1.6,0], [0,90,4.5]))
        self.hook.child_id = 0
        self.hook.set_ohe(PartPrimitiveType.HookOfCurtainRing)
        self.children = [self.hook]
        self.children_varnames = ["hook"]
        self.primitive_types = ["Hook"]
        self.is_concave = True
        self.init_mesh_metrics()
