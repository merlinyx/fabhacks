from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
import os

class OpenHandleBasket(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.OpenHandleBasket
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "OpenHandleBasket.obj")
        self.hook = Hook({"arc_radius": 87.5, "arc_angle":180, "thickness": 12.7}, parent_part=self)
        self.hook.set_frame(Frame([0,0,200], [-90,0,90])) # 90,0,0
        self.hook.child_id = 0
        self.hook.set_ohe(PartPrimitiveType.HookOfOpenHandleBasket)
        self.children = [self.hook]
        self.children_varnames = ["hook"]
        self.primitive_types = ["Hook"]
        self.is_concave = True
        self.child_symmetry_groups = [0]
        self.init_mesh_metrics()
