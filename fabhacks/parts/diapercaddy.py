from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
import os

class DiaperCaddy(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.DiaperCaddy
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "DiaperCaddy.obj")
        # self.hook1 = Hook({"arc_radius": 16.2, "arc_angle": 150, "thickness": 4, "is_open": False}, parent_part=self)
        self.hook1 = Hook({"arc_radius": 16.2, "arc_angle": 150, "thickness": 4, "is_open": False, "can_flex":True}, parent_part=self)
        self.hook1.set_frame(Frame([165,128.4,311], [255,-45,-90]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfDiaperCaddy)
        # self.hook2 = Hook({"arc_radius": 16.2, "arc_angle": 150, "thickness": 4, "is_open": False}, parent_part=self)
        self.hook2 = Hook({"arc_radius": 16.2, "arc_angle": 150, "thickness": 4, "is_open": False, "can_flex":True}, parent_part=self)
        self.hook2.set_frame(Frame([165,-357,311], [-105,-45,90]))
        self.hook2.child_id = 1
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfDiaperCaddy)
        self.children = [self.hook1, self.hook2]
        self.children_varnames = ["hook1", "hook2"]
        self.children_pairwise_ranges = {(31, 32): (492.36900329589844, 502.6057843634157)}
        self.primitive_types = ["Hook"]
        self.is_concave = True
        self.init_mesh_metrics()
