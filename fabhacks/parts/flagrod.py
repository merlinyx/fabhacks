from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
import os

class FlagRod(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.FlagRod
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "FlagRod.obj")
        self.rod = Rod({"length": 280, "radius": 6.35, "centerline": True}, parent_part=self)
        self.rod.set_frame(Frame([0,0,140], [0,0,0]))
        self.rod.child_id = 0
        self.rod.set_ohe(PartPrimitiveType.RodOfFlagRod)
        self.children = [self.rod]
        self.children_varnames = ["rod"]
        self.primitive_types = ["Rod"]
        self.init_mesh_metrics()
