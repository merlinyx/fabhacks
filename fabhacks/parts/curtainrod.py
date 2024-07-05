from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
import os

class ShowerCurtainRod(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.ShowerCurtainRod
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "ShowerCurtainRod.obj")
        self.rod = Rod({"radius": 6.5, "length": 351}, parent_part=self)
        self.rod.set_frame(Frame([200,127,2133.5], [0,90,0]))
        self.rod.child_id = 0
        self.rod.set_ohe(PartPrimitiveType.RodOfShowerCurtainRod)
        self.children = [self.rod]
        self.children_varnames = ["rod"]
        self.primitive_types = ["Rod"]
        self.init_mesh_metrics()
