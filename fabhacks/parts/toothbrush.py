from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hemisphere import Hemisphere
from ..primitives.rod import Rod
# from primitives.point import Point
import os

class Toothbrush(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.Toothbrush
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Toothbrush.obj")
        self.hemisphere = Hemisphere({"radius": 9}, parent_part=self)
        self.hemisphere.set_frame(Frame([0,188,29], [180,0,0]))
        self.hemisphere.child_id = 0
        self.hemisphere.set_ohe(PartPrimitiveType.HemisphereOfToothbrush)
        self.rod = Rod({"length": 12, "radius": 2.8, "centerline": True}, parent_part=self)
        self.rod.set_frame(Frame([0,46,6], [106,0,0]))
        self.rod.child_id = 1
        self.rod.set_ohe(PartPrimitiveType.RodOfToothbrush)
        self.children = [self.hemisphere, self.rod]
        self.children_varnames = ["hemisphere", "rod"]
        self.children_pairwise_ranges = {(64, 65): (129.60213048646153, 158.65422299332403)}
        self.primitive_types = ["Hemisphere", "Rod"]
        self.init_mesh_metrics()
        self.rod = Rod({"length": 12, "radius": 2.8}, parent_part=self)
