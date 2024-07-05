from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
import os

class ClosetRods(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.ClosetRods
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "ClosetRods.obj")
        self.rod1 = Rod({"length": 170, "radius": 6.35, "is_open": False, "is_open_for_connect": True}, parent_part=self)
        self.rod1.set_frame(Frame([0,-175,0], [90,0,180]))
        self.rod1.child_id = 0
        self.rod1.set_ohe(PartPrimitiveType.Rod1OfClosetRods)
        self.rod2 = Rod({"length": 170, "radius": 6.35, "is_open": False, "is_open_for_connect": True}, parent_part=self)
        self.rod2.set_frame(Frame([0,175,0], [90,0,180]))
        self.rod2.child_id = 1
        self.rod2.set_ohe(PartPrimitiveType.Rod2OfClosetRods)
        self.children = [self.rod1, self.rod2]
        self.children_varnames = ["rod1", "rod2"]
        self.children_pairwise_ranges = {(74, 75): (200.0, 500.0)}
        # self.children_pairwise_ranges = {(74, 75): (362, 532)}
        self.primitive_types = ["Rod"]
        self.child_symmetry_groups = [0, 0]
        self.init_mesh_metrics()
