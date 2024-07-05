from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
import os

class BackSeats(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.BackSeats
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "BackSeats.obj")
        self.rod1 = Rod({"length": 37, "radius": 5.7, "is_open": False}, parent_part=self)
        self.rod1.set_frame(Frame([50.8,556.9,763], [0,0,90]))
        self.rod1.child_id = 0
        self.rod1.set_ohe(PartPrimitiveType.Rod1OfBackSeats)
        self.rod2 = Rod({"length": 37, "radius": 5.7, "is_open": False}, parent_part=self)
        self.rod2.set_frame(Frame([677.8,556.9,763], [0,0,-90]))
        self.rod2.child_id = 1
        self.rod2.set_ohe(PartPrimitiveType.Rod2OfBackSeats)
        self.children = [self.rod1, self.rod2]
        self.children_varnames = ["rod1", "rod2"]
        self.children_pairwise_ranges = {(25, 26): (638.4000015258789, 639.2342252588853)}
        self.primitive_types = ["Rod"]
        self.is_concave = True
        self.init_mesh_metrics()
