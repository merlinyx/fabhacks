from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
import os

class CarSeat(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.CarSeat
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "CarSeat.obj")
        self.rod1 = Rod({"length": 37, "radius": 5.7}, parent_part=self)
        self.rod1.set_frame(Frame([50.8,556.9,763], [0,0,180]))
        self.rod1.child_id = 0
        self.rod1.set_ohe(PartPrimitiveType.Rod1OfCarSeat)
        self.rod2 = Rod({"length": 37, "radius": 5.7}, parent_part=self)
        self.rod2.set_frame(Frame([-50.8,556.9,763], [0,0,180]))
        self.rod2.child_id = 1
        self.rod2.set_ohe(PartPrimitiveType.Rod2OfCarSeat)
        self.children = [self.rod1, self.rod2]
        self.children_varnames = ["rod1", "rod2"]
        self.children_pairwise_ranges = {(23, 24): (101.5999984741211, 106.71641150276069)}
        self.primitive_types = ["Rod"]
        self.is_concave = True
        self.init_mesh_metrics()
