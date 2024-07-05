from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
from ..primitives.surface import Surface
import os

class Basket(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.Basket
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Basket.obj")
        self.rod1 = Rod({"length": 72, "radius": 3, "is_open": False}, parent_part=self)
        self.rod1.set_frame(Frame([124.5,0,252], [90,180,0]))
        self.rod1.child_id = 0
        self.rod1.set_ohe(PartPrimitiveType.Rod1OfBasket)
        self.rod2 = Rod({"length": 72, "radius": 3, "is_open": False}, parent_part=self)
        self.rod2.set_frame(Frame([-124.5,0,252], [90,180,180]))
        self.rod2.child_id = 1
        self.rod2.set_ohe(PartPrimitiveType.Rod2OfBasket)
        self.surface = Surface({"width": 248, "length": 248, "is_open": False}, parent_part=self)
        self.surface.set_frame(Frame([0,0,6], [0,0,0]))
        self.surface.child_id = 2
        self.surface.set_ohe(PartPrimitiveType.SurfaceOfBasket)
        self.children = [self.rod1, self.rod2, self.surface]
        self.children_varnames = ["rod1", "rod2", "surface"]
        self.children_pairwise_ranges = {(1, 2): (249.0, 256.976625594911), (1, 3): (243.41989036236802, 365.60937196623524), (2, 3): (243.41989036236802, 365.60937196623524)}
        self.primitive_types = ["Rod", "Surface"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 0, 1]
        self.init_mesh_metrics()
