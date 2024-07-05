from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
from ..primitives.surface import Surface
import os

class Tack(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.Tack
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Tack.obj")
        self.rod1 = Rod({"length": 21, "radius": 3.5}, parent_part=self)
        self.rod1.set_frame(Frame([18.5,0,0], [0,90,0]))
        self.rod1.child_id = 0
        self.rod1.set_ohe(PartPrimitiveType.Rod1OfTack)
        self.rod2 = Rod({"length": 28.6, "radius": 1}, parent_part=self)
        self.rod2.set_frame(Frame([-10.2,0,0], [0,90,0]))
        self.rod2.child_id = 1
        self.rod2.set_ohe(PartPrimitiveType.Rod2OfTack)
        self.surface = Surface({"width": 10, "length": 10}, parent_part=self)
        self.surface.set_frame(Frame([4,0,0], [0,90,0]))
        self.surface.child_id = 2
        self.surface.set_ohe(PartPrimitiveType.SurfaceOfTack)
        self.children = [self.rod1, self.rod2, self.surface]
        self.children_varnames = ["rod1", "rod2", "surface"]
        self.primitive_types = ["Rod", "Surface"]
        self.children_pairwise_ranges = {(61, 62): (7.261564128939813, 50.64409634792505), (61, 63): (5.200370164197623, 25.385959542717906), (62, 63): (1.6628056299440828, 27.760068068199416)}
        self.is_concave = True
        self.init_mesh_metrics()
