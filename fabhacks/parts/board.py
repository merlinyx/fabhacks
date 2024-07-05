from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.surface import Surface
import os

class Board(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.Board
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Board.obj")
        self.surface1 = Surface({"width": 180, "length": 280}, parent_part=self)
        self.surface1.set_frame(Frame([0,-10,0], [90,0,0]))
        self.surface1.child_id = 0
        self.surface1.set_ohe(PartPrimitiveType.Surface1OfBoard)
        self.surface2 = Surface({"width": 180, "length": 280}, parent_part=self)
        self.surface2.set_frame(Frame([0,0,0], [90,0,0]))
        self.surface2.child_id = 1
        self.surface2.set_ohe(PartPrimitiveType.Surface2OfBoard)
        self.children = [self.surface1, self.surface2]
        self.children_varnames = ["surface1", "surface2"]
        self.children_pairwise_ranges = {(16, 17): (10.000000189989805, 293.8757836130351)}
        self.primitive_types = ["Surface"]
        self.is_concave = True
        self.init_mesh_metrics()
