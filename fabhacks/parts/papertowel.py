from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.tube import Tube
import os

class PaperTowelRoll(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.PaperTowelRoll
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "PaperTowelRoll.obj")
        self.tube = Tube({"inner_radius": 21, "thickness": 29, "length": 279.4, "centerline": True}, parent_part=self)
        self.tube.set_frame(Frame([0,0,-140], [0,0,0]))
        self.tube.child_id = 0
        self.tube.set_ohe(PartPrimitiveType.TubeOfPaperTowelRoll)
        self.children = [self.tube]
        self.children_varnames = ["tube"]
        self.primitive_types = ["Tube"]
        self.is_concave = True
        self.init_mesh_metrics()
