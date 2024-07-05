from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.ring import Ring
import os

class RingS(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.RingS
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "RingS.obj")
        self.ring = Ring({"arc_radius": 43.3, "thickness": 3.3}, parent_part=self)
        self.ring.set_frame(Frame([0,0,0], [0,90,0]))
        self.ring.child_id = 0
        self.ring.set_ohe(PartPrimitiveType.RingOfRingS)
        self.children = [self.ring]
        self.children_varnames = ["ring"]
        self.primitive_types = ["Ring"]
        self.is_concave = True
        self.child_symmetry_groups = [0]
        self.init_mesh_metrics()
