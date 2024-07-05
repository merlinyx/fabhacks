from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.ring import Ring
import os

class ObstacleRingL(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.ObstacleRingL
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "ObstacleRingL.obj")
        self.ring = Ring({"arc_radius": 195, "thickness": 3}, parent_part=self)
        self.ring.set_frame(Frame([0,0,0],[0,90,0]))
        self.ring.child_id = 0
        self.ring.set_ohe(PartPrimitiveType.RingOfObstacleRingL)
        self.children = [self.ring]
        self.children_varnames = ["ring"]
        self.primitive_types = ["Ring"]
        self.is_concave = True
        self.child_symmetry_groups = [0]
        self.attach_length = 400
        self.init_mesh_metrics()
