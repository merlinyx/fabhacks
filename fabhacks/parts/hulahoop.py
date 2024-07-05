from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.ring import Ring
from ..primitives.hook import Hook
import os

class HulaHoop(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.HulaHoop
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "HulaHoop.obj")
        # self.ring = Hole({"arc_radius": 345.75, "thickness": 11.5}, parent_part=self)
        self.ring = Ring({"arc_radius": 345.75, "thickness": 11.5, "can_flex": True}, parent_part=self)
        # self.ring = Hook({"arc_angle": 320, "arc_radius": 345.75, "thickness": 11.5, "can_flex": True, "is_open": False, "is_open_for_connect": True, "phi_ranges": [30, 150]}, parent_part=self)
        self.ring.set_frame(Frame())
        self.ring.child_id = 0
        self.ring.set_ohe(PartPrimitiveType.RingOfHulaHoop)
        self.children = [self.ring]
        self.children_varnames = ["ring"]
        self.primitive_types = ["Ring"]
        # self.primitive_types = ["Hook"]
        self.child_symmetry_groups = [0]
        self.is_concave = True
        self.attach_length = 680
        # self.mass_ratio = 0.085
        self.init_mesh_metrics()
