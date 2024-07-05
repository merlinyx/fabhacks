from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
import os

class RingLink(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.RingLink
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "RingLink.obj")
        self.hook1 = Hook({"arc_radius": 18, "arc_angle": 150, "thickness": 4, "can_flex": True, "is_open": False, "is_open_for_connect": True}, parent_part=self)
        self.hook1.set_frame(Frame([13.5,0,0], [-105,90,0]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfRingLink)
        self.hook2 = Hook({"arc_radius": 18, "arc_angle": 150, "thickness": 4, "can_flex": True, "is_open": False, "is_open_for_connect": True}, parent_part=self)
        self.hook2.set_frame(Frame([-13.5,0,0], [75,90,0]))
        self.hook2.child_id = 1
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfRingLink)
        self.children = [self.hook1, self.hook2]
        self.children_varnames = ["hook1", "hook2"]
        self.children_pairwise_ranges = {(99, 100): (38.30978775024414, 54.95925558634855)}
        self.primitive_types = ["Hook"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 0]
        self.attach_length = 58
        self.init_mesh_metrics()
