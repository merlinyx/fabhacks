from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.ring import Ring
import os

class SodaCanTab(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.SodaCanTab
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "SodaCanTab.obj")
        self.ring1 = Ring({"arc_radius": 7, "thickness": 0.7, "centerline": True}, parent_part=self)
        self.ring1.set_frame(Frame([0,0,0.64], [0,90,0]))
        self.ring1.child_id = 0
        self.ring1.set_ohe(PartPrimitiveType.Ring1OfSodaCanTab)
        self.ring2 = Ring({"arc_radius": 3.5, "thickness": 0.5, "centerline": True}, parent_part=self)
        self.ring2.set_frame(Frame([0,12.7,0.64], [0,90,0]))
        self.ring2.child_id = 1
        self.ring2.set_ohe(PartPrimitiveType.Ring2OfSodaCanTab)
        self.children = [self.ring1, self.ring2]
        self.children_varnames = ["ring1", "ring2"]
        self.children_pairwise_ranges = {(59, 60): (12.699999809265137, 12.699999809265137)}
        self.primitive_types = ["Ring"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 0]
        self.init_mesh_metrics()
