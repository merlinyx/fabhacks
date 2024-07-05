from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
from ..primitives.rod import Rod
import os

class Hanger(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.Hanger
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Hanger.obj")
        # params based on hanger.scad
        self.hook1 = Hook({"arc_radius": 20, "arc_angle": 180, "thickness": 3.1, "ranges": [0, 270]}, parent_part=self)
        self.hook1.set_frame(Frame([0,0,83], [-45,0,0]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfHanger)
        self.rod = Rod({"length": 390, "radius": 3.5, "is_open": False}, parent_part=self)
        self.rod.set_frame(Frame([0,0,-110], [-90,180,0]))
        self.rod.child_id = 1
        self.rod.set_ohe(PartPrimitiveType.RodOfHanger)
        # self.neck = Rod({"length": 45, "radius": 3, "centerline": True, "is_open": False}, parent_part=self)
        # self.neck.set_frame(Frame([0,0,28], [0,180,0]))
        # self.neck.child_id = 2
        # self.neck.set_ohe(PartPrimitiveType.NeckOfHanger)
        self.hook2 = Hook({"arc_radius": 12.5, "arc_angle": 160, "thickness": 3.1, "is_open": False}, parent_part=self)
        self.hook2.set_frame(Frame([0,194.1,-97.5], [160,0,0]))
        self.hook2.child_id = 2
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfHanger)
        self.hook3 = Hook({"arc_radius": 12.5, "arc_angle": 160, "thickness": 3.1, "is_open": False}, parent_part=self)
        self.hook3.set_frame(Frame([0,-194.1,-97.5], [0,0,0]))
        self.hook3.child_id = 3
        self.hook3.set_ohe(PartPrimitiveType.Hook3OfHanger)
        self.children = [self.hook1, self.rod, self.hook2, self.hook3]
        self.children_varnames = ["hook1", "rod", "hook2", "hook3"]
        self.children_pairwise_ranges = {(40, 41): (178.12319718635385, 272.85642613758273), (40, 42): (261.1500763022817, 291.35620784622), (40, 43): (244.88783157685987, 291.3458080514316), (41, 42): (22.304563510939726, 375.66910130244673), (41, 43): (22.304563510939726, 375.66910130244673), (42, 43): (388.7197265625, 406.999755859375)}
        self.primitive_types = ["Hook", "Rod"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 1, 2, 2] # It's a little wrong to call the side hooks the same
        self.init_mesh_metrics()
