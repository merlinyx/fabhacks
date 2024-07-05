from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
import os

class SmallBirdFeederEnv(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.SmallBirdFeederEnv
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "SmallBirdFeederEnv.obj")
        self.hook1 = Hook({"arc_radius": 12.5, "arc_angle": 180, "thickness": 1.25}, parent_part=self)
        self.hook1.set_frame(Frame([0,-245,-61], [90,0,0]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfSmallBirdFeederEnv)
        self.hook2 = Hook({"arc_radius": 12.5, "arc_angle": 180, "thickness": 1.25}, parent_part=self)
        self.hook2.set_frame(Frame([0,245,-61], [90,0,0]))
        self.hook2.child_id = 1
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfSmallBirdFeederEnv)
        self.children = [self.hook1, self.hook2]
        self.children_varnames = ["hook1", "hook2"]
        self.children_pairwise_ranges = {(71, 72): (467.5108947753906, 512.4891357421875)}
        self.primitive_types = ["Hook"]
        self.is_concave = True
        self.init_mesh_metrics()
