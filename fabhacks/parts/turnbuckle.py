from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
from solid import *
import os

class Turnbuckle(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.Turnbuckle
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Turnbuckle.obj")
        self.hook1 = Hook({"arc_radius": 5.75 , "arc_angle": 220, "thickness": 1.5}, parent_part=self)
        # self.hook1.set_frame(Frame([2,63.5,0], [175,90,0]))
        self.hook1.set_frame(Frame([1.45,63.5,-1.45], [175,135,0]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfTurnbuckle)
        self.hook2 = Hook({"arc_radius": 5.75 , "arc_angle": 220, "thickness": 1.5, "can_flex": True}, parent_part=self)
        # self.hook2.set_frame(Frame([0,-63.5,2], [60,0,0]))
        self.hook2.set_frame(Frame([-2,-63.5,0], [-5,90,0]))
        self.hook2.child_id = 1
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfTurnbuckle)
        self.children = [self.hook1, self.hook2]
        self.children_varnames = ["hook1", "hook2"]
        self.children_pairwise_ranges = {(103, 104): (121.88044510883729, 135.53325257395863)}
        self.primitive_types = ["Hook"]
        self.is_concave = True
        self.init_mesh_metrics()
