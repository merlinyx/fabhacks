from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.clip import Clip
from ..primitives.hook import Hook
import os

class ClothesClip(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.ClothesClip
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "ClothesClip.obj")
        self.hook = Hook({"arc_radius": 3.8, "arc_angle": 360, "thickness": 1}, parent_part=self)
        self.hook.set_frame(Frame([0,-4.6,5], [0,90,0]))
        self.hook.child_id = 0
        self.hook.set_ohe(PartPrimitiveType.HookOfClothesClip)
        self.clip = Clip({"width": 10, "height": 32, "thickness": 1.5, "dist": 2.4, "open_gap": 7.8}, parent_part=self)
        self.clip.set_frame(Frame([0,16,5], [90,180,0]))
        self.clip.child_id = 1
        self.clip.set_ohe(PartPrimitiveType.ClipOfClothesClip)
        self.children = [self.hook, self.clip]
        self.children_varnames = ["hook", "clip"]
        self.children_pairwise_ranges = {(27, 28): (2.1176180651688283, 39.08285421062852)}
        self.primitive_types = ["Hook", "Clip"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 1]
        self.init_mesh_metrics()
