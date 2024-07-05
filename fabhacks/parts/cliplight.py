from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.clip import Clip
import os

class ClipLight(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.ClipLight
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "ClipLight.obj")
        self.clip = Clip({"width": 30, "height": 30, "thickness": 1, "dist": 13.5, "open_gap": 13.5}, parent_part=self)
        self.clip.set_frame(Frame([23.6,0,14.05], [0,90,0]))
        self.clip.child_id = 0
        self.clip.set_ohe(PartPrimitiveType.ClipOfClipLight)
        self.children = [self.clip]
        self.children_varnames = ["clip"]
        self.children_pairwise_ranges = {}
        self.primitive_types = ["Clip"]
        self.is_concave = True
        self.child_symmetry_groups = [0]
        self.init_mesh_metrics()
