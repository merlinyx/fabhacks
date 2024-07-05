from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.clip import Clip
from ..primitives.ring import Ring
import os

class BinderClip(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.BinderClip
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "BinderClip.obj")
        self.clip = Clip({"width": 19, "height": 11, "thickness": 0.1, "dist": 8.2, "open_gap": 3.5}, parent_part=self)
        self.clip.set_frame(Frame([15,1.2,3.5], [90,0,90]))
        self.clip.child_id = 0
        self.clip.set_ohe(PartPrimitiveType.ClipOfBinderClip)
        self.ring1 = Ring({"arc_radius": 3.6, "thickness": 0.6, "centerline": True}, parent_part=self)
        self.ring1.set_frame(Frame([28,-5.3,3.5], [90,0,-102]))
        self.ring1.child_id = 1
        self.ring1.set_ohe(PartPrimitiveType.Ring1OfBinderClip)
        self.ring2 = Ring({"arc_radius": 3.6, "thickness": 0.6, "centerline": True}, parent_part=self)
        self.ring2.set_frame(Frame([28,7.8,3.5], [90,0,102]))
        self.ring2.child_id = 2
        self.ring2.set_ohe(PartPrimitiveType.Ring2OfBinderClip)
        self.children = [self.clip, self.ring1, self.ring2]
        self.children_varnames = ["clip", "ring1", "ring2"]
        self.children_pairwise_ranges = {(10, 11): (10.007274230859181, 19.505952267632587), (10, 12): (10.07251390271822, 19.539502876378265), (11, 12): (13.100000381469727, 13.100000381469727)}
        self.primitive_types = ["Clip", "Ring"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 1, 1]
        self.init_mesh_metrics()
