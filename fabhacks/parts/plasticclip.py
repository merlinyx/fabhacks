from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.clip import Clip
from ..primitives.hemisphere import Hemisphere
import os

class PlasticClip(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.PlasticClip
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "PlasticClip.obj")
        self.clip = Clip({"width": 17, "height": 8, "thickness": 1, "dist": 13, "open_gap": 12}, parent_part=self)
        self.clip.set_frame(Frame([0,26,0], [90,0,180]))
        self.clip.child_id = 0
        self.clip.set_ohe(PartPrimitiveType.ClipOfPlasticClip)
        self.hemisphere1 = Hemisphere({"radius": 8.5}, parent_part=self)
        self.hemisphere1.set_frame(Frame([-12,-21.7,0], [-90,0,0]))
        self.hemisphere1.child_id = 1
        self.hemisphere1.set_ohe(PartPrimitiveType.Hemisphere1OfPlasticClip)
        self.hemisphere2 = Hemisphere({"radius": 8.5}, parent_part=self)
        self.hemisphere2.set_frame(Frame([12,-21.7,0], [-90,0,0]))
        self.hemisphere2.child_id = 2
        self.hemisphere2.set_ohe(PartPrimitiveType.Hemisphere2OfPlasticClip)
        self.children = [self.clip, self.hemisphere1, self.hemisphere2]
        self.children_varnames = ["clip", "hemisphere1", "hemisphere2"]
        self.children_pairwise_ranges = {(54, 55): (44.24331922450512, 61.45922431640474), (54, 56): (44.24331922450512, 61.45922431640474), (55, 56): (7.289457321166992, 40.82617718995033)}
        self.primitive_types = ["Clip", "Hemisphere"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 1, 1]
        self.init_mesh_metrics()
