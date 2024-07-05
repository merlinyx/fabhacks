from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
# from ..primitives.ring import Hole
from ..primitives.hook import Hook
from ..primitives.surface import Surface
import os

class TwoholeBasket(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.TwoholeBasket
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "TwoholeBasket.obj")
        self.hook1 = Hook({"arc_radius": 19, "arc_angle":180, "thickness": 1}, parent_part=self)
        self.hook1.set_frame(Frame([117.5,0,178.2], [-90,0,-90]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfTwoholeBasket)
        self.hook2 = Hook({"arc_radius": 19, "arc_angle":180, "thickness": 1}, parent_part=self)
        self.hook2.set_frame(Frame([117.5,-340,178.2], [-90,0,90]))
        self.hook2.child_id = 1
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfTwoholeBasket)
        self.surface = Surface({"width": 340, "length": 235}, parent_part=self)
        self.surface.set_frame(Frame([117.5,-170,4]))
        self.surface.child_id = 2
        self.surface.set_ohe(PartPrimitiveType.SurfaceOfTwoholeBasket)
        self.children = [self.hook1, self.hook2, self.surface]
        self.children_varnames = ["hook1", "hook2", "surface"]
        self.children_pairwise_ranges = {(66, 67): (340.0, 341.89873678331725), (66, 68): (175.872232924224, 388.71213965612304), (67, 68): (175.872232924224, 388.71213965612304)}
        self.primitive_types = ["Hook", "Surface"]
        self.is_concave = True
        self.init_mesh_metrics()
