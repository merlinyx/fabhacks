from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.ring import Ring
from ..primitives.surface import Surface
import os

class BasketBase(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.BasketBase
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "BasketBase.obj")
        self.ring1 = Ring({"arc_radius": 3.5, "thickness": 1}, parent_part=self)
        self.ring1.set_frame(Frame([0,-151.8,43.1], [0,0,-90]))
        self.ring1.child_id = 0
        self.ring1.set_ohe(PartPrimitiveType.Ring1OfBasketBase)
        self.ring2 = Ring({"arc_radius": 3.5, "thickness": 1}, parent_part=self)
        self.ring2.set_frame(Frame([0,-0.7,43.1], [0,180,-90]))
        self.ring2.child_id = 1
        self.ring2.set_ohe(PartPrimitiveType.Ring2OfBasketBase)
        self.surface = Surface({"width": 118, "length": 141}, parent_part=self)
        self.surface.set_frame(Frame([0,-76,-41], [0,0,90]))
        self.surface.child_id = 2
        self.surface.set_ohe(PartPrimitiveType.SurfaceOfBasketBase)
        self.children = [self.ring1, self.ring2, self.surface]
        self.children_varnames = ["ring1", "ring2", "surface"]
        self.children_pairwise_ranges = {(4, 5): (151.1000230703606, 151.18270716955297), (4, 6): (82.74649012640832, 171.23248688390314), (5, 6): (82.66539109706147, 170.83022784798376)}
        self.primitive_types = ["Ring", "Surface"]
        self.is_concave = True
        self.init_mesh_metrics()
