from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
from ..primitives.surface import Surface
import os

class HandleBasket(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.HandleBasket
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "HandleBasket.obj")
        self.rod = Rod({"radius": 3, "length": 153, "is_open": False}, parent_part=self)
        self.rod.set_frame(Frame([-31.4,178,297.1], [0,90,90]))
        self.rod.child_id = 0
        self.rod.set_ohe(PartPrimitiveType.RodOfHandleBasket)
        self.surface = Surface({"width": 118, "length": 141}, parent_part=self)
        self.surface.set_frame(Frame([-31.3,177.9,143], [0,0,90]))
        self.surface.child_id = 1
        self.surface.set_ohe(PartPrimitiveType.SurfaceOfHandleBasket)
        self.children = [self.rod, self.surface]
        self.children_varnames = ["rod", "surface"]
        self.children_pairwise_ranges = {(38, 39): (154.05065645971393, 208.86231653716212)}
        self.primitive_types = ["Rod", "Surface"]
        self.is_concave = True
        self.init_mesh_metrics()
