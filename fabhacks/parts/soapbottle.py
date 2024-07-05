from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
from ..primitives.surface import Surface
import os

class SoapBottle(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.SoapBottle
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "SoapBottle.obj")
        self.rod = Rod({"length": 140, "radius": 38.5}, parent_part=self)
        self.rod.set_frame(Frame([0,0,70], [0,0,0]))
        self.rod.child_id = 0
        self.rod.set_ohe(PartPrimitiveType.RodOfSoapBottle)
        self.surface = Surface({"length": 77, "width": 77}, parent_part=self)
        self.surface.set_frame(Frame([0,0,-1], [0,0,0]))
        self.surface.child_id = 1
        self.surface.set_ohe(PartPrimitiveType.SurfaceOfSoapBottle)
        self.children = [self.rod, self.surface]
        self.children_varnames = ["rod", "surface"]
        self.children_pairwise_ranges = {(57, 58): (10.488760250705955, 154.98122862788213)}
        self.primitive_types = ["Rod", "Surface"]
        self.child_symmetry_groups = [0, 1]
        self.init_mesh_metrics()
