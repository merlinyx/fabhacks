from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
from ..primitives.edge import Edge
import os

class Pen(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.Pen
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Pen.obj")
        self.rod = Rod({"length": 137, "radius": 5.9}, parent_part=self)
        self.rod.set_frame(Frame([68.4,30,0], [90,180,90]))
        self.rod.child_id = 0
        self.rod.set_ohe(PartPrimitiveType.RodOfPen)
        self.edge = Edge({"width": 5.5, "length": 47, "height": 6.1}, parent_part=self)
        self.edge.set_frame(Frame([113.5,30,-8.2], [0,0,0]))
        self.edge.child_id = 1
        self.edge.set_ohe(PartPrimitiveType.EdgeOfPen)
        self.children = [self.rod, self.edge]
        self.children_varnames = ["rod", "edge"]
        self.children_pairwise_ranges = {(51, 52): (3.5857115526894363, 128.6257995473691)}
        self.primitive_types = ["Rod", "Edge"]
        self.init_mesh_metrics()
