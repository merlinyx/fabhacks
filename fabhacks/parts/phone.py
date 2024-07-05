from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.edge import Edge
import os

class Phone(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.Phone
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Phone.obj")
        self.edge = Edge({"width": 65, "length": 153, "height": 8}, parent_part=self)
        self.edge.set_frame(Frame([-9.7,-4,-6.8], [90,90,0]))
        self.edge.child_id = 0
        self.edge.set_ohe(PartPrimitiveType.EdgeOfPhone)
        self.children = [self.edge]
        self.children_varnames = ["edge"]
        self.primitive_types = ["Edge"]
        self.init_mesh_metrics()
