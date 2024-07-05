from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
import os

class BasketHandle(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.BasketHandle
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "BasketHandle.obj")
        self.rod1 = Rod({"radius": 3, "length": 153, "is_open": False}, parent_part=self)
        self.rod1.set_frame(Frame([12.7,-3,20.5], [0,90,0]))
        self.rod1.child_id = 0
        self.rod1.set_ohe(PartPrimitiveType.Rod1OfBasketHandle)
        self.rod2 = Rod({"radius": 2, "length": 6.5}, parent_part=self)
        self.rod2.set_frame(Frame([-60.5,-3.1,-50], [0,90,0]))
        self.rod2.child_id = 1
        self.rod2.set_ohe(PartPrimitiveType.Rod2OfBasketHandle)
        self.rod3 = Rod({"radius": 2, "length": 6.5}, parent_part=self)
        self.rod3.set_frame(Frame([85.8,-3.1,-50], [0,-90,0]))
        self.rod3.child_id = 2
        self.rod3.set_ohe(PartPrimitiveType.Rod3OfBasketHandle)
        self.children = [self.rod1, self.rod2, self.rod3]
        self.children_varnames = ["rod1", "rod2", "rod3"]
        self.children_pairwise_ranges = {(7, 8): (70.56544648570349, 159.94726596706997), (7, 9): (70.5615033436725, 159.8575174782156), (8, 9): (140.56470489501953, 152.03529357910156)}
        self.primitive_types = ["Rod"]
        self.is_concave = True
        self.init_mesh_metrics()
