from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.rod import Rod
from ..primitives.hook import Hook
import os

class ReadingRoom(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.ReadingRoom
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "ReadingRoom.obj")
        # self.rod1 = Rod({"length": 750, "radius": 15}, parent_part=self)
        # self.rod1.set_frame(Frame([0, 100, 1283], [0,90,0])) # uncomment if it is smaller nook example for debugging
        self.rod1 = Rod({"length": 1500, "radius": 10}, parent_part=self)
        self.rod1.set_frame(Frame([-175, 1030, 1283], [0,90,0]))
        self.rod1.child_id = 0
        self.rod1.set_ohe(PartPrimitiveType.Rod1OfReadingRoom)
        self.rod2 = Rod({"length": 20, "radius": 2.5}, parent_part=self)
        self.rod2.set_frame(Frame([72.2, -1375, 1293.7], [120,0,0]))
        self.rod2.child_id = 1
        self.rod2.set_ohe(PartPrimitiveType.Rod2OfReadingRoom)
        # self.rod3 = Rod({"radius": 12.7, "length": 2700}, parent_part=self)
        # self.rod3.set_frame(Frame([-1300,-50,1275], [90,0,0]))
        # self.rod3.child_id = 2
        # self.rod3.set_ohe(PartPrimitiveType.Rod3OfReadingRoom)
        self.children = [self.rod1, self.rod2]#, self.rod3]
        self.children_varnames = ["rod1", "rod2"]#, "rod3"]
        self.children_pairwise_ranges = {(95, 96): (2408.8176080323974, 2588.732818868977)}
        self.primitive_types = ["Rod"]
        self.child_symmetry_groups = [0, 1]#, 2]
        self.init_mesh_metrics()
