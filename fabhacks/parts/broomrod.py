from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.tube import Tube
from .material import Material
import os

class BroomRod(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.BroomRod
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "BroomRod.obj")
        self.tube = Tube({"length": 387, "inner_radius": 8.5, "thickness": 1.5}, parent_part=self)
        self.tube.set_frame(Frame([0,0,197.5], [0,0,0]))
        self.tube.child_id = 0
        self.tube.set_ohe(PartPrimitiveType.TubeOfBroomRod)
        # self.tube1 = Tube({"length": 30, "inner_radius": 8.5, "thickness": 1.5}, parent_part=self)
        # self.tube1.set_frame(Frame([0,0,15], [0,0,-90]))
        # self.tube1.child_id = 0
        # self.tube1.set_ohe(PartPrimitiveType.Tube1OfBroomRod)
        # self.tube2 = Tube({"length": 335, "inner_radius": 8.5, "thickness": 1.5}, parent_part=self)
        # self.tube2.set_frame(Frame([0,0,197.5], [0,0,-90]))
        # self.tube2.child_id = 1
        # self.tube2.set_ohe(PartPrimitiveType.Tube2OfBroomRod)
        # self.tube3 = Tube({"length": 30, "inner_radius": 8.5, "thickness": 1.5}, parent_part=self)
        # self.tube3.set_frame(Frame([0,0,380], [0,0,-90]))
        # self.tube3.child_id = 2
        # self.tube3.set_ohe(PartPrimitiveType.Tube3OfBroomRod)
        self.children = [self.tube]#[self.tube1, self.tube2, self.tube3]
        self.children_varnames = ["tube"]#["tube1", "tube2", "tube3"]
        # self.children_pairwise_ranges = {(18, 19): (3.613862991333008, 361.38613680005074), (18, 20): (335.5940570831299, 394.40594270825386), (19, 20): (3.613861083984375, 361.3861389160156)}
        self.primitive_types = ["Tube"]
        self.is_concave = True
        self.init_mesh_metrics()
