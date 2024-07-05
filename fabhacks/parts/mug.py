from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
from ..primitives.tube import Tube
import os

class Mug(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.Mug
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Mug.obj")
        self.hook = Hook({"arc_radius": 30.5, "arc_angle": 200, "thickness": 6, "is_open": False}, parent_part=self)
        self.hook.set_frame(Frame([49.5,0,50], [10,0,90]))
        self.hook.child_id = 0
        self.hook.set_ohe(PartPrimitiveType.HookOfMug)
        self.tube = Tube({"length": 100, "inner_radius": 43, "thickness": 2.2}, parent_part=self)
        self.tube.set_frame(Frame([0,0,50], [0,0,0]))
        self.tube.child_id = 1
        self.tube.set_ohe(PartPrimitiveType.TubeOfMug)
        self.children = [self.hook, self.tube]
        self.children_varnames = ["hook", "tube"]
        self.children_pairwise_ranges = {(48, 49): (4.840790672434061, 128.86480752525281)}
        self.primitive_types = ["Hook", "Tube"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 1]
        self.init_mesh_metrics()
