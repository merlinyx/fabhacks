from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
from ..primitives.rod import Rod
import os

class FabricHanger(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.FabricHanger
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "FabricHanger.obj")
        # params based on onshape model
        self.hook1 = Hook({"arc_radius": 25, "arc_angle": 180, "thickness": 6.5, "ranges": [0, 270]}, parent_part=self)
        self.hook1.set_frame(Frame([0,0,65], [-90,0,90]))
        self.hook1.child_id = 0
        self.hook1.set_ohe(PartPrimitiveType.Hook1OfFabricHanger)
        self.rod = Rod({"length": 385, "radius": 6.5, "is_open": False}, parent_part=self)
        self.rod.set_frame(Frame([0,0,-110], [0,90,0]))
        self.rod.child_id = 1
        self.rod.set_ohe(PartPrimitiveType.RodOfFabricHanger)
        self.hook2 = Hook({"arc_radius": 15, "arc_angle": 160, "thickness": 6.5, "is_open": False}, parent_part=self)
        self.hook2.set_frame(Frame([193.5,0,-95], [0,0,90]))
        self.hook2.child_id = 2
        self.hook2.set_ohe(PartPrimitiveType.Hook2OfFabricHanger)
        self.hook3 = Hook({"arc_radius": 15, "arc_angle": 160, "thickness": 6.5, "is_open": False}, parent_part=self)
        self.hook3.set_frame(Frame([-193.5,0,-95], [160,0,90]))
        self.hook3.child_id = 3
        self.hook3.set_ohe(PartPrimitiveType.Hook3OfFabricHanger)
        self.children = [self.hook1, self.rod, self.hook2, self.hook3]
        self.children_varnames = ["hook1", "rod", "hook2", "hook3"]
        self.children_pairwise_ranges = {(91, 92): (175.26765740580703, 262.4550695407978), (91, 93): (234.80369958327432, 278.0778877057271), (91, 94): (234.22721719266872, 278.0813528081442), (92, 93): (25.590907630692286, 372.2188875044992), (92, 94): (25.590907630692286, 372.2188875044992), (93, 94): (387.469970703125, 403.9997863769531)}
        self.primitive_types = ["Hook", "Rod"]
        self.is_concave = True
        self.child_symmetry_groups = [0, 1, 2, 2]
        self.init_mesh_metrics()
