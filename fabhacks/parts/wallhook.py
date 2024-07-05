from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
from ..primitives.surface import Surface
import os

class WallHook(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.WallHook
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "WallHook.obj")
        self.hook = Hook({"arc_radius": 12.5, "arc_angle": 180, "thickness": 1.25}, parent_part=self)
        self.hook.set_frame(Frame([0,-16,-61], [90,0,0]))
        self.hook.child_id = 0
        self.hook.set_ohe(PartPrimitiveType.HookOfWallHook)
        self.surface = Surface({"length": 68, "width": 68}, parent_part=self)
        self.surface.set_frame(Frame([0,0,0], [90,-90,0]))
        self.surface.child_id = 1
        self.surface.set_ohe(PartPrimitiveType.SurfaceOfWallHook)
        self.surface.mcf_factory.set_fixed_under_physics(-1, True)
        self.children = [self.hook, self.surface]
        self.children_varnames = ["hook", "surface"]
        self.children_pairwise_ranges = {(69, 70): (31.7640629110215, 107.89194197042269)}
        self.primitive_types = ["Hook", "Surface"]
        self.child_symmetry_groups = [0, 1]
        self.is_concave = True
        self.init_mesh_metrics()
