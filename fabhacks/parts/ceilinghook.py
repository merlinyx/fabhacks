from .part import Part, PartType, PartPrimitiveType
from ..assembly.frame import Frame
from ..primitives.hook import Hook
from ..primitives.surface import Surface
import os

#https://www.rejuvenation.com/products/small-open-loop-swag-hook/?catalogId=84&sku=6725051&cm_ven=PLA&cm_cat=Google&cm_pla=Hardware%20%3E%20Hooks%20%26%20Racks&cm_ite=6725051_14334675725&gclid=Cj0KCQjw7JOpBhCfARIsAL3bobcXuA6pLlpnStWXwtU6C2yp-1jc1QKnfHe480JtCsw3eTZG-DRtDd8aAmFpEALw_wcB
class CeilingHook(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.CeilingHook
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "CeilingHook.obj")
        self.hook = Hook({"arc_radius": 25, "arc_angle": 300, "thickness": 2.5}, parent_part=self)
        self.hook.set_frame(Frame([0,0,-61.5], [120,0,90]))
        self.hook.child_id = 0
        self.hook.set_ohe(PartPrimitiveType.HookOfCeilingHook)
        self.children = [self.hook]
        self.children_varnames = ["hook"]
        self.primitive_types = ["Hook"]
        self.child_symmetry_groups = [0]
        self.is_concave = True
        self.init_mesh_metrics()
