from .backseats import BackSeats
from .basket import Basket
from .basketbase import BasketBase
from .baskethandle import BasketHandle
from .binderclip import BinderClip
from .birdfeeder import BirdFeeder
from .birdfeederenv import BirdFeederEnv
from .board import Board
from .broomrod import BroomRod
from .cable import Cable
from .carseat import CarSeat
from .ceilinghook import CeilingHook
from .cliplight import ClipLight
from .closetrods import ClosetRods
from .clothesclip import ClothesClip
from .curtainring import CurtainRing
from .curtainrod import ShowerCurtainRod
from .diapercaddy import DiaperCaddy
from .doublehook import DoubleHook
from .doublehookmed import DoubleHookMedium
from .fabrichanger import FabricHanger
from .flagrod import FlagRod
from .handlebasket import HandleBasket
from .hanger import Hanger
from .hookeyeleft import HookEyeLeft
from .hookeyeleftstudy import HookEyeLeftS
from .hookeyeright import HookEyeRight, HookEyeConnector
from .hulahoop import HulaHoop
from .longtensionrod import LongTensionRod
from .mug import Mug
from .obsringm import ObstacleRingM
from .obsringl import ObstacleRingL
from .openhandlebasket import OpenHandleBasket
from .papertowel import PaperTowelRoll
from .paramturnbuckle import ParamTurnbuckle
from .pen import Pen
from .phone import Phone
from .plasticclip import PlasticClip
from .readingroom import ReadingRoom
from .ringl import RingL
from .ringlink import RingLink
from .ringm import RingM
from .rings import RingS
from .ringxs import RingXS
from .roundhandlebasket import RoundHandleBasket
from .smallbirdfeederenv import SmallBirdFeederEnv
from .soapbottle import SoapBottle
from .sodacantab import SodaCanTab
from .tack import Tack
from .toothbrush import Toothbrush
from .towelhangingenv import TowelHangingEnv
from .turnbuckle import Turnbuckle
from .twoholebasket import TwoholeBasket
from .wallhook import WallHook

from .part import PartType

class PartsLibrary:
    def __init__(self, parts_list):
        self.parts = []
        for part_info in parts_list:
            pre_init_parts = []
            for _ in range(part_info["num"]):
                pre_init_parts.append(part_info["constructor"]())
            self.parts.append({"name": part_info["constructor"].__name__, "parts": pre_init_parts, "num": part_info["num"], "curr_num": 0})
        self.end_envs = None
        self.end_connected = []

    def set_end_envs(self, end_envs):
        self.end_envs = end_envs
        self.end_connected = [False] * len(end_envs)

    def all_end_connected(self):
        for connected in self.end_connected:
            if not connected:
                return False
        return True

    def used_parts(self):
        full_list = []
        for part in self.parts:
            for i in range(part["curr_num"]):
                full_list.append(part["parts"][i])
        end_parts = []
        end_ids = []
        for i, connected in enumerate(self.end_connected):
            if connected:
                if self.end_envs[i].parent.id not in end_parts:
                    end_parts.append(self.end_envs[i].parent.id)
                    end_ids.append(i)
        for end_id in end_ids:
            full_list.append(self.end_envs[end_id].parent)
        return full_list

    def all_parts(self):
        full_list = []
        for part in self.parts:
            if part["num"] > 0:
                full_list.append(part["parts"][part["curr_num"]])
        if not self.all_end_connected():
            end_parts = []
            end_ids = []
            for i, connected in enumerate(self.end_connected):
                if not connected:
                    if self.end_envs[i].parent.id not in end_parts:
                        end_parts.append(self.end_envs[i].parent.id)
                        end_ids.append(i)
            for end_id in end_ids:
                full_list.append(self.end_envs[end_id].parent)
        return full_list

    def has_part(self, part_name):
        for part in self.parts:
            if part["name"] == part_name:
                return part["num"] > 0
        for i, end_env in enumerate(self.end_envs):
            if end_env.parent.__class__.__name__ == part_name:
                return self.end_connected[i]
        return False
    
    def use_part(self, part_name):
        for part in self.parts:
            if part["name"] == part_name:
                part["num"] -= 1
                used_part = part["parts"][part["curr_num"]]
                part["curr_num"] += 1
                used_part.name_in_assembly = "PART_{}_{}".format(part_name.lower(), part["curr_num"])
                return False, used_part
        for i, end_env in enumerate(self.end_envs):
            if end_env.parent.__class__.__name__ == part_name:
                self.end_connected[i] = True
                return True, end_env.parent
        return False, None

class PartsLibraryG(PartsLibrary):
    def __init__(self, parts_list):
        self.parts = parts_list
        self.end_envs = None
        self.end_parts = {}
        self.constructed_parts = {}

    def set_end_envs(self, end_envs):
        self.end_envs = end_envs
        self.end_parts = {}
        for end_env in self.end_envs:
            if end_env.parent.id not in self.end_parts:
                self.end_parts[end_env.parent.id] = end_env.parent

    def all_parts(self, assembly_ends=[]):
        full_list = []
        for part in self.parts:
            full_list.append(part["constructor"]())
        end_list = []
        if len(assembly_ends) > 0:
            assembly_end_ids = set([end.parent.id for end in assembly_ends])
            for end_env_id in self.end_parts.keys():
                if end_env_id not in assembly_end_ids:
                    end_list.append(self.end_parts[end_env_id])
        else:
            end_list = self.end_parts.values()
        return full_list, end_list

    def all_parts_for_search(self):
        pre_init_parts = []
        for part_info in self.parts:
            for _ in range(part_info["num"]):
                pre_init_parts.append(part_info["constructor"]())
        return pre_init_parts

    def get_part(self, part_name, force_new=False):
        if part_name in self.constructed_parts and not force_new:
            return self.constructed_parts[part_name]
        for part in self.parts:
            if part["constructor"].__name__ == part_name:
                if "Param" in part_name:
                    if force_new:
                        return part["constructor"](part["constructor"].default_part_params, part["constructor"].default_scafpath)
                    else:
                        self.constructed_parts[part_name] = part["constructor"](part["constructor"].default_part_params, part["constructor"].default_scafpath)
                        return self.constructed_parts[part_name]
                else:
                    if force_new:
                        return part["constructor"]()
                    else:
                        self.constructed_parts[part_name] = part["constructor"]()
                        return self.constructed_parts[part_name]
        return None

    def has_parts(self, assembly, new_part=None):
        counts = {}
        for _, part in assembly.part_graph.nodes(data="part"):
            if part.name_in_assembly.startswith("ENV_"):
                key_str = part.__class__.__name__
            else:
                key_str = part.name_in_assembly.split("_")[0]
            counts[key_str] = counts.get(key_str, 0) + 1
        if new_part is not None:
            if new_part.name_in_assembly.startswith("ENV_"):
                key_str = new_part.__class__.__name__
            else:
                key_str = new_part.name_in_assembly.split("_")[0]
            counts[key_str] = counts.get(key_str, 0) + 1
        for part in self.parts:
            if part["constructor"].__name__ in counts:
                if counts[part["constructor"].__name__] > part["num"]:
                    return False
        for part in self.end_parts:
            if part.__class__.__name__ in counts:
                if counts[part.__class__.__name__] > 1:
                    return False
        return True

part_to_constructor = {
    PartType.BackSeats: BackSeats,
    PartType.Basket: Basket,
    PartType.BasketBase: BasketBase,
    PartType.BasketHandle: BasketHandle,
    PartType.BinderClip: BinderClip,
    PartType.BirdFeeder: BirdFeeder,
    PartType.BirdFeederEnv: BirdFeederEnv,
    PartType.Board: Board,
    PartType.BroomRod: BroomRod,
    PartType.Cable: Cable,
    PartType.CarSeat: CarSeat,
    PartType.CeilingHook: CeilingHook,
    PartType.ClipLight: ClipLight,
    PartType.ClosetRods: ClosetRods,
    PartType.ClothesClip: ClothesClip,
    PartType.CurtainRing: CurtainRing,
    PartType.ShowerCurtainRod: ShowerCurtainRod,
    PartType.DiaperCaddy: DiaperCaddy,
    PartType.DoubleHook: DoubleHook,
    PartType.DoubleHookMedium: DoubleHookMedium,
    PartType.FlagRod: FlagRod,
    PartType.FabricHanger: FabricHanger,
    PartType.HandleBasket: HandleBasket,
    PartType.Hanger: Hanger,
    PartType.HookEyeLeft: HookEyeLeft,
    PartType.HookEyeLeftS: HookEyeLeftS,
    PartType.HookEyeConnector: HookEyeConnector,
    PartType.HookEyeRight: HookEyeRight,
    PartType.HulaHoop: HulaHoop,
    PartType.Mug: Mug,
    PartType.ObstacleRingL: ObstacleRingL,
    PartType.ObstacleRingM: ObstacleRingM,
    PartType.OpenHandleBasket: OpenHandleBasket,
    PartType.PaperTowelRoll: PaperTowelRoll,
    PartType.ParamTurnbuckle: ParamTurnbuckle,
    PartType.Pen: Pen,
    PartType.Phone: Phone,
    PartType.PlasticClip: PlasticClip,
    PartType.ReadingRoom: ReadingRoom,
    PartType.RingL: RingL,
    PartType.RingLink: RingLink,
    PartType.RingM: RingM,
    PartType.RingS: RingS,
    PartType.RingXS: RingXS,
    PartType.RoundHandleBasket: RoundHandleBasket,
    PartType.SmallBirdFeederEnv: SmallBirdFeederEnv,
    PartType.SoapBottle: SoapBottle,
    PartType.SodaCanTab: SodaCanTab,
    PartType.Tack: Tack,
    PartType.Toothbrush: Toothbrush,
    PartType.TwoholeBasket: TwoholeBasket,
    PartType.WallHook: WallHook,
}

part_to_english_name = {
    PartType.BackSeats: "back seats",
    PartType.Basket: "basket",
    PartType.BasketBase: "basket base",
    PartType.BasketHandle: "basket handle",
    PartType.BinderClip: "binder clip",
    PartType.BirdFeeder: "bird feeder",
    PartType.BirdFeederEnv: "bird feeder environment",
    PartType.Board: "board",
    PartType.BroomRod: "broom rod",
    PartType.Cable: "cable",
    PartType.CarSeat: "car seat",
    PartType.CeilingHook: "ceiling hook",
    PartType.ClipLight: "clippable light",
    PartType.ClothesClip: "clothes clip",
    PartType.CurtainRing: "shower curtain ring",
    PartType.ShowerCurtainRod: "shower curtain rod",
    PartType.DiaperCaddy: "diaper caddy",
    PartType.DoubleHook: "double hook",
    PartType.DoubleHookMedium: "medium double hook",
    PartType.FlagRod: "flag rod",
    PartType.FabricHanger: "vintage crocheted hanger",
    PartType.HandleBasket: "basket with a handle",
    PartType.Hanger: "hanger",
    PartType.HookEyeLeft: "hookeye connector (left facing)",
    PartType.HookEyeLeftS: "hookeye connector (left facing)",
    PartType.HookEyeConnector: "hookeye connector",
    PartType.HookEyeRight: "hookeye connector (right facing)",
    PartType.HulaHoop: "hula hoop",
    PartType.Mug: "mug",
    PartType.ObstacleRingL: "obstacle ring (size L)",
    PartType.ObstacleRingM: "obstacle ring (size M)",
    PartType.OpenHandleBasket: "open handle basket",
    PartType.PaperTowelRoll: "paper towel roll",
    PartType.ParamTurnbuckle: "extendable turnbuckle",
    PartType.Pen: "pen",
    PartType.Phone: "phone",
    PartType.PlasticClip: "plastic clip",
    PartType.ReadingRoom: "reading room",
    PartType.RingL: "toy toss ring (size L)",
    PartType.RingLink: "ring link",
    PartType.RingM: "toy toss ring (size M)",
    PartType.RingS: "toy toss ring (size S)",
    PartType.RingXS: "toy toss ring (size XS)",
    PartType.RoundHandleBasket: "round handle basket",
    PartType.SmallBirdFeederEnv: "small bird feeder environment",
    PartType.SoapBottle: "soap bottle",
    PartType.SodaCanTab: "soda can tab",
    PartType.Tack: "tack",
    PartType.Toothbrush: "toothbrush",
    PartType.TwoholeBasket: "two-hole basket",
    PartType.WallHook: "sticky wall hook",
}

env_to_description = {
    PartType.RoundHandleBasket: "the hook refers to the round handle at the middle of the basket",
    PartType.ClosetRods: "Note: it refers to the environment setup where two towel rods are already on the wall, and rod1 and rod2 are rod parts of the two towel rods",
    PartType.BirdFeederEnv: "Note: it refers to the environment setup where two sticky hooks are already on two walls, and hook1 and hook2 are the sticky hooks' free hook ends",
    PartType.BackSeats: "Note: rod1 is the right metal rod on the left seat, and rod2 is the left metal rod on the right seat",
    PartType.DiaperCaddy: "the hooks refer to the two round handles on the diaper caddy",
    PartType.Mug: "hook refers to the handle of the mug, tube refers to the cup of the mug",
    PartType.Cable: "Note: rod1 and rod2 are two segments on the charger cable, and we only want to connect to rod1",
    PartType.BinderClip: "ring1 and ring2 refer to the two loops that the binderclip has",
    PartType.Basket: "rod1 and rod2 refer to the two handles of the basket, surface refers to the inside bottom surface of the basket",
    PartType.SoapBottle: "surface refers to the bottom surface of the bottle, rod refers to the body of the bottle",
    PartType.ReadingRoom: "rod1 refers to the shelf's edge, rod2 refers to the camera's neck",
    PartType.Hanger: "Note: hook1 is the top hook of the hanger, hook2 and hook3 are the side bent of the hanger, rod is the bottom rod of the hanger",
    PartType.FabricHanger: "Note: hook1 is the top hook of the hanger, hook2 and hook3 are the side bent of the hanger, rod is the bottom rod of the hanger",
}

def complete_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor": BasketBase, "num": 1})
    parts_list.append({"constructor": BasketHandle, "num": 1})
    parts_list.append({"constructor": BackSeats, "num": 1})
    parts_list.append({"constructor": BirdFeeder, "num": 1})
    parts_list.append({"constructor": BirdFeederEnv, "num": 1})
    parts_list.append({"constructor": Cable, "num": 1})
    parts_list.append({"constructor": CarSeat, "num": 1})
    parts_list.append({"constructor": ClosetRods, "num": 1})
    parts_list.append({"constructor": ClipLight, "num": 1})
    parts_list.append({"constructor": DiaperCaddy, "num": 1})
    parts_list.append({"constructor": LongTensionRod, "num": 1})
    parts_list.append({"constructor": HandleBasket, "num": 1})
    parts_list.append({"constructor": HulaHoop, "num": 1})
    parts_list.append({"constructor": Mug, "num": 1})
    parts_list.append({"constructor": PaperTowelRoll, "num": 1})
    parts_list.append({"constructor": Pen, "num": 1})
    parts_list.append({"constructor": Phone, "num": 1})
    parts_list.append({"constructor": SmallBirdFeederEnv, "num": 1})
    parts_list.append({"constructor": SoapBottle, "num": 1})
    parts_list.append({"constructor": Toothbrush, "num": 1})
    parts_list.append({"constructor": TowelHangingEnv, "num": 1})
    parts_list.append({"constructor": TwoholeBasket, "num": 1})
    parts_list.append({"constructor": OpenHandleBasket, "num": 1})
    parts_list.append({"constructor": RoundHandleBasket, "num": 1})
    parts_list.append({"constructor": ReadingRoom, "num": 1})
    parts_list.append({"constructor": Basket, "num": 1})
    parts_list.append({"constructor": BinderClip, "num": 1})
    parts_list.append({"constructor": BroomRod, "num": 1})
    parts_list.append({"constructor": ClothesClip, "num": 1})
    parts_list.append({"constructor": CurtainRing, "num": 1})
    parts_list.append({"constructor": ShowerCurtainRod, "num": 1})
    parts_list.append({"constructor": DoubleHook, "num": 1})
    parts_list.append({"constructor": DoubleHookMedium, "num": 1})
    parts_list.append({"constructor": FabricHanger, "num": 1})
    parts_list.append({"constructor": FlagRod, "num": 1})
    parts_list.append({"constructor": Hanger, "num": 1})
    parts_list.append({"constructor": HookEyeLeft, "num": 1})
    parts_list.append({"constructor": HookEyeRight, "num": 1})
    parts_list.append({"constructor": PlasticClip, "num": 1})
    parts_list.append({"constructor": Turnbuckle, "num": 1})
    parts_list.append({"constructor": SodaCanTab, "num": 1})
    parts_list.append({"constructor": WallHook, "num": 1})
    parts_list.append({"constructor": RingLink, "num": 1})
    parts_list.append({"constructor": RingXS, "num": 1})
    parts_list.append({"constructor": RingS, "num": 1})
    parts_list.append({"constructor": RingM, "num": 1})
    parts_list.append({"constructor": RingL, "num": 1})
    parts_list.append({"constructor": ObstacleRingM, "num": 1})
    parts_list.append({"constructor": ObstacleRingL, "num": 1})
    parts_list.append({"constructor": TwoholeBasket, "num": 1})
    return partslibrary_constructor(parts_list)

def env_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor": BackSeats, "num": 1})
    parts_list.append({"constructor": BirdFeeder, "num": 1})
    parts_list.append({"constructor": BirdFeederEnv, "num": 1})
    parts_list.append({"constructor": Cable, "num": 1})
    parts_list.append({"constructor": CarSeat, "num": 1})
    parts_list.append({"constructor": ClosetRods, "num": 1})
    parts_list.append({"constructor": ClipLight, "num": 1})
    parts_list.append({"constructor": DiaperCaddy, "num": 1})
    parts_list.append({"constructor": LongTensionRod, "num": 1})
    parts_list.append({"constructor": HandleBasket, "num": 1})
    parts_list.append({"constructor": HulaHoop, "num": 1})
    parts_list.append({"constructor": Mug, "num": 1})
    parts_list.append({"constructor": PaperTowelRoll, "num": 1})
    parts_list.append({"constructor": Pen, "num": 1})
    parts_list.append({"constructor": Phone, "num": 1})
    parts_list.append({"constructor": SmallBirdFeederEnv, "num": 1})
    parts_list.append({"constructor": SoapBottle, "num": 1})
    parts_list.append({"constructor": Toothbrush, "num": 1})
    parts_list.append({"constructor": TowelHangingEnv, "num": 1})
    parts_list.append({"constructor": TwoholeBasket, "num": 1})
    parts_list.append({"constructor": OpenHandleBasket, "num": 1})
    parts_list.append({"constructor": RoundHandleBasket, "num": 1})
    parts_list.append({"constructor": ReadingRoom, "num": 1})
    return partslibrary_constructor(parts_list)

def full_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor": Basket, "num": 1})
    parts_list.append({"constructor": BinderClip, "num": 1})
    parts_list.append({"constructor": BroomRod, "num": 1})
    parts_list.append({"constructor": ClothesClip, "num": 1})
    parts_list.append({"constructor": CurtainRing, "num": 1})
    parts_list.append({"constructor": ShowerCurtainRod, "num": 1})
    parts_list.append({"constructor": DoubleHook, "num": 1})
    parts_list.append({"constructor": DoubleHookMedium, "num": 1})
    parts_list.append({"constructor": FabricHanger, "num": 1})
    parts_list.append({"constructor": FlagRod, "num": 1})
    parts_list.append({"constructor": Hanger, "num": 1})
    parts_list.append({"constructor": HookEyeLeft, "num": 1})
    parts_list.append({"constructor": HookEyeRight, "num": 1})
    parts_list.append({"constructor": PlasticClip, "num": 1})
    parts_list.append({"constructor": Turnbuckle, "num": 1})
    parts_list.append({"constructor": SodaCanTab, "num": 1})
    parts_list.append({"constructor": WallHook, "num": 1})
    parts_list.append({"constructor": RingLink, "num": 1})
    parts_list.append({"constructor": RingXS, "num": 1})
    parts_list.append({"constructor": RingS, "num": 1})
    parts_list.append({"constructor": RingM, "num": 1})
    parts_list.append({"constructor": RingL, "num": 1})
    parts_list.append({"constructor": ObstacleRingM, "num": 1})
    parts_list.append({"constructor": ObstacleRingL, "num": 1})
    parts_list.append({"constructor": TwoholeBasket, "num": 1})
    return partslibrary_constructor(parts_list)

def demo_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : DoubleHook, "num" : 1})
    parts_list.append({"constructor" : Hanger, "num" : 1})
    return partslibrary_constructor(parts_list)

def demo_larger_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : DoubleHook, "num" : 1})
    parts_list.append({"constructor" : SodaCanTab, "num": 1})
    parts_list.append({"constructor" : HookEyeLeft, "num" : 1})
    parts_list.append({"constructor" : HookEyeRight, "num" : 1})
    parts_list.append({"constructor" : Hanger, "num" : 1})
    return partslibrary_constructor(parts_list)

def clip_toothbrush_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : PlasticClip, "num" : 1})
    parts_list.append({"constructor" : BinderClip, "num" : 1})
    parts_list.append({"constructor" : ClothesClip, "num" : 1})
    return partslibrary_constructor(parts_list)

def clip_cable_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : BinderClip, "num" : 1})
    return partslibrary_constructor(parts_list)

def clip_cable_larger_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : PlasticClip, "num" : 1})
    parts_list.append({"constructor" : BinderClip, "num" : 1})
    parts_list.append({"constructor" : ClothesClip, "num" : 1})
    parts_list.append({"constructor" : DoubleHook, "num" : 1})
    return partslibrary_constructor(parts_list)

def bathroom_organizer_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : DoubleHook, "num" : 2})
    parts_list.append({"constructor" : Basket, "num" : 1})
    return partslibrary_constructor(parts_list)

def basket_v2_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : HookEyeLeft, "num" : 1})
    parts_list.append({"constructor" : HookEyeRight, "num" : 1})
    parts_list.append({"constructor" : Basket, "num" : 1})
    return partslibrary_constructor(parts_list)

def basket_v2_larger_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : HookEyeLeft, "num" : 2})
    parts_list.append({"constructor" : HookEyeRight, "num" : 2})
    parts_list.append({"constructor" : DoubleHook, "num" : 2})
    parts_list.append({"constructor" : Basket, "num" : 1})
    return partslibrary_constructor(parts_list)

def scarf_organizer_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : DoubleHook, "num" : 3})
    parts_list.append({"constructor" : Hanger, "num" : 1})
    return partslibrary_constructor(parts_list)

def mug_hanger_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : DoubleHook, "num" : 3})
    return partslibrary_constructor(parts_list)

def mug_hanger_larger_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : DoubleHook, "num" : 3})
    parts_list.append({"constructor" : HookEyeLeft, "num" : 1})
    parts_list.append({"constructor" : HookEyeRight, "num" : 1})
    parts_list.append({"constructor" : Hanger, "num" : 1})
    return partslibrary_constructor(parts_list)

def basket_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : HookEyeLeft, "num" : 1})
    parts_list.append({"constructor" : HookEyeRight, "num" : 1})
    parts_list.append({"constructor" : TwoholeBasket, "num" : 1})
    return partslibrary_constructor(parts_list)

def cantab_hangers_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : SodaCanTab, "num" : 2})
    parts_list.append({"constructor" : Hanger, "num" : 3})
    return partslibrary_constructor(parts_list)

def handle_basket_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : BasketHandle, "num" : 1})
    parts_list.append({"constructor" : HookEyeLeft, "num" : 1})
    parts_list.append({"constructor" : HookEyeRight, "num" : 1})
    parts_list.append({"constructor" : WallHook, "num" : 1})
    return partslibrary_constructor(parts_list)

def handle_basket_variation_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : HookEyeLeft, "num" : 1})
    parts_list.append({"constructor" : HookEyeRight, "num" : 1})
    parts_list.append({"constructor" : WallHook, "num" : 1})
    return partslibrary_constructor(parts_list)

def bird_feeder_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : HookEyeRight, "num" : 2})
    parts_list.append({"constructor" : DoubleHook, "num" : 2})
    parts_list.append({"constructor" : Hanger, "num" : 1})
    return partslibrary_constructor(parts_list)

def bird_feeder_study_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : HookEyeLeft, "num" : 1})
    parts_list.append({"constructor" : HookEyeRight, "num" : 1})
    parts_list.append({"constructor" : DoubleHook, "num" : 1})
    parts_list.append({"constructor" : Hanger, "num" : 1})
    return partslibrary_constructor(parts_list)

def small_bird_feeder_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : DoubleHook, "num" : 2})
    parts_list.append({"constructor" : Hanger, "num" : 1})
    return partslibrary_constructor(parts_list)

def paper_towel_holder_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : WallHook, "num" : 2})
    parts_list.append({"constructor" : HookEyeLeft, "num" : 2})
    parts_list.append({"constructor" : BroomRod, "num" : 1})
    return partslibrary_constructor(parts_list)

def backseat_diaper_caddy_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : DoubleHook, "num" : 4})
    return partslibrary_constructor(parts_list)

def reading_nook_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : RingLink, "num" : 10})
    parts_list.append({"constructor" : ObstacleRingM, "num" : 6})
    parts_list.append({"constructor" : ObstacleRingL, "num" : 6})
    parts_list.append({"constructor" : DoubleHook, "num" : 4})
    parts_list.append({"constructor" : ParamTurnbuckle, "num" : 2})
    return partslibrary_constructor(parts_list)

def clip_light_library(partslibrary_constructor=PartsLibrary):
    parts_list = []
    parts_list.append({"constructor" : FabricHanger, "num" : 2})
    parts_list.append({"constructor" : Turnbuckle, "num" : 4})
    parts_list.append({"constructor" : RingXS, "num" : 4})
    parts_list.append({"constructor" : RingS, "num" : 4})
    parts_list.append({"constructor" : RingM, "num" : 4})
    parts_list.append({"constructor" : RingL, "num" : 4})
    return partslibrary_constructor(parts_list)
