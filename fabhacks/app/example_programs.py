from ..assembly.assembly import Assembly
from ..assembly.assembly_R import AssemblyR
from ..assembly.frame import Frame
from ..primitives.edge import Edge
from ..primitives.surface import Surface
from ..primitives.rod import Rod
from ..primitives.ring import Ring
from ..primitives.hook import Hook
from ..primitives.point import Point
from ..parts.clothesclip import ClothesClip
from ..parts.curtainring import CurtainRing
from ..parts.binderclip import BinderClip
from ..parts.doublehook import DoubleHook
from ..parts.doublehookmed import DoubleHookMedium
from ..parts.hanger import Hanger
from ..parts.tack import Tack
from ..parts.soapbottle import SoapBottle
from ..parts.sodacantab import SodaCanTab
from ..parts.phone import Phone
from ..parts.pen import Pen
from ..parts.cable import Cable
from ..parts.board import Board
from ..parts.basket import Basket
from ..parts.mug import Mug
from ..parts.toothbrush import Toothbrush
from ..parts.plasticclip import PlasticClip
from ..parts.twoholebasket import TwoholeBasket
from ..parts.hookeyeleft import HookEyeLeft
from ..parts.hookeyeright import HookEyeRight
from ..parts.basketbase import BasketBase
from ..parts.baskethandle import BasketHandle
from ..parts.handlebasket import HandleBasket
from ..parts.wallhook import WallHook
from ..parts.papertowel import PaperTowelRoll
from ..parts.flagrod import FlagRod
from ..parts.curtainrod import ShowerCurtainRod
from ..parts.broomrod import BroomRod
from ..parts.carseat import CarSeat
from ..parts.birdfeeder import BirdFeeder
from ..parts.diapercaddy import DiaperCaddy
from ..parts.backseats import BackSeats
from ..parts.birdfeederenv import BirdFeederEnv
from ..parts.smallbirdfeederenv import SmallBirdFeederEnv
from ..parts.openhandlebasket import OpenHandleBasket
from ..parts.closetrods import ClosetRods
from ..parts.roundhandlebasket import RoundHandleBasket
from ..parts.hookeyeleftstudy import HookEyeLeftS
from ..parts.paramhanger import ParamHanger
from ..parts.paramturnbuckle import ParamTurnbuckle
from ..parts.cliplight import ClipLight
from ..parts.ringl import RingL
from ..parts.ringm import RingM
from ..parts.rings import RingS
from ..parts.ringxs import RingXS
from ..parts.ceilinghook import CeilingHook
from ..parts.fabrichanger import FabricHanger
from ..parts.readingroom import ReadingRoom
from ..parts.hulahoop import HulaHoop
from ..parts.ringlink import RingLink
from ..parts.obsringm import ObstacleRingM
from ..parts.obsringl import ObstacleRingL
from ..parts.turnbuckle import Turnbuckle
from ..parts.longtensionrod import LongTensionRod
from ..parts.towelhangingenv import TowelHangingEnv
from ..parts.env import Environment

from importlib_resources import files

scadfpath = str(files("fabhacks").joinpath("parts", "paramscad", "M4.scad"))

def part_viewing():
    ASSEMBLY = AssemblyR()
    # part = RoundHandleBasket()

    # part_params = {
    #     "length": 419.1,
    #     "height": 110.9,
    #     "bar_d": 12.7,
    #     # "bend_d", "hook_d", "hook_h"
    # }
    # scadfile = str(files('fabhacks').joinpath('parts', 'paramscad', 'hanger.scad'))
    # part = ParamHanger(part_params, scadfile)

    # part_params = {
    #     "extended_length": 20
    # }
    # scadfile = str(files('fabhacks').joinpath('parts', 'paramscad', 'M4.scad'))
    # part = ParamTurnbuckle(part_params, scadfile)

    # part = ClipLight()

    # part = RingL()
    # part = RingM()
    # part = RingS()
    # part = RingXS()

    # part = CeilingHook()

    # part = FabricHanger()

    # part = ReadingRoom()
    # part = HulaHoop()
    # part = ObstacleRingM()
    # part = ObstacleRingL()
    # part = Turnbuckle()
    # part = LongTensionRod()

    # part = TowelHangingEnv()
    part = PlasticClip()

    start_env = Environment({"part": part})
    start_frame = Frame()
    ASSEMBLY.add(start_env, start_frame)
    return ASSEMBLY

#0 ex0
def demo_program_R():
    ASSEMBLYR_demo = AssemblyR()
    rod = Rod({"length": 500, "radius": 10})
    ENV_start = Environment({"rod": rod})
    start_frame = Frame([0,0,400], [90,0,90])
    ASSEMBLYR_demo.add(ENV_start.rod, start_frame)
    PART_hanger = Hanger()
    ASSEMBLYR_demo.attach(PART_hanger.hook1, ENV_start.rod)
    PART_doublehook = DoubleHook()
    ASSEMBLYR_demo.attach(PART_doublehook.hook1, PART_hanger.rod)
    ring = CurtainRing()
    ENV_end = Environment({"ring": ring})
    ASSEMBLYR_demo.attach(ENV_end.hook, PART_doublehook.hook2)
    end_frame = Frame([0,2.1,142], [87,0,0])
    ASSEMBLYR_demo.end_with(ENV_end.hook, frame=end_frame)
    return ASSEMBLYR_demo

#ex1
def clip_toothbrush_2con_R():
    # toothbrush
    # plasticclip
    # environment - no start or end, just a surface
    ASSEMBLY_clip_toothbrush = AssemblyR()
    surface = Surface({"width": 400, "length": 400})
    ENV_start = Environment({"surface": surface})
    start_frame = Frame()
    ASSEMBLY_clip_toothbrush.add(ENV_start.surface, start_frame)
    PART_clip = PlasticClip()
    ASSEMBLY_clip_toothbrush.attach(PART_clip.hemisphere1, ENV_start.surface)
    ASSEMBLY_clip_toothbrush.connect(PART_clip.hemisphere2, ENV_start.surface)
    PART_toothbrush = Toothbrush()
    ENV_end = Environment({"toothbrush": PART_toothbrush})
    ASSEMBLY_clip_toothbrush.attach(ENV_end.rod, PART_clip.clip)
    ASSEMBLY_clip_toothbrush.connect(ENV_end.hemisphere, ENV_start.surface)
    end_point = Point({"minpos": [-20,-20,30], "maxpos": [20,20,50]})
    end_frame = Frame([-10, -62.5, 50], [-65,0,0])
    ASSEMBLY_clip_toothbrush.end_with(ENV_end.rod, end_frame)#point=end_point)
    return ASSEMBLY_clip_toothbrush

#ex2
def clip_cable_R():
    # binderclip
    # cable
    # environment - start is table(edge), end is cable
    # pairs: rod-(ring clip)-edge; rod-ring, clip-edge
    ASSEMBLY_clip_cable = AssemblyR()
    edge = Edge({"width": 100, "length": 200, "height": 1.5})
    ENV_start = Environment({"edge": edge})
    start_frame = Frame([0,0,150],[0,0,0])
    ASSEMBLY_clip_cable.add(ENV_start.edge, start_frame)
    PART_binderclip = BinderClip()
    ASSEMBLY_clip_cable.attach(PART_binderclip.clip, ENV_start.edge, is_fixed_joint=True)
    cable = Cable()
    ENV_end = Environment({"cable": cable})
    ASSEMBLY_clip_cable.attach(ENV_end.rod1, PART_binderclip.ring2)
    ASSEMBLY_clip_cable.connect(ENV_end.rod1, PART_binderclip.ring1)
    # end_frame = Frame([0,57.6,153], [0,0,0])
    # end_frame = Frame([42,57.6,153], [0,0,-160])
    end_frame = Frame([0,57.5,163], [0,0,0])
    ASSEMBLY_clip_cable.end_with(ENV_end.rod2, frame=end_frame)
    return ASSEMBLY_clip_cable

#ex3
def bathroom_organizer_v2_R():
    # door
    # hookeye left and right
    # basket
    # bottles
    # environment - start is door(edge), end could be bottles or basket
    # edge = Edge({"width": 300, "length": 500, "height": 10})
    # pairs: edge-(hook hook)-(rod surface)-surface; edge-hook, hook-rod, surface-surface
    ASSEMBLY_bathroom_organizer = AssemblyR()
    rod = Rod({"length": 500, "radius": 5})
    ENV_start = Environment({"door": rod})
    start_frame = Frame([0,0,500], [90,0,90])
    ASSEMBLY_bathroom_organizer.add(ENV_start.door, start_frame)
    PART_hookeye1 = HookEyeLeftS()
    ASSEMBLY_bathroom_organizer.attach(PART_hookeye1.ring, ENV_start.door)
    PART_basket = Basket()
    ASSEMBLY_bathroom_organizer.attach(PART_basket.rod1, PART_hookeye1.hook)
    PART_hookeye2 = HookEyeLeftS()
    ASSEMBLY_bathroom_organizer.attach(PART_hookeye2.ring, ENV_start.door, alignment="flip")
    ASSEMBLY_bathroom_organizer.connect(PART_hookeye2.hook, PART_basket.rod2)
    soapbottle = SoapBottle()
    ENV_end = Environment({"soapbottle": soapbottle})
    ASSEMBLY_bathroom_organizer.attach(ENV_end.surface, PART_basket.surface)
    end_frame = Frame([0,0,253], [0,0,180])
    ASSEMBLY_bathroom_organizer.end_with(ENV_end.rod, end_frame)
    return ASSEMBLY_bathroom_organizer

#ex5
def mug_hanger3_R():
    # rod
    # doublehook
    # mug
    # environment - start could be rod or wall(surface), end is mug
    # pairs: rod-(hook hook)-hook; rod-hook, hook-hook
    ASSEMBLY_mug_hanger = AssemblyR()
    rod = Rod({"length": 500, "radius": 2})
    surface = Surface({"length": 800, "width": 600})
    ENV_start = Environment({"rod": rod})
    ENV_wall = Environment({"wall": surface})
    start_frame = Frame([0,0,200], [90,0,90])
    ASSEMBLY_mug_hanger.add(ENV_start.rod, start_frame)
    wall_frame = Frame([0,50,0], [90,0,0])
    ASSEMBLY_mug_hanger.add(ENV_wall.wall, wall_frame)
    PART_doublehook1 = DoubleHook()
    PART_doublehook2 = DoubleHook()
    PART_doublehook3 = DoubleHook()
    ASSEMBLY_mug_hanger.attach(PART_doublehook1.hook2, ENV_start.rod)
    ASSEMBLY_mug_hanger.attach(PART_doublehook2.hook2, PART_doublehook1.hook1)
    ASSEMBLY_mug_hanger.attach(PART_doublehook3.hook1, PART_doublehook2.hook1)
    mug = Mug()
    ENV_end = Environment({"mug": mug})
    ASSEMBLY_mug_hanger.attach(ENV_end.hook, PART_doublehook3.hook2)
    end_frame = Frame([0,0,50], [-35,0,-90])
    ASSEMBLY_mug_hanger.end_with(ENV_end.hook, end_frame)
    return ASSEMBLY_mug_hanger

def debug_connection_program():
    ASSEMBLY_debug = AssemblyR()

    # multiparts env initialization not right, frames are not changing
    # wallhook1 = WallHook()
    # wallhook1.frame = Frame([366, 0, 300], [360, 0, 0])
    # wallhook2 = WallHook()
    # wallhook2.frame = Frame([0, 0, 300], [360, 0, 0])
    # ENV_start = Environment({"hook1": wallhook1, "hook2": wallhook2})

    # surface = Surface({"width": 400, "length": 400})
    # ENV_start = Environment({"surface": surface})
    # start_frame = Frame([0,0,0], [0,0,0])
    # ASSEMBLY_debug.add(ENV_start, start_frame)
    # toothbrush = Toothbrush()
    # ENV_end = Environment({"toothbrush": toothbrush})
    # ASSEMBLY_debug.attach(ENV_end.hemisphere, ENV_start.surface)
    # ASSEMBLY_debug.end_with(ENV_end, Frame())

    rod = Rod({"length": 500, "radius": 5})
    ENV_start = Environment({"rod": rod})
    start_frame = Frame([0,0,500], [90,0,90])
    ASSEMBLY_debug.add(ENV_start.rod, start_frame)

    # part_params = {
    #     "extended_length": 20
    # }
    # scadfile = str(files('fabhacks').joinpath('parts', 'paramscad', 'M4.scad'))
    # turnbuckle = ParamTurnbuckle(part_params, scadfile)
    # turnbuckle2 = ParamTurnbuckle(part_params, scadfile)
    hook1 = DoubleHook()
    hook2 = DoubleHook()
    # ASSEMBLY_debug.attach(turnbuckle.hook2, ENV_start.rod)
    # ASSEMBLY_debug.attach(turnbuckle2.hook2, turnbuckle.hook1)
    # ASSEMBLY_debug.attach(turnbuckle.hook1, ENV_start.rod)
    ASSEMBLY_debug.attach(hook1.hook1, ENV_start.rod)
    ASSEMBLY_debug.attach(hook2.hook1, ENV_start.rod, alignment="flip")
    # ASSEMBLY_debug.attach(hook2.hook2, hook1.hook2)

    # PART_backseat = BackSeats()
    # ENV_start = Environment({"backseat": PART_backseat})
    # start_frame = Frame([0,0,0], [0,0,0])
    # ASSEMBLY_debug.add(ENV_start, start_frame)
    # PART_doublehook3 = DoubleHook()
    # ASSEMBLY_debug.attach(PART_doublehook3.hook1, ENV_start.rod1)
    # PART_doublehook4 = DoubleHook()
    # ASSEMBLY_debug.attach(PART_doublehook4.hook1, ENV_start.rod2)

    # PART_diaper_caddy = DiaperCaddy()
    # ASSEMBLY_debug.attach(PART_diaper_caddy.hook2, PART_doublehook3.hook2)
    # ASSEMBLY_debug.connect(PART_diaper_caddy.hook1, PART_doublehook4.hook2)

    # PART_broomrod = BroomRod()
    # ENV_start = Environment({"rod": PART_broomrod})
    # start_frame = Frame([0,0,500], [90,0,90])
    # ASSEMBLY_debug.add(ENV_start, start_frame)
    # PART_paper_towel_roll = PaperTowelRoll()
    # ENV_end = Environment({"paper_towel_roll": PART_paper_towel_roll})
    # ASSEMBLY_debug.attach(ENV_end.tube, ENV_start.tube)
    # ringM = ObstacleRingM()
    # ASSEMBLY_debug.attach(ringM.ring, ENV_start.tube)
    # doublehook = DoubleHook()
    # ASSEMBLY_debug.attach(doublehook.hook1, ENV_start.tube)
    # ASSEMBLY_debug.end_with(ENV_end, Frame())

    # PART_hookeyeleft_1 = HookEyeLeft()
    # PART_hookeyeright_1 = HookEyeRight()
    # PART_basket_1 = Basket()
    # rod = Rod({"length": 500, "radius": 5})
    # ENV_start = Environment({"rod": rod})
    # start_frame = Frame([0,0,500], [90,0,90])
    # ASSEMBLY_debug.add(ENV_start.rod, start_frame)
    # ASSEMBLY_debug.attach(PART_hookeyeleft_1.ring, ENV_start.rod)
    # ASSEMBLY_debug.attach(PART_hookeyeright_1.ring, ENV_start.rod)
    # ASSEMBLY_debug.attach(PART_basket_1.rod1, PART_hookeyeright_1.hook)
    # ASSEMBLY_debug.connect(PART_hookeyeleft_1.hook, PART_basket_1.rod2)

    # binderclip connect
    # rod = Rod({"length": 100, "radius": 1, "centerline": True})
    # ENV_start = Environment({"rod": rod})
    # PART_binderclip = BinderClip()
    # ASSEMBLY_debug.add(ENV_start.rod, Frame(pos=[0,0,500], rot=[90,0,90]))
    # ASSEMBLY_debug.attach(PART_binderclip.ring1, ENV_start.rod)
    # ASSEMBLY_debug.connect(PART_binderclip.ring2, ENV_start.rod)
    # rod and three hooks
    # rod = Rod({"length": 500, "radius": 10})
    # ENV_start = Environment({"rod": rod})
    # ASSEMBLY_debug.add(ENV_start.rod, Frame(pos=[0,0,500], rot=[90,0,90]))
    # PART_basket = TwoholeBasket()
    # PART_basket_ = TwoholeBasket()
    # PART_basket_.frame = Frame()
    # ASSEMBLY_debug.attach(PART_basket.hook2, ENV_start.rod)
    # ASSEMBLY_debug.connect(PART_basket.hook1, ENV_start.rod)
    # PART_hanger_1 = Hanger()
    # PART_hanger_1_ = Hanger()
    # PART_hanger_1_.frame = Frame()
    # ASSEMBLY_debug.attach(PART_hanger_1.hook1, ENV_start.rod)
    # PART_mug = Mug()
    # PART_mug_ = Mug()
    # PART_mug_.frame = Frame()
    # ASSEMBLY_debug.add(PART_mug_.rod, Frame())
    # ASSEMBLY_debug.attach(PART_mug.hook, ENV_start.rod)
    # PART_curtainring1 = CurtainRing()
    # PART_curtainring1_ = CurtainRing()
    # PART_curtainring1_.frame = Frame()
    # ASSEMBLY_debug.add(PART_curtainring1_.hook, Frame())
    # ASSEMBLY_debug.attach(PART_curtainring1.hook, ENV_start.rod)
    # PART_hookeyeleft = HookEyeLeft()
    # PART_hookeyeright = HookEyeRight()
    # PART_hookeyeleft_ = HookEyeLeft()
    # PART_hookeyeleft_.frame = Frame()
    # ASSEMBLY_debug.attach(PART_hookeyeleft.ring, ENV_start.rod)
    # PART_basket = Basket()
    # ASSEMBLY_debug.attach(PART_basket.rod1, PART_hookeyeleft.hook)
    # ASSEMBLY_debug.attach(PART_hookeyeright.hook, PART_basket.rod2)
    # ASSEMBLY_debug.connect(PART_hookeyeright.ring, ENV_start.rod)
    # PART_clothesclip = ClothesClip()
    # ASSEMBLY_debug.attach(PART_clothesclip.hook, ENV_start.rod)
    # PART_doublehook = DoubleHook()
    # ASSEMBLY_debug.attach(PART_doublehook.hook1, ENV_start.rod)
    # end_frame = Frame()
    # ASSEMBLY_debug.end_with(PART_basket.surface, frame=end_frame)

    # rod and three rings
    # PART_binderclip = BinderClip()
    # ASSEMBLY_debug.attach(PART_binderclip.ring1, ENV_start.rod)
    # PART_hookeyeleft = HookEyeLeft()
    # ASSEMBLY_debug.attach(PART_hookeyeleft.ring, ENV_start.rod)
    # PART_sodacantab = SodaCanTab()
    # ASSEMBLY_debug.attach(PART_sodacantab.ring2, ENV_start.rod)
    # ASSEMBLY_debug.end_with(PART_sodacantab.ring1, frame=Frame())

    # hooks and hooks, not marked is Frame(rot=[90,90,0])
    # PART_doublehook = DoubleHook()
    # PART_basket = TwoholeBasket()
    # ENV_start = Environment({"doublehook": PART_doublehook})
    # ENV_start = Environment({"basket": PART_basket})
    # start_frame = Frame([0,0,300], [0,0,0])
    # ASSEMBLY_debug.add(ENV_start, start_frame)
    # ASSEMBLY_debug.add(ENV_start.hook2, start_frame)
    # PART_mug = Mug()
    # ASSEMBLY_debug.attach(PART_mug.hook, ENV_start.hook2)
    # PART_hanger_1 = Hanger()
    # ASSEMBLY_debug.attach(PART_hanger_1.hook, ENV_start.hook2) #Frame(rot=[180,90,0]) with doublehook, Frame(rot=[-90,0,270]) with basket
    # PART_basket = TwoholeBasket()
    # ASSEMBLY_debug.attach(PART_basket.hook1, ENV_start.hook2)
    # ASSEMBLY_debug.attach(PART_basket.hook1, ENV_start.hook1)
    # ASSEMBLY_debug.attach(PART_basket.hook2, ENV_start.hook2)
    # ASSEMBLY_debug.attach(PART_basket.hook2, ENV_start.hook1)
    # PART_curtainring1 = CurtainRing()
    # ASSEMBLY_debug.attach(PART_curtainring1.hook, ENV_start.hook2) #Frame(rot=[90,90,180])
    # PART_hookeyeleft = HookEyeLeft()
    # ASSEMBLY_debug.attach(PART_hookeyeleft.hook, ENV_start.hook2)
    # PART_hookeyeright = HookEyeRight()
    # ASSEMBLY_debug.attach(PART_hookeyeright.hook, ENV_start.hook2)

    # PART_hanger_1 = Hanger()
    # PART_basket = TwoholeBasket()
    # ENV_start = Environment({"hanger": PART_hanger_1})
    # start_frame = Frame([0,0,200], [0,0,0])
    # ASSEMBLY_debug.add(ENV_start.hook, start_frame)
    # PART_hanger_2 = Hanger()
    # PART_hanger_2.frame = Frame()
    # ASSEMBLY_debug.add(PART_hanger_2.rod, Frame())
    # PART_doublehook = DoubleHook()
    # ASSEMBLY_debug.attach(PART_doublehook.hook1, ENV_start.rod) #Frame(rot=[90,90,0])

    return ASSEMBLY_debug

#ex11
def bird_feeder_variation1():
    ASSEMBLY_bird_feeder = AssemblyR()
    PART_bird_feeder_env = BirdFeederEnv()
    ENV_start = Environment({"bird_feeder_env": PART_bird_feeder_env})
    ASSEMBLY_bird_feeder.add(ENV_start, Frame())
    PART_hookeye1 = HookEyeRight()
    ASSEMBLY_bird_feeder.attach(PART_hookeye1.ring, ENV_start.hook1)
    PART_hookeye2 = HookEyeRight()
    ASSEMBLY_bird_feeder.attach(PART_hookeye2.ring, ENV_start.hook2)
    PART_doublehook1 = DoubleHook()
    ASSEMBLY_bird_feeder.attach(PART_doublehook1.hook1, PART_hookeye1.hook)
    PART_doublehook2 = DoubleHook()
    ASSEMBLY_bird_feeder.attach(PART_doublehook2.hook1, PART_hookeye2.hook)
    PART_hanger = Hanger()
    ASSEMBLY_bird_feeder.attach(PART_hanger.hook2, PART_doublehook1.hook2)
    ASSEMBLY_bird_feeder.connect(PART_hanger.hook3, PART_doublehook2.hook2)
    PART_bird_feeder = BirdFeeder()
    ENV_end = Environment({"bird_feeder": PART_bird_feeder})
    ASSEMBLY_bird_feeder.attach(ENV_end.ring, PART_hanger.hook1)
    end_frame = Frame([0,330,-352.3], [0,0,90])
    ASSEMBLY_bird_feeder.end_with(ENV_end.ring, frame=end_frame)
    return ASSEMBLY_bird_feeder

#ex12
def paper_towel_holder():
    # wall
    # 2 wallhooks
    # flag rod
    # paper towel roll
    ASSEMBLY_paper_towel_holder = AssemblyR()
    ENV_start = Environment({"env": TowelHangingEnv()})
    wall_frame = Frame([0,0,300], [0,0,0])
    ASSEMBLY_paper_towel_holder.add(ENV_start, wall_frame)

    PART_hookeye1 = HookEyeLeft()
    ASSEMBLY_paper_towel_holder.attach(PART_hookeye1.ring, ENV_start.hook1)
    PART_hookeye2 = HookEyeLeft()
    ASSEMBLY_paper_towel_holder.attach(PART_hookeye2.ring, ENV_start.hook2)

    PART_broomrod = BroomRod()
    ASSEMBLY_paper_towel_holder.attach(PART_broomrod.tube, PART_hookeye1.hook, is_fixed_joint=True)
    PART_paper_towel_roll = PaperTowelRoll()
    ENV_end = Environment({"paper_towel_roll": PART_paper_towel_roll})
    ASSEMBLY_paper_towel_holder.attach(ENV_end.tube, PART_broomrod.tube)
    ASSEMBLY_paper_towel_holder.connect(PART_hookeye2.hook, PART_broomrod.tube, is_fixed_joint=True)
    end_frame = Frame([53,0,160], [-90,-60,0])
    ASSEMBLY_paper_towel_holder.end_with(ENV_end.tube, end_frame)
    return ASSEMBLY_paper_towel_holder

#ex13
def backseat_diaper_caddy():
    # 2 backseats
    # 2 doublehooks or more
    # connected to both hooks of the diaper caddy
    # diaper caddy
    ASSEMBLY_backseat_diaper_caddy = AssemblyR()
    PART_backseat = BackSeats()
    ENV_start = Environment({"backseat": PART_backseat})
    start_frame = Frame([0,0,0], [0,0,0])
    ASSEMBLY_backseat_diaper_caddy.add(ENV_start, start_frame)
    PART_doublehook3 = DoubleHook()
    ASSEMBLY_backseat_diaper_caddy.attach(PART_doublehook3.hook1, ENV_start.rod1)
    PART_doublehook4 = DoubleHook()
    ASSEMBLY_backseat_diaper_caddy.attach(PART_doublehook4.hook2, ENV_start.rod2)
    PART_doublehook1 = DoubleHook()
    ASSEMBLY_backseat_diaper_caddy.attach(PART_doublehook1.hook1, PART_doublehook3.hook2)
    PART_doublehook2 = DoubleHook()
    ASSEMBLY_backseat_diaper_caddy.attach(PART_doublehook2.hook1, PART_doublehook4.hook1)
    PART_diaper_caddy = DiaperCaddy()
    ENV_end = Environment({"diaper_caddy": PART_diaper_caddy})
    ASSEMBLY_backseat_diaper_caddy.attach(ENV_end.hook2, PART_doublehook1.hook2)
    ASSEMBLY_backseat_diaper_caddy.connect(ENV_end.hook1, PART_doublehook2.hook2)
    end_frame = Frame([474, 645, 388], [0, -18, -90])
    ASSEMBLY_backseat_diaper_caddy.end_with(ENV_end, frame=end_frame)
    return ASSEMBLY_backseat_diaper_caddy

#ex15
def bathcloth_basket2():
    AssemblyUI_1 = AssemblyR()
    eyehook1 = HookEyeLeftS()
    eyehook2 = HookEyeLeftS()
    PART_ENV_part = ClosetRods()
    ENV_start = Environment({"part": PART_ENV_part})
    start_frame = Frame([0.0,0.0,200.0], [0.0,0.0,-90.0])
    AssemblyUI_1.add(ENV_start, start_frame)
    AssemblyUI_1.attach(eyehook1.ring, ENV_start.rod1, alignment="flip")
    AssemblyUI_1.attach(eyehook2.ring, ENV_start.rod2)
    PART_ENV_part = RoundHandleBasket()
    ENV_end = Environment({"part": PART_ENV_part})
    end_frame = Frame([0,0,100], [-75,0,90])
    AssemblyUI_1.attach(ENV_end.hook, eyehook1.hook)
    AssemblyUI_1.connect(eyehook2.hook, ENV_end.hook) #original connect
    # AssemblyUI_1.connect(ENV_end.hook, eyehook2.hook) #reversed connect doesn't work?
    AssemblyUI_1.end_with(ENV_end.hook, frame=end_frame)
    return AssemblyUI_1

# 17. hula hoop reading nook
def reading_nook():
    assembly = AssemblyR()
    PART_room = ReadingRoom()
    ENV_start = Environment({"part": PART_room})
    start_frame = Frame([0,0,0], [0,0,0])
    assembly.add(ENV_start, start_frame)
    PART_hulahoop = HulaHoop()
    ENV_end = Environment({"part": PART_hulahoop})

    PART_doublehook1 = DoubleHook()
    PART_doublehook2 = DoubleHook()

    PART_obsringL1 = ObstacleRingL()
    PART_obsringL2 = ObstacleRingL()
    PART_obsringL3 = ObstacleRingL()
    PART_obsringL4 = ObstacleRingL()
    PART_obsringL5 = ObstacleRingL()
    PART_obsringL6 = ObstacleRingL()
    PART_obsringL7 = ObstacleRingL()

    PART_ringlink1 = RingLink()
    PART_ringlink2 = RingLink()
    PART_ringlink3 = RingLink()
    PART_ringlink4 = RingLink()
    PART_ringlink5 = RingLink()
    PART_ringlink6 = RingLink()
    PART_ringlink7 = RingLink()

    assembly.attach(PART_doublehook1.hook2, ENV_start.rod1, is_fixed_joint=True)
    assembly.attach(PART_obsringL1.ring, PART_doublehook1.hook1)
    assembly.attach(PART_ringlink1.hook1, PART_obsringL1.ring)
    assembly.attach(PART_obsringL2.ring, PART_ringlink1.hook2)
    assembly.attach(PART_ringlink2.hook2, PART_obsringL2.ring)
    assembly.attach(PART_obsringL3.ring, PART_ringlink2.hook1)
    assembly.attach(PART_ringlink3.hook1, PART_obsringL3.ring)

    assembly.attach(PART_doublehook2.hook2, ENV_start.rod1, is_fixed_joint=True)
    assembly.attach(PART_obsringL4.ring, PART_doublehook2.hook1)
    assembly.attach(PART_ringlink4.hook1, PART_obsringL4.ring)
    assembly.attach(PART_obsringL5.ring, PART_ringlink4.hook2)
    assembly.attach(PART_ringlink5.hook2, PART_obsringL5.ring)
    assembly.attach(PART_obsringL6.ring, PART_ringlink5.hook1)
    assembly.attach(PART_ringlink6.hook1, PART_obsringL6.ring)

    part_params = {"extended_length": 40}
    scadfile = str(files("fabhacks").joinpath("parts", "paramscad", "M4.scad"))
    PART_turnbuckle = ParamTurnbuckle(part_params, scadfile)
    # PART_turnbuckle = Turnbuckle()
    assembly.attach(PART_turnbuckle.hook1, ENV_start.rod2, alignment="flip")
    assembly.attach(PART_obsringL7.ring, PART_turnbuckle.hook2)
    assembly.attach(PART_ringlink7.hook2, PART_obsringL7.ring)#, alignment="flip")

    assembly.attach(ENV_end.ring, PART_ringlink7.hook1)
    assembly.connect(ENV_end.ring, PART_ringlink3.hook2, is_fixed_joint=True)#, alignment="flip")
    assembly.connect(ENV_end.ring, PART_ringlink6.hook2, is_fixed_joint=True)#, alignment="flip")

    end_frame = Frame([0, -550, 940], [0,-80,90]) #ognookcon
    assembly.end_with(ENV_end.ring, frame=end_frame)

    return assembly

def clip_light_526(): # XS
    AssemblyR_526 = AssemblyR()
    part_params = {'extended_length': 25.135}
    ParamTurnbuckle_3771 = ParamTurnbuckle(part_params, scadfpath)
    RingXS_3772 = RingXS()
    part_params = {'extended_length': 25.135}
    ParamTurnbuckle_3773 = ParamTurnbuckle(part_params, scadfpath)
    RingXS_3774 = RingXS()
    part_params = {'extended_length': 25.135}
    ParamTurnbuckle_3775 = ParamTurnbuckle(part_params, scadfpath)
    RingXS_3776 = RingXS()
    part_params = {'extended_length': 25.135}
    ParamTurnbuckle_3777 = ParamTurnbuckle(part_params, scadfpath)
    RingXS_3778 = RingXS()
    FabricHanger_3779 = FabricHanger()
    PART_ENV_clip_light = ClipLight()
    ENV_end = Environment({"clip_light": PART_ENV_clip_light})
    PART_ENV_ceiling_hook = CeilingHook()
    ENV_start = Environment({"ceiling_hook": PART_ENV_ceiling_hook})
    start_frame = Frame([0,0,1000], [0,0,0])
    AssemblyR_526.add(ENV_start, start_frame)
    AssemblyR_526.attach(ParamTurnbuckle_3771.hook1, ENV_start.hook, alignment="default")
    AssemblyR_526.attach(RingXS_3772.ring, ParamTurnbuckle_3771.hook2, alignment="default")
    AssemblyR_526.attach(ParamTurnbuckle_3773.hook1, RingXS_3772.ring, alignment="default")
    AssemblyR_526.attach(RingXS_3774.ring, ParamTurnbuckle_3773.hook2, alignment="default")
    AssemblyR_526.attach(ParamTurnbuckle_3775.hook1, RingXS_3774.ring, alignment="default")
    AssemblyR_526.attach(RingXS_3776.ring, ParamTurnbuckle_3775.hook2, alignment="default")
    AssemblyR_526.attach(ParamTurnbuckle_3777.hook1, RingXS_3776.ring, alignment="default")
    AssemblyR_526.attach(RingXS_3778.ring, ParamTurnbuckle_3777.hook2, alignment="default")
    AssemblyR_526.attach(FabricHanger_3779.hook1, RingXS_3778.ring, alignment="default")
    AssemblyR_526.attach(ENV_end.clip, FabricHanger_3779.rod, alignment="default")
    end_frame = Frame([0,0,0], [180,90,0])
    AssemblyR_526.end_with(ENV_end.clip, frame=end_frame)
    return AssemblyR_526

def clip_light_550(): #S
    AssemblyR_550 = AssemblyR()
    part_params = {'extended_length': 4.57}
    ParamTurnbuckle_4035 = ParamTurnbuckle(part_params, scadfpath)
    RingS_4036 = RingS()
    part_params = {'extended_length': 4.57}
    ParamTurnbuckle_4037 = ParamTurnbuckle(part_params, scadfpath)
    RingS_4038 = RingS()
    part_params = {'extended_length': 4.57}
    ParamTurnbuckle_4039 = ParamTurnbuckle(part_params, scadfpath)
    RingS_4040 = RingS()
    part_params = {'extended_length': 4.57}
    ParamTurnbuckle_4041 = ParamTurnbuckle(part_params, scadfpath)
    RingS_4042 = RingS()
    FabricHanger_4043 = FabricHanger()
    PART_ENV_clip_light = ClipLight()
    ENV_end = Environment({"clip_light": PART_ENV_clip_light})
    PART_ENV_ceiling_hook = CeilingHook()
    ENV_start = Environment({"ceiling_hook": PART_ENV_ceiling_hook})
    start_frame = Frame([0,0,1000], [0,0,0])
    AssemblyR_550.add(ENV_start, start_frame)
    AssemblyR_550.attach(ParamTurnbuckle_4035.hook1, ENV_start.hook, alignment="default")
    AssemblyR_550.attach(RingS_4036.ring, ParamTurnbuckle_4035.hook2, alignment="default")
    AssemblyR_550.attach(ParamTurnbuckle_4037.hook1, RingS_4036.ring, alignment="default")
    AssemblyR_550.attach(RingS_4038.ring, ParamTurnbuckle_4037.hook2, alignment="default")
    AssemblyR_550.attach(ParamTurnbuckle_4039.hook1, RingS_4038.ring, alignment="default")
    AssemblyR_550.attach(RingS_4040.ring, ParamTurnbuckle_4039.hook2, alignment="default")
    AssemblyR_550.attach(ParamTurnbuckle_4041.hook1, RingS_4040.ring, alignment="default")
    AssemblyR_550.attach(RingS_4042.ring, ParamTurnbuckle_4041.hook2, alignment="default")
    AssemblyR_550.attach(FabricHanger_4043.hook1, RingS_4042.ring, alignment="default")
    AssemblyR_550.attach(ENV_end.clip, FabricHanger_4043.rod, alignment="default")
    end_frame = Frame([0,0,0], [180,90,0])
    AssemblyR_550.end_with(ENV_end.clip, frame=end_frame)
    return AssemblyR_550

def clip_light_459(): #M
    AssemblyR_459 = AssemblyR()
    part_params = {'extended_length': 43.415000000000006}
    ParamTurnbuckle_3124 = ParamTurnbuckle(part_params, scadfpath)
    RingM_3125 = RingM()
    part_params = {'extended_length': 43.415000000000006}
    ParamTurnbuckle_3126 = ParamTurnbuckle(part_params, scadfpath)
    RingM_3127 = RingM()
    part_params = {'extended_length': 43.415000000000006}
    ParamTurnbuckle_3128 = ParamTurnbuckle(part_params, scadfpath)
    RingM_3129 = RingM()
    FabricHanger_3130 = FabricHanger()
    PART_ENV_clip_light = ClipLight()
    ENV_end = Environment({"clip_light": PART_ENV_clip_light})
    PART_ENV_ceiling_hook = CeilingHook()
    ENV_start = Environment({"ceiling_hook": PART_ENV_ceiling_hook})
    start_frame = Frame([0,0,1000], [0,0,0])
    AssemblyR_459.add(ENV_start, start_frame)
    AssemblyR_459.attach(ParamTurnbuckle_3124.hook1, ENV_start.hook, alignment="default")
    AssemblyR_459.attach(RingM_3125.ring, ParamTurnbuckle_3124.hook2, alignment="default")
    AssemblyR_459.attach(ParamTurnbuckle_3126.hook1, RingM_3125.ring, alignment="default")
    AssemblyR_459.attach(RingM_3127.ring, ParamTurnbuckle_3126.hook2, alignment="default")
    AssemblyR_459.attach(ParamTurnbuckle_3128.hook1, RingM_3127.ring, alignment="default")
    AssemblyR_459.attach(RingM_3129.ring, ParamTurnbuckle_3128.hook2, alignment="default")
    AssemblyR_459.attach(FabricHanger_3130.hook1, RingM_3129.ring, alignment="flip")
    AssemblyR_459.attach(ENV_end.clip, FabricHanger_3130.rod, alignment="default", is_fixed_joint=True)
    end_frame = Frame([0,0,0], [180,90,0])
    AssemblyR_459.end_with(ENV_end.clip, frame=end_frame)
    return AssemblyR_459

def clip_light_487(): #L
    AssemblyR_487 = AssemblyR()
    part_params = {'extended_length': 27.42}
    ParamTurnbuckle_3376 = ParamTurnbuckle(part_params, scadfpath)
    RingL_3377 = RingL()
    part_params = {'extended_length': 27.42}
    ParamTurnbuckle_3378 = ParamTurnbuckle(part_params, scadfpath)
    RingL_3379 = RingL()
    part_params = {'extended_length': 27.42}
    ParamTurnbuckle_3380 = ParamTurnbuckle(part_params, scadfpath)
    RingL_3381 = RingL()
    FabricHanger_3382 = FabricHanger()
    PART_ENV_clip_light = ClipLight()
    ENV_end = Environment({"clip_light": PART_ENV_clip_light})
    PART_ENV_ceiling_hook = CeilingHook()
    ENV_start = Environment({"ceiling_hook": PART_ENV_ceiling_hook})
    start_frame = Frame([0,0,1000], [0,0,0])
    AssemblyR_487.add(ENV_start, start_frame)
    AssemblyR_487.attach(ParamTurnbuckle_3376.hook1, ENV_start.hook, alignment="default")
    AssemblyR_487.attach(RingL_3377.ring, ParamTurnbuckle_3376.hook2, alignment="default")
    AssemblyR_487.attach(ParamTurnbuckle_3378.hook1, RingL_3377.ring, alignment="default")
    AssemblyR_487.attach(RingL_3379.ring, ParamTurnbuckle_3378.hook2, alignment="default")
    AssemblyR_487.attach(ParamTurnbuckle_3380.hook1, RingL_3379.ring, alignment="default")
    AssemblyR_487.attach(RingL_3381.ring, ParamTurnbuckle_3380.hook2, alignment="default")
    AssemblyR_487.attach(FabricHanger_3382.hook1, RingL_3381.ring, alignment="flip")
    AssemblyR_487.attach(ENV_end.clip, FabricHanger_3382.rod, alignment="default")
    end_frame = Frame([0,0,0], [180,90,0])
    AssemblyR_487.end_with(ENV_end.clip, frame=end_frame)
    return AssemblyR_487