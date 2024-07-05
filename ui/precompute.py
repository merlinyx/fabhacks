from scipy.spatial.distance import cdist
from fabhacks.parts.backseats import BackSeats
from fabhacks.parts.basket import Basket
from fabhacks.parts.basketbase import BasketBase
from fabhacks.parts.baskethandle import BasketHandle
from fabhacks.parts.binderclip import BinderClip
from fabhacks.parts.birdfeeder import BirdFeeder
from fabhacks.parts.birdfeederenv import BirdFeederEnv
from fabhacks.parts.board import Board
from fabhacks.parts.broomrod import BroomRod
from fabhacks.parts.cable import Cable
from fabhacks.parts.carseat import CarSeat
from fabhacks.parts.clothesclip import ClothesClip
from fabhacks.parts.curtainring import CurtainRing
from fabhacks.parts.curtainrod import ShowerCurtainRod
from fabhacks.parts.diapercaddy import DiaperCaddy
from fabhacks.parts.doublehook import DoubleHook
from fabhacks.parts.doublehookmed import DoubleHookMedium
from fabhacks.parts.flagrod import FlagRod
from fabhacks.parts.handlebasket import HandleBasket
from fabhacks.parts.hanger import Hanger
from fabhacks.parts.hookeyeleft import HookEyeLeft
from fabhacks.parts.hookeyeright import HookEyeRight
from fabhacks.parts.mug import Mug
from fabhacks.parts.papertowel import PaperTowelRoll
from fabhacks.parts.pen import Pen
from fabhacks.parts.phone import Phone
from fabhacks.parts.plasticclip import PlasticClip
from fabhacks.parts.soapbottle import SoapBottle
from fabhacks.parts.sodacantab import SodaCanTab
from fabhacks.parts.tack import Tack
from fabhacks.parts.toothbrush import Toothbrush
from fabhacks.parts.twoholebasket import TwoholeBasket
from fabhacks.parts.wallhook import WallHook
from fabhacks.parts.smallbirdfeederenv import SmallBirdFeederEnv
from fabhacks.parts.openhandlebasket import OpenHandleBasket
from fabhacks.parts.closetrods import ClosetRods
from fabhacks.parts.roundhandlebasket import RoundHandleBasket
from fabhacks.parts.hookeyeleftstudy import HookEyeLeftS
from fabhacks.parts.cliplight import ClipLight
from fabhacks.parts.ringl import RingL
from fabhacks.parts.ringm import RingM
from fabhacks.parts.rings import RingS
from fabhacks.parts.ringxs import RingXS
from fabhacks.parts.ceilinghook import CeilingHook
from fabhacks.parts.fabrichanger import FabricHanger
from fabhacks.parts.readingroom import ReadingRoom
from fabhacks.parts.hulahoop import HulaHoop
from fabhacks.parts.ringlink import RingLink
from fabhacks.parts.obsringl import ObstacleRingL
from fabhacks.parts.obsringm import ObstacleRingM
from fabhacks.parts.hookeyeright import HookEyeConnector
from fabhacks.parts.turnbuckle import Turnbuckle
from fabhacks.parts.longtensionrod import LongTensionRod
from fabhacks.parts.towelhangingenv import TowelHangingEnv

def precompute_for_part(part_constructor, n=None):
    results_dict = {}
    part = part_constructor()
    for c in part.children:
        c.get_new_mcf()
    for i in range(len(part.children) - 1):
        childi = part.children[i]
        XA = childi.mcf_factory.generate_mcf_samples(sample_size=n)
        for j in range(i + 1, len(part.children)):
            childj = part.children[j]
            XB = childj.mcf_factory.generate_mcf_samples(sample_size=n)
            dists = cdist(XA, XB)
            results_dict[(childi.ohe.value, childj.ohe.value)] = (dists.min(), dists.max())
    print("self.children_pairwise_ranges =", results_dict)

# Precomputes the pairwise distances between the primitives of each part.
list_of_constructors = [BackSeats,Basket,BasketBase,BasketHandle,BinderClip,BirdFeeder,BirdFeederEnv,Board,BroomRod,Cable,CarSeat,ClothesClip,CurtainRing,ShowerCurtainRod,DiaperCaddy,DoubleHook,DoubleHookMedium,FlagRod,HandleBasket,Hanger,HookEyeLeft,HookEyeRight,Mug,PaperTowelRoll,Pen,Phone,PlasticClip,SoapBottle,SodaCanTab,Tack,Toothbrush,TwoholeBasket,WallHook,SmallBirdFeederEnv,OpenHandleBasket,ClosetRods,RoundHandleBasket,HookEyeLeftS,ClipLight,RingL,RingM,RingS,RingXS,CeilingHook,FabricHanger,ReadingRoom,HulaHoop,RingLink,ObstacleRingM,ObstacleRingL,HookEyeConnector,Turnbuckle,LongTensionRod,TowelHangingEnv]
for constructor in list_of_constructors:
    print(constructor.__name__)
    precompute_for_part(constructor)
