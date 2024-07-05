from ..assembly.frame import Frame
from .material import Material
import os
import igl
import numpy as np
import dill as pickle
from importlib_resources import files
from enum import Enum, auto

class PartType(Enum):
    BackSeats = auto()
    Basket = auto()
    BasketBase = auto()
    BasketHandle = auto()
    BinderClip = auto()
    BirdFeeder = auto()
    BirdFeederEnv = auto()
    Board = auto() # unused
    BroomRod = auto()
    Cable = auto()
    CarSeat = auto()
    ClothesClip = auto() # unused
    CurtainRing = auto()
    ShowerCurtainRod = auto()
    DiaperCaddy = auto()
    DoubleHook = auto()
    DoubleHookMedium = auto()
    FlagRod = auto()
    HandleBasket = auto()
    Hanger = auto()
    HookEyeLeft = auto()
    HookEyeRight = auto()
    Mug = auto()
    PaperTowelRoll = auto()
    Pen = auto() # unused
    Phone = auto() # unused
    PlasticClip = auto()
    SoapBottle = auto()
    SodaCanTab = auto()
    Tack = auto() # unused
    Toothbrush = auto()
    TwoholeBasket = auto()
    WallHook = auto()
    SmallBirdFeederEnv = auto()
    OpenHandleBasket = auto()
    ClosetRods = auto()
    RoundHandleBasket = auto()
    HookEyeLeftS = auto()
    ParamTurnbuckle = auto()
    ParamHanger = auto()
    ClipLight = auto()
    RingL = auto()
    RingM = auto()
    RingS = auto()
    RingXS = auto()
    CeilingHook = auto()
    FabricHanger = auto()
    ReadingRoom = auto()
    HulaHoop = auto()
    RingLink = auto()
    ObstacleRingM = auto()
    ObstacleRingL = auto()
    HookEyeConnector = auto()
    Turnbuckle = auto()
    LongTensionRod = auto()
    TowelHangingEnv = auto()

    def __str__(self):
        return str(self.value)

class PartPrimitiveType(Enum):
    Rod1OfBasket = auto()
    Rod2OfBasket = auto()
    SurfaceOfBasket = auto()
    Ring1OfBasketBase = auto()
    Ring2OfBasketBase = auto()
    SurfaceOfBasketBase = auto()
    Rod1OfBasketHandle = auto()
    Rod2OfBasketHandle = auto()
    Rod3OfBasketHandle = auto()
    ClipOfBinderClip = auto()
    Ring1OfBinderClip = auto()
    Ring2OfBinderClip = auto()
    RingOfBirdFeeder = auto()
    Hook1OfBirdFeederEnv = auto()
    Hook2OfBirdFeederEnv = auto()
    Surface1OfBoard = auto()
    Surface2OfBoard = auto()
    TubeOfBroomRod = auto()
    # Tube1OfBroomRod = auto()
    Tube2OfBroomRod = auto()
    Tube3OfBroomRod = auto()
    Rod1OfCable = auto()
    Rod2OfCable = auto()
    Rod1OfCarSeat = auto()
    Rod2OfCarSeat = auto()
    Rod1OfBackSeats = auto()
    Rod2OfBackSeats = auto()
    # Rod3OfBackSeats = auto()
    # Rod4OfBackSeats = auto()
    HookOfClothesClip = auto()
    ClipOfClothesClip = auto()
    HookOfCurtainRing = auto()
    RodOfShowerCurtainRod = auto()
    Hook1OfDiaperCaddy = auto()
    Hook2OfDiaperCaddy = auto()
    Hook1OfDoubleHook = auto()
    Hook2OfDoubleHook = auto()
    Hook1OfDoubleHookMedium = auto()
    Hook2OfDoubleHookMedium = auto()
    RodOfFlagRod = auto()
    RodOfHandleBasket = auto()
    SurfaceOfHandleBasket = auto()
    Hook1OfHanger = auto()
    RodOfHanger = auto()
    # NeckOfHanger = auto()
    Hook2OfHanger = auto()
    Hook3OfHanger = auto()
    HookOfHookEyeLeft = auto()
    RingOfHookEyeLeft = auto()
    HookOfHookEyeRight = auto()
    RingOfHookEyeRight = auto()
    HookOfMug = auto()
    TubeOfMug = auto()
    TubeOfPaperTowelRoll = auto()
    RodOfPen = auto()
    EdgeOfPen = auto()
    EdgeOfPhone = auto()
    ClipOfPlasticClip = auto()
    Hemisphere1OfPlasticClip = auto()
    Hemisphere2OfPlasticClip = auto()
    RodOfSoapBottle = auto()
    SurfaceOfSoapBottle = auto()
    Ring1OfSodaCanTab = auto()
    Ring2OfSodaCanTab = auto()
    Rod1OfTack = auto()
    Rod2OfTack = auto()
    SurfaceOfTack = auto()
    HemisphereOfToothbrush = auto()
    RodOfToothbrush = auto()
    Hook1OfTwoholeBasket = auto()
    Hook2OfTwoholeBasket = auto()
    SurfaceOfTwoholeBasket = auto()
    HookOfWallHook = auto()
    SurfaceOfWallHook = auto()
    Hook1OfSmallBirdFeederEnv = auto()
    Hook2OfSmallBirdFeederEnv = auto()
    HookOfOpenHandleBasket = auto()
    Rod1OfClosetRods = auto()
    Rod2OfClosetRods = auto()
    HookOfRoundHandleBasket = auto()
    HookOfHookEyeLeftS = auto()
    RingOfHookEyeLeftS = auto()
    Hook1OfParamTurnbuckle = auto()
    Hook2OfParamTurnbuckle = auto()
    Hook1OfParamHanger = auto()
    RodOfParamHanger = auto()
    # NeckOfHanger = auto()
    Hook2OfParamHanger = auto()
    Hook3OfParamHanger = auto()
    ClipOfClipLight = auto()
    RingOfRingL = auto()
    RingOfRingM = auto()
    RingOfRingS = auto()
    RingOfRingXS = auto()
    HookOfCeilingHook = auto()
    Hook1OfFabricHanger = auto()
    RodOfFabricHanger = auto()
    Hook2OfFabricHanger = auto()
    Hook3OfFabricHanger = auto()
    Rod1OfReadingRoom = auto()
    Rod2OfReadingRoom = auto()
    Rod3OfReadingRoom = auto()
    RingOfHulaHoop = auto()
    Hook1OfRingLink = auto()
    Hook2OfRingLink = auto()
    RingOfObstacleRingM = auto()
    RingOfObstacleRingL = auto()
    Hook1OfTurnbuckle = auto()
    Hook2OfTurnbuckle = auto()
    RodOfLongTensionRod = auto()
    Hook1OfTowelHangingEnv = auto()
    Hook2OfTowelHangingEnv = auto()

    def __str__(self):
        return str(self.value)

part_to_pprim = {
    PartType.Basket: [PartPrimitiveType.Rod1OfBasket, PartPrimitiveType.Rod2OfBasket, PartPrimitiveType.SurfaceOfBasket],
    PartType.BasketBase: [PartPrimitiveType.Ring1OfBasketBase, PartPrimitiveType.Ring2OfBasketBase, PartPrimitiveType.SurfaceOfBasketBase],
    PartType.BasketHandle: [PartPrimitiveType.Rod1OfBasketHandle, PartPrimitiveType.Rod2OfBasketHandle, PartPrimitiveType.Rod3OfBasketHandle],
    PartType.BinderClip: [PartPrimitiveType.ClipOfBinderClip, PartPrimitiveType.Ring1OfBinderClip, PartPrimitiveType.Ring2OfBinderClip],
    PartType.BirdFeeder: [PartPrimitiveType.RingOfBirdFeeder],
    PartType.BirdFeederEnv: [PartPrimitiveType.Hook1OfBirdFeederEnv, PartPrimitiveType.Hook2OfBirdFeederEnv],
    PartType.Board: [PartPrimitiveType.Surface1OfBoard, PartPrimitiveType.Surface2OfBoard],
    PartType.BroomRod: [PartPrimitiveType.TubeOfBroomRod],#[PartPrimitiveType.Tube1OfBroomRod, PartPrimitiveType.Tube2OfBroomRod, PartPrimitiveType.Tube3OfBroomRod],
    PartType.Cable: [PartPrimitiveType.Rod1OfCable, PartPrimitiveType.Rod2OfCable],
    PartType.CarSeat: [PartPrimitiveType.Rod1OfCarSeat, PartPrimitiveType.Rod2OfCarSeat],
    PartType.BackSeats: [PartPrimitiveType.Rod1OfBackSeats, PartPrimitiveType.Rod2OfBackSeats],
    PartType.ClothesClip: [PartPrimitiveType.HookOfClothesClip, PartPrimitiveType.ClipOfClothesClip],
    PartType.CurtainRing: [PartPrimitiveType.HookOfCurtainRing],
    PartType.ShowerCurtainRod: [PartPrimitiveType.RodOfShowerCurtainRod],
    PartType.DiaperCaddy: [PartPrimitiveType.Hook1OfDiaperCaddy, PartPrimitiveType.Hook2OfDiaperCaddy],
    PartType.DoubleHook: [PartPrimitiveType.Hook1OfDoubleHook, PartPrimitiveType.Hook2OfDoubleHook],
    PartType.DoubleHookMedium: [PartPrimitiveType.Hook1OfDoubleHookMedium, PartPrimitiveType.Hook2OfDoubleHookMedium],
    PartType.FlagRod: [PartPrimitiveType.RodOfFlagRod],
    PartType.HandleBasket: [PartPrimitiveType.RodOfHandleBasket, PartPrimitiveType.SurfaceOfHandleBasket],
    PartType.Hanger: [PartPrimitiveType.Hook1OfHanger, PartPrimitiveType.RodOfHanger, PartPrimitiveType.Hook2OfHanger, PartPrimitiveType.Hook3OfHanger],
    PartType.HookEyeLeft: [PartPrimitiveType.HookOfHookEyeLeft, PartPrimitiveType.RingOfHookEyeLeft],
    PartType.HookEyeRight: [PartPrimitiveType.HookOfHookEyeRight, PartPrimitiveType.RingOfHookEyeRight],
    PartType.Mug: [PartPrimitiveType.HookOfMug, PartPrimitiveType.TubeOfMug],
    PartType.PaperTowelRoll: [PartPrimitiveType.TubeOfPaperTowelRoll],
    PartType.Pen: [PartPrimitiveType.RodOfPen, PartPrimitiveType.EdgeOfPen],
    PartType.Phone: [PartPrimitiveType.EdgeOfPhone],
    PartType.PlasticClip: [PartPrimitiveType.ClipOfPlasticClip, PartPrimitiveType.Hemisphere1OfPlasticClip, PartPrimitiveType.Hemisphere2OfPlasticClip],
    PartType.SoapBottle: [PartPrimitiveType.RodOfSoapBottle, PartPrimitiveType.SurfaceOfSoapBottle],
    PartType.SodaCanTab: [PartPrimitiveType.Ring1OfSodaCanTab, PartPrimitiveType.Ring2OfSodaCanTab],
    PartType.Tack: [PartPrimitiveType.Rod1OfTack, PartPrimitiveType.Rod2OfTack, PartPrimitiveType.SurfaceOfTack],
    PartType.Toothbrush: [PartPrimitiveType.HemisphereOfToothbrush, PartPrimitiveType.RodOfToothbrush],
    PartType.TwoholeBasket: [PartPrimitiveType.Hook1OfTwoholeBasket, PartPrimitiveType.Hook2OfTwoholeBasket, PartPrimitiveType.SurfaceOfTwoholeBasket],
    PartType.WallHook: [PartPrimitiveType.HookOfWallHook, PartPrimitiveType.SurfaceOfWallHook],
    PartType.SmallBirdFeederEnv: [PartPrimitiveType.Hook1OfSmallBirdFeederEnv, PartPrimitiveType.Hook2OfSmallBirdFeederEnv],
    PartType.OpenHandleBasket: [PartPrimitiveType.HookOfOpenHandleBasket],
    PartType.ClosetRods: [PartPrimitiveType.Rod1OfClosetRods, PartPrimitiveType.Rod2OfClosetRods],
    PartType.RoundHandleBasket: [PartPrimitiveType.HookOfRoundHandleBasket],
    PartType.HookEyeLeftS: [PartPrimitiveType.HookOfHookEyeLeftS, PartPrimitiveType.RingOfHookEyeLeftS],
    PartType.ParamTurnbuckle: [PartPrimitiveType.Hook1OfParamTurnbuckle, PartPrimitiveType.Hook2OfParamTurnbuckle],
    PartType.ParamHanger: [PartPrimitiveType.Hook1OfParamHanger, PartPrimitiveType.RodOfParamHanger, PartPrimitiveType.Hook2OfParamHanger, PartPrimitiveType.Hook3OfParamHanger],
    PartType.ClipLight: [PartPrimitiveType.ClipOfClipLight],
    PartType.RingL: [PartPrimitiveType.RingOfRingL],
    PartType.RingM: [PartPrimitiveType.RingOfRingM],
    PartType.RingS: [PartPrimitiveType.RingOfRingS],
    PartType.RingXS: [PartPrimitiveType.RingOfRingXS],
    PartType.CeilingHook: [PartPrimitiveType.HookOfCeilingHook],
    PartType.FabricHanger: [PartPrimitiveType.Hook1OfFabricHanger, PartPrimitiveType.RodOfFabricHanger, PartPrimitiveType.Hook2OfFabricHanger, PartPrimitiveType.Hook3OfFabricHanger],
    PartType.ReadingRoom: [PartPrimitiveType.Rod1OfReadingRoom, PartPrimitiveType.Rod2OfReadingRoom],#, PartPrimitiveType.Hook1OfReadingRoom, PartPrimitiveType.Hook2OfReadingRoom],
    PartType.HulaHoop: [PartPrimitiveType.RingOfHulaHoop],
    PartType.RingLink: [PartPrimitiveType.Hook1OfRingLink, PartPrimitiveType.Hook2OfRingLink],
    PartType.ObstacleRingM: [PartPrimitiveType.RingOfObstacleRingM],
    PartType.ObstacleRingL: [PartPrimitiveType.RingOfObstacleRingL],
    PartType.HookEyeConnector: [PartPrimitiveType.HookOfHookEyeRight, PartPrimitiveType.RingOfHookEyeRight],
    PartType.Turnbuckle: [PartPrimitiveType.Hook1OfTurnbuckle, PartPrimitiveType.Hook2OfTurnbuckle],
    PartType.LongTensionRod: [PartPrimitiveType.RodOfLongTensionRod],
    PartType.TowelHangingEnv: [PartPrimitiveType.Hook1OfTowelHangingEnv, PartPrimitiveType.Hook2OfTowelHangingEnv],
}

class Part:
    # auxiliary id for __eq__
    pid = 0
    # temp folder for meshes
    temp_dir = "temp"
    # number of part-primitive types
    npptype = len(PartPrimitiveType)

    def __init__(self):
        self.id = Part.pid
        self.tid = None # should be one of PartType enum, except for Environment Part
        Part.pid += 1

        self.children = []
        self.children_pairwise_ranges = {}
        self.children_varnames = []
        self.name_in_assembly = None
        self.primitive_init_program = None
        self.constructor_program = None
        self.primitive_types = []
        self.frame = Frame()
        self.V = None
        self.F = None
        self.bbox = None
        self.com_offset = None
        self.attach_length = 0
        self.is_concave = False
        self.is_fixed = False
        self.mass = 0
        self.mass_ratio = 1.
        self.material = Material(1.)

    def __eq__(self, other):
        if isinstance(other, Part):
            return self.id == other.id
        return False
    
    def __str__(self):
        return self.__class__.__name__ + "_" + str(self.id)

    def set_from(self, other):
        self.id = other.id
        self.tid = other.tid
        self.children = other.children
        self.children_varnames = other.children_varnames
        self.children_pairwise_ranges = other.children_pairwise_ranges
        # self.name_in_assembly = other.name_in_assembly
        self.primitive_init_program = other.primitive_init_program
        self.constructor_program = other.constructor_program
        self.primitive_types = other.primitive_types
        self.frame = other.frame
        self.V = other.V
        self.F = other.F
        self.bbox = other.bbox
        self.com_offset = other.com_offset
        self.is_concave = other.is_concave
        self.is_fixed = other.is_fixed
        self.part_obj_path = other.part_obj_path
        self.mass = other.mass
        self.material = other.material
        for i in range(len(self.children)):
            self.children[i].parent = self
            setattr(self, self.children_varnames[i], self.children[i])

    def connectors(self):
        conns = []
        for child in self.children:
            if not child.connected:
                conns.append(child)
            elif not child.single_conn:
                conns.append(child)
        return conns

    def get_child_varname(self, child_id):
        return self.children_varnames[child_id]

    def compute_mass(self, V, F):
        V2 = np.ndarray(shape = (V.shape[0] + 1, 3), dtype=np.float32)
        V2[:-1, :] = V
        V2[-1, :] = [0, 0, 0]
        F2 = np.ndarray(shape = (F.shape[0], 4), dtype=np.int32)
        F2[:, :3] = F
        F2[:, 3] = V.shape[0]
        vol = igl.volume(V2, F2)
        totalV = np.abs(np.sum(vol))
        self.mass = self.material.get_mass(totalV)

    # http://melax.github.io/volint.html
    def compute_inertia(self):
        com = np.zeros(3)
        volume = 0.
        for i in range(self.F.shape[0]):
            tri = self.F[i, :]
            A = np.array([self.V[tri[0], :], self.V[tri[1], :], self.V[tri[2], :]])
            vol = np.linalg.det(A)
            com += vol * (self.V[tri[0], :] + self.V[tri[1], :] + self.V[tri[2], :])
            volume += vol
        com /= volume * 4.

        volume = 0.
        diag = np.zeros(3)
        offd = np.zeros(3)
        for i in range(self.F.shape[0]):
            tri = self.F[i, :]
            A = np.array([self.V[tri[0], :]-com, self.V[tri[1], :]-com, self.V[tri[2], :]-com])
            vol = np.linalg.det(A)
            normal = np.cross(self.V[tri[1], :] - self.V[tri[0], :], self.V[tri[2], :] - self.V[tri[0], :])
            com_ctr = (self.V[tri[0], :] + self.V[tri[1], :] + self.V[tri[2], :])/3. - com
            if np.dot(normal, com_ctr) < 0:
                volume -= vol
            else:
                volume += vol
            for j in range(3):
                j1 = (j + 1) % 3
                j2 = (j + 2) % 3
                diag[j] += (A[0,j]*A[1,j] + A[1,j]*A[2,j] + A[2,j]*A[0,j] + \
                            A[0,j]*A[0,j] + A[1,j]*A[1,j] + A[2,j]*A[2,j]) * vol
                offd[j] += (A[0,j1]*A[1,j2] + A[1,j1]*A[2,j2] + A[2,j1]*A[0,j2] + \
                            A[0,j1]*A[2,j2] + A[1,j1]*A[0,j2] + A[2,j1]*A[1,j2] + \
                            A[0,j1]*A[0,j2]*2 + A[1,j1]*A[1,j2]*2 + A[2,j1]*A[2,j2]*2) * vol
        diag /= volume * (60. / 6.)
        offd /= volume * (120. / 6.)
        inertia_attrib = {"ixx":str(diag[1]+diag[2]), "ixy":str(-offd[2]), "ixz":str(-offd[1]), \
                          "iyy":str(diag[0]+diag[2]), "iyz":str(-offd[0]), "izz":str(diag[0]+diag[1])}
        return inertia_attrib

    def compute_com(self, V, F):
        com = np.zeros(3)
        volume = 0.
        for i in range(F.shape[0]):
            tri = F[i, :]
            A = np.array([V[tri[0], :], V[tri[1], :], V[tri[2], :]])
            vol = np.linalg.det(A)
            com += vol * (V[tri[0], :] + V[tri[1], :] + V[tri[2], :])
            volume += vol
        com /= volume * 4.
        self.com_offset = com

    def init_mesh_metrics(self, classname=None):
        if classname is None:
            classname = self.__class__.__name__.lower()
        mesh_path = str(files('fabhacks').joinpath("parts", "meshes", "{}.off".format(classname)))
        if not os.path.exists(mesh_path):
            mesh_path = str(files('fabhacks').joinpath("parts", "meshes", "{}.obj".format(classname)))
        if not os.path.exists(mesh_path):
            mesh_path = str(files('fabhacks').joinpath("parts", "meshes", "{}.stl".format(classname)))
        if not os.path.exists(self.part_obj_path):
            V, F = igl.read_triangle_mesh(mesh_path)
            igl.write_obj(self.part_obj_path, V, F)

        mesh_metrics_path = str(files('fabhacks').joinpath("parts", "meshes", "{}.metrics".format(classname)))
        metrics = {}
        if os.path.exists(mesh_metrics_path):
            with open(mesh_metrics_path, 'rb') as f:
                metrics = pickle.load(f)
            self.com_offset = metrics["com_offset"]
            self.attach_length = metrics["attach_length"]
            self.mass = metrics["mass"]
            self.bbox = metrics["bbox"]
        else:
            V, F = igl.read_triangle_mesh(mesh_path)
            self.compute_com(V, F)
            metrics["com_offset"] = self.com_offset
            if self.attach_length == 0:
                self.attach_length = .5 * igl.bounding_box_diagonal(V)
            metrics["attach_length"] = self.attach_length
            self.compute_mass(V, F)
            metrics["mass"] = self.mass# * self.mass_ratio
            self.bbox, _ = igl.bounding_box(V) # returns BV (2^3 by 3 numpy array), BF
            metrics["bbox"] = self.bbox
            with open(mesh_metrics_path, 'wb') as f:
                pickle.dump(metrics, f)

    def init_mesh(self):
        self.V, self.F = igl.read_triangle_mesh(self.part_obj_path)

    def get_transformed_VF(self, reload=False):
        if reload or self.V is None or self.F is None:
            self.init_mesh()
        return self.frame.apply_transform(self.V), self.F
