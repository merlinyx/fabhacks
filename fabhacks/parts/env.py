from .part import Part
from ..assembly.frame import Frame
import igl
import os
import numpy as np
import dill as pickle
from importlib_resources import files

class EnvPartPrimitiveType:
    """ ENV-specific enum. """
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return self.name

class Environment(Part):
    # input: a dictionary of format {name: primitive/part}
    def __init__(self, input):
        super().__init__()
        self.VF_needs_reload = False
        self.tid = 0
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Environment{}".format(self.id))
        self.constructor_program = ""
        if len(input.keys()) == 1:
            # handle the case where the environment is a single primitive or part
            name = list(input.keys())[0]
            # classname = "Env{}".format(self.id) + name
            self.part_obj_path += name + ".obj"
            if isinstance(input[name], Part):
                Part.pid -= 1
                self.set_from(input[name])
                self.primitive_init_program = "PART_ENV_{} = {}()\n".format(name, input[name].__class__.__name__)
                self.constructor_program = "{{\"{}\": PART_ENV_{}}}".format(name, name)
            else:
                setattr(self, name, input[name])
                self.primitive_init_program = "{} = {}({})\n".format(name, input[name].__class__.__name__, input[name].program)
                self.constructor_program = "{{\"{}\": {}}}".format(name, name)
                input[name].parent = self
                self.children = [input[name]]
                self.children_varnames = [name]
                self.primitive_types = [input[name].__class__.__name__]
                self.children[0].set_frame(Frame())
                self.children[0].child_id = 0
                self.children[0].set_ohe(EnvPartPrimitiveType(name.capitalize() + "OfEnvironment" + str(self.id), Part.npptype))
                Part.npptype += 1
                self.is_concave = self.children[0].is_concave
                self.init_mesh_metrics(input)#, classname)
        elif len(input.keys()) > 1:
            # handle the case where the environment is a combination of primitives and parts
            self.children = []
            self.children_varnames = []
            child_types = []
            self.primitive_init_program = ""
            self.constructor_program = "{"
            cid = 0
            concave = False
            # classname = "Env{}".format(self.id)
            for name in input.keys():
                if isinstance(input[name], Part):
                    self.primitive_init_program += "PART_ENV_{} = {}()\n".format(name, input[name].__class__.__name__)
                    self.constructor_program += "{{\"{}\": PART_ENV_{}}}, ".format(name, name)
                    for i, child in enumerate(input[name].children):
                        if child.is_concave:
                            concave = True
                        child.parent = self
                        child.child_id = cid
                        child_varname = input[name].children_varnames[i] + str(cid)
                        setattr(self, child_varname, child)
                        child.set_ohe(EnvPartPrimitiveType(child_varname.capitalize() + "OfEnvironment" + str(self.id), Part.npptype))
                        Part.npptype += 1
                        self.children.append(child)
                        self.children_varnames.append(child_varname)
                        child_types.append(child.__class__.__name__)
                        cid += 1
                else:
                    self.constructor_program += "{{\"{}\": {}, ".format(name, name)
                    if input[name].is_concave:
                        concave = True
                    input[name].parent = self
                    input[name].child_id = cid
                    self.primitive_init_program += "{} = {}({})\n".format(name, input[name].__class__.__name__, input[name].program)
                    child_varname = name + str(cid)
                    setattr(self, child_varname, input[name])
                    input[name].set_ohe(EnvPartPrimitiveType(child_varname.capitalize() + "OfEnvironment" + str(self.id), Part.npptype))
                    Part.npptype += 1
                    self.children.append(input[name])
                    self.children_varnames.append(child_varname)
                    child_types.append(input[name].__class__.__name__)
                    cid += 1
                self.part_obj_path += name
                # classname += name
            self.part_obj_path += ".obj"
            self.constructor_program = self.constructor_program[:-2] + "}\n"
            self.primitive_types = set(child_types)
            self.is_concave = concave
            self.init_mesh_metrics(input)#, classname)
        else:
            raise(Exception("Environment must have at least one part or primitive"))
        self.child_symmetry_groups = np.arange(len(self.children)).tolist()

    def init_mesh_metrics(self, input):#, classname):
        # handle the case where the environment is a single primitive or part
        if len(input.keys()) == 1:
            if self.children[0].V is None or self.children[0].F is None:
                self.children[0].init_mesh()
            self.V, self.F = self.children[0].V, self.children[0].F
            self.children[0].V = None
            self.children[0].F = None
        elif len(input.keys()) > 1:
            # handle the case where the environment is a combination of primitives and parts
            Vs = []
            Fs = []
            num_verts = 0
            offset = []
            self.VF_needs_reload = True
            for name in input.keys():
                if input[name].V is None or input[name].F is None:
                    input[name].init_mesh()
                Vs.append(input[name].V)
                Fs.append(input[name].F)
                offset.append(num_verts)
                num_verts += len(input[name].V)
                input[name].V = None
                input[name].F = None
            self.V = np.concatenate(Vs)
            newFs =[]
            for i, F in enumerate(Fs):
                newFs.append(F + offset[i])
            self.F = np.concatenate(newFs)
        # if not os.path.exists(self.part_obj_path):
        igl.write_obj(self.part_obj_path, self.V, self.F)

        # mesh_metrics_path = str(files('fabhacks').joinpath("parts", "meshes", "{}.metrics".format(classname)))
        metrics = {}
        # if os.path.exists(mesh_metrics_path):
        #     with open(mesh_metrics_path, 'rb') as f:
        #         metrics = pickle.load(f)
        #     self.com_offset = metrics["com_offset"]
        #     self.attach_length = metrics["attach_length"]
        #     self.mass = metrics["mass"]
        #     self.bbox = metrics["bbox"]
        # else:
        self.compute_com(self.V, self.F)
        metrics["com_offset"] = self.com_offset
        if self.attach_length == 0:
            self.attach_length = .5 * igl.bounding_box_diagonal(self.V)
        metrics["attach_length"] = self.attach_length
        self.compute_mass(self.V, self.F)
        metrics["mass"] = self.mass# * self.mass_ratio
        self.bbox, _ = igl.bounding_box(self.V) # returns BV (2^3 by 3 numpy array), BF
        metrics["bbox"] = self.bbox
        # with open(mesh_metrics_path, 'wb') as f:
        #     pickle.dump(metrics, f)

        if self.is_concave:
            self.create_vhacd_mesh()
        self.V = None
        self.F = None
