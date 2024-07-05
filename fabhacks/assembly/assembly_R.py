from ..parts.part import Part
from ..parts.parampart import ParamPart
from ..parts.env import Environment
from ..primitives.primitive import Primitive
from ..primitives.dimensions import *
from ..primitives.point import Point
from ..primitives.mate_connector_frame import ParameterFlag, FixedFloatParameter
from ..solver.brute_force_solver import SolverOptions
from .frame import Frame, AlignmentFrame
from .assembly import Assembly
from .helpers import *
from .debug import *
from scipy.optimize import minimize, linprog
from scipy.sparse import *
from scipy import stats
from collections import deque
import networkx
import numpy as np
import polyscope as ps
import copy
from polyscope import imgui as psim
import ete3
import dill as pickle
import os
import time
import xxhash

from ..physics_solver.constraints import *
from ..physics_solver.potentials import *
from ..physics_solver.solver import *
import sys

pickle_file_path = "temp.pickle"
show_frames_global = False
include_physics = True

# Assembly using reduced mate connector frames
class AssemblyR:
    aid = 0
    PLACEHOLDER = "PARTS_CONSTRUCTION_PLACEHOLDER\n"
    SIGMA_INIT = 1e2
    G_WEIGHT = 1e-9
    G_CONSTANT = 9.8
    LINPROG_EPSILON = 1e-3

    USE_OLD_STYLE_HASH = False
    USE_PATH_HASH = False
    SKIP_FEASIBILITY = False

    @classmethod
    def set_skip_feasibility(cls, value : bool):
        AssemblyR.SKIP_FEASIBILITY = value

    @classmethod
    def set_old_hash_style(cls, value : bool):
        AssemblyR.USE_OLD_STYLE_HASH = value

    @classmethod
    def set_use_path_hash(cls, value : bool):
        AssemblyR.USE_PATH_HASH = value

    def __init__(self, name="AssemblyR"):
        self.constraints_tol = 1e-3
        self.objective_tol = 1e-3
        self.part_graph = networkx.DiGraph() # the reason for using DiGraph: __str__() and has_floater()
        self.primitive_graph = networkx.DiGraph()
        self.primitive_graph_for_loops = networkx.Graph()
        self.starts = [] #parts
        self.connects = []
        self.ends = [] #primitives
        self.end_frames = []
        self.hash_str = None
        self.approx_length = 0
        self.se_length = None
        # Objectives (soft constraints)
        self.minimize_considered_failed = False
        self.last10convals = deque(maxlen=10)
        self.constraint_list = []
        self.initial_guess_params = []
        self.con_check_initial_guess = None
        self.constraints_val = None
        self.user_objective_val = None
        self.gravitational_val = None
        self.sigma = AssemblyR.SIGMA_INIT # might need more complex strategy
        self.solve_prepared = False
        self.autosolve = False
        # Program
        self.id = AssemblyR.aid
        AssemblyR.aid += 1
        self.assembly_name = name + "_" + str(self.id)
        self.end_program = ""
        self.program = "{} = AssemblyR()\n".format(self.assembly_name)
        self.program += "{}".format(AssemblyR.PLACEHOLDER) # placeholder for part construction program

    def __str__(self):
        subtrees = {id:ete3.Tree(name=str(part)) for id, part in self.part_graph.nodes(data="part")}
        edges = []
        for u, v in self.part_graph.edges():
            if self.part_graph.edges[u, v]["is_connect_edge"]:
                continue
            edges.append((u, v))
        [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), edges)]
        root = list(subtrees.keys())[0]
        tree = subtrees[root]
        return tree.get_ascii()

    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.constraints_tol, self.objective_tol, self.part_graph, self.primitive_graph, self.primitive_graph_for_loops, self.starts, \
                self.connects, self.ends, self.end_frames, self.hash_str, self.approx_length, self.se_length, \
                self.last10convals, self.minimize_considered_failed, self.constraint_list, self.con_check_initial_guess, \
                self.initial_guess_params, self.constraints_val, self.user_objective_val, self.gravitational_val, \
                self.sigma, self.solve_prepared, self.autosolve, self.id, self.assembly_name, self.end_program, self.program)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.constraints_tol, self.objective_tol, self.part_graph, self.primitive_graph, self.primitive_graph_for_loops, self.starts, \
        self.connects, self.ends, self.end_frames, self.hash_str, self.approx_length, self.se_length, \
        self.last10convals, self.minimize_considered_failed, self.constraint_list, self.con_check_initial_guess, \
        self.initial_guess_params, self.constraints_val, self.user_objective_val, self.gravitational_val, \
        self.sigma, self.solve_prepared, self.autosolve, self.id, self.assembly_name, self.end_program, self.program = state

    #=====================================vPROGRAM RELATED METHODSv=====================================
    def insert_part_construction_program(self, overwrite=False):
        part_construction_program = ""
        for id, part in self.part_graph.nodes(data="part"):
            if id in self.starts or id in self.end_part_ids():
                continue
            if isinstance(part, ParamPart):
                part_construction_program += "part_params = {}\n".format(part.part_params)
                part_construction_program += "scadfpath = '{}'\n".format(part.openscad_fpath)
                part_construction_program += "{} = {}(part_params, scadfpath)\n".format(part.name_in_assembly, part.__class__.__name__)
            else:
                part_construction_program += "{} = {}()\n".format(part.name_in_assembly, part.__class__.__name__)
        part_construction_program += self.end_program
        full_program = self.program.replace(AssemblyR.PLACEHOLDER, part_construction_program)
        if overwrite:
            self.program = full_program
        return full_program

    def prepare_params(self, subgraph=None):
        def process_param(params_dict, param, primitive):
            params_dict[param.index] = param
            # the flag is default to BOUNDED_AND_OPEN
            # if the parameter is unbounded then it will have been set to UNBOUNDED already
            # so here we only update it if it is physically clamped
            if (not param.unbounded) and primitive.has_bounds():
                param.flag = ParameterFlag.BOUNDED_AND_CLAMPED

        primitive_graph_to_use = self.primitive_graph if subgraph is None else subgraph

        for _, entity in primitive_graph_to_use.nodes(data="primitive"):
            for mcfparam in entity.mcf_factory.MCFParams:
                mcfparam.reset_slider_param_indices()

        self.initial_guess_params = []
        params_dict = {}
        # process connect() constraints
        if subgraph is None:
            for c in self.connects:
                centity1, centity1fid, centity2, centity2fid, _ = c
                e1prefix = centity1.get_prefix_chain(self.id)
                for prefix in e1prefix:
                    entity1id, entity1fid, entity2id, entity2fid, _ = prefix
                    gentity1 = self.primitive_graph.nodes(data="primitive")[entity1id]
                    gentity2 = self.primitive_graph.nodes(data="primitive")[entity2id]
                    gentity1.mcf_factory.MCFParams[entity1fid].set_slider_param_indices(len(params_dict.items()))
                    for p in gentity1.mcf_factory.MCFParams[entity1fid].slider_parameters():
                        process_param(params_dict, p, gentity1)
                    gentity2.mcf_factory.MCFParams[entity2fid].set_slider_param_indices(len(params_dict.items()))
                    for p in gentity2.mcf_factory.MCFParams[entity2fid].slider_parameters():
                        process_param(params_dict, p, gentity2)
                centity1.mcf_factory.MCFParams[centity1fid].set_slider_param_indices(len(params_dict.items()))
                for p in centity1.mcf_factory.MCFParams[centity1fid].slider_parameters():
                    process_param(params_dict, p, centity1)
                e2prefix = centity2.get_prefix_chain(self.id)
                for prefix in e2prefix:
                    entity1id, entity1fid, entity2id, entity2fid, _ = prefix
                    gentity1 = self.primitive_graph.nodes(data="primitive")[entity1id]
                    gentity2 = self.primitive_graph.nodes(data="primitive")[entity2id]
                    gentity1.mcf_factory.MCFParams[entity1fid].set_slider_param_indices(len(params_dict.items()))
                    for p in gentity1.mcf_factory.MCFParams[entity1fid].slider_parameters():
                        process_param(params_dict, p, gentity1)
                    gentity2.mcf_factory.MCFParams[entity2fid].set_slider_param_indices(len(params_dict.items()))
                    for p in gentity2.mcf_factory.MCFParams[entity2fid].slider_parameters():
                        process_param(params_dict, p, gentity2)
                centity2.mcf_factory.MCFParams[centity2fid].set_slider_param_indices(len(params_dict.items()))
                for p in centity2.mcf_factory.MCFParams[centity2fid].slider_parameters():
                    process_param(params_dict, p, centity2)
        # process attach() dofs
        should_skip = False
        for e1, e2, data in primitive_graph_to_use.edges.data():
            if data["is_part_edge"]:
                continue
            if primitive_graph_to_use.has_edge(e2, e1):
                if should_skip:
                    should_skip = False
                    continue
                else:
                    should_skip = True
            if "conn_info" in data:
                entity1, entity1fid, entity2, entity2fid, _ = data["conn_info"]
                entity1.mcf_factory.MCFParams[entity1fid].set_slider_param_indices(len(params_dict.items()))
                for p in entity1.mcf_factory.MCFParams[entity1fid].slider_parameters():
                    process_param(params_dict, p, entity1)
                entity2.mcf_factory.MCFParams[entity2fid].set_slider_param_indices(len(params_dict.items()))
                for p in entity2.mcf_factory.MCFParams[entity2fid].slider_parameters():
                    process_param(params_dict, p, entity2)
        self.initial_guess_params = list(params_dict.values())
    #=====================================^PROGRAM RELATED METHODS^=====================================

    #=====================================vPRUNING RELATED METHODSv=====================================
    def hash(self):
        if self.hash_str is None:
            if AssemblyR.USE_PATH_HASH:
                self.hash_str = str(self.paths_hash())
                return self.hash_str
            subtrees = {id:ete3.Tree(name=part.__class__.__name__) for id, part in self.part_graph.nodes(data="part")}
            edges = []
            for u, v in self.part_graph.edges():
                if self.part_graph.edges[u, v]["is_connect_edge"]:
                    continue
                edges.append((u, v))
            [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), edges)]
            root = list(subtrees.keys())[0]
            tree = subtrees[root]
            tree.ladderize()
            if AssemblyR.USE_OLD_STYLE_HASH:
                ohe_map = {id:str(primitive.tid)+"/"+str(primitive.parent.tid) for id, primitive in self.primitive_graph.nodes(data="primitive")}
            else:
                ohe_map = {id:str(primitive.ohe.value) for id, primitive in self.primitive_graph.nodes(data="primitive")}
            edge_strs = [ohe_map[p1] + "-" + ohe_map[p2] for p1, p2, data in self.primitive_graph.edges.data() if not data["is_part_edge"]]
            edge_strs.sort()
            self.hash_str = tree.get_ascii() + "\n" + "_".join(edge_strs)
        return self.hash_str
    
    def paths_hash(self):
        old_graph = self.primitive_graph
        new_to_old = {k:v for k,v in enumerate(old_graph.nodes)}
        node_remap = {v:k for k,v in enumerate(old_graph.nodes)}
        nodes = np.arange(len(node_remap))
        parents = [old_graph.nodes[new_to_old[n]]['primitive'].parent for n in nodes]
        is_start = [old_graph.nodes[new_to_old[n]]['primitive'].parent.name_in_assembly == 'ENV_start' for n in nodes]
        idx_in_parent = [old_graph.nodes[new_to_old[n]]['primitive'].child_id for n in nodes]
        sym_nodes = ['_'.join(map(str, [parent.tid, parent.child_symmetry_groups[idx]])) for parent,idx in zip(parents, idx_in_parent)]
        sym_nodes = sym_nodes + ['sink']
        edges = [(node_remap[u], node_remap[v], {"is_connect_edge": old_graph.has_edge(u,v) and old_graph.has_edge(v,u)}) for u,v in old_graph.edges]
        graph = networkx.Graph()
        graph.add_edges_from(edges)
        sink = len(nodes)
        graph.add_edges_from([(n, sink, {"is_connect_edge": False}) for n in nodes])
        starts = [n for n in nodes if is_start[n]] # most suspect part -- needs to have the correct order
        # make all unique simple paths to sink node, stringify and sort for unique id
        paths_strings = []
        for start in starts:
            spaths = []
            for p in networkx.all_simple_paths(graph, start, sink):
                pstr = sym_nodes[p[0]]
                for i in range(1, len(p)):
                    if graph.edges[p[i - 1], p[i]]["is_connect_edge"]:
                        pstr += "::"
                    else:
                        pstr += ":"
                    pstr += sym_nodes[p[i]]
                spaths.append(pstr)
            paths_strings.append('-'.join(sorted(spaths)))
        all_paths_string = '#'.join(paths_strings)
        paths_hash = xxhash.xxh64(all_paths_string).intdigest()
        return paths_hash

    def loops(self, connect_edge=None):
        cycle_bases = list(networkx.cycle_basis(self.primitive_graph_for_loops))
        cycle_strs = []
        cycles = []
        for cycle in cycle_bases:
            should_skip = True
            for i in range(len(cycle) - 1):
                if self.primitive_graph.has_edge(cycle[i], cycle[i + 1]):
                    if not self.primitive_graph.edges[cycle[i], cycle[i + 1]]["is_part_edge"]:
                        should_skip = False
                elif self.primitive_graph.has_edge(cycle[i + 1], cycle[i]):
                    if not self.primitive_graph.edges[cycle[i + 1], cycle[i]]["is_part_edge"]:
                        should_skip = False
                else:
                    raise Exception("Cycle edge not found")
                if connect_edge is not None:
                    if (cycle[i] == connect_edge[0].id and cycle[i + 1] == connect_edge[1].id) or\
                       (cycle[i] == connect_edge[1].id and cycle[i + 1] == connect_edge[0].id):
                        should_skip = False
                if not should_skip: break
            if should_skip: continue # this is a part edge only cycle or wasn't introduced by the connect_edge
            cycle_str_list = []
            for id in cycle:
                primitive = self.primitive_graph.nodes(data="primitive")[id]
                cycle_str_list.append(str(primitive.ohe.value))
            cycle_strs.append("-".join(cycle_str_list))
            cycles.append(cycle)
        return cycle_strs, cycles

    def end_part_ids(self):
        return set([end.parent.id for end in self.ends])

    def all_start_end_connected(self):
        for s in self.starts:
            part = self.part_graph.nodes(data="part")[s]
            for c in part.children:
                if not self.primitive_graph.nodes[c.id]["connected"]:
                    return False
        return True

    def all_end_connected(self):
        for s in self.starts:
            for e in self.end_part_ids():
                if not networkx.has_path(self.part_graph, s, e):
                    return False
        return True

    def has_floater(self):
        path_parts_set = set()
        for s in self.starts:
            path_parts_set.add(s) # any start environment part is not a floater
            s_c_ids = [c.id for c in self.part_graph.nodes(data="part")[s].children]
            for s_c in self.part_graph.nodes(data="part")[s].children:
                for e in self.ends:
                    if not self.primitive_graph.has_node(e.id):
                        return False
                    paths = networkx.all_simple_paths(self.primitive_graph, s_c.id, e.id)
                    for path in paths:
                        should_skip = False
                        for i in range(len(path) - 1):
                            if self.primitive_graph.edges[path[i], path[i + 1]]["is_part_edge"]\
                                and path[i] in s_c_ids and path[i + 1] in s_c_ids:
                                should_skip = True
                                break
                        if not should_skip:
                            for p in path:
                                path_parts_set.add(self.primitive_graph.nodes[p]['primitive'].parent.id)
        all_parts_set = set(self.part_graph.nodes())
        return len(all_parts_set - path_parts_set) > 0
        # for node in self.part_graph:
        #     if self.part_graph.out_degree(node) == 0 and self.part_graph.nodes(data="part")[node].__class__.__name__ != "Environment":
        #         return True
        # return False

    def approx_length_diff(self, end_frames=None):
        if self.se_length is None:
            self.se_length = {}
            if end_frames is None:
                end_frames = self.end_frames
            for s in self.starts:
                for s_c in self.part_graph.nodes(data="part")[s].children:
                    lengths = []
                    for ef in end_frames:
                        if isinstance(ef, Frame):
                            diffvec = np.array(s_c.frame().pos) - np.array(ef.pos)
                            lengths.append(np.linalg.norm(diffvec))
                    self.se_length[s_c.id] = np.mean(lengths)
        length_diffs = []
        for s in self.starts:
            for s_c in self.part_graph.nodes(data="part")[s].children:
                desc = networkx.descendants(self.primitive_graph, s_c.id)
                max_min_dist = max([self.primitive_graph.nodes[d]["min_dist"] for d in desc])
                length_diffs.append(abs(self.se_length[s_c.id] - max_min_dist))
        return max(length_diffs)

    def approx_length_within_range(self): # unused
        return self.approx_length_diff() / max(self.se_length.values()) < 0.2

    def connectors(self):
        conns = []
        for node_id, primitive in self.primitive_graph.nodes(data="primitive"):
            primitive_connected = self.primitive_graph.nodes[node_id]["connected"]
            if not primitive_connected:
                conns.append(primitive)
            elif not primitive.single_conn:
                conns.append(primitive)
        return conns

    def compute_critical_dim(self, entity1, entity2):
        if entity1.__class__.__name__ == "Edge":
            if entity2.__class__.__name__ == "Clip":
                self.primitive_graph.nodes[entity1.id]["critical_dim"] -= entity2.width
        # elif entity1.__class__.__name__ == "Ring":
        #     if entity2.__class__.__name__ == "Rod":
        #         self.primitive_graph.nodes[entity1.id]["critical_dim"] -= entity2.radius
        #     elif entity2.__class__.__name__ == "Tube":
        #         self.primitive_graph.nodes[entity1.id]["critical_dim"] -= entity2.generalized_radius()
        #     elif entity2.__class__.__name__ == "Hook":
        #         self.primitive_graph.nodes[entity1.id]["critical_dim"] -= entity2.thickness
        elif entity1.__class__.__name__ == "Rod" or entity1.__class__.__name__ == "Tube":
            cd1 = self.primitive_graph.nodes[entity1.id]["critical_dim"]
            if entity2.__class__.__name__ == "Hook":
                self.primitive_graph.nodes[entity1.id]["critical_dim"] -= entity2.thickness * 2
            elif entity2.__class__.__name__ == "Ring":
                self.primitive_graph.nodes[entity1.id]["critical_dim"] -= entity2.thickness * 2
            elif entity2.__class__.__name__ == "Clip":
                self.primitive_graph.nodes[entity1.id]["critical_dim"] -= entity2.width
            elif entity2.__class__.__name__ == "Tube":
                # self.primitive_graph.nodes[entity1.id]["critical_dim"] -= entity2.length
                cd2 = self.primitive_graph.nodes[entity2.id]["critical_dim"]
                cd1 -= cd2
                if cd1 > 0:
                    self.primitive_graph.nodes[entity1.id]["critical_dim"] = cd1
                else:
                    cd1 += cd2
        elif entity1.__class__.__name__ == "Surface":
            cd1 = self.primitive_graph.nodes[entity1.id]["critical_dim"]
            if entity2.__class__.__name__ == "Surface":
                cd2 = self.primitive_graph.nodes[entity2.id]["critical_dim"]
                cd1[0] -= cd2[0]
                cd1[1] -= cd2[1]
                if cd1[0] > 0 and cd1[1] > 0:
                    self.primitive_graph.nodes[entity1.id]["critical_dim"] = cd1
                else:
                    cd1[0] += cd2[0]
                    cd1[1] += cd2[1]
            elif entity2.__class__.__name__ == "Hemisphere":
                cd1[0] -= entity2.radius * 2
                cd1[1] -= entity2.radius * 2
                if cd1[0] > 0 and cd1[1] > 0:
                    self.primitive_graph.nodes[entity1.id]["critical_dim"] = cd1
    #=====================================^PRUNING RELATED METHODS^=====================================

    #=====================================vVISUALIZATION AND UIv=====================================
    def visualize(self, show_frames=False, after_sim_or_reset=False, load_from_file=False):
        if not after_sim_or_reset:
            self.forward_propagate_frames(should_redistribute=False)

        # visualize the primitives
        m = self.primitive_graph.order()
        id_to_index = {id:i for i, id in enumerate(self.primitive_graph.nodes())}
        primitive_nodes = np.zeros((m, 3))
        primitive_edges = np.array([[id_to_index[p1], id_to_index[p2]] for p1, p2, data in self.primitive_graph.edges.data() if not data["is_part_edge"]]).astype(int)
        for id, primitive in self.primitive_graph.nodes(data="primitive"):
            primitive_nodes[id_to_index[id]] = primitive.get_global_frame().pos
            V, F = primitive.get_transformed_VF(reload=load_from_file)
            ps_mesh = ps.register_surface_mesh("Primitive{}-{}".format(id, primitive.__class__.__name__), V, F)
            ps_mesh.set_transparency(0.5)
        if primitive_edges.size > 0:
            primitive_net = ps.register_curve_network("primitive network", primitive_nodes, primitive_edges, enabled=False)
            primitive_net.set_transparency(0.8)

            pxvec, pyvec, pzvec = np.zeros((m, 3)), np.zeros((m, 3)), np.zeros((m, 3))
            for id, primitive in self.primitive_graph.nodes(data="primitive"):
                rot_ = primitive.get_global_frame().rot
                rot_mat = xyz_matrix(rot_[0], rot_[1], rot_[2])
                pxvec[id_to_index[id]] = rot_mat[:,0]
                pyvec[id_to_index[id]] = rot_mat[:,1]
                pzvec[id_to_index[id]] = rot_mat[:,2]
            primitive_net.add_vector_quantity("primitive frame x", pxvec, enabled=True, length=0.05, color=(0.2, 0.5, 0.5))
            primitive_net.add_vector_quantity("primitive frame y", pyvec, enabled=True, length=0.05, color=(0.8, 0.8, 0.6))
            primitive_net.add_vector_quantity("primitive frame z", pzvec, enabled=True, length=0.05, color=(0.2, 0.5, 0.2))

        # visualize the parts
        n = self.part_graph.order()
        id_to_index = {id:i for i, id in enumerate(self.part_graph.nodes())}
        part_nodes = np.zeros((n, 3))
        part_edges = np.array([[id_to_index[e[0]], id_to_index[e[1]]] for e in self.part_graph.edges]).astype(int)
        for id, part in self.part_graph.nodes(data="part"):
            part_nodes[id_to_index[id]] = part.frame.pos
            V, F = None, None
            if isinstance(part, Environment) and part.VF_needs_reload:
                Vs = []
                Fs = []
                num_verts = 0
                offset = []
                for child in part.children:
                    childV, childF = child.get_transformed_VF(reload=load_from_file)
                    Vs.append(childV)
                    Fs.append(childF)
                    offset.append(num_verts)
                    num_verts += len(childV)
                V = np.concatenate(Vs)
                for i, F in enumerate(Fs):
                    F += offset[i]
                F = np.concatenate(Fs)
            else:
                V, F = part.get_transformed_VF(reload=load_from_file)
            ps_mesh = ps.register_surface_mesh("Part{}-{}".format(id, part.__class__.__name__), V, F)
            ps_mesh.set_transparency(0.4)
        if part_edges.size > 0:
            part_net = ps.register_curve_network("part network", part_nodes, part_edges, enabled=False)
            part_net.set_transparency(0.8)

            xvec, yvec, zvec = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
            for id, part in self.part_graph.nodes(data="part"):
                rot_ = part.frame.rot
                rot_mat = xyz_matrix(rot_[0], rot_[1], rot_[2])
                xvec[id_to_index[id]] = rot_mat[:,0]
                yvec[id_to_index[id]] = rot_mat[:,1]
                zvec[id_to_index[id]] = rot_mat[:,2]
            part_net.add_vector_quantity("part frame x", xvec, enabled=True, length=0.05, color=(0.5, 0.2, 0.5))
            part_net.add_vector_quantity("part frame y", yvec, enabled=True, length=0.05, color=(0.8, 0.6, 0.6))
            part_net.add_vector_quantity("part frame z", zvec, enabled=True, length=0.05, color=(0.2, 0.2, 0.5))

        # visualize the joint frames
        if not show_frames:
            return
        N = 0
        frames, e1f, e2f = [], [], []
        for p1, p2, data in self.part_graph.edges(data=True):
            # if self.part_graph.has_edge(p2, p1) or p1 in self.end_part_ids() or p2 in self.end_part_ids():
            if "joint_frame" in data:
                N += 1
                frames.append(data["joint_frame"])
                part1 = self.part_graph.nodes(data="part")[p1]
                part2 = self.part_graph.nodes(data="part")[p2]
                e1f.append(part1.frame.transform_frame(data["p1mcf"]))
                e2f.append(part2.frame.transform_frame(data["p2mcf"]))
        points, e1points, e2points = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3))
        pxvec, pyvec, pzvec = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3))
        e1xvec, e1yvec, e1zvec = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3))
        e2xvec, e2yvec, e2zvec = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3))
        # for i, frame in enumerate(frames):
        #     points[i, :] = frame.pos
        #     rot_ = frame.rot
        #     rot_mat = xyz_matrix(rot_[0], rot_[1], rot_[2])
        #     pxvec[i] = rot_mat[:,0]
        #     pyvec[i] = rot_mat[:,1]
        #     pzvec[i] = rot_mat[:,2]
        # ps_cloud1 = ps.register_point_cloud("joint frames", points, enabled=True)
        # ps_cloud1.add_vector_quantity("joint frame x", pxvec, enabled=True, length=0.05, radius=0.001, color=(1, 0, 0))
        # ps_cloud1.add_vector_quantity("joint frame y", pyvec, enabled=True, length=0.05, radius=0.001, color=(0, 1, 0))
        # ps_cloud1.add_vector_quantity("joint frame z", pzvec, enabled=True, length=0.05, radius=0.001, color=(0, 0, 1))
        for i, frame in enumerate(e1f):
            e1points[i, :] = frame.pos
            rot_ = frame.rot
            rot_mat = xyz_matrix(rot_[0], rot_[1], rot_[2])
            e1xvec[i] = rot_mat[:,0]
            e1yvec[i] = rot_mat[:,1]
            e1zvec[i] = rot_mat[:,2]
        for i, frame in enumerate(e2f):
            e2points[i, :] = frame.pos
            rot_ = frame.rot
            rot_mat = xyz_matrix(rot_[0], rot_[1], rot_[2])
            e2xvec[i] = rot_mat[:,0]
            e2yvec[i] = rot_mat[:,1]
            e2zvec[i] = rot_mat[:,2]
        ps_cloud2 = ps.register_point_cloud("entity1 frames", e1points, enabled=True)
        ps_cloud2.add_vector_quantity("entity1f x", e1xvec, enabled=True, length=0.05, radius=0.001, color=(1, 0, 0))
        ps_cloud2.add_vector_quantity("entity1f y", e1yvec, enabled=True, length=0.05, radius=0.001, color=(0, 1, 0))
        ps_cloud2.add_vector_quantity("entity1f z", e1zvec, enabled=True, length=0.05, radius=0.001, color=(0, 0, 1))
        ps_cloud3 = ps.register_point_cloud("entity2 frames", e2points, enabled=True)
        ps_cloud3.add_vector_quantity("entity2f x", e2xvec, enabled=True, length=0.05, radius=0.001, color=(0.8, 0.2, 0.2))
        ps_cloud3.add_vector_quantity("entity2f y", e2yvec, enabled=True, length=0.05, radius=0.001, color=(0.2, 0.8, 0.2))
        ps_cloud3.add_vector_quantity("entity2f z", e2zvec, enabled=True, length=0.05, radius=0.001, color=(0.2, 0.2, 0.8))

    def polyscope_callback(self):
        global show_frames_global, pickle_file_path, include_physics

        def set_params(entity1, entity1fid, entity2, entity2fid, values1, values2):
            if len(values1) > 0:
                entity1.mcf_factory.MCFParams[entity1fid].set_slider_parameters(values1)
            if len(values2) > 0:
                entity2.mcf_factory.MCFParams[entity2fid].set_slider_parameters(values2)

        def redraw():
            ps.remove_all_structures()
            self.visualize(show_frames=show_frames_global)

        def recompute():
            # self.compute_initial_guess_frames(should_redistribute=False)
            self.end_objective_with_constraints_grav_potential()
            if len(self.connects) > 0 and self.autosolve:
                # self.solve_constraints_only(SolverOptions())
                self.check_constraints_feasibility(SolverOptions())
                self.solve_prepared = False
                self.con_check_initial_guess = None
            redraw()

        click_p = None
        if ps.have_selection():
            name = ps.get_selection()[0]
            if "Primitive" in name:
                click_selected_name = name
                click_pid = int(click_selected_name.split("-")[0][9:])
                click_p = self.primitive_graph.nodes[click_pid]["primitive"]

        psim.PushItemWidth(150)
        psim.TextUnformatted("Connector Frame Parameters")
        psim.Separator()
        should_recompute = False
        store_c = None
        values1, values2 = [], []
        for e1, e2, data in self.primitive_graph.edges.data():
            if data["is_part_edge"]:# or self.primitive_graph.has_edge(e2, e1):
                continue
            if "conn_info" in data:
                entity1, entity1fid, entity2, entity2fid, alignment_frame = data["conn_info"]
                conn_params1 = entity1.mcf_factory.MCFParams[entity1fid].slider_parameters()
                conn_params2 = entity2.mcf_factory.MCFParams[entity2fid].slider_parameters()
                psim.PushItemWidth(200)
                if len(alignment_frame.axes) > 0:
                    psim.TextUnformatted("Conn. alignment - {}".format(alignment_frame))
                    nf = len(alignment_frame.axes)
                    for axis, name in alignment_frame.named_axes.items():
                        if psim.Button("Flip Alignment {} ({} {})".format(name, entity1.id, entity2.id)):
                            alignment_frame.flip_axis(axis)
                            # should_recompute = True
                        nf = nf - 1
                        if nf > 0:
                            psim.SameLine()
                values1 = []
                for p in conn_params1:
                    if isinstance(p, FixedFloatParameter):
                        continue
                    new_name = p.name
                    if click_p is not None and e1 == click_p.id:
                        new_name = "(selected) " + new_name
                    changed, p.value = psim.SliderFloat(
                        new_name, p.value, v_min=p.min_value, v_max=p.max_value)
                    values1.append(p.value)
                    if changed: should_recompute = True
                values2 = []
                for p in conn_params2:
                    if isinstance(p, FixedFloatParameter):
                        continue
                    new_name = p.name
                    if click_p is not None and e2 == click_p.id:
                        new_name = "(selected) " + new_name
                    changed, p.value = psim.SliderFloat(
                        new_name, p.value, v_min=p.min_value, v_max=p.max_value)
                    values2.append(p.value)
                    if changed: should_recompute = True
                if should_recompute:
                    store_c = [entity1, entity1fid, entity2, entity2fid, values1, values2]
                psim.Separator()
                psim.PopItemWidth()
        if should_recompute:
            set_params(*store_c)
            recompute()
        changed, self.autosolve = psim.Checkbox("Enable Autosolve", self.autosolve)
        psim.SameLine()
        changed, include_physics = psim.Checkbox("Optim. Incl. Phys.", include_physics)
        if psim.Button("Run Optimization"):
            if store_c is not None:
                set_params(*store_c)
            self.solve(SolverOptions(), include_physics=include_physics)
            redraw()
        psim.SameLine()
        if psim.Button("Run Physics Only"):
            if store_c is not None:
                set_params(*store_c)
            self.solve_with_static_solve(SolverOptions())
            redraw()
        if psim.Button("Run check_constraints()"):
            if store_c is not None:
                set_params(*store_c)
            self.check_constraints_feasibility(SolverOptions())
            redraw()
        psim.SameLine()
        if psim.Button("Reset"):
            for _, entity in self.primitive_graph.nodes(data="primitive"):
                for mcfparam in entity.mcf_factory.MCFParams:
                    mcfparam.reset_slider_param_values()
            self.solve_prepared = False
            self.con_check_initial_guess = None
            self.sigma = AssemblyR.SIGMA_INIT
            self.forward_propagate_frames(should_redistribute=False)
            ps.remove_all_structures()
            self.visualize(show_frames=show_frames_global, after_sim_or_reset=True)
        psim.SameLine()
        if psim.Button("Reset to Manual Guess"):
            self.set_saved_initial_guess()
            redraw()
        changed, show_frames_global = psim.Checkbox("Show Frames", show_frames_global)
        if changed:
            redraw()
        psim.SameLine()
        if psim.Button("Print Frames"):
            self.print_frames()
        psim.SameLine()
        if psim.Button("Print Params"):
            self.print_params()
        psim.SameLine()
        if psim.Button("Print Constraints"):
            self.print_constraints()
        psim.Separator()
        psim.TextUnformatted("User Objective {}".format(self.user_objective_val))
        psim.TextUnformatted("Weighted Constraints Penalty {} * {}".format(self.constraints_val, self.sigma))
        psim.TextUnformatted("Weighted Gravitational Potential {} * {}".format(self.gravitational_val, AssemblyR.G_WEIGHT))
        psim.Separator()
        if psim.Button("Save Pickle"):
            self.save_pickle(filepath=pickle_file_path)
        psim.SameLine()
        if psim.Button("Load Pickle"):
            try:
                self.load_pickle(pickle_file_path)
            except Exception as e:
                print(e)
                ps.warning("Failed to load pickle file")
        psim.SameLine()
        changed, pickle_file_path = psim.InputText("Pickle File Path", pickle_file_path)
        psim.PopItemWidth()

    def bring_up_ui(self, screenshot=False):
        ps.remove_all_structures()
        self.visualize()
        ps.init()
        ps.set_user_callback(self.polyscope_callback)
        if screenshot:
            ps.screenshot()
            return
        else:
            ps.show()

    def print_frames(self):
        for _, part in self.part_graph.nodes(data="part"):
            print("part:", part)
            print("part frame:", part.frame)
        for _, primitive in self.primitive_graph.nodes(data="primitive"):
            print("primitive:", primitive)
            print("primitive frame:", primitive.get_global_frame())

    def print_params(self, reset=True):
        if reset:
            self.solve_prepared = False
        self.prepare_for_minimize()
        for p in self.initial_guess_params:
            print(p.index, p.name, p.value)
        for p in self.initial_guess_params:
            print(f"initial_guess[{p.index}] = {p.value:.4f}")

    def print_constraints(self):
        self.prepare_for_minimize()
        initial_guess = []
        for p in self.initial_guess_params:
            initial_guess.append(p.value)
        nparts = len(self.part_graph.nodes(data="part"))
        nparams = len(initial_guess)
        config = np.zeros(6*len(self.part_graph) + nparams)
        config[6*nparts:6*nparts + nparams] = initial_guess
        static_solve_constraint_list = self.static_solve_constraints(nparts, self.part_graph, self.primitive_graph)
        print("------------------")
        print("static solve constraints")
        print("------------------")
        for c in static_solve_constraint_list:
            print("cval ", c.print_g(config))
        print("------------------")
        print("scipy optimize constraints")
        print("------------------")
        for c in self.constraint_list:
            cfun = c['fun']
            args = c['args']
            print(c['type'])  
            print(args)
            if c['type'] == 'eq':
                args = args[0], True
            print(cfun(initial_guess, *args))

    def print_connect_constraints(self, x, c, verbose=True):
        centity1, centity1fid, centity2, centity2fid, calignment_frame = c
        for param in centity1.mcf_factory.MCFParams[centity1fid].slider_parameters():
            param.value = x[param.index]
        for param in centity2.mcf_factory.MCFParams[centity2fid].slider_parameters():
            param.value = x[param.index]
        centity1f = centity1.mcf_factory.eval_mcf(centity1fid)
        centity2f = centity2.mcf_factory.eval_mcf(centity2fid)
        f1 = centity1.parent.frame.transform_frame(centity1f)#.transform_frame(calignment_frame))
        f2 = centity2.parent.frame.transform_frame(centity2f)
        output = "compute_connect_constraint_params\n"
        output += f"{centity1} {centity1fid} {centity2} {centity2fid}\n"
        output += "f1 global\n"
        output += f"{f1.homogeneous_mat()}\n"
        output += "f2 global\n"
        output += f"{f2.homogeneous_mat()}\n"
        output += "p1parent\n"
        output += f"{centity1.parent.frame.homogeneous_mat()}\n"
        output += "p2parent\n"
        output += f"{centity2.parent.frame.homogeneous_mat()}\n"
        return output

    def write_constraints(self, filename):
        self.prepare_for_minimize()
        initial_guess = []
        for p in self.initial_guess_params:
            initial_guess.append(p.value)
        nparts = len(self.part_graph.nodes(data="part"))
        nparams = len(initial_guess)
        config = np.zeros(6*len(self.part_graph) + nparams)
        config[6*nparts:6*nparts + nparams] = initial_guess
        static_solve_constraint_list = self.static_solve_constraints(nparts, self.part_graph, self.primitive_graph)
        with open(f"{filename}_staticsolvecon.txt", "w") as f:
            f.write("------------------\n")
            f.write("static solve constraints\n")
            f.write("------------------\n")
            for c in static_solve_constraint_list:
                f.write(c.print_g(config, return_str=True))
        with open(f"{filename}_scipyoptcon.txt", "w") as f:
            f.write("------------------\n")
            f.write("scipy optimize constraints\n")
            f.write("------------------\n")
            for c in self.constraint_list:
                cfun = c['fun']
                args = c['args']
                f.write(c['type'] + "\n")
                for arg in args:
                    f.write(str(arg))
                f.write("\n")
                if c['type'] == 'eq':
                    args = args[0], True
                    f.write(self.print_connect_constraints(initial_guess, *args) + "\n")
                else:
                    f.write(str(cfun(initial_guess, *args)) + "\n")

    def save_pickle(self, filepath="", outdir="temp"):
        os.makedirs(outdir, exist_ok=True)
        if len(filepath) == 0:
            filepath = self.assembly_name + ".pickle"
        with open(os.path.join(outdir, filepath), 'wb') as f:
            pickle.dump(self, f)

    def load_pickle(self, filepath, no_ui=False, outdir="temp"):
        temp_file_path = os.path.join(outdir, filepath)
        if not os.path.exists(temp_file_path):
            ps.warning("File does not exist: {}".format(temp_file_path))
            return
        with open(temp_file_path, 'rb') as f:
            self = pickle.load(f)
        if not no_ui:
            ps.set_user_callback(self.polyscope_callback)
            ps.remove_all_structures()
            self.visualize(load_from_file=True)
    #=====================================^VISUALIZATION AND UI^=====================================

    #=====================================vDSL SYNTAXv=====================================
    def check_if_in_part_graph(self, part_to_check):
        if not self.part_graph.has_node(part_to_check.id):
            self.part_graph.add_node(part_to_check.id, part=part_to_check)
            for i in range(len(part_to_check.children)):
                child_i = part_to_check.children[i]
                if not self.primitive_graph.has_node(child_i.id):
                    self.primitive_graph.add_node(
                        child_i.id, primitive=child_i, connected=False, critical_dim=child_i.critical_dim(), min_dist=float('inf'))
                    self.primitive_graph_for_loops.add_node(child_i.id)
            if len(part_to_check.children) > 1:
                for i in range(len(part_to_check.children) - 1):
                    child_i = part_to_check.children[i]
                    for j in range(i + 1, len(part_to_check.children)):
                        child_j = part_to_check.children[j]
                        self.primitive_graph.add_edge(child_i.id, child_j.id, is_part_edge=True)
                        self.primitive_graph.add_edge(child_j.id, child_i.id, is_part_edge=True)
                        self.primitive_graph_for_loops.add_edge(child_i.id, child_j.id)
            return False
        return True

    def add(self, entity, frame):
        if isinstance(entity, Part):
            for c in entity.children:
                c.set_reduced(False)
            entity.frame = frame
            entity.is_fixed = True
            entity.mass = 0.
            if not self.check_if_in_part_graph(entity):
                self.starts.append(entity.id)
            for c in entity.children:
                self.primitive_graph.nodes[c.id]["min_dist"] = 0.
            # program update
            entity.name_in_assembly = "ENV_start"
            self.program += entity.primitive_init_program
            self.program += "ENV_start = Environment({})\n".format(entity.constructor_program)
            self.program += "start_frame = {}\n".format(frame.get_program())
            self.program += "{}.add(ENV_start, start_frame)\n".format(self.assembly_name)
            # end of program update
        else:
            assert isinstance(entity, Primitive)
            start_env = entity.parent
            for c in start_env.children:
                c.set_reduced(False)
            if len(start_env.children) == 1:
                start_env.frame = frame
            else:
                entity.set_frame(frame)
            start_env.is_fixed = True
            start_env.mass = 0.
            if not self.check_if_in_part_graph(start_env):
                self.starts.append(start_env.id)
            for c in start_env.children:
                self.primitive_graph.nodes[c.id]["min_dist"] = 0.
            # program update
            start_env.name_in_assembly = "ENV_start"
            self.program += start_env.primitive_init_program
            self.program += "ENV_start = Environment({})\n".format(start_env.constructor_program)
            self.program += "start_frame = {}\n".format(frame.get_program())
            child_varname = start_env.get_child_varname(entity.child_id)
            self.program += \
                "{}.add(ENV_start.{}, start_frame)\n".format(self.assembly_name, child_varname)
            # end of program update
        self.solve_prepared = False
        self.con_check_initial_guess = None

    def process_mcf(self, ctype, gentity1, gentity2, is_fixed_joint=False):
        gentity1mcfid, gentity2mcfid = None, None
        if ctype == "Surface_Surface":
            # use only one set of the surface parameters
            w1 = gentity1.width
            w2 = gentity2.width
            h1 = gentity1.length
            h2 = gentity2.length
            # check which surface is smaller and use only base frame for its mcf
            # and compute a reduced range for the larger surface's mcf parameters
            if w1 < w2 and h1 < h2:
                gentity1mcfid = gentity1.get_new_mcf(use_base_frame=True)
                gentity2mcfid = gentity2.get_new_mcf(ranges=[(w1/(2*w2), 1-w1/(2*w2)), (h1/(2*h2), 1-h1/(2*h2)), (0., 360.)])
            else:
                assert w2 <= w1 and h2 <= h1
                gentity1mcfid = gentity1.get_new_mcf(ranges=[(w2/(2*w1), 1-w2/(2*w1)), (h2/(2*h1), 1-h2/(2*h1)), (0., 360.)])
                gentity2mcfid = gentity2.get_new_mcf(use_base_frame=True)
        elif ctype == "HookFlex_HookFlex":
            # use only one of the hook's theta + phi parameters
            gentity1mcfid = gentity1.get_new_mcf(use_base_frame=True)
            gentity2mcfid = gentity2.get_new_mcf()
        elif ctype == "Tube_TubeCtl" or ctype == "TubeCtl_Rod" or ctype == "TubeCtl_Tube" or ctype == "Rod_TubeCtl":
            l1 = gentity1.length
            l2 = gentity2.length
            angle_range = (90., 90.) if "Rod" in ctype else (0., 360.)
            if l1 < l2:
                t_range = (0.5 - (l2-l1) / l2 * 0.5, 0.5 + (l2-l1) / l2 * 0.5)
                gentity1mcfid = gentity1.get_new_mcf(use_base_frame=True)
                gentity2mcfid = gentity2.get_new_mcf(ranges=[t_range, angle_range])
            else:
                t_range = (0.5 - (l1-l2) / l1 * 0.5, 0.5 + (l1-l2) / l1 * 0.5)
                gentity1mcfid = gentity1.get_new_mcf(ranges=[t_range, angle_range])
                gentity2mcfid = gentity2.get_new_mcf(use_base_frame=True)
        else:
            gentity1mcfid = gentity1.get_new_mcf()
            gentity2mcfid = gentity2.get_new_mcf()
            if "Clip" in ctype:
                if not gentity1.mcf_factory.is_fixed_under_physics(gentity1mcfid):
                    gentity1.mcf_factory.set_fixed_under_physics(gentity1mcfid, True)
                if not gentity2.mcf_factory.is_fixed_under_physics(gentity2mcfid):
                    gentity2.mcf_factory.set_fixed_under_physics(gentity2mcfid, True)
        if is_fixed_joint:
            if not gentity1.mcf_factory.is_fixed_under_physics(gentity1mcfid):
                gentity1.mcf_factory.set_fixed_under_physics(gentity1mcfid, True)
            if not gentity2.mcf_factory.is_fixed_under_physics(gentity2mcfid):
                gentity2.mcf_factory.set_fixed_under_physics(gentity2mcfid, True)
        return gentity1mcfid, gentity2mcfid

    def attach(self, entity, environment_reference, is_fixed_joint=False, alignment="default"):
        a_part = entity.parent
        b_part = environment_reference.parent
        self.check_if_in_part_graph(a_part)
        for c in a_part.children:
            diffvec = np.array(c.frame().pos) - np.array(entity.frame().pos)
            new_min_dist = np.linalg.norm(diffvec) + self.primitive_graph.nodes[environment_reference.id]["min_dist"]
            if new_min_dist < self.primitive_graph.nodes[c.id]["min_dist"]:
                self.primitive_graph.nodes[c.id]["min_dist"] = new_min_dist
        self.approx_length += a_part.attach_length
        assert b_part.id in self.part_graph.nodes()
        part_conn_info = (environment_reference.child_id, entity.child_id)
        self.part_graph.add_edge(
            b_part.id, a_part.id,\
            joint_type=get_joint_type(environment_reference, entity, is_fixed_joint),\
            joint_frame=Frame(), conn_info=part_conn_info, is_connect_edge=False)
        gentity1 = self.primitive_graph.nodes(data="primitive")[environment_reference.id]
        gentity1_connected = self.primitive_graph.nodes(data="connected")[environment_reference.id]
        if gentity1_connected and gentity1.single_conn:
            raise Exception("Cannot connect to a single-conn primitive that already has a connection.")
        gentity2 = self.primitive_graph.nodes(data="primitive")[entity.id]
        ctype = gentity1.ctype_str + "_" + gentity2.ctype_str
        alignment_frame = copy.deepcopy(Assembly.connection_frame_alignment.get(ctype, AlignmentFrame()))
        if alignment == "flip":
            alignment_frame.flip_axis(alignment_frame.axes[0])
        elif alignment != "default":
            raise Exception("Invalid alignment option: {}".format(alignment))
        gentity1mcfid, gentity2mcfid = self.process_mcf(ctype, gentity1, gentity2, is_fixed_joint)
        connect_info = (gentity1, gentity1mcfid, gentity2, gentity2mcfid, alignment_frame)
        for c in gentity2.parent.children:
            c.set_prefix_chain(self.id, gentity1.get_prefix_chain(self.id) + [(gentity1.id, gentity1mcfid, gentity2.id, gentity2mcfid, alignment_frame)])
        self.primitive_graph.add_edge(gentity1.id, gentity2.id, conn_info=connect_info, is_part_edge=False)
        self.primitive_graph_for_loops.add_edge(gentity1.id, gentity2.id)
        self.primitive_graph.nodes[entity.id]["connected"] = True
        self.primitive_graph.nodes[environment_reference.id]["connected"] = True
        if self.primitive_graph.nodes[entity.id]["critical_dim"] is not None:
            self.compute_critical_dim(entity, environment_reference)
        if self.primitive_graph.nodes[environment_reference.id]["critical_dim"] is not None:
            self.compute_critical_dim(environment_reference, entity)
        # print("setting connected to true for {} and {}".format(gentity1, gentity2))
        if gentity1.parent.is_fixed and is_fixed_joint:
            gentity2.parent.is_fixed = True
            gentity2.parent.mass = 0
        self.solve_prepared = False
        self.con_check_initial_guess = None
        # program update
        a_part_name = a_part.name_in_assembly
        if a_part_name is None:
            a_part_name = "ENV_end"
        a_child_varname = a_part.get_child_varname(entity.child_id)
        b_part_name = b_part.name_in_assembly
        if b_part_name is None:
            b_part_name = "ENV_end"
        b_child_varname = b_part.get_child_varname(environment_reference.child_id)
        self.program += \
            "{}.attach({}.{}, {}.{}, alignment=\"{}\")\n".format(
                self.assembly_name, a_part_name, a_child_varname, b_part_name, b_child_varname, alignment)
        # end of program update

    def can_connect(self, entity1, entity2):
        gentity1 = self.primitive_graph.nodes(data="primitive")[entity1.id]
        gentity1_connected = self.primitive_graph.nodes(data="connected")[entity1.id]
        gentity2 = self.primitive_graph.nodes(data="primitive")[entity2.id]
        gentity2_connected = self.primitive_graph.nodes(data="connected")[entity2.id]
        gentity12_has_edge = self.primitive_graph.has_edge(gentity1.id, gentity2.id)\
            and (not self.primitive_graph.edges[gentity1.id, gentity2.id]["is_part_edge"])
        gentity21_has_edge = self.primitive_graph.has_edge(gentity2.id, gentity1.id)\
            and (not self.primitive_graph.edges[gentity2.id, gentity1.id]["is_part_edge"])
        gentity12_connected = gentity12_has_edge or gentity21_has_edge
        return not (gentity1_connected and gentity1.single_conn or gentity2_connected and gentity2.single_conn or gentity12_connected)

    def connect(self, entity1, entity2, is_fixed_joint=False, alignment="default"):
        for c in entity1.parent.children:
            diffvec = np.array(c.frame().pos) - np.array(entity1.frame().pos)
            new_min_dist = np.linalg.norm(diffvec) + self.primitive_graph.nodes[entity2.id]["min_dist"]
            if new_min_dist < self.primitive_graph.nodes[c.id]["min_dist"]:
                self.primitive_graph.nodes[c.id]["min_dist"] = new_min_dist
        for c in entity2.parent.children:
            diffvec = np.array(c.frame().pos) - np.array(entity2.frame().pos)
            new_min_dist = np.linalg.norm(diffvec) + self.primitive_graph.nodes[entity1.id]["min_dist"]
            if new_min_dist < self.primitive_graph.nodes[c.id]["min_dist"]:
                self.primitive_graph.nodes[c.id]["min_dist"] = new_min_dist
        if self.part_graph.has_edge(entity1.parent.id, entity2.parent.id):
            self.part_graph.add_edge(entity2.parent.id, entity1.parent.id, is_connect_edge=True)
        elif self.part_graph.has_edge(entity2.parent.id, entity1.parent.id):
            self.part_graph.add_edge(entity1.parent.id, entity2.parent.id, is_connect_edge=True)
        else:
            self.part_graph.add_edge(entity2.parent.id, entity1.parent.id, is_connect_edge=True)
            self.part_graph.add_edge(entity1.parent.id, entity2.parent.id, is_connect_edge=True)
        gentity1 = self.primitive_graph.nodes(data="primitive")[entity1.id]
        gentity2 = self.primitive_graph.nodes(data="primitive")[entity2.id]
        ctype = gentity1.__class__.__name__ + "_" + gentity2.__class__.__name__
        alignment_frame = copy.deepcopy(Assembly.connection_frame_alignment.get(ctype, AlignmentFrame()))
        if alignment == "flip":
            alignment_frame.flip_axis(alignment_frame.axes[0])
        elif alignment != "default":
            raise Exception("Invalid alignment option: {}".format(alignment))
        gentity1mcfid, gentity2mcfid = self.process_mcf(ctype, gentity1, gentity2, is_fixed_joint)
        self.connects.append((gentity1, gentity1mcfid, gentity2, gentity2mcfid, alignment_frame))
        self.primitive_graph.add_edge(gentity1.id, gentity2.id, conn_info=self.connects[-1], is_part_edge=False)
        ctype = gentity2.__class__.__name__ + "_" + gentity1.__class__.__name__
        alignment_frame = copy.deepcopy(Assembly.connection_frame_alignment.get(ctype, AlignmentFrame()))
        if alignment == "flip":
            alignment_frame.flip_axis(alignment_frame.axes[0])
        elif alignment != "default":
            raise Exception("Invalid alignment option: {}".format(alignment))
        self.primitive_graph.add_edge(gentity2.id, gentity1.id, conn_info=(gentity2, gentity2mcfid, gentity1, gentity1mcfid, alignment_frame), is_part_edge=False)
        self.primitive_graph_for_loops.add_edge(gentity1.id, gentity2.id)
        self.primitive_graph.nodes[entity1.id]["connected"] = True
        self.primitive_graph.nodes[entity2.id]["connected"] = True
        if self.primitive_graph.nodes[entity1.id]["critical_dim"] is not None:
            self.compute_critical_dim(entity1, entity2)
        if self.primitive_graph.nodes[entity2.id]["critical_dim"] is not None:
            self.compute_critical_dim(entity2, entity1)
        if gentity1.parent.is_fixed and is_fixed_joint:
            gentity2.parent.is_fixed = True
            gentity2.parent.mass = 0
        elif gentity2.parent.is_fixed and is_fixed_joint:
            gentity1.parent.is_fixed = True
            gentity1.parent.mass = 0
        self.solve_prepared = False
        self.con_check_initial_guess = None
        # program update
        a_part = entity1.parent
        b_part = entity2.parent
        a_part_name = a_part.name_in_assembly
        if a_part_name is None:
            a_part_name = "ENV_end"
        a_child_varname = a_part.get_child_varname(entity1.child_id)
        b_part_name = b_part.name_in_assembly
        if b_part_name is None:
            b_part_name = "ENV_end"
        b_child_varname = b_part.get_child_varname(entity2.child_id)
        self.program += \
            "{}.connect({}.{}, {}.{}, alignment=\"{}\")\n".format(
                self.assembly_name, a_part_name, a_child_varname, b_part_name, b_child_varname, alignment)
        # end of program update

    def end_with(self, end_entity, frame=None, point=None):
        if isinstance(end_entity, Part):
            self.ends.append(end_entity.children[0])
            self.check_if_in_part_graph(end_entity)
            self.solve_prepared = False
            self.con_check_initial_guess = None
            # program update
            varname = ""
            if end_entity.name_in_assembly is None:
                end_entity.name_in_assembly = "ENV_end"
            self.end_program += end_entity.primitive_init_program
            self.end_program += "ENV_end = Environment({})\n".format(end_entity.constructor_program)
            if frame is not None:
                end_entity.frame = frame
                child_frame = end_entity.children[0].get_global_frame()
                self.end_frames.append(child_frame)
                varname = "frame"
                self.program += "end_frame = {}\n".format(frame.get_program())
            elif point is not None:
                self.end_frames.append(copy.deepcopy(point))
                varname = "point"
                self.program += "end_point = Point({})\n".format(point.get_program())
            else:
                raise Exception("Must specify either a frame or a point, not both or neither.")
            child_varname = end_entity.get_child_varname(0)
            self.program += \
                "{}.end_with(ENV_end.{}, {}={})\n".format(self.assembly_name, child_varname, varname, "end_" + varname)
            # end of program update
        else:
            assert isinstance(end_entity, Primitive)
            end_env = end_entity.parent
            self.ends.append(end_entity)
            self.check_if_in_part_graph(end_env)
            self.solve_prepared = False
            self.con_check_initial_guess = None
            # program update
            varname = ""
            if end_env.name_in_assembly is None:
                end_env.name_in_assembly = "ENV_end"
            self.end_program += end_env.primitive_init_program
            self.end_program += "ENV_end = Environment({})\n".format(end_env.constructor_program)
            if frame is not None:
                self.end_frames.append(copy.deepcopy(frame))
                parent_frame_mat = frame.homogeneous_mat() @ np.linalg.inv(end_entity.frame().homogeneous_mat())
                end_env.frame.set_from_mat(parent_frame_mat)
                varname = "frame"
                self.program += "end_frame = {}\n".format(frame.get_program())
            elif point is not None:
                self.end_frames.append(copy.deepcopy(point))
                varname = "point"
                self.program += "end_point = Point({})\n".format(point.get_program())
            else:
                raise Exception("Must specify either a frame or a point, not both or neither.")
            child_varname = end_env.get_child_varname(end_entity.child_id)
            self.program += \
                "{}.end_with(ENV_end.{}, {}={})\n".format(self.assembly_name, child_varname, varname, "end_" + varname)
            # end of program update
    #=====================================^DSL SYNTAX^=====================================

    #=====================================vUSER OBJECTIVE FUNCTIONSv=====================================
    def check_end_objective(self):
        user_objective = 0.
        objective_guided_feedback = ""
        for i in range(len(self.ends)):
            curr_end_frame = self.ends[i].get_global_frame()
            if isinstance(self.end_frames[i], Point):
                if not self.end_frames[i].contains(curr_end_frame.pos):
                    user_objective += 1e6
                    objective_guided_feedback = f"The target object is not within the range specified by {self.end_frames[i]}"
            else:
                end_sym = Assembly.primitive_symmetry[self.ends[i].__class__.__name__]
                diffvec = curr_end_frame.diff_with(self.end_frames[i])
                for sym_axis, sym in enumerate(end_sym):
                    sym_axis += 3
                    if sym == 0:
                        diffvec[sym_axis] = 0
                    else:
                        if abs(diffvec[sym_axis]) > sym:
                            diffvec[sym_axis] %= sym
                        else:
                            if abs(diffvec[sym_axis] + sym) < abs(diffvec[sym_axis]):
                                diffvec[sym_axis] += sym
                            if abs(diffvec[sym_axis] - sym) < abs(diffvec[sym_axis]):
                                diffvec[sym_axis] -= sym
                # print("diff", diffvec)
                user_objective += np.dot(diffvec, diffvec)
                if user_objective >= 1e3:
                    # diffvec = curr_end_frame.diff_with(self.end_frames[i])
                    max_diff_coord = np.argmax(np.abs(diffvec))
                    child_varname = self.ends[i].parent.get_child_varname(self.ends[i].child_id)
                    if max_diff_coord < 3:
                        position_word = ["x", "y", "z"][max_diff_coord]
                        relative_word = [["to the left", "to the right"], ["too far", "too close"], ["too low", "too high"]][max_diff_coord]
                        relative = relative_word[diffvec[max_diff_coord] > 0]
                        objective_guided_feedback = f"the resulting frame of ENV_end.{child_varname} is {relative} in the {position_word} direction"
                    else:
                        position_word = ["x", "y", "z"][max_diff_coord - 3]
                        # relative_word = [["rotated clockwise", "rotated counterclockwise"], ["tilted down", "tilted up"], ["tilted left", "tilted right"]][max_diff_coord - 3]
                        # relative = relative_word[diffvec[max_diff_coord] > 0]
                        # self.objective_guided_feedback = f"the resulting frame of ENV_end is {relative} around the {position_word} axis"
                        deg = diffvec[max_diff_coord - 3]
                        if deg < 0:
                            deg += 360
                        objective_guided_feedback = f"the resulting frame of ENV_end.{child_varname} is incorrectly rotated by another {deg} degrees around the {position_word} axis"
        return user_objective, objective_guided_feedback

    def end_objective(self):
        user_objective = 0.
        for i in range(len(self.ends)):
            curr_end_frame = self.ends[i].get_global_frame()
            if isinstance(self.end_frames[i], Point):
                if not self.end_frames[i].contains(curr_end_frame.pos):
                    user_objective += 1e6
            else:
                end_sym = Assembly.primitive_symmetry[self.ends[i].__class__.__name__]
                diffvec = curr_end_frame.diff_with(self.end_frames[i])
                for i, sym in enumerate(end_sym):
                    if sym == 0:
                        diffvec[i + 3] = 0
                    else:
                        if abs(diffvec[i + 3]) > sym:
                            diffvec[i + 3] %= sym
                        else:
                            if abs(diffvec[i + 3] + sym) < abs(diffvec[i + 3]):
                                diffvec[i + 3] += sym
                            if abs(diffvec[i + 3] - sym) < abs(diffvec[i + 3]):
                                diffvec[i + 3] -= sym
                # print("diff", diffvec)
                user_objective += np.dot(diffvec, diffvec)
        return user_objective

    def end_objective_params(self, x):
        self.set_frames_from_x(x)
        return self.end_objective()
    #=====================================^USER OBJECTIVE FUNCTIONS^=====================================

    #=====================================vSOFT CONSTRAINTS AS OBJECTIVEv=====================================
    def gravitational_potential(self):
        potential = 0.
        for _, part in self.part_graph.nodes(data="part"):
            part_frame = part.frame
            potential += part.mass * AssemblyR.G_CONSTANT * (part_frame.pos[2] + part.com_offset[2])
        return potential

    def end_objective_with_constraints_grav_potential_x(self, x):
        self.set_frames_from_x(x)
        self.user_objective_val = self.end_objective()
        self.constraints_val = constraints_penalty(x, self.constraint_list)
        self.gravitational_val = self.gravitational_potential()
        result = self.user_objective_val + self.sigma * self.constraints_val + self.gravitational_val * AssemblyR.G_WEIGHT
        # print("user_objective_val", self.user_objective_val)
        # print("constraints_val {:.4e}".format(self.constraints_val), "sigma", self.sigma)
        # print("gravitational_val {:.4e}".format(self.gravitational_val))
        return result

    def end_objective_with_constraints_grav_potential(self):
        x = np.zeros(len(self.initial_guess_params))
        for p in self.initial_guess_params:
            x[p.index] = p.value
        return self.end_objective_with_constraints_grav_potential_x(x)

    def end_objective_with_constraints(self, x):
        self.set_frames_from_x(x)
        self.user_objective_val = self.end_objective()
        self.constraints_val = constraints_penalty(x, self.constraint_list)
        return self.user_objective_val + self.sigma * self.constraints_val

    def grav_potential_with_constraints(self, x):
        self.set_frames_from_x(x)
        self.constraints_val = constraints_penalty(x, self.constraint_list)
        self.gravitational_val = self.gravitational_potential()
        return self.sigma * self.constraints_val + self.gravitational_val * AssemblyR.G_WEIGHT

    def constraints_penalty_only(self, x):
        self.set_frames_from_x(x)
        self.constraints_val = constraints_penalty(x, self.constraint_list)
        return self.sigma * self.constraints_val

    def gravitational_potential_only(self, x):
        self.set_frames_from_x(x)
        self.gravitational_val = self.gravitational_potential()
        return self.gravitational_val * AssemblyR.G_WEIGHT

    #=====================================SOFT CONSTRAINTS AS OBJECTIVE=====================================

    def forward_propagate_frames(self, should_redistribute=True):
        # use a queue-based impl for the breadth-first part_graph traversal
        queue = []
        # enqueue the start parts
        for start in self.starts:
            if start not in queue:
                queue.append(start)
        if should_redistribute:
            # pre-set the mcf slider parameters to be evenly distributed over their ranges
            for _, primitive in self.primitive_graph.nodes(data="primitive"):
                primitive.mcf_factory.distribute_slider_parameters()

        new_queue = []
        pcpairs = []
        while len(queue) > 0:
            # pop from queue and explore the connected parts
            current_part = queue.pop(0)
            for child_part in self.part_graph.neighbors(current_part):
                if self.part_graph.edges[current_part, child_part]["is_connect_edge"]:
                    continue
                pcpairs.append((current_part, child_part))
                if child_part not in new_queue:
                    new_queue.append(child_part)

            if len(queue) == 0:
                # time to process the pcpairs that's connecting current layer and the next layer
                for pcpair in pcpairs:
                    part1id, part2id = pcpair
                    part1 = self.part_graph.nodes(data="part")[part1id]
                    part2 = self.part_graph.nodes(data="part")[part2id]
                    for entity1 in part1.children:
                        for entity2 in part2.children:
                            if self.primitive_graph.has_edge(entity1.id, entity2.id) and self.primitive_graph.has_edge(entity2.id, entity1.id):
                                # print("skipping", pcpair, entity1, entity2)
                                continue
                            elif self.primitive_graph.has_edge(entity1.id, entity2.id):
                                conn_info = self.primitive_graph.edges[entity1.id, entity2.id]["conn_info"]
                                part1 = entity1.parent
                                part2 = entity2.parent
                                # print("connecting part{}-primitive{} to part{}-primitive{}".format(part1.id, entity1.id, part2.id, entity2.id))
                                # print(part1, entity1)
                                # print(part2, entity2)
                                entity1fid, entity2fid = conn_info[1], conn_info[3]
                                if entity1 != conn_info[0]:
                                    entity1fid, entity2fid = conn_info[3], conn_info[1]
                                entity1f = entity1.mcf_factory.eval_mcf(entity1fid)
                                entity2f = entity2.mcf_factory.eval_mcf(entity2fid)
                                entity1global = part1.frame.transform_frame(entity1f.transform_frame(conn_info[4])) # alignment frame
                                # # v for joint_frame debugging
                                self.part_graph.edges[part1.id, part2.id]["joint_frame"] = entity1global
                                self.part_graph.edges[part1.id, part2.id]["p1mcf"] = entity1f#.transform_frame(conn_info[4])
                                self.part_graph.edges[part1.id, part2.id]["p2mcf"] = entity2f
                                # # ^ for joint_frame debugging
                                part2_frame_mat = entity1global.homogeneous_mat() @ np.linalg.inv(entity2f.homogeneous_mat())
                                part2.frame.set_from_mat(part2_frame_mat)
                # clear out the temporary queue and pcpairs list
                queue.extend(new_queue)
                new_queue.clear()
                pcpairs.clear()

    def compute_initial_guess_frames(self, should_redistribute=True):
        self.forward_propagate_frames(should_redistribute=should_redistribute)
        return self.end_objective_with_constraints_grav_potential()

    def static_solve_constraints(self, nparts, part_graph_to_use, primitive_graph_to_use):    
        id_to_index = {id:i for i, id in enumerate(part_graph_to_use.nodes())}    
        origmats = []
        for _, part in part_graph_to_use.nodes(data="part"):
            mat = part.frame.homogeneous_mat()
            origmats.append(mat)
        constraints = []
        # Set the soft constraints
        for id_i, part1 in part_graph_to_use.nodes(data="part"):
            for id_j, part2 in part_graph_to_use.nodes(data="part"):
                if id_i == id_j:
                    continue
                i = id_to_index[id_i]
                j = id_to_index[id_j]
                for entity1 in part1.children:
                    for entity2 in part2.children:
                        addConstraint = False
                        if primitive_graph_to_use.has_edge(entity1.id, entity2.id) and primitive_graph_to_use.has_edge(entity2.id, entity1.id):  
                            # Only include the constraint once?
                            if i < j:
                                addConstraint = True                            
                                
                        elif primitive_graph_to_use.has_edge(entity1.id, entity2.id):                            
                            addConstraint = True
                            
                        if addConstraint:
                            conn_info = primitive_graph_to_use.edges[entity1.id, entity2.id]["conn_info"]
                            if entity1 != conn_info[0]:
                                sys.exit("Why are these orders not consistent??")
                            entity1fid, entity2fid = conn_info[1], conn_info[3]   
                            conn1indices = entity1.mcf_factory.MCFParams[entity1fid].indices(start_index=6*nparts)
                            conn2indices = entity2.mcf_factory.MCFParams[entity2fid].indices(start_index=6*nparts)
                            nparams1 = entity1.mcf_factory.numParams()
                            nparams2 = entity2.mcf_factory.numParams()
                            assert(len(conn1indices) == nparams1)
                            assert(len(conn2indices) == nparams2)
                            connectionmat = conn_info[4].homogeneous_mat()
                            constraints.append(ConnectorConstraint(
                                id_to_index, origmats, entity1, entity2, entity1fid, entity2fid,
                                connectionmat, 6*i, 6*j, conn1indices, conn2indices, nparams1, nparams2))
        return constraints

    def static_solve(self, initial_guess, maxiters, no_objective=False, subpartgraph=None, subprimgraph=None):
        part_graph_to_use = self.part_graph if subpartgraph is None else subpartgraph
        primitive_graph_to_use = self.primitive_graph if subprimgraph is None else subprimgraph

        # We will linearize about the initial guess
        self.set_frames_from_x(initial_guess)        
        nparts = len(part_graph_to_use.nodes(data="part"))
        nparams = len(initial_guess)
        
        config = np.zeros(6*nparts + nparams)
        config[6*nparts:6*nparts + nparams] = initial_guess 
        
        constraints = []
        DOFbounds = []
        
        # Set the bound constraints
        id_to_index = {id:i for i, id in enumerate(part_graph_to_use.nodes())}
        for start in self.starts:
            #constraints.append(FixedPartConstraint(6*start))
            start_idx = id_to_index[start]
            for i in range(6):
                DOFbounds.append(DOFBound(6*start_idx+i, 0.0, 0.0, True))

        for param in self.initial_guess_params:
            idx = param.index
            if idx == -1: continue
            pbounds = param.bounds()
            flag = param.flag
            entity = primitive_graph_to_use.nodes(data="primitive")[param.pid]
            if param.fixed_under_physics:
                # Pin the parameter for now
                if Assembly.verbose: print("Adding pin bounds for parameter ", idx, ", type ", entity.mcf_factory.__class__, ": ", pbounds)
                DOFbounds.append(DOFBound(6*nparts + idx, initial_guess[idx], initial_guess[idx], True))
            elif flag is not ParameterFlag.UNBOUNDED:
                if Assembly.verbose: print("Adding bounds for parameter ", idx, ", type ", entity.mcf_factory.__class__, ": ", pbounds)
                DOFbounds.append(DOFBound(6*nparts + idx, pbounds[0], pbounds[1], flag == ParameterFlag.BOUNDED_AND_CLAMPED))

        origmats = []
        for _, part in part_graph_to_use.nodes(data="part"):
            mat = part.frame.homogeneous_mat()
            origmats.append(mat)

        # Set the soft constraints
        for id_i, part1 in part_graph_to_use.nodes(data="part"):
            for id_j, part2 in part_graph_to_use.nodes(data="part"):
                if id_i == id_j:
                    continue
                i = id_to_index[id_i]
                j = id_to_index[id_j]
                for entity1 in part1.children:
                    for entity2 in part2.children:
                        addConstraint = False
                        if primitive_graph_to_use.has_edge(entity1.id, entity2.id) and primitive_graph_to_use.has_edge(entity2.id, entity1.id):  
                            # Only include the constraint once?
                            if i < j:
                                addConstraint = True                            
                                
                        elif primitive_graph_to_use.has_edge(entity1.id, entity2.id):                            
                            addConstraint = True
                            
                        if addConstraint:
                            conn_info = primitive_graph_to_use.edges[entity1.id, entity2.id]["conn_info"]
                            if entity1 != conn_info[0]:
                                sys.exit("Why are these orders not consistent??")
                            entity1fid, entity2fid = conn_info[1], conn_info[3]   
                            conn1indices = entity1.mcf_factory.MCFParams[entity1fid].indices(start_index=6*nparts)
                            conn2indices = entity2.mcf_factory.MCFParams[entity2fid].indices(start_index=6*nparts)
                            # print(conn1indices)
                            # print(conn2indices)
                            nparams1 = entity1.mcf_factory.numParams()
                            nparams2 = entity2.mcf_factory.numParams()
                            assert(len(conn1indices) == nparams1)
                            assert(len(conn2indices) == nparams2)
                            connectionmat = conn_info[4].homogeneous_mat()
                            constraints.append(ConnectorConstraint(
                                id_to_index, origmats, entity1, entity2, entity1fid, entity2fid,
                                connectionmat, 6*i, 6*j, conn1indices, conn2indices, nparams1, nparams2))
        
        # Set the potentials
        potentials = []

        # Set the multiconn potential
        # for multiconn_c in self.constraint_list:
        #     if multiconn_c['type'] == 'ineq':
        #         mcf1ids, mcf2ids, tdists, dists = multiconn_c['args']
        #         potentials.append(MultiConnPotential(mcf1ids, mcf2ids, tdists, dists, nparts))

        # in mm
        g = 1e-6 * np.array([0, 0, -9.8])
        potentials.append(GravityPotential(part_graph_to_use.nodes(data="part"), origmats, g, id_to_index))
        
        # Set the objective
        objectives = []
        if not no_objective:
            for i in range(len(self.ends)):
                end_sym = Assembly.primitive_symmetry[self.ends[i].__class__.__name__]
                mask = [True, True, True]
                for j, sym in enumerate(end_sym):
                    if sym == 0:
                        mask[j] = False
                objectives.append(TargetPotential(self.ends[i], origmats, self.end_frames[i].homogeneous_mat(), mask))

        params = SolverParams(10000, 1, maxiters, 1e-6)  
        ans = staticSolver(nparts, potentials, constraints, objectives, DOFbounds, origmats, config, params, verbose = Assembly.verbose)

        self.set_frames_from_x(ans[1])
        config[:6*nparts] = np.zeros(6*nparts)
        config[6*nparts:] = ans[1]
        residuals = np.zeros(len(constraints))
        ridx = 0
        for c in constraints:
            residuals[ridx] = np.linalg.norm(c.g(config))
            ridx = ridx + 1

        if Assembly.verbose:
            print("Constraint residuals: ", residuals)
            # for i in range(nparts):
            #     randaxisangle = np.random.random_sample(3)
            #     randtrans = np.random.random_sample(3)
            #     origmats[i] = origmats[i] @ homogeneousMatFromTranslation(randtrans) @ homogeneousMatFromAxisAngle(randaxisangle)
            # matsbackup = list(origmats)

            # ##Finite difference checks
            # idx = 0
            # grandtotal = 0
            # for o in objectives:
            
            #     eps = 1e-6
            #     pert = np.random.random_sample(config.size)
            #     for i in range(nparts):
            #         pert[6*i+3:6*i+6] = np.zeros(3)
            #     fwd = config + eps * pert            
            #     back = config - eps * pert
                
            #     ofwd = o.V(fwd)
            #     oback = o.V(back)
            #     ograd = o.dV(config)
            #     print(".....")
            #     print((ofwd-oback)/(2.0*eps))
            #     print(ograd @ pert)
            #     print(".....")
                
            #     for i in range(nparts):
            #         mat = matsbackup[i]                
            #         mat = mat @ homogeneousMatFromTranslation(fwd[6*i:6*i+3])
            #         origmats[i] = mat                    
            #     dgfwd = o.dV(fwd)
            #     for i in range(nparts):
            #         mat = matsbackup[i]                
            #         mat = mat @ homogeneousMatFromTranslation(back[6*i:6*i+3])
            #         origmats[i] = mat
            #     dgback = o.dV(back)
            #     findiff = (dgfwd-dgback)/(2.0 * eps)                        
            #     for i in range(nparts):
            #         origmats[i] = matsbackup[i]
            #     hg = scipy.sparse.coo_array(o.HV(config), (config.size, config.size))
            #     row = hg @ pert
            #     print(findiff.tolist())            
            #     print("=====")            
            #     print(row.tolist())
            #     error = np.linalg.norm(findiff - row)
            #     print(error)
            #     print("#####\n")            
            #     grandtotal = grandtotal + error
                
            # for c in constraints:
            #     eps = 1e-6
            #     pert = np.random.random_sample(config.size)
            #     for i in range(nparts):
            #         pert[6*i+3:6*i+6] = np.zeros(3)
            #     fwd = config + eps * pert
            #     back = config - eps * pert
                
            #     print("cval ", c.g(config))            
            #     ofwd = c.g(fwd)
            #     oback = c.g(back)
            #     ograd = scipy.sparse.coo_array(c.dg(config), (c.dimension(), config.size))
            #     print(".....")
            #     ofindiff = (ofwd-oback)/(2.0*eps)
            #     print(ofindiff)
            #     print((ograd @ pert))
            #     error = np.linalg.norm(ofindiff - (ograd @ pert))
            #     print(error)
            #     print(".....")
            #     grandtotal = grandtotal + error
                
            #     for i in range(nparts):
            #         mat = matsbackup[i]                
            #         mat = mat @ homogeneousMatFromTranslation(fwd[6*i:6*i+3])
            #         origmats[i] = mat                    
            #     dgfwd = scipy.sparse.coo_array(c.dg(fwd), (c.dimension(), config.size))
            #     for i in range(nparts):
            #         mat = matsbackup[i]                
            #         mat = mat @ homogeneousMatFromTranslation(back[6*i:6*i+3])
            #         origmats[i] = mat
            #     dgback = scipy.sparse.coo_array(c.dg(back), (c.dimension(), config.size))
            #     findiff = (dgfwd-dgback)/(2.0 * eps)
            #     for i in range(nparts):
            #         origmats[i] = matsbackup[i]
            #     print("Constraint ", idx, ": dimension: ", c.dimension())
            #     idx = idx + 1
            #     totalerror = 0
            #     for i in range(c.dimension()):
            #         hg = scipy.sparse.coo_array(c.Hg(config)[i], (config.size, config.size))
            #         row = hg @ pert
            #         print(findiff.toarray()[i,:].tolist())            
            #         print("=====")            
            #         print(row.tolist())
            #         error = np.linalg.norm(findiff.toarray()[i,:] - row)
            #         print(error)
            #         print("#####")
            #         totalerror = totalerror + error
            #     print("Total error: ", totalerror, "\n")            
            #     grandtotal = grandtotal + totalerror
            # print("Grand total: ", grandtotal)

        if not isinstance(ans[2], str):
            msg1, dofidx, msg2 = ans[2]
            ans = ans[0], ans[1], msg1 + self.initial_guess_params[dofidx].name + msg2
        print(ans)
        if no_objective:
            if ans[0] == float('inf'):
                return ans
            return self.end_objective(), ans[1], ans[2]
        return ans

    #==============================TWO STEP WITH BRUTE FORCE INITIAL GUESSES================================
    def solve_with_two_steps(self, initial_guess, xbounds, maxiters):
        batchiters = 100
        nbatch = int(maxiters / batchiters)
        min_objective = float('inf')
        min_objective_x = None
        self.sigma = AssemblyR.SIGMA_INIT
        opt_options = {'disp':Assembly.verbose,'maxiter':batchiters}
        for i in range(nbatch):
            # 1e-4 clamping for the objective to avoid the roundoff issue
            res = minimize(
                self.gravitational_potential_only,
                method="Powell",
                x0=initial_guess,
                bounds=xbounds,
                options=opt_options)
            self.set_frames_from_x(res.x)
            res = minimize(
                self.end_objective_with_constraints_grav_potential_x,
                method="Powell",
                x0=initial_guess,
                bounds=xbounds,
                options=opt_options)
            if min_objective > self.user_objective_val:
                min_objective = self.user_objective_val
                min_objective_x = res.x
                initial_guess = min_objective_x
                self.set_frames_from_x(initial_guess)
            if self.constraints_val is not None and self.constraints_val < self.constraints_tol or min_objective < self.objective_tol:
                break
            else:
                self.sigma *= 2 # this update could be considered as a dual variable and be computed from mirror descent
        print("two step best objective: {}".format(min_objective))
        return min_objective, min_objective_x
    #==============================TWO STEP WITH BRUTE FORCE INITIAL GUESSES================================

    def generate_initial_guess_list(self, options):
        initial_guess_list = []
        initial_guess = np.zeros(len(self.initial_guess_params))
        for p in self.initial_guess_params:
            initial_guess[p.index] = p.value
        if options.parameter_sweep and (options.ninitialization is None or options.ninitialization > 1):
            # USE THE FULL INITIAL GUESS LIST
            initial_guess_list = generate_initial_guesses(self.initial_guess_params, k=options.ninitialization-1, random=options.random_init)
            if self.con_check_initial_guess is not None and len(self.con_check_initial_guess) > 0:
                itertools.chain([self.con_check_initial_guess], initial_guess_list)
            else:
                itertools.chain([initial_guess], initial_guess_list)
                # initial_guess_list = generate_initial_guesses(self.initial_guess_params, k=options.ninitialization, random=options.random_init)
        else:
            if self.con_check_initial_guess is not None and len(self.con_check_initial_guess) > 0:
                initial_guess_list = [self.con_check_initial_guess]
            else:
                # USE ONLY THE DEFAULT AS INITIAL GUESS (because full initial guess is too long)
                initial_guess_list.append(initial_guess)
        return initial_guess_list

    def solve_constraints_only(self, options):
        if len(self.connects) == 0:
            return True

        start_time = time.time()

        # this includes the slider parameters as part of the independent variables to be solved for
        # eq, zero
        def compute_connect_constraint_params(x, c, verbose=False):
            centity1, centity1fid, centity2, centity2fid, calignment_frame = c
            for param in centity1.mcf_factory.MCFParams[centity1fid].slider_parameters():
                param.value = x[param.index]
            for param in centity2.mcf_factory.MCFParams[centity2fid].slider_parameters():
                param.value = x[param.index]
            centity1f = centity1.mcf_factory.eval_mcf(centity1fid)
            centity2f = centity2.mcf_factory.eval_mcf(centity2fid)
            f1 = centity1.parent.frame.transform_frame(centity1f.transform_frame(calignment_frame))
            f2 = centity2.parent.frame.transform_frame(centity2f)
            self.part_graph.edges[centity1.parent.id, centity2.parent.id]["joint_frame"] = f1
            self.part_graph.edges[centity1.parent.id, centity2.parent.id]["p1mcf"] = centity1f
            self.part_graph.edges[centity1.parent.id, centity2.parent.id]["p2mcf"] = centity2f
            if verbose:
                print("compute_connect_constraint_params")
                print(centity1, centity1fid, centity2, centity2fid)
                print("f1 global")
                print((centity1.parent.frame.transform_frame(centity1f)).homogeneous_mat())
                print("centity1 parent")
                print(repr(centity1.parent.frame.homogeneous_mat()))
                print("centity1f")
                print(repr(centity1f.homogeneous_mat()))
                print("f2 global")
                print(f2.homogeneous_mat())
                print("p1parent", centity1.parent.frame.homogeneous_mat())
                print("p2parent", centity2.parent.frame.homogeneous_mat())
            diffvec = f1.diff_with(f2)
            end_sym = Assembly.primitive_symmetry[centity1.__class__.__name__]
            for sym_axis, sym in enumerate(end_sym):
                sym_axis += 3
                if sym == 0:
                    diffvec[sym_axis] = 0
                else:
                    if abs(diffvec[sym_axis]) > sym:
                        diffvec[sym_axis] %= sym
                    else:
                        if abs(diffvec[sym_axis] + sym) < abs(diffvec[sym_axis]):
                            diffvec[sym_axis] += sym
                        if abs(diffvec[sym_axis] - sym) < abs(diffvec[sym_axis]):
                            diffvec[sym_axis] -= sym
            return np.dot(diffvec, diffvec)

        def process_param(params_dict, param, primitive):
            params_dict[param.index] = param
            # the flag is default to BOUNDED_AND_OPEN
            # if the parameter is unbounded then it will have been set to UNBOUNDED already
            # so here we only update it if it is physically clamped
            if (not param.unbounded) and primitive.has_bounds():
                param.flag = ParameterFlag.BOUNDED_AND_CLAMPED

        for _, entity in self.primitive_graph.nodes(data="primitive"):
            for mcfparam in entity.mcf_factory.MCFParams:
                mcfparam.reset_slider_param_indices()

        self.initial_guess_params = []
        params_dict = {}
        # process connect() constraints
        for c in self.connects:
            centity1, centity1fid, centity2, centity2fid, _ = c
            e1prefix = centity1.get_prefix_chain(self.id)
            for prefix in e1prefix:
                entity1id, entity1fid, entity2id, entity2fid, _ = prefix
                gentity1 = self.primitive_graph.nodes(data="primitive")[entity1id]
                gentity2 = self.primitive_graph.nodes(data="primitive")[entity2id]
                gentity1.mcf_factory.MCFParams[entity1fid].set_slider_param_indices(len(params_dict.items()))
                for p in gentity1.mcf_factory.MCFParams[entity1fid].slider_parameters():
                    process_param(params_dict, p, gentity1)
                gentity2.mcf_factory.MCFParams[entity2fid].set_slider_param_indices(len(params_dict.items()))
                for p in gentity2.mcf_factory.MCFParams[entity2fid].slider_parameters():
                    process_param(params_dict, p, gentity2)
            centity1.mcf_factory.MCFParams[centity1fid].set_slider_param_indices(len(params_dict.items()))
            for p in centity1.mcf_factory.MCFParams[centity1fid].slider_parameters():
                process_param(params_dict, p, centity1)
            e2prefix = centity2.get_prefix_chain(self.id)
            for prefix in e2prefix:
                entity1id, entity1fid, entity2id, entity2fid, _ = prefix
                gentity1 = self.primitive_graph.nodes(data="primitive")[entity1id]
                gentity2 = self.primitive_graph.nodes(data="primitive")[entity2id]
                gentity1.mcf_factory.MCFParams[entity1fid].set_slider_param_indices(len(params_dict.items()))
                for p in gentity1.mcf_factory.MCFParams[entity1fid].slider_parameters():
                    process_param(params_dict, p, gentity1)
                gentity2.mcf_factory.MCFParams[entity2fid].set_slider_param_indices(len(params_dict.items()))
                for p in gentity2.mcf_factory.MCFParams[entity2fid].slider_parameters():
                    process_param(params_dict, p, gentity2)
            centity2.mcf_factory.MCFParams[centity2fid].set_slider_param_indices(len(params_dict.items()))
            for p in centity2.mcf_factory.MCFParams[centity2fid].slider_parameters():
                process_param(params_dict, p, centity2)
        self.initial_guess_params = list(params_dict.values())

        self.constraint_list = []
        # process connect() constraints
        for c in self.connects:
            self.constraint_list.append(
                {'type':'eq', 'fun': compute_connect_constraint_params, 'args': (c, False)})

        xbounds = [(0,0)] * len(self.initial_guess_params)
        for p in self.initial_guess_params:
            xbounds[p.index] = p.bounds()
        # To get flags for each parameter:
        # flags = [p.flag for p in self.initial_guess_params] # one of ParameterFlag values

        initial_guess_list = self.generate_initial_guess_list(options)
        overall_min_obj = float('inf')
        overall_min_obj_x = None
        for initial_guess in initial_guess_list:
            min_objective, min_objective_x = self.solve_constraints_only_with_initial_guess(initial_guess, xbounds, options.maxiters)
            if min_objective <= overall_min_obj:
                overall_min_obj = min_objective
                overall_min_obj_x = min_objective_x
                # print(min_objective)
                if self.constraints_val < self.constraints_tol:
                    return True
        if overall_min_obj_x is None:
            print("No valid continuous parameters")
            return False
        else:
            self.end_objective_with_constraints_grav_potential_x(overall_min_obj_x)
            print("\nbest x:", overall_min_obj_x)
            print("best constraints penalty:", self.constraints_val)
        print("time elapsed:", time.time() - start_time)
        return False

    def minimize_callback(self, xk):
        self.last10convals.append(constraints_penalty(xk, self.constraint_list))
        if self.last10convals[-1] < 1e-6:
            self.minimize_considered_failed = False
            self.con_check_initial_guess = xk
            raise StopIteration("constraints_penalty < 1e-6")
            # return True
        if len(self.last10convals) == 10:
            l = list(self.last10convals)
            try:
                res = stats.linregress(l[1:], l[:-1])
                if abs(res.slope) < 1:
                    self.minimize_considered_failed = True
                    raise StopIteration("slope = {} < 1".format(res.slope))
                    # return True
            except ValueError:
                self.minimize_considered_failed = True
                raise StopIteration("slope = 0 < 1")

    def check_constraints_feasibility(self, options):
        if AssemblyR.SKIP_FEASIBILITY:
            return True

        print("check_constraints_feasibility()")
        start_time = time.time()

        self.prepare_for_minimize()
        self.con_check_initial_guess = None

        batchiters = 100
        nbatch = int(options.maxiters / batchiters)
        opt_options = {'disp':Assembly.verbose,'maxiter':batchiters}
        initial_guess = np.zeros(len(self.initial_guess_params))

        xbounds = [(0,0)] * len(self.initial_guess_params)
        for p in self.initial_guess_params:
            initial_guess[p.index] = p.value
            xbounds[p.index] = p.bounds()

        initial_guess_list = self.generate_initial_guess_list(options)
        overall_min_objective = float('inf')
        overall_min_objective_x = None
        for ni, initial_guess in enumerate(initial_guess_list):
            min_objective = float('inf')
            min_objective_x = None
            self.sigma = AssemblyR.SIGMA_INIT
            self.last10convals.clear()
            for _ in range(nbatch):
                self.minimize_considered_failed = False
                # 1e-4 clamping for the objective to avoid the roundoff issue
                res = None
                try:
                    res = minimize(
                        self.constraints_penalty_only,
                        method="Powell",
                        x0=initial_guess,
                        bounds=xbounds,
                        options=opt_options,
                        callback=self.minimize_callback)
                except StopIteration as e: # temporary solution: callback ignored by scipy 1.10 and fix is in 1.11
                    # print(e.value)
                    if self.minimize_considered_failed:
                        break
                    else:
                        min_objective = self.last10convals[-1]
                        min_objective_x = self.con_check_initial_guess
                        self.set_frames_from_x(min_objective_x)
                        print("(True) connect constraint value:", min_objective)
                        # print(self.con_check_initial_guess)
                        # print("time elapsed:", time.time() - start_time)
                        # self.set_frames_from_x(res.x)
                        return True, ni + 1, time.time() - start_time
                if res is not None and min_objective > self.constraints_val:
                    min_objective = self.constraints_val
                    min_objective_x = res.x
                    initial_guess = min_objective_x
                    # self.set_frames_from_x(res.x)
                if min_objective < self.constraints_tol:
                    print("(True) connect constraint value:", min_objective)
                    self.con_check_initial_guess = min_objective_x
                    # print(self.con_check_initial_guess)
                    # print("time elapsed:", time.time() - start_time)
                    # self.set_frames_from_x(min_objective_x)
                    return True, ni + 1, time.time() - start_time
                else:
                    self.sigma *= 2
            if not self.minimize_considered_failed and overall_min_objective > min_objective:
                overall_min_objective = min_objective
                overall_min_objective_x = min_objective_x
                if overall_min_objective < self.constraints_tol:
                    print("(True) connect constraint value:", overall_min_objective)
                    self.con_check_initial_guess = overall_min_objective_x
                    # print(self.con_check_initial_guess)
                    # print("time elapsed:", time.time() - start_time)
                    # self.set_frames_from_x(overall_min_objective_x)
                    return True, ni + 1, time.time() - start_time
        print("(False) connect constraint value:", overall_min_objective)
        # print("time elapsed:", time.time() - start_time)
        return False, ni + 1, time.time() - start_time

    def check_constraints_with_linear_program(self, cycles):
        start_time = time.time()
        overall_bounds = []
        nrows_per_cycle = []
        for cycle in cycles:
            nparams_for_cycle = 0
            bounds = []
            bound_pairs = set()
            for i in range(1, len(cycle) - 1):
                p = self.primitive_graph.nodes(data="primitive")[cycle[i]]
                bound = None
                if len(p.parent.children) == 1: # single primitive part
                    bound = (0, p.parent.attach_length)
                if bound is not None:
                    bounds.append((bound[0] - AssemblyR.LINPROG_EPSILON, bound[1] + AssemblyR.LINPROG_EPSILON))
                    nparams_for_cycle += 1
                    bound_pairs.add((cycle[i], cycle[i]))
            cycle = cycle + cycle
            for i in range(len(cycle) - 1):
                if (not self.primitive_graph.has_edge(cycle[i], cycle[i + 1])) and (not self.primitive_graph.has_edge(cycle[i + 1], cycle[i])):
                    if cycle[i] != cycle[i + 1]:
                        raise ValueError("Cycle edge not found", cycle[i], cycle[i + 1])
                p1 = self.primitive_graph.nodes(data="primitive")[cycle[i]]
                p2 = self.primitive_graph.nodes(data="primitive")[cycle[i + 1]]
                bound = None
                if p1.parent == p2.parent: # for same-part edges
                    if (cycle[i], cycle[i + 1]) in bound_pairs:
                        # print("edge looping started! no more new edges need to be considered")
                        break
                    if p1 != p2: # same part but different primitive edge distance range can be looked up from the range dict
                        key = (p1.ohe.value, p2.ohe.value)
                        if key in self.part_graph.nodes(data="part")[p1.parent.id].children_pairwise_ranges:
                            bound = self.part_graph.nodes(data="part")[p1.parent.id].children_pairwise_ranges.get(key, None)
                        else:
                            key = (p2.ohe.value, p1.ohe.value)
                            bound = self.part_graph.nodes(data="part")[p1.parent.id].children_pairwise_ranges.get(key, None)
                            if bound is None:
                                raise ValueError("No range found for same part edge of", p1.parent)
                    else: # same part and same primitive edge distance range can be directly computed
                        print("this should never be reached")
                else: # for different-part edges
                    # TODO: check if this is correct
                    if isinstance(p1.parent, Environment) and not isinstance(p2.parent, Environment): # single environment part primitive
                        bound = p1.mcf_factory.get_mcf_range()
                if bound is not None:
                    bounds.append((bound[0] - AssemblyR.LINPROG_EPSILON, bound[1] + AssemblyR.LINPROG_EPSILON))
                    nparams_for_cycle += 1
                    bound_pairs.add((cycle[i], cycle[i + 1]))
            overall_bounds += bounds
            nrows_per_cycle.append(nparams_for_cycle)
            assert nparams_for_cycle == len(bounds)
        nvariables = sum(nrows_per_cycle)
        A = np.zeros((nvariables, nvariables))
        curr_row = 0
        for d in nrows_per_cycle:
            A[curr_row:curr_row + d, curr_row:curr_row + d] = -np.ones((d, d)) + 2 * np.eye(d)
            curr_row += d
        b = np.zeros(nvariables)
        res = linprog(np.ones(nvariables), A_ub=A, b_ub=b, bounds=overall_bounds)
        time_elapsed = time.time() - start_time
        print("time elapsed:", time_elapsed)
        return res.success, time_elapsed

    def solve_with_initial_guess(self, initial_guess, xbounds, maxiters):
        batchiters = 100
        nbatch = int(maxiters / batchiters)
        opt_options = {'disp':Assembly.verbose,'maxiter':batchiters}
        min_objective = float('inf')
        min_objective_x = None
        self.sigma = AssemblyR.SIGMA_INIT
        for i in range(nbatch):
            # 1e-4 clamping for the objective to avoid the roundoff issue
            res = minimize(
                self.end_objective_with_constraints_grav_potential_x,
                method="Powell",
                x0=initial_guess,
                bounds=xbounds,
                options=opt_options)
            if min_objective > self.user_objective_val:
                min_objective = self.user_objective_val
                min_objective_x = res.x
                initial_guess = min_objective_x
                for p in self.initial_guess_params:
                    p.value = min_objective_x[p.index]
            if self.constraints_val is not None and self.constraints_val < self.constraints_tol or min_objective < self.objective_tol:
                break
            else:
                self.sigma *= 2
        print("single step best objective:", min_objective)
        return min_objective, min_objective_x

    def solve_constraints_only_with_initial_guess(self, initial_guess, xbounds, maxiters):
        batchiters = 500
        nbatch = int(maxiters / batchiters)
        overall_min_obj = float('inf')
        overall_min_obj_x = None
        self.sigma = AssemblyR.SIGMA_INIT
        for i in range(nbatch):
            # 1e-4 clamping for the objective to avoid the roundoff issue
            opt_options = {'disp':Assembly.verbose,'maxiter':batchiters}
            res = minimize(
                self.constraints_penalty_only,
                method="Powell",
                x0=initial_guess,
                bounds=xbounds,
                options=opt_options)
            if overall_min_obj > res.fun:
                overall_min_obj = res.fun
                overall_min_obj_x = res.x
                initial_guess = overall_min_obj_x
                for p in self.initial_guess_params:
                    p.value = overall_min_obj_x[p.index]
            if self.constraints_val is not None and self.constraints_val < self.constraints_tol or overall_min_obj < self.objective_tol:
                break
            else:
                self.sigma *= 2
        self.end_objective_with_constraints_grav_potential_x(overall_min_obj_x)
        # print("\n\tbest x:", overall_min_obj_x, "\n")
        # print(self.constraints_val)
        return overall_min_obj, overall_min_obj_x

    def set_frames_from_x(self, x):
        for p in self.initial_guess_params:
            p.set_value(x[p.index])
        self.forward_propagate_frames(should_redistribute=False)

    def prepare_for_minimize(self, subgraph=None):
        if self.solve_prepared:
            return

        # this includes the slider parameters as part of the independent variables to be solved for
        # eq, zero
        def compute_connect_constraint_params(x, c, verbose=False):
            centity1, centity1fid, centity2, centity2fid, calignment_frame = c
            for param in centity1.mcf_factory.MCFParams[centity1fid].slider_parameters():
                param.value = x[param.index]
            for param in centity2.mcf_factory.MCFParams[centity2fid].slider_parameters():
                param.value = x[param.index]
            centity1f = centity1.mcf_factory.eval_mcf(centity1fid)
            centity2f = centity2.mcf_factory.eval_mcf(centity2fid)
            f1 = centity1.parent.frame.transform_frame(centity1f.transform_frame(calignment_frame))
            f2 = centity2.parent.frame.transform_frame(centity2f)
            self.part_graph.edges[centity1.parent.id, centity2.parent.id]["joint_frame"] = f1
            self.part_graph.edges[centity1.parent.id, centity2.parent.id]["p1mcf"] = centity1f
            self.part_graph.edges[centity1.parent.id, centity2.parent.id]["p2mcf"] = centity2f
            diffvec = f1.diff_with(f2)
            if verbose:
                print("compute_connect_constraint_params")
                print(centity1, centity1fid, centity2, centity2fid)
                print("f1 global")
                print((centity1.parent.frame.transform_frame(centity1f)).homogeneous_mat())
                print("centity1 parent")
                print(repr(centity1.parent.frame.homogeneous_mat()))
                print("centity1f")
                print(repr(centity1f.homogeneous_mat()))
                print("f2 global")
                print(f2.homogeneous_mat())
                print("p1parent", centity1.parent.frame.homogeneous_mat())
                print("p2parent", centity2.parent.frame.homogeneous_mat())
            end_sym = Assembly.primitive_symmetry[centity1.__class__.__name__]
            for sym_axis, sym in enumerate(end_sym):
                sym_axis += 3
                if sym == 0:
                    diffvec[sym_axis] = 0
                else:
                    if abs(diffvec[sym_axis]) > sym:
                        diffvec[sym_axis] %= sym
                    else:
                        if abs(diffvec[sym_axis] + sym) < abs(diffvec[sym_axis]):
                            diffvec[sym_axis] += sym
                        if abs(diffvec[sym_axis] - sym) < abs(diffvec[sym_axis]):
                            diffvec[sym_axis] -= sym
            # debugging code to compute exact position of hemisphere on toothbrush
            # if centity1.id == 4 and centity2.id == 0:
            #     idealf1 = f2.transform_frame(calignment_frame)
            #     tframe1 = centity1.mcf_factory.eval_mcft(centity1fid)
            #     idealframe = Frame()
            #     PF = centity1.parent.frame.homogeneous_mat()
            #     F2 = f2.homogeneous_mat()
            #     AF = calignment_frame.homogeneous_mat()
            #     TF = tframe1.homogeneous_mat()
            #     idealframe.set_from_mat(np.linalg.inv(PF) @ F2 @ np.linalg.inv(AF) @ np.linalg.inv(TF))
            #     print(idealframe)
            return np.dot(diffvec, diffvec)

        # ineq, nonnegative
        def compute_multiconn_constraint(x, mcf1ids, mcf2ids, tdists, dists):
            # this assumes that the assembly connection order matters (from left to right)
            assert len(mcf1ids) == len(mcf2ids) and len(mcf1ids) == len(tdists) and len(mcf1ids) == len(dists)
            # for i in range(len(mcf1ids)):
            #     print(abs(x[mcf2ids[i]] - x[mcf1ids[i]]))
            #     print(tdists[i])
            #     print(dists[i])
            #     print((abs(x[mcf2ids[i]] - x[mcf1ids[i]]) - tdists[i]) * dists[i])
            return sum([(abs(x[mcf2ids[i]] - x[mcf1ids[i]]) - tdists[i]) * dists[i] for i in range(len(mcf1ids))])

        self.prepare_params(subgraph=subgraph)
        self.constraint_list = []
        # process connect() constraints
        if subgraph is None:
            for c in self.connects:
                self.constraint_list.append(
                    {'type':'eq', 'fun': compute_connect_constraint_params, 'args': (c, False)})
        # handle the constraints between parameters of the same mcf when there are multiple connections
        primitive_graph_to_use = self.primitive_graph if subgraph is None else subgraph
        for node_id, entity in primitive_graph_to_use.nodes(data="primitive"):
            entity_connected = primitive_graph_to_use.nodes[node_id]["connected"]
            if entity_connected and len(entity.mcf_factory.MCFParams) > 1:
                entity_cd = entity.critical_dim()
                connector_cds = {}
                # rationale: if the primitives on the same part connects to another part, the multiconn constraint
                # is automatically satisfied because we work with rigid parts
                connector_neighbor_parts = set()
                for p1, p2, data in primitive_graph_to_use.edges(data=True):
                    if data["is_part_edge"]:
                        continue
                    if p1 == node_id:
                        neighbor_entity = primitive_graph_to_use.nodes[p2]["primitive"]
                        if neighbor_entity.parent.id not in connector_neighbor_parts:
                            connector_neighbor_parts.add(neighbor_entity.parent.id)
                            connector_cds[data["conn_info"][1]] = neighbor_entity.connector_critical_dim()
                    elif p2 == node_id:
                        neighbor_entity = primitive_graph_to_use.nodes[p1]["primitive"]
                        if neighbor_entity.parent.id not in connector_neighbor_parts:
                            connector_neighbor_parts.add(neighbor_entity.parent.id)
                            connector_cds[data["conn_info"][3]] = neighbor_entity.connector_critical_dim()
                if len(connector_cds) > 1:
                    fids = list(connector_cds.keys())
                    for i in range(len(fids) - 1):
                        for j in range(i + 1, len(fids)):
                            fid = fids[i]
                            next_fid = fids[j]
                            slider_params = entity.mcf_factory.MCFParams[fid].critical_dim_slider_parameters()
                            next_slider_params = entity.mcf_factory.MCFParams[next_fid].critical_dim_slider_parameters()
                            assert len(slider_params) == len(next_slider_params)
                            cd = connector_cds[fid]
                            next_cd = connector_cds[next_fid]
                            mcf1ids, mcf2ids, tdists, entity_cds = [], [], [], []
                            for i, p in enumerate(slider_params):
                                mcf1ids.append(p.index)
                                pnext = next_slider_params[i]
                                mcf2ids.append(pnext.index)
                                if len(slider_params) > 1:
                                    tdist = (cd[i] + next_cd[i]) / entity_cd[i]
                                    tdists.append(tdist)
                                    entity_cds.append(entity_cd[i])
                                else:
                                    tdist = (cd + next_cd) / entity_cd
                                    tdists.append(tdist)
                                    entity_cds.append(entity_cd)
                            if len(mcf1ids) > 0:
                                self.constraint_list.append(
                                    {'type':'ineq', 'fun': compute_multiconn_constraint, 'args': (mcf1ids, mcf2ids, tdists, entity_cds)})
        self.solve_prepared = True

    def solve_with_static_solve(self, options):
        # return 1e6
        start_time = time.time()
        self.prepare_for_minimize()
        message = ""

        xbounds = [(0,0)] * len(self.initial_guess_params)
        for p in self.initial_guess_params:
            xbounds[p.index] = p.bounds()

        initial_guess_list = self.generate_initial_guess_list(options)
        overall_min_obj = float('inf')
        overall_min_obj_x = None
        # nbatch = 5
        for initial_guess in initial_guess_list:
            # for i in range(nbatch):
            static_solve_objective, static_solve_x, message = self.static_solve(initial_guess, options.maxsiters, no_objective=True)
            if static_solve_objective != float('inf'):
                min_objective = static_solve_objective
                min_objective_x = static_solve_x
            else:
                continue
            if min_objective <= overall_min_obj and (self.constraints_val is None or self.constraints_val < self.constraints_tol):
                overall_min_obj = min_objective
                overall_min_obj_x = min_objective_x
                initial_guess = min_objective_x
        if overall_min_obj_x is None:
            print("No valid continuous parameters")
        else:
            self.end_objective_with_constraints_grav_potential_x(overall_min_obj_x)
            print("\nbest x:", overall_min_obj_x)
            print("best user obj:", self.user_objective_val)
        print("time elapsed:", time.time() - start_time)
        return (self.user_objective_val, message)

    def set_initial_guess_for_multichains(self, options=SolverOptions()):
        start_env_primitives = self.part_graph.nodes(data="part")[self.starts[0]].children
        se_p_ids = [p.id for p in start_env_primitives]
        subgraph_nodes_list = []
        part_nodes_list = []
        connect_edge_list = []
        primitive_double_edge_list = []
        # first find the chains from start to end
        for se_p in start_env_primitives:
            for ee_p in self.ends:
                # print(se_p, ee_p)
                for path in networkx.all_simple_paths(self.primitive_graph_for_loops, source=se_p.id, target=ee_p.id):
                    should_skip = False
                    for i in range(len(path) - 1):
                        if self.primitive_graph.edges[path[i], path[i + 1]]["is_part_edge"]\
                            and path[i] in se_p_ids and path[i + 1] in se_p_ids:
                            should_skip = True
                            break
                    if not should_skip:
                        # print(path)
                        # for p in path:
                        #     print(self.primitive_graph.nodes(data="primitive")[p])
                        subgraph_nodes_list.append(path)
                        part_nodes_list.append(list(set([self.primitive_graph.nodes(data="primitive")[p].parent.id for p in path])))
                        part1 = self.primitive_graph.nodes(data="primitive")[path[-2]].parent.id
                        part2 = self.primitive_graph.nodes(data="primitive")[path[-1]].parent.id
                        primitive_double_edge_list.append((path[-2], path[-1]))
                        # print("primitive double edge", path[-2], path[-1])
                        connect_edge_list.append([part1, part2])
                        # print("connect_edge:", part1, part2)
        # save the original "is_connect_edge"
        is_connect_edge_save = {}
        for part1, part2 in connect_edge_list:
            is_connect_edge_save[part1, part2] = self.part_graph.edges[part1, part2]["is_connect_edge"]
        # then optimize each chain separately
        primitive_graph_save = self.primitive_graph.copy()
        for ni, subgraph_nodes in enumerate(subgraph_nodes_list):
            print(f"solving for subgraph {ni}")
            start_time = time.time()
            subgraph = self.primitive_graph.subgraph(subgraph_nodes)
            def filter_edge_ni(n1, n2):
                if (n2, n1) == primitive_double_edge_list[ni]:
                    return False
                return True
            subgraph = networkx.subgraph_view(subgraph, filter_edge=filter_edge_ni)
            self.primitive_graph = networkx.subgraph_view(primitive_graph_save, filter_edge=filter_edge_ni)
            subpartgraph = self.part_graph.subgraph(part_nodes_list[ni])
            for ei in range(len(connect_edge_list)):
                part1, part2 = connect_edge_list[ei]
                new_connect_edge = False if ei == ni else True # make sure end is considered "attached" to the current chain
                if ei == ni:
                    subpartgraph.edges[part1, part2]["is_connect_edge"] = new_connect_edge
                self.part_graph.edges[part1, part2]["is_connect_edge"] = new_connect_edge

            self.solve_prepared = False
            self.prepare_for_minimize(subgraph)
            xbounds = [(0,0)] * len(self.initial_guess_params)
            for p in self.initial_guess_params:
                xbounds[p.index] = p.bounds()

            initial_guess_list = self.generate_initial_guess_list(options)
            overall_min_obj = float('inf')
            overall_min_obj_x = None
            message = None
            for initial_guess in initial_guess_list:
                static_solve_objective, static_solve_x, message = self.static_solve(initial_guess, options.maxsiters, no_objective=True, subpartgraph=subpartgraph, subprimgraph=subgraph)
                print("static solve:", static_solve_objective)
                min_objective, min_objective_x = self.solve_with_initial_guess(static_solve_x, xbounds, options.maxiters)
                if min_objective <= overall_min_obj:
                    overall_min_obj = min_objective
                    overall_min_obj_x = min_objective_x
                    print("min solve obj:", min_objective)
                    if min_objective < 1e-6:
                        break
            if overall_min_obj_x is None:
                if message is None:
                    message = "No valid continuous parameters"
                else:
                    if type(message) != str:
                        message = f"Optimization Terminated: Connector Parameter {self.initial_guess_params[message[1]]} Fell Off!"
                    else:
                        message = "No valid continuous parameters; " + message
            else:
                self.end_objective_with_constraints_grav_potential_x(overall_min_obj_x)
                print("\nbest x:", overall_min_obj_x)
                print("best user obj:", self.user_objective_val)
                # self.print_params(reset=False)
                # self.bring_up_ui(screenshot=True)
            print("time elapsed:", time.time() - start_time)
            print(message)
        # write back the original "is_connect_edge"
        for (part1, part2), saved_boolean in is_connect_edge_save.items():
            self.part_graph.edges[part1, part2]["is_connect_edge"] = saved_boolean
        self.primitive_graph = primitive_graph_save
        self.forward_propagate_frames(should_redistribute=False)
        self.solve_prepared = False
        print("multichain end")

    def set_saved_initial_guess(self, options=SolverOptions()):
        start_time = time.time()
        self.prepare_for_minimize()
        message = ""

        xbounds = [(0,0)] * len(self.initial_guess_params)
        for p in self.initial_guess_params:
            xbounds[p.index] = p.bounds()

        initial_guess_list = self.generate_initial_guess_list(options)
        message = None
        for initial_guess in initial_guess_list:
            print(len(initial_guess))
            initial_guess = set_manual_initial_guess(initial_guess)
            self.end_objective_with_constraints_grav_potential_x(initial_guess)

        print("time elapsed:", time.time() - start_time)
        return (self.user_objective_val, message)

    def solve(self, options, include_physics=True):
        start_time = time.time()
        each_solve_times = []
        self.prepare_for_minimize()
        message = ""
        solve_success = False

        xbounds = [(0,0)] * len(self.initial_guess_params)
        for p in self.initial_guess_params:
            xbounds[p.index] = p.bounds()
        # To get flags for each parameter:
        # flags = [p.flag for p in self.initial_guess_params] # one of ParameterFlag values

        initial_guess_list = self.generate_initial_guess_list(options)
        overall_min_obj = float('inf')
        overall_min_obj_x = None
        message = None
        for initial_guess in initial_guess_list:
            min_objective, min_objective_x = self.solve_with_initial_guess(initial_guess, xbounds, options.maxiters)
            if include_physics:
                if len(self.constraint_list) > 0 and self.constraints_val < self.constraints_tol or len(self.constraint_list) == 0:
                    min_objective, min_objective_x, message = self.static_solve(min_objective_x, options.maxsiters, no_objective=True)
            if len(each_solve_times) == 0:
                each_solve_times.append(time.time() - start_time)
            else:
                each_solve_times.append(time.time() - start_time - sum(each_solve_times))
            if min_objective <= overall_min_obj and (self.constraints_val is None or self.constraints_val < self.constraints_tol):
                overall_min_obj = min_objective
                overall_min_obj_x = min_objective_x
                print("updated overall_min", min_objective)
                if min_objective < 1e-6 or "Optimization Succeeded" in message:
                    solve_success = True
                    break
        if overall_min_obj_x is None:
            print("No valid continuous parameters")
            solve_success = False
            if message is None:
                message = "No valid continuous parameters"
            else:
                if type(message) != str:
                    message = f"Optimization Terminated: Connector Parameter {self.initial_guess_params[message[1]]} Fell Off!"
                else:
                    message = "No valid continuous parameters; " + message
        else:
            self.end_objective_with_constraints_grav_potential_x(overall_min_obj_x)
            print("\nbest x:", overall_min_obj_x)
            print("best user obj:", self.user_objective_val)
            solve_success = True
        total_time = time.time() - start_time
        print("total time elapsed:", total_time)
        print("initial_guesses tried:", len(each_solve_times))
        print("time per initial guess solve:", each_solve_times)
        return (self.user_objective_val, message)
        # comment above and uncomment below for test_solver.py
        # return total_time, each_solve_times, solve_success

    def solve_with_flips(self, options):
        # return 1e6
        start_time = time.time()
        self.prepare_for_minimize()
        message = ""

        xbounds = [(0,0)] * len(self.initial_guess_params)
        for p in self.initial_guess_params:
            xbounds[p.index] = p.bounds()
        # To get flags for each parameter:
        # flags = [p.flag for p in self.initial_guess_params] # one of ParameterFlag values

        initial_guess_list = []
        if self.con_check_initial_guess is not None and len(self.con_check_initial_guess) > 0:
            initial_guess_list = [self.con_check_initial_guess]
        else:
            initial_guess = np.zeros(len(self.initial_guess_params))
            for p in self.initial_guess_params:
                initial_guess[p.index] = p.value
            initial_guess_list.append(initial_guess)

        # Generate alignment flips combinations
        axes_options = []
        axes = {}
        id_map = {}
        for e1, e2, data in self.primitive_graph.edges.data():
            # if data["is_part_edge"]:
            #     continue
            if self.primitive_graph.has_edge(e2, e1):
                continue
            if "conn_info" in data:
                alignment_frame = data["conn_info"][4]
                axes[e1, e2] = [[]] + alignment_frame.axes
                axes_options.append(axes[e1, e2])
                id_map[e1, e2] = len(axes_options) - 1
        alignment_flips = itertools.product(*axes_options)

        overall_min_obj = float('inf')
        overall_min_obj_x = None
        curr_min_obj = float('inf')
        best_flip_assignment = None
        message = None
        for initial_guess in initial_guess_list:
            for alignment_flip in alignment_flips:
                print(alignment_flip)
                # flip based on current flip
                for e1, e2, data in self.primitive_graph.edges.data():
                    if (e1, e2) in id_map:
                        alignment_frame = data["conn_info"][4]
                        alignment_frame.flip_axis(alignment_flip[id_map[e1, e2]])
                if options.use_two_step: # Note: use_two_step is static solve here
                    min_objective, min_objective_x, message = self.static_solve(initial_guess, options.maxsiters)
                    curr_min_obj = min(curr_min_obj, min_objective)
                else:
                    if options.parameter_sweep:
                        min_objective, min_objective_x = self.solve_constraints_only_with_initial_guess(initial_guess, xbounds, options.maxiters)
                    else:
                        min_objective, min_objective_x = self.solve_with_initial_guess(initial_guess, xbounds, options.maxiters)
                    if min_objective < curr_min_obj:
                        curr_min_obj = min_objective
                        overall_min_obj = min_objective
                        overall_min_obj_x = min_objective_x
                        best_flip_assignment = alignment_flip
                        # static_solve_objective, static_solve_x, message = self.static_solve(min_objective_x, options.maxsiters, no_objective=True)
                        # if static_solve_objective != float('inf'):
                        #     min_objective = static_solve_objective
                        #     min_objective_x = static_solve_x
                        # else:
                        #     continue
                print(curr_min_obj)
                if min_objective <= overall_min_obj and (self.constraints_val is None or self.constraints_val < self.constraints_tol):
                    overall_min_obj = min_objective
                    overall_min_obj_x = min_objective_x
                    print(min_objective)
                    if (not options.parameter_sweep) and min_objective < 1e-6:
                        break
                # flip everything back
                for e1, e2, data in self.primitive_graph.edges.data():
                    if (e1, e2) in id_map:
                        alignment_frame = data["conn_info"][4]
                        alignment_frame.flip_axis(alignment_flip[id_map[e1, e2]])
        if overall_min_obj_x is None:
            if message is None:
                message = "No valid continuous parameters"
            else:
                if type(message) != str:
                    message = f"Optimization Terminated: Connector Parameter {self.initial_guess_params[message[1]]} Fell Off!"
                else:
                    message = "No valid continuous parameters; " + message
        else:
            for e1, e2, data in self.primitive_graph.edges.data():
                if (e1, e2) in id_map:
                    entity1, entity1fid, entity2, entity2fid, alignment_frame = data["conn_info"]
                    alignment_frame.flip_axis(best_flip_assignment[id_map[e1, e2]])
                    if len(best_flip_assignment[id_map[e1, e2]]) > 0:
                        print("flipping", entity1, entity1fid, entity2, entity2fid)
            print("\nbest x:", overall_min_obj_x)
            if options.parameter_sweep:
                print("found a best flip alignment assignment")
                self.constraints_penalty_only(overall_min_obj_x)
                print("best constraint penalty:", self.constraints_val)
            if not options.parameter_sweep:
                self.end_objective_with_constraints_grav_potential_x(overall_min_obj_x)
                print("best user obj:", self.user_objective_val)
        print("time elapsed:", time.time() - start_time)
        return (self.constraints_val if options.parameter_sweep else self.user_objective_val, message)
