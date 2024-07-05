from numpy import *
from ..primitives import *
from ..primitives.primitive import Primitive
from ..primitives.point import Point
from ..parts.part import Part
from .frame import Frame, AlignmentFrame
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import copy
import networkx
import polyscope as ps
from polyscope import imgui as psim
import ete3

class Assembly:
    # alignment of frames between primitives
    connection_frame_alignment = {
        'Rod_Hook':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Rod_HookFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Hook_Rod':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'HookFlex_Rod':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Rod_Ring':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Rod_RingFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Ring_Rod':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'RingFlex_Rod':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Rod_RingCtl':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'RingCtl_Rod':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        # 'Rod_Rod':AlignmentFrame(rot=[0,90,0]),
        'Rod_Tube':AlignmentFrame(rot=[0,90,0]),
        'Tube_Rod':AlignmentFrame(rot=[0,90,0]),
        'Rod_TubeCtl':AlignmentFrame(rot=[0,0,0]),
        'TubeCtl_Rod':AlignmentFrame(rot=[0,0,0]),
        'RodCtl_Hook':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'RodCtl_HookFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Hook_RodCtl':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'HookFlex_RodCtl':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'RodCtl_RingCtl':AlignmentFrame(rot=[0,0,90], axes=[[2]]),
        'RingCtl_RodCtl':AlignmentFrame(rot=[0,0,90], axes=[[2]]),
        'RodCtl_Ring':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'RodCtl_RingFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Ring_RodCtl':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'RingFlex_RodCtl':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        # 'Tube_Tube':AlignmentFrame(rot=[0,90,0]),
        'Tube_Hook':AlignmentFrame(rot=[0,0,90], axes=[[2]]),
        'Tube_HookFlex':AlignmentFrame(rot=[0,0,90], axes=[[2]]),
        'Hook_Tube':AlignmentFrame(rot=[0,0,90], axes=[[2]]),
        'HookFlex_Tube':AlignmentFrame(rot=[0,0,90], axes=[[2]]),
        'Tube_Ring':AlignmentFrame(rot=[0,0,90], axes=[[2]]),
        'Tube_RingFlex':AlignmentFrame(rot=[0,0,90], axes=[[2]]),
        'Ring_Tube':AlignmentFrame(rot=[0,0,90], axes=[[2]]),
        'RingFlex_Tube':AlignmentFrame(rot=[0,0,90], axes=[[2]]),
        # 'Tube_TubeCtl':AlignmentFrame(rot=[180,0,0]),
        # 'TubeCtl_Tube':AlignmentFrame(rot=[180,0,0]),
        'Hook_Hook':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Hook_Ring':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Hook_RingFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Ring_Hook':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'RingFlex_Hook':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Hook_RingCtl':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'HookFlex_RingCtl':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'RingCtl_Hook':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'RingCtl_HookFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Hook_Edge':AlignmentFrame(rot=[90,0,-90], axes=[[2]]),
        'HookFlex_Edge':AlignmentFrame(rot=[90,0,-90], axes=[[2]]),
        'Edge_Hook':AlignmentFrame(rot=[-90,0,90], axes=[[2]]),
        'Edge_HookFlex':AlignmentFrame(rot=[-90,0,90], axes=[[2]]),
        'Hook_HookFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'HookFlex_Hook':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'HookFlex_HookFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'HookFlex_Ring':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'HookFlex_RingFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Ring_HookFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'RingFlex_HookFlex':AlignmentFrame(rot=[180,0,90], axes=[[2]]),
        'Edge_Clip':AlignmentFrame(rot=[0,-90,-90], axes=[[1,2]]),
        'Clip_Edge':AlignmentFrame(rot=[0,90,90], axes=[[1,2]]),
        'Clip_Rod':AlignmentFrame(rot=[0,180,0], axes=[[1]]),
        'Rod_Clip':AlignmentFrame(rot=[0,180,0], axes=[[1]]),
        'Clip_RodCtl':AlignmentFrame(rot=[0,180,0], axes=[[1]]),
        'RodCtl_Clip':AlignmentFrame(rot=[0,180,0], axes=[[1]]),
        'Surface_Surface':AlignmentFrame(rot=[0,0,0], axes=[[0]]),
    }
    # rotational symmetry of each primitive
    primitive_symmetry = {
        'Clip':[360., 360., 180.],
        'Edge':[360., 180., 360.],
        'Ring':[0., 180., 360.],
        'Hook':[360., 360., 360.],
        'Rod':[360., 180., 0.],
        'Surface':[360., 360., 180.],
        'Tube':[360., 0., 360.],
        'Hemisphere':[360., 360., 0.]
    }
    # whether to enable printing for minimize() calls
    verbose = False

    def __init__(self):
        self.tol = 1e-3 # a hack for simulation
        self.starts = []
        self.ends = []
        self.end_frames = []
        self.part_graph = networkx.DiGraph()
        self.primitive_graph = networkx.Graph()
        self.constraints = []

    def __str__(self):
        root = 0
        subtrees = {id:ete3.Tree(name=str(part)) for id, part in self.part_graph.nodes(data="part")}
        [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), self.part_graph.edges())]
        tree = subtrees[root]
        return tree.get_ascii()

    def visualize(self):
        # visualize the parts
        part_nodes = zeros((Part.pid, 3))
        part_edges = array([list(e) for e in self.part_graph.edges]).astype(int)
        for id, part in self.part_graph.nodes(data="part"):
            part_nodes[id] = part.frame.pos
            V, F = part.get_transformed_VF()
            ps_mesh = ps.register_surface_mesh("Part{}-{}".format(id, part.__class__.__name__), V, F)
            ps_mesh.set_transparency(0.4)
        if part_edges.size > 0:
            part_net = ps.register_curve_network("part network", part_nodes, part_edges, enabled=False)
            part_net.set_transparency(0.8)

            xvec, yvec, zvec = zeros((Part.pid, 3)), zeros((Part.pid, 3)), zeros((Part.pid, 3))
            for id, part in self.part_graph.nodes(data="part"):
                rot_mat = Rotation.from_euler('xyz', part.frame.rot, degrees=True).as_matrix()
                xvec[id] = rot_mat[0]
                yvec[id] = rot_mat[1]
                zvec[id] = rot_mat[2]
            part_net.add_vector_quantity("part frame x", xvec, enabled=True, length=0.05, color=(0.5, 0.2, 0.5))
            part_net.add_vector_quantity("part frame y", yvec, enabled=True, length=0.05, color=(0.8, 0.6, 0.6))
            part_net.add_vector_quantity("part frame z", zvec, enabled=True, length=0.05, color=(0.2, 0.2, 0.5))

        # visualize the primitives
        primitive_nodes = zeros((Primitive.pid, 3))
        primitive_edges = array([list(e) for e in self.primitive_graph.edges]).astype(int)
        for id, primitive in self.primitive_graph.nodes(data="primitive"):
            primitive_nodes[id] = primitive.get_global_frame().pos
            V, F = primitive.get_transformed_VF()
            ps_mesh = ps.register_surface_mesh("Primitive{}-{}".format(id, primitive.__class__.__name__), V, F)
            ps_mesh.set_transparency(0.5)
        if primitive_edges.size > 0:
            primitive_net = ps.register_curve_network("primitive network", primitive_nodes, primitive_edges, enabled=False)
            primitive_net.set_transparency(0.8)

            pxvec, pyvec, pzvec = zeros((Primitive.pid, 3)), zeros((Primitive.pid, 3)), zeros((Primitive.pid, 3))
            for id, primitive in self.primitive_graph.nodes(data="primitive"):
                rot_mat = Rotation.from_euler('xyz', primitive.get_global_frame().rot, degrees=True).as_matrix()
                pxvec[id] = rot_mat[0]
                pyvec[id] = rot_mat[1]
                pzvec[id] = rot_mat[2]
            primitive_net.add_vector_quantity("primitive frame x", pxvec, enabled=True, length=0.05, color=(0.2, 0.5, 0.5))
            primitive_net.add_vector_quantity("primitive frame y", pyvec, enabled=True, length=0.05, color=(0.8, 0.8, 0.6))
            primitive_net.add_vector_quantity("primitive frame z", pzvec, enabled=True, length=0.05, color=(0.2, 0.5, 0.2))

    def polyscope_callback(self):
        # print(psim.GetMousePos())
        psim.PushItemWidth(150)
        psim.TextUnformatted("Connector Frame Parameters")
        psim.Separator()
        should_recompute = False
        for c in self.constraints:
            if len(c) != 4:
                continue
            entity1, entity1fid, entity2, entity2fid = c
            conn_params1 = entity1.mcf_factory.MCFParams[entity1fid].slider_parameters()
            conn_params2 = entity2.mcf_factory.MCFParams[entity2fid].slider_parameters()
            psim.PushItemWidth(200)
            values1 = []
            for p in conn_params1:
                changed, p.value = psim.SliderFloat(
                    p.name, p.value, v_min=p.min_value, v_max=p.max_value)
                values1.append(p.value)
                if changed: should_recompute = True
            values2 = []
            for p in conn_params2:
                changed, p.value = psim.SliderFloat(
                    p.name, p.value, v_min=p.min_value, v_max=p.max_value)
                values2.append(p.value)
                if changed: should_recompute = True
            psim.Separator()
            psim.PopItemWidth()
        psim.PopItemWidth()
        if should_recompute:
            entity1.mcf_factory.MCFParams[entity1fid].set_slider_parameters(values1)
            entity2.mcf_factory.MCFParams[entity2fid].set_slider_parameters(values2)
            self.compute_initial_guess_frames(should_redistribute=False)
            ps.remove_all_structures()
            self.visualize()

    def print_frames(self):
        for _, part in self.part_graph.nodes(data="part"):
            print("part:", part)
            print("part frame:", part.frame)

    def connectors(self):
        conns = []
        for _, primitive in self.primitive_graph.nodes(data="primitive"):
            if not primitive.connected:
                conns.append(primitive)
            elif not primitive.single_conn:
                conns.append(primitive)
        return conns

    def check_if_in_part_graph(self, part_to_check):
        if not self.part_graph.has_node(part_to_check.id):
            self.part_graph.add_node(part_to_check.id, part=part_to_check, pidx=self.part_graph.number_of_nodes())
            for i in range(len(part_to_check.children)):
                child_i = part_to_check.children[i]
                if not self.primitive_graph.has_node(child_i.id):
                    self.primitive_graph.add_node(child_i.id, primitive=child_i)
            if len(part_to_check.children) > 1:
                for i in range(len(part_to_check.children) - 1):
                    child_i = part_to_check.children[i]
                    for j in range(i + 1, len(part_to_check.children)):
                        child_j = part_to_check.children[j]
                        self.primitive_graph.add_edge(child_i.id, child_j.id)

    def start_with(self, start_entity, frame):
        self.starts.append(start_entity)
        self.check_if_in_part_graph(self.starts[-1].parent)
        self.starts[-1].parent.frame = frame
        self.starts[-1].parent.is_fixed = True
        self.starts[-1].parent.mass = 0.
        self.constraints.append([self.starts[-1], "start"])
    
    def end_with(self, end_entity, frame=None, point=None):
        self.ends.append(end_entity)
        self.check_if_in_part_graph(self.ends[-1].parent)
        if frame is not None:
            self.ends[-1].parent.frame = copy.deepcopy(frame)
            self.end_frames.append(copy.deepcopy(frame))
        if point is not None:
            self.ends[-1].parent.point = copy.deepcopy(point)
            self.end_frames.append(copy.deepcopy(point))

    def add(self, part):
        self.check_if_in_part_graph(part)

    def connect(self, entity1, entity2, is_fixed_joint=False):
        self.check_if_in_part_graph(entity1.parent)
        self.check_if_in_part_graph(entity2.parent)
        self.part_graph.add_edge(entity1.parent.id, entity2.parent.id, is_fixed_joint=is_fixed_joint)
        gentity1 = self.primitive_graph.nodes(data="primitive")[entity1.id]
        gentity2 = self.primitive_graph.nodes(data="primitive")[entity2.id]
        if gentity1.connected and gentity1.single_conn or gentity2.connected and gentity2.single_conn:
            raise Exception("Cannot connect two single-conn primitives that already have a connection.")
        gentity1mcfid = gentity1.mcf_factory.new_mcf()
        gentity2mcfid = gentity2.mcf_factory.new_mcf()
        self.constraints.append([gentity1, gentity1mcfid, gentity2, gentity2mcfid])
        ctype = gentity1.ctype_str + "_" + gentity2.ctype_str
        alignment_frame = copy.deepcopy(Assembly.connection_frame_alignment.get(ctype, AlignmentFrame()))
        self.primitive_graph.add_edge(gentity1.id, gentity2.id, constraint=self.constraints[-1], alignment_frame=alignment_frame)
        gentity1.connected = True
        gentity2.connected = True
        if gentity1.parent.is_fixed and is_fixed_joint:
            gentity2.parent.is_fixed = True
            gentity2.parent.mass = 0
        elif gentity2.parent.is_fixed and is_fixed_joint:
            gentity1.parent.is_fixed = True
            gentity1.parent.mass = 0
    
    def end_objective(self, x, eindices):
        res = 0.
        for i, eidx in enumerate(eindices):
            idx = eidx * 6
            curr_end_frame = Frame(x[idx:idx+3], x[idx+3:idx+6]).transform_frame(self.ends[i].frame())
            if isinstance(self.end_frames[i], Point):
                if not self.end_frames[i].contains(curr_end_frame.pos):
                    res += 1e6 # add an arbitrarily large penalty
            else:
                end_sym = Assembly.primitive_symmetry[self.ends[i].__class__.__name__]
                diffvec = curr_end_frame.diff_with(self.end_frames[i])
                for i, sym in enumerate(end_sym):
                    if sym == 0:
                        diffvec[i + 3] = 0
                    elif sym > 0 and sym < 360:
                        while diffvec[i + 3] >= sym:
                            diffvec[i + 3] -= sym
                        while diffvec[i + 3] <= -sym:
                            diffvec[i + 3] += sym
                res += dot(diffvec, diffvec)
        return res

    def forward_propagate_frames(self, nparts, should_redistribute=True):
        # use a queue-based impl for the breadth-first part_graph traversal
        propagated_frames = zeros(nparts * 6)
        queue = []
        # enqueue the start parts
        for start in self.starts:
            if start.parent.id not in queue:
                queue.append(start.parent.id)
            sidx = self.part_graph.nodes(data="pidx")[start.parent.id]
            propagated_frames[sidx*6:sidx*6+6] = start.parent.frame.to_numpy_array()
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
                            if self.primitive_graph.has_edge(entity1.id, entity2.id):
                                constraint = self.primitive_graph.edges[entity1.id, entity2.id]["constraint"]
                                assert len(constraint) == 4
                                part1 = entity1.parent
                                part2 = entity2.parent
                                # print("connecting part{}-primitive{} to part{}-primitive{}".format(part1.id, entity1.id, part2.id, entity2.id))
                                # print(part1, entity1)
                                # print(part2, entity2)
                                entity1fid, entity2fid = constraint[1], constraint[3]
                                if entity1 != constraint[0]:
                                    entity1fid, entity2fid = constraint[3], constraint[1]
                                entity1f = entity1.mcf_factory.eval_mcf(entity1fid)
                                entity2f = entity2.mcf_factory.eval_mcf(entity2fid)
                                alignment_frame = self.primitive_graph.edges[entity1.id, entity2.id]["alignment_frame"]
                                if alignment_frame is None:
                                    ctype = entity1.ctype_str + "_" + entity2.ctype_str
                                    alignment_frame = copy.deepcopy(Assembly.connection_frame_alignment.get(ctype, AlignmentFrame()))
                                entity1global = part1.frame.transform_frame(entity1f.transform_frame(alignment_frame))
                                part2_frame_mat = entity1global.homogeneous_mat() @ linalg.inv(entity2f.homogeneous_mat())
                                # print(part2_frame_mat)
                                part2.frame.set_from_mat(part2_frame_mat)
                                p2idx = self.part_graph.nodes(data="pidx")[part2.id]
                                propagated_frames[p2idx * 6:p2idx * 6 + 6] = part2.frame.to_numpy_array()
                # clear out the temporary queue and pcpairs list
                queue.extend(new_queue)
                new_queue.clear()
                pcpairs.clear()

        return propagated_frames

    def compute_initial_guess_frames(self, should_redistribute=True):
        nparts = self.part_graph.number_of_nodes()
        initial_guess = self.forward_propagate_frames(nparts, should_redistribute=should_redistribute)
        eindices = []
        for i in range(len(self.ends)):
            eindices.append(self.part_graph.nodes(data="pidx")[self.ends[i].parent.id])
        return self.end_objective(initial_guess, eindices)

    def solve(self, options):
        # this includes the slider parameters as part of the independent variables to be solved for
        # eq, zero
        def compute_connect_constraint_params(x, c, p1idx, e1f_params_indices, p2idx, e2f_params_indices, alignment_frame):
            p1frame = Frame(x[p1idx * 6:(p1idx * 6 + 3)], x[(p1idx * 6 + 3):(p1idx * 6 + 6)])
            p2frame = Frame(x[p2idx * 6:(p2idx * 6 + 3)], x[(p2idx * 6 + 3):(p2idx * 6 + 6)])
            e1f_params_values = []
            for i1 in e1f_params_indices:
                e1f_params_values.append(x[i1])
            e2f_params_values = []
            for i2 in e2f_params_indices:
                e2f_params_values.append(x[i2])
            entity1, entity1fid, entity2, entity2fid = c
            entity1.mcf_factory.MCFParams[entity1fid].set_slider_parameters(e1f_params_values)
            entity2.mcf_factory.MCFParams[entity2fid].set_slider_parameters(e2f_params_values)
            entity1f = entity1.mcf_factory.eval_mcf(entity1fid)
            entity2f = entity2.mcf_factory.eval_mcf(entity2fid)
            f1 = p1frame.transform_frame(entity1f.transform_frame(alignment_frame))
            f2 = p2frame.transform_frame(entity2f)
            diffvec = f1.diff_with(f2)
            return dot(diffvec, diffvec)

        # this essentially is a vector-valued constraint but in scipy.optimize.minimize
        # there's no way to specify a vector-valued constraint (unlike in nlopt, there's mconstraint)
        # eq, zero
        def compute_connect_constraint(x, entity1f, p1idx, entity2f, p2idx, alignment_frame):
            p1frame = Frame(x[p1idx * 6:(p1idx * 6 + 3)], x[(p1idx * 6 + 3):(p1idx * 6 + 6)])
            p2frame = Frame(x[p2idx * 6:(p2idx * 6 + 3)], x[(p2idx * 6 + 3):(p2idx * 6 + 6)])
            # primitive frame won't change and will only depend on parent frame
            f1 = p1frame.transform_frame(entity1f.transform_frame(alignment_frame))
            f2 = p2frame.transform_frame(entity2f)
            diffvec = f1.diff_with(f2)
            return dot(diffvec, diffvec)

        # eq, zero
        def compute_startend_constraint(x, pidx, fidx, entity):
            f1 = x[pidx * 6:(pidx+1) * 6]
            f2 = entity.parent.frame.to_numpy_array()
            return f1[fidx] - f2[fidx]

        # ineq, nonnegative
        def compute_multiconn_constraint(x, mcf1id, mcf2id, dist):
            return x[mcf2id] - x[mcf1id] - dist

        nparts = self.part_graph.number_of_nodes()
        constraint_list = []
        len_x = nparts * 6
        initial_guess_params = []
        for c in self.constraints:
            # process constraints
            if len(c) == 4:
                entity1, entity1fid, entity2, entity2fid = c
                sel_i1 = e1f_params_indices = entity1.mcf_factory.MCFParams[entity1fid].set_slider_param_indices(len_x + len(initial_guess_params))
                initial_guess_params.extend(entity1.mcf_factory.MCFParams[entity1fid].slider_parameters(sel_i1))
                sel_i2 = e2f_params_indices = entity2.mcf_factory.MCFParams[entity2fid].set_slider_param_indices(len_x + len(initial_guess_params))
                initial_guess_params.extend(entity2.mcf_factory.MCFParams[entity2fid].slider_parameters(sel_i2))
                ctype = entity1.ctype_str + "_" + entity2.ctype_str
                p1idx = self.part_graph.nodes(data="pidx")[entity1.parent.id]
                p2idx = self.part_graph.nodes(data="pidx")[entity2.parent.id]
                alignment_frame = self.primitive_graph.edges[entity1.id, entity2.id]["alignment_frame"]
                if alignment_frame is None:
                    alignment_frame = copy.deepcopy(Assembly.connection_frame_alignment.get(ctype, AlignmentFrame()))
                # old constraint:
                # constraint_list.append(
                #     {'type':'eq', 'fun': compute_connect_constraint,\
                #      'args': (entity1f, p1idx, entity2f, p2idx, alignment_frame)})
                # new constraint with parameters:
                constraint_list.append(
                    {'type':'eq', 'fun': compute_connect_constraint_params,\
                     'args': (c, p1idx, e1f_params_indices, p2idx, e2f_params_indices, alignment_frame)})
            elif len(c) == 2:
                entity, type = c
                assert type == "start"
                pidx = self.part_graph.nodes(data="pidx")[entity.parent.id]
                for i in range(6):
                    constraint_list.append(
                        {'type':'eq', 'fun': compute_startend_constraint, 'args': (pidx, i, entity)})
        # handle the constraints between parameters of the same mcf when there are multiple connections
        conn_indices = [p.index for p in initial_guess_params]
        for _, entity in self.primitive_graph.nodes(data="primitive"):
            if entity.connected and len(entity.mcf_factory.MCFParams) > 1:
                for fid in range(len(entity.mcf_factory.MCFParams) - 1):
                    slider_params = entity.mcf_factory.MCFParams[fid].slider_parameters()
                    next_slider_params = entity.mcf_factory.MCFParams[fid + 1].slider_parameters()
                    assert len(slider_params) == len(next_slider_params)
                    for i, p in enumerate(slider_params):
                        mcf1id = p.index
                        pnext = next_slider_params[i]
                        mcf2id = pnext.index
                        if mcf1id == -1 or mcf2id == -1 or mcf1id == mcf2id \
                            or p.varname() != pnext.varname() \
                            or mcf1id not in conn_indices or mcf2id not in conn_indices:
                            continue
                        # for now just use a fixed value and try to implement the case-by-case dist later
                        dist = (p.max_value - p.min_value) * 0.05
                        constraint_list.append(
                            {'type':'ineq', 'fun': compute_multiconn_constraint, 'args': (mcf1id, mcf2id, dist)})

        initial_guess = zeros(len_x + len(initial_guess_params))
        xbounds = []
        initial_guess_frames = self.forward_propagate_frames(nparts)
        initial_guess[:len_x] = initial_guess_frames
        for i in range(nparts):
            for _ in range(3):
                xbounds.append((None, None))
            for _ in range(3):
                xbounds.append((-360, 360))
        for idx in range(len(initial_guess_params)):
            initial_guess[idx + len_x] = initial_guess_params[idx].value
            xbounds.append((initial_guess_params[idx].min_value, initial_guess_params[idx].max_value))

        # show_options(method="SLSQP")
        batchiters = 100
        nbatch = int(options.maxiters / batchiters)
        min_objective = float('inf')
        min_objective_x = None
        eindices = []
        for i in range(len(self.ends)):
            eindices.append(self.part_graph.nodes(data="pidx")[self.ends[i].parent.id])
        for i in range(nbatch):
            opt_options = {'disp':Assembly.verbose,'maxiter':batchiters,'ftol':1e-6}
            res = minimize(
                self.end_objective,
                args=(eindices,),
                method="SLSQP",
                x0=initial_guess,
                constraints=constraint_list,
                bounds=xbounds,
                options=opt_options)
            # msg = res.message
            # print(res.status, msg)
            if min_objective > res.fun:
                min_objective = res.fun
                min_objective_x = res.x
            initial_guess = min_objective_x
        # print(min_objective_x)
        for p in initial_guess_params:
            p.value = min_objective_x[p.index]
        for part_id, part in self.part_graph.nodes(data="part"):
            pidx = self.part_graph.nodes(data="pidx")[part_id]
            part.frame = Frame(min_objective_x[pidx * 6:pidx * 6 + 3], min_objective_x[(pidx * 6 + 3):(pidx * 6 + 6)])
        return min_objective
