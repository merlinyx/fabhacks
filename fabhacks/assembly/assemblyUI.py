from .assembly_R import AssemblyR
from ..parts.env import Environment
from ..primitives.primitive import Primitive
from ..parts.part import Part
from ..primitives.mate_connector_frame import FixedFloatParameter

import polyscope as ps
from polyscope import imgui as psim
import numpy as np
import copy
import networkx

class AssemblyUI(AssemblyR):
    def __init__(self, name="AssemblyUI"):
        super().__init__(name)

    def end_connected(self, pid):
        for s in self.starts:
            if networkx.has_path(self.part_graph, s, pid):
                return True
        return False

    def visualize(self, after_sim_or_reset=False, load_from_file=False):
        if not after_sim_or_reset:
            self.forward_propagate_frames(should_redistribute=False)

        primitive_meshes = []
        # visualize the primitives
        m = self.primitive_graph.order()
        id_to_index = {id:i for i, id in enumerate(self.primitive_graph.nodes())}
        primitive_nodes = np.zeros((m, 3))
        for id, primitive in self.primitive_graph.nodes(data="primitive"):
            primitive_nodes[id_to_index[id]] = primitive.get_global_frame().pos
            V, F = primitive.get_transformed_VF(reload=load_from_file)
            ps_mesh = ps.register_surface_mesh("Primitive{}-{}".format(id, primitive.__class__.__name__), V, F)
            ps_mesh.set_transparency(0.5)
            primitive_meshes.append(ps_mesh)

        # visualize the parts
        n = self.part_graph.order()
        id_to_index = {id:i for i, id in enumerate(self.part_graph.nodes())}
        part_nodes = np.zeros((n, 3))
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
            ps_mesh.set_transparency(0.2)

        return primitive_meshes

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
                self.end_frames.append(copy.deepcopy(frame))
                end_entity.frame = frame
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
                end_env.frame.set_from_mat(frame.homogeneous_mat() @ np.linalg.inv(end_entity.frame().homogeneous_mat()))
                varname = "frame"
                self.end_program += "end_frame = {}\n".format(frame.get_program())
            elif point is not None:
                self.end_frames.append(copy.deepcopy(point))
                varname = "point"
                self.end_program += "end_point = Point({})\n".format(point.get_program())
            else:
                raise Exception("Must specify either a frame or a point, not both or neither.")
            child_varname = end_env.get_child_varname(end_entity.child_id)
            self.end_program += \
                "{}.end_with(ENV_end.{}, {}={})\n".format(self.assembly_name, child_varname, varname, "end_" + varname)
            # end of program update

    def can_attach(self, solver, part, assembly_conn=None):
        assembly_connectors = self.connectors()
        if assembly_conn is not None:
            assembly_connectors = [assembly_conn]
        part_connectors = part.connectors()
        for part_conn in part_connectors:
            part_conn_id = part_conn.tid
            for assembly_conn in assembly_connectors:
                assembly_conn_id = assembly_conn.tid
                if solver.connectivity_graph.has_edge(part_conn_id, assembly_conn_id):
                    dim_constraint_check = solver.connectivity_graph.edges[part_conn_id, assembly_conn_id]["dim_constraint"]
                    if dim_constraint_check is None or dim_constraint_check(part_conn, assembly_conn, assembly=self):
                        return True
        return False

    def polyscope_callback(self, clicked_primitive, options, env_setup=False):
        ninit = 1

        def set_params(entity1, entity1fid, entity2, entity2fid, values1, values2):
            if len(values1) > 0:
                entity1.mcf_factory.MCFParams[entity1fid].set_slider_parameters(values1)
            if len(values2) > 0:
                entity2.mcf_factory.MCFParams[entity2fid].set_slider_parameters(values2)

        def recompute():
            self.compute_initial_guess_frames(should_redistribute=False)
            # if len(self.connects) > 0:
                # self.solve_constraints_only(options)
                # self.solve_prepared = False
                # self.con_check_initial_guess = None
            ps.remove_all_structures()
            self.visualize()
        if env_setup:
            return
        psim.PushItemWidth(150)
        psim.TextUnformatted("Connector Frame Parameters")
        psim.Separator()
        if clicked_primitive is not None:
            psim.TextUnformatted("Selected Primitive: {}".format(clicked_primitive))
        should_recompute = False
        store_c = None
        values1, values2 = [], []
        for e1, e2, data in self.primitive_graph.edges.data():
            if data["is_part_edge"]:
                continue
            if "conn_info" in data:
                entity1, entity1fid, entity2, entity2fid, alignment_frame = data["conn_info"]
                conn_params1 = entity1.mcf_factory.MCFParams[entity1fid].slider_parameters()
                conn_params2 = entity2.mcf_factory.MCFParams[entity2fid].slider_parameters()
                psim.PushItemWidth(120)
                if len(alignment_frame.axes) > 0:
                    psim.TextUnformatted("Conn. alignment - {}".format(alignment_frame))
                    nf = len(alignment_frame.axes)
                    for axis, name in alignment_frame.named_axes.items():
                        if psim.Button("Flip Alignment {} ({} {})".format(name, entity1.id, entity2.id)):
                            alignment_frame.flip_axis(axis)
                            print(entity1, entity2)
                            print(alignment_frame)
                            should_recompute = True
                        nf = nf - 1
                        if nf > 0:
                            psim.SameLine()
                values1 = []
                for p in conn_params1:
                    if isinstance(p, FixedFloatParameter):
                        continue
                    new_name = p.name
                    if clicked_primitive is not None and e1 == clicked_primitive.id:
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
                    if clicked_primitive is not None and e2 == clicked_primitive.id:
                        new_name = "(selected) " + new_name
                    changed, p.value = psim.SliderFloat(
                        new_name, p.value, v_min=p.min_value, v_max=p.max_value)
                    values2.append(p.value)
                    if changed: should_recompute = True
                if should_recompute:
                    store_c = [entity1, entity1fid, entity2, entity2fid, values1, values2]
                psim.Separator()
                psim.PopItemWidth()
        if should_recompute and self.autosolve:
            set_params(*store_c)
            recompute()
        changed, self.autosolve = psim.Checkbox("Enable Autosolve", self.autosolve)
        if psim.Button("Run Static Solve"):
            self.solve_with_static_solve(options)
            ps.remove_all_structures()
            self.visualize()
        if self.all_start_end_connected() and self.all_end_connected():
            psim.SameLine()
            if psim.Button("Run Optimization"):
                if store_c is not None:
                    set_params(*store_c)
                print(self.solve(options))
                ps.remove_all_structures()
                self.visualize()
            changed, ninit = psim.InputFloat("# Initial Guesses", ninit)
            if changed:
                options.ninitialization = int(ninit)
                if options.ninitialization < 1:
                    options.ninitialization = 1
                if options.ninitialization > 1:
                    options.parameter_sweep = True
        if psim.Button("Reset Parameters"):
            for _, entity in self.primitive_graph.nodes(data="primitive"):
                for mcfparam in entity.mcf_factory.MCFParams:
                    mcfparam.reset_slider_param_values()
            self.solve_prepared = False
            self.con_check_initial_guess = None
            self.sigma = AssemblyR.SIGMA_INIT
            self.forward_propagate_frames(should_redistribute=True)
            ps.remove_all_structures()
            self.visualize(after_sim_or_reset=True)
        psim.SameLine()
        if psim.Button("Print Program"):
            print(self.insert_part_construction_program())
        psim.Separator()
        psim.TextUnformatted("User Objective {}".format(self.user_objective_val))
        psim.TextUnformatted("Weighted Constraints Penalty {} * {}".format(self.constraints_val, self.sigma))
        psim.TextUnformatted("Weighted Gravitational Potential {} * {}".format(self.gravitational_val, AssemblyR.G_WEIGHT))
