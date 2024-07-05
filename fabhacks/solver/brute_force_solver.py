import copy
import networkx
from ..assembly.assembly import Assembly
from ..assembly.frame import Frame
from ..primitives.rod import Rod
from ..primitives.ring import Ring
from ..primitives.primitive import Primitive
from ..primitives.dimensions import *
from ..parts.env import Environment
from ..parts.library import *
import polyscope as ps
from time import time
import argparse
import os
import datetime

class SolverOptions:
    def __init__(self, assembly_name="ASSEMBLY", output_dir=".", use_initial_guess=False,
                 use_two_step=False, parameter_sweep=False, ninitialization=None, random_init=False, timeout=float('inf'),
                 max_depth=10, max_parts=7, max_connections=2, maxiters=1000, maxsiters=1000, use_sol_hash=True,
                 prune_search=True, heuristic_search=True, learn_heuristic=True, early_return_threshold=1,
                 heuristic_threshold=3, heuristic_epochs=30, heuristic_lr=1e-3, log_root=None, exp_name=None,
                 no_solve=False, goal_hash=None, precomputed_loops_file_prefix=None,
                 epsilon_greedy=False, search_epsilon=0.2):
        self.assembly_name = assembly_name
        self.output_dir = output_dir
        self.use_initial_guess = use_initial_guess
        self.use_two_step = use_two_step
        self.parameter_sweep = parameter_sweep
        self.ninitialization = ninitialization if self.parameter_sweep else 1
        self.random_init = random_init
        self.timeout = timeout
        self.max_depth = max_depth
        self.max_parts = max_parts
        self.max_connections = max_connections
        self.maxiters = maxiters
        self.maxsiters = maxsiters
        self.early_return_threshold = early_return_threshold

        self.use_sol_hash = use_sol_hash
        self.prune_search = prune_search
        self.heuristic_search = heuristic_search
        self.learn_heuristic = learn_heuristic
        
        self.heuristic_threshold = heuristic_threshold # How many solutions needed to switch to learned heuristic
        self.heuristic_epochs = heuristic_epochs
        self.heuristic_lr = heuristic_lr

        self.no_solve = no_solve
        self.goal_hash = goal_hash
        self.precomputed_loops_file_prefix = precomputed_loops_file_prefix
        self.epsilon_greedy = epsilon_greedy
        self.search_epsilon = search_epsilon

        if log_root is None:
            log_root = os.path.join(os.path.abspath(os.getcwd()), 'logs')

        if exp_name is None:
            dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            flags = []
            flags.append(assembly_name)
            if self.use_initial_guess:
                flags.append('ig')
            if self.use_two_step:
                flags.append('2s')
            if not self.parameter_sweep:
                flags.append('nosweep')
            flags.append(f't{timeout}')
            flags.append(f'd{max_depth}')
            if not self.use_sol_hash:
                flags.append('nohash')
            if self.prune_search:
                flags.append('prune')
            if self.heuristic_search:
                flags.append('heuristic')
            if self.learn_heuristic:
                flags.append('learned')
            
            exp_name = '_'.join([dt] + flags)
        
        self.log_root = log_root
        self.exp_name = exp_name

        self.tb_logdir = os.path.join(self.log_root, 'tb_logs', self.exp_name)
        self.log_file_path = os.path.join(self.log_root, f'{self.assembly_name}-{self.exp_name}.pickle')

        # This is useful when debugging, but will mess with parallel runs -
        # we need to make sure we don't re-use names for now, and maybe later-on
        # use another means to check for same experiment name but different
        # example
        """
        num = 0
        while os.path.exists(self.tb_logdir) or os.path.exists(self.log_file_path):
            name = f'{exp_name}_{num}'
            num += 1
            self.tb_logdir = os.path.join(self.log_root, 'tb_logs', name)
            self.log_file_path = os.path.join(self.log_root, f'{name}.pickle')
        """


        

# v prune v #
def loop_in_loops(loop, loops):
    for l in loops:
        if len(loop) == len(l) and loop in l + "-" + l:
            return True
    return False

def one_loop_in_loops(created_loops, loops):
    for loop in created_loops:
        if loop_in_loops(loop, loops):
            return True
    return False

def all_loops_in_loops(created_loops, loops):
    assert len(created_loops) != 0
    for loop in created_loops:
        if not loop_in_loops(loop, loops):
            return False
    return True

class Solution:
    # auxiliary id for __eq__
    sid = 0

    def __init__(self, parts_library, assembly_name="ASSEMBLY"):
        self.id = Solution.sid
        Solution.sid += 1
        self.assembly = Assembly()
        self.parts_lib = parts_library
        self.assembly_name = assembly_name
        self.program = "{} = Assembly()\n".format(self.assembly_name)

    def __eq__(self, other):
        if isinstance(other, Solution):
            return self.id == other.id
        return False

    def add_start(self, start_env, start_frame, child_id):
        start_env.name_in_assembly = "ENV_start"
        self.assembly.start_with(start_env.children[child_id], start_frame[child_id])
        self.program += start_env.primitive_init_program
        self.program += "ENV_start = Environment({})\n".format(start_env.constructor_program)
        self.program += "start_frame = {}\n".format(start_frame[child_id].get_program())
        child_varname = start_env.get_child_varname(child_id)
        self.program += \
            "{}.start_with(ENV_start.{}, start_frame)\n".format(self.assembly_name, child_varname)

    def add_end(self, end_env, end_frame, child_id):
        end_env.name_in_assembly = "ENV_end"
        self.assembly.end_with(end_env.children[child_id], end_frame[child_id])
        self.program += end_env.primitive_init_program
        self.program += "ENV_end = Environment({})\n".format(end_env.constructor_program)
        self.program += "end_frame = {}\n".format(end_frame[child_id].get_program())
        child_varname = end_env.get_child_varname(child_id)
        self.program += \
            "{}.end_with(ENV_end.{}, end_frame)\n".format(self.assembly_name, child_varname)

    def add_part(self, part_name):
        part_to_add = self.parts_lib.use_part(part_name)
        self.assembly.add(part_to_add)
        self.program += "{} = {}()\n".format(part_to_add.name_in_assembly, part_name)
        return part_to_add

    def connect_part(self, a_part, a_child, b_part, b_child):
        # remove the used connection ports
        self.assembly.connect(a_part.children[a_child], b_part.children[b_child])
        a_part_name = a_part.name_in_assembly
        a_child_varname = a_part.get_child_varname(a_child)
        b_part_name = b_part.name_in_assembly
        b_child_varname = b_part.get_child_varname(b_child)
        self.program += \
            "{}.connect({}.{}, {}.{})\n".format(
                self.assembly_name, a_part_name, a_child_varname, b_part_name, b_child_varname)


class BruteForceSolver:
    # brute force search to find all suitable solution (up to a certain depth) as a baseline
    def __init__(self, start_envs, start_frames, end_envs, end_frames, parts_library, options=SolverOptions()):
        self.start_envs = start_envs
        self.start_frames = start_frames
        assert(len(self.start_frames) == len(self.start_envs))
        self.end_envs = end_envs
        self.end_frames = end_frames
        assert(len(self.end_frames) == len(self.end_envs))
        self.parts_library = parts_library
        self.options = options
        self.connectivity_graph = networkx.Graph()
        self.infeasible_loops = set()
        self.feasible_loops = set()
        if options.precomputed_loops_file_prefix is not None:
            self.load_precomputed_loops(options.precomputed_loops_file_prefix)
        self.init_graph()

    def load_precomputed_loops(self, prefix):
        with open(os.path.join(prefix, "infeasible_loops.txt"), "r") as f:
            for line in f:
                self.infeasible_loops.add(line.rstrip())
        with open(os.path.join(prefix, "feasible_loops.txt"), "r") as f:
            for line in f:
                self.feasible_loops.add(line.rstrip())

    def init_graph(self):
        self.connectivity_graph.add_node(0, type="Rod")
        self.connectivity_graph.add_node(1, type="Hook")
        self.connectivity_graph.add_node(2, type="Ring")
        self.connectivity_graph.add_node(3, type="Clip")
        self.connectivity_graph.add_node(4, type="Tube")
        self.connectivity_graph.add_node(5, type="Edge")
        self.connectivity_graph.add_node(6, type="Surface")
        self.connectivity_graph.add_node(7, type="Hemisphere")
        self.connectivity_graph.add_edge(0, 1, dim_constraint=rod_hook_dim_constraint)
        self.connectivity_graph.add_edge(0, 2, dim_constraint=rod_ring_dim_constraint)
        self.connectivity_graph.add_edge(0, 3, dim_constraint=rod_clip_dim_constraint)
        self.connectivity_graph.add_edge(0, 4, dim_constraint=rod_tube_dim_constraint)
        # self.connectivity_graph.add_edge(0, 6, dim_constraint=rod_surface_dim_constraint)
        self.connectivity_graph.add_edge(1, 1, dim_constraint=hook_hook_dim_constraint)
        self.connectivity_graph.add_edge(1, 2, dim_constraint=hook_ring_dim_constraint)
        self.connectivity_graph.add_edge(1, 4, dim_constraint=hook_tube_dim_constraint)
        # self.connectivity_graph.add_edge(1, 5, dim_constraint=hook_edge_dim_constraint)
        self.connectivity_graph.add_edge(2, 4, dim_constraint=ring_tube_dim_constraint)
        # self.connectivity_graph.add_edge(2, 6, dim_constraint=ring_surface_dim_constraint)
        self.connectivity_graph.add_edge(3, 4, dim_constraint=clip_tube_dim_constraint)
        self.connectivity_graph.add_edge(3, 5, dim_constraint=clip_edge_dim_constraint)
        self.connectivity_graph.add_edge(4, 4, dim_constraint=tube_tube_dim_constraint)
        self.connectivity_graph.add_edge(6, 6, dim_constraint=surface_surface_dim_constraint)
        self.connectivity_graph.add_edge(6, 7, dim_constraint=surface_hemisphere_dim_constraint)

    def solve(self):
        # use a queue-based impl for the brute force breadth-first search
        queue = []
        solutions = []
        # enqueue the start part
        for start_env in self.start_envs:
            curr_sol = Solution(self.parts_library)
            curr_sol.add_start(start_env.parent, self.start_frames, start_env.child_id)
        queue.append(curr_sol)

        while len(queue) > 0:
            # pop from queue and explore the children connectors
            curr_sol = queue.pop(0)
            parts = curr_sol.parts_lib
            assembly_connectors = curr_sol.assembly.connectors()
            next_parts = parts.all_parts()
            cannot_connect_further = True

            for part in next_parts:
                part_connectors = part.connectors()
                for part_conn in part_connectors:
                    part_conn_id = part_conn.tid
                    for assembly_conn in assembly_connectors:
                        assembly_conn_id = assembly_conn.tid
                        if self.connectivity_graph.has_edge(part_conn_id, assembly_conn_id):
                            dim_constraint_check = self.connectivity_graph.edges[part_conn_id, assembly_conn_id]["dim_constraint"]
                            if dim_constraint_check is None or dim_constraint_check(part_conn, assembly_conn, assembly=curr_sol.assembly):
                                # enqueue back the new subassembly with each possible connection
                                cannot_connect_further = False
                                new_sol = copy.deepcopy(curr_sol)
                                added_part = new_sol.add_part(part.__class__.__name__)
                                new_sol.connect_part(added_part, part_conn.child_id, assembly_conn.parent, assembly_conn.child_id)
                                queue.append(new_sol)
            if cannot_connect_further:
                solutions.append(curr_sol)
        # print(len(solutions))

        # check each possible solution about whether it connects with the end part
        full_solutions = []
        for sol in solutions:
            if networkx.diameter(sol.assembly.part_graph) > self.options.max_depth:
                continue
            conns = sol.assembly.connectors()
            for conn in conns:
                conn_id = conn.tid
                for end_env in self.end_envs:
                    end_conn_id = end_env.tid
                    if self.connectivity_graph.has_edge(conn_id, end_conn_id):
                        full_sol = copy.deepcopy(sol)
                        full_sol.add_end(end_env.parent, self.end_frames, end_env.child_id)
                        full_sol.connect_part(end_env.parent, end_env.child_id, conn.parent, conn.child_id)
                        full_solutions.append(full_sol)
        # print(len(full_solutions))

        min_objective = float("inf")
        best_solution = None
        for full_sol in full_solutions:
            obj = full_sol.assembly.compute_initial_guess_frames() if self.options.use_initial_guess else full_sol.assembly.solve(self.options)
            if min_objective > obj:
                min_objective = obj
                best_solution = full_sol
        return best_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brute force solver for macgyverism assembly solving.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="whether to enable full printing")
    parser.add_argument("--assembly_name", type=str, default="ASSEMBLY", metavar="N",
                        help="assembly name to store as files")
    parser.add_argument("--use_initial_guess", action="store_true", default=False,
                        help="whether to use frame propagation instead of running the solver")
    parser.add_argument("--maxiters", type=int, default=500, metavar="I",
                        help="max number of iterations for the solver (default: 500)")
    parser.add_argument("--max_depth", type=int, default=5, metavar="D",
                        help="max depth of the assembly explored by the solver (default: 5)")
    args = parser.parse_args()
    Primitive.verbose = args.verbose
    Assembly.verbose = args.verbose
    solver_options = SolverOptions(assembly_name=args.assembly_name, use_initial_guess=args.use_initial_guess, maxiters=args.maxiters, max_depth=args.max_depth)

    rod = Rod({"length": 500, "radius": 10})
    ENV_start = Environment({"rod": rod})
    ring = Ring({"arc_radius": 12, "thickness": 3})
    ENV_end = Environment({"ring": ring})
    start_frame = Frame([0,0,0], [90,0,90])
    end_frame = Frame([0,0,-210], [0,0,0])
    parts = demo_library()
    time_start = time()
    solver = BruteForceSolver([ENV_start], [start_frame], [ENV_end], [end_frame], parts, options=solver_options)
    solution = solver.solve()
    time_end = time()
    print(solution.program)
    print(time_end - time_start)
    solution.assembly.solve(solver_options)
    solution.assembly.visualize()
    # init and show the window
    ps.set_program_name("macgyvering")
    ps.set_print_prefix("MACGYVER")
    ps.set_up_dir("z_up")
    ps.init()
    ps.show()
