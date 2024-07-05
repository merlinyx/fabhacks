import OpenGL.GL as gl
import argparse
import copy
import dill as pickle
import os
import polyscope as ps
from PIL import Image
from collections import deque
from polyscope import imgui as psim

from ..assembly.assembly import Assembly
from ..assembly.assemblyUI import AssemblyUI
from ..assembly.frame import Frame
from ..parts.library import *
from ..parts.env import Environment
from ..parts.part import PartType, part_to_pprim
from ..primitives.primitive import Primitive
from ..primitives.edge import Edge
from ..primitives.rod import Rod
from ..primitives.ring import Ring
from ..primitives.hook import Hook
from ..primitives.surface import Surface
from ..primitives.tube import Tube
from ..solver.brute_force_solver import SolverOptions, BruteForceSolver

available_env_setups = ["demo", "toothbrush", "clipcable", "soapbottle", "mughanger", "birdfeeder", 
                        "papertowelroll", "diapercaddy", "studybasket", "readingnook", "cliplight"]

def load_texture_image(image_path):
    image = Image.open(image_path)
    textureData = image.convert("RGBA").tobytes()

    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image.width, image.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, textureData)

    return texture, image.width, image.height

def load_textures():
    part_textures = {} # maps from part.ohe to texture, width, height
    prim_textures = {} # maps from primitive.ohe to texture, width, height
    ppindex = 0
    for part in list(PartType):
        part_name = part.name.lower()
        image_path = os.path.join("fabhacks", "parts", "icons", part_name + ".png")
        if not os.path.exists(image_path):
            continue
        texture, width, height = load_texture_image(image_path)
        part_textures[part.value] = {"texture": texture, "width": width, "height": height}
        pprimitives = part_to_pprim[part]
        for i, pprim in enumerate(pprimitives):
            prim_name = part_name + str(i + 1)
            image_path = os.path.join("fabhacks", "parts", "icons", prim_name + ".png")
            texture, width, height = load_texture_image(image_path)
            prim_textures[pprim.value] = {"texture": texture, "width": width, "height": height}
            ppindex += 1
    return part_textures, prim_textures

def clear_textures(textures):
    for texture_info in textures.values():
        gl.glDeleteTextures(texture_info["texture"])

def env_setup(arglib):
    assembly = AssemblyUI()
    if arglib == "demo":
        rod = Rod({"length": 500, "radius": 10})
        ENV_start = Environment({"rod": rod})
        start_frame = Frame([0,0,400], [90,0,90])
        assembly.add(ENV_start.rod, start_frame)
        ring = CurtainRing()
        ENV_end = Environment({"ring": ring})
        end_frame = Frame([0,2.1,142], [87,0,0])
        assembly.end_with(ENV_end.hook, frame=end_frame)
    elif arglib == "toothbrush":
        surface = Surface({"width": 400, "length": 400})
        ENV_start = Environment({"surface": surface})
        start_frame = Frame()
        assembly.add(ENV_start.surface, start_frame)
        toothbrush = Toothbrush()
        ENV_end = Environment({"toothbrush": toothbrush})
        end_frame = Frame([-10, -62.5, 50], [-65,0,0])
        assembly.end_with(ENV_end.rod, frame=end_frame)
    elif arglib == "clipcable":
        edge = Edge({"width": 100, "length": 200, "height": 1.5})
        ENV_start = Environment({"edge": edge})
        start_frame = Frame([0,0,150],[0,0,0])
        assembly.add(ENV_start.edge, start_frame)
        cable = Cable()
        ENV_end = Environment({"cable": cable})
        end_frame = Frame([0,57.6,163], [0,0,0])
        assembly.end_with(ENV_end, frame=end_frame)
    elif arglib == "soapbottle":
        ENV_start = Environment({"rod": Rod({"length": 500, "radius": 5})})
        start_frame = Frame([0,0,500], [90,0,90])
        assembly.add(ENV_start, start_frame)
        ENV_end = Environment({"soapbottle": SoapBottle()})
        end_frame = Frame([0,0,260], [0,0,180])
        assembly.end_with(ENV_end, end_frame)
    elif arglib == "mughanger":
        ENV_start = Environment({"rod": Rod({"length": 500, "radius": 2})})
        start_frame = Frame([0,0,200], [90,0,90])
        assembly.add(ENV_start, start_frame)
        ENV_end = Environment({"mug": Mug()})
        end_frame = Frame([0,0,50], [-35,0,-90])
        assembly.end_with(ENV_end.hook, frame=end_frame)
    elif arglib == "smallbirdfeeder":
        PART_bird_feeder_env = SmallBirdFeederEnv()
        ENV_start = Environment({"bird_feeder_env": PART_bird_feeder_env})
        assembly.add(ENV_start, Frame())
        PART_bird_feeder = BirdFeeder()
        ENV_end = Environment({"bird_feeder": PART_bird_feeder})
        end_frame = Frame([0,0,-314.2], [0,0,90])
        assembly.end_with(ENV_end.ring, frame=end_frame)
    elif arglib == "papertowelroll":
        ENV_start = Environment({"env": TowelHangingEnv()})
        wall_frame = Frame([0,0,300], [0,0,0])
        assembly.add(ENV_start, wall_frame)
        PART_paper_towel_roll = PaperTowelRoll()
        ENV_end = Environment({"paper_towel_roll": PART_paper_towel_roll})
        end_frame = Frame([53,0,160], [-90,-60,0])
        assembly.end_with(ENV_end.tube, frame=end_frame)
    elif arglib == "diapercaddy":
        PART_backseat = BackSeats()
        ENV_start = Environment({"backseat": PART_backseat})
        start_frame = Frame([0,0,0], [0,0,0])
        assembly.add(ENV_start, start_frame)
        PART_diaper_caddy = DiaperCaddy()
        ENV_end = Environment({"diaper_caddy": PART_diaper_caddy})
        end_frame = Frame([124.3,580,717.1], [-135.5,-40,20.5])
        assembly.end_with(ENV_end.hook2, frame=end_frame)
    elif arglib == "birdfeeder":
        PART_bird_feeder_env = BirdFeederEnv()
        ENV_start = Environment({"bird_feeder_env": PART_bird_feeder_env})
        assembly.add(ENV_start, Frame())
        PART_bird_feeder = BirdFeeder()
        ENV_end = Environment({"bird_feeder": PART_bird_feeder})
        end_frame = Frame([0,330,-352.3], [0,0,90])
        assembly.end_with(ENV_end.ring, frame=end_frame)
    elif arglib == "cad":
        PART_carseat = CarSeat()
        ENV_start = Environment({"part": PART_carseat})
        assembly.add(ENV_start, Frame())
        PART_basket = OpenHandleBasket()
        ENV_end = Environment({"part": PART_basket})
        end_frame = Frame([0.0,645.2361909794959,624.1851652578137], [-89.99999999999999,-15.000000000000009,89.99999999999999])
        assembly.end_with(ENV_end.hook, frame=end_frame)
    elif arglib == "cad2":
        PART_closetrods = ClosetRods()
        ENV_start = Environment({"part": PART_closetrods})
        start_frame = Frame([0.0,0.0,200.0], [0.0,0.0,-90.0])
        assembly.add(ENV_start, start_frame)
        PART_basket = OpenHandleBasket()
        ENV_end = Environment({"part": PART_basket})
        end_frame = Frame([0.0,0.0,50.0], [-90,0.0,90])
        assembly.end_with(ENV_end.hook, frame=end_frame)
    elif arglib == "study":
        PART_closetrods = ClosetRods()
        ENV_start = Environment({"part": PART_closetrods})
        start_frame = Frame([0.0,0.0,200.0], [0.0,0.0,-90.0])
        assembly.add(ENV_start, start_frame)
        PART_basket = RoundHandleBasket()
        ENV_end = Environment({"part": PART_basket})
        end_frame = Frame([0.0,0.0,100.0], [-75,0.0,90])
        assembly.end_with(ENV_end.hook, frame=end_frame)
    elif arglib == "readingnook":
        ENV_start = Environment({"room": ReadingRoom()})
        start_frame = Frame([0,0,0], [0,0,0])
        assembly.add(ENV_start, start_frame)
        ENV_end = Environment({"hulahoop": HulaHoop()})
        end_frame = Frame([0, -550, 1000], [0,90,90])
        assembly.end_with(ENV_end, frame=end_frame)
    elif arglib == "cliplight":
        ENV_start = Environment({"ceiling_hook": CeilingHook()})
        start_frame = Frame([0,0,1000], [0,0,0])
        assembly.add(ENV_start, start_frame)
        ENV_end = Environment({"clip_light": ClipLight()})
        end_frame = Frame([0,0,0], [180,90,0])
        assembly.end_with(ENV_end, end_frame)
    else:
        raise ValueError("arglib not recognized")
    return assembly

def show_help_marker(desc):
    psim.SameLine()
    psim.TextDisabled("(?)")
    if psim.IsItemHovered():
        psim.BeginTooltip()
        psim.PushTextWrapPos(450.)
        psim.TextUnformatted(desc)
        psim.PopTextWrapPos()
        psim.EndTooltip()

def primitive_env(primitive, params=None):
    p = None
    if primitive == "Edge":
        p = Edge(params)
    elif primitive == "Rod":
        p = Rod(params)
    elif primitive == "Surface":
        p = Surface(params)
    elif primitive == "Tube":
        p = Tube(params)
    elif primitive == "Ring":
        p = Ring(params)
    elif primitive == "Hook":
        p = Hook(params)
    else:
        raise ValueError("env primitive not recognized")
    return Environment({primitive.lower(): p})

def part_env(part):
    return Environment({"part": part})

def viewing_assembly(env):
    ASSEMBLY = AssemblyUI()
    ASSEMBLY.add(env, Frame())
    return ASSEMBLY

def update_view(start_added, assembly_for_viewing, assembly, reset=True, solver_options=None, after_connect=False):
    ps.remove_all_structures()
    if start_added:
        if after_connect:
            assembly.solve(solver_options if solver_options is not None else SolverOptions())
        else:
            assembly.solve_with_static_solve(solver_options if solver_options is not None else SolverOptions())
        assembly.visualize(after_sim_or_reset=reset)
    assembly_for_viewing.visualize()
    if reset:
        ps.reset_camera_to_home_view()

def can_attach_to_clicked_primitive(click_selected_primitive, c, click_p, solver, assembly):
    if click_selected_primitive is not None:
        part_conn_id = c.tid
        assembly_conn_id = click_p.tid
        if solver.connectivity_graph.has_edge(part_conn_id, assembly_conn_id):
            dim_constraint_check = solver.connectivity_graph.edges[part_conn_id, assembly_conn_id]["dim_constraint"]
            return (dim_constraint_check is None or dim_constraint_check(c, click_p, assembly=assembly))
        else:
            return False
    return True

def explorer():
    parser = argparse.ArgumentParser(description="Design explorer for home hacks.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="whether to enable full printing")
    parser.add_argument("--assembly_name", type=str, default="ASSEMBLY", metavar="N",
                        help="assembly name to store as files")
    parser.add_argument("--output_dir", type=str, default=".", help="where to save the output files")
    parser.add_argument("--use_initial_guess", action="store_true", default=False,
                        help="whether to use frame propagation instead of running the solver")
    parser.add_argument("--use_two_step", action="store_true", default=False,
                        help="whether to use two-step optimization (physics) in the solver")
    parser.add_argument("--use_larger_library", action="store_true", default=False,
                        help="whether to use larger library of parts for ex0,2,3,5")
    parser.add_argument("--parameter_sweep", action="store_true", default=False,
                        help="whether to use all initial guesses from parameter sweep")
    parser.add_argument("--ninitialization", type=int, default=None,
                        help="number of initializations to try; only used when parameter_sweep is True")
    parser.add_argument("--maxiters", type=int, default=1000, metavar="I",
                        help="max number of iterations for the solver (default: 500)")
    parser.add_argument("--maxsiters", type=int, default=1000, metavar="I",
                        help="max number of simulation iterations for the solver (default: 500)")
    parser.add_argument("--library", type=str, default="demo", metavar="L",
                        help="the library of parts to use (default: demo). " +\
                        "Current list of example libraries are: " + ",".join(available_env_setups) + ".")
    args = parser.parse_args()

    if args.use_larger_library and args.library not in [0,2,3,5]:
        raise ValueError("use_larger_library option is only available for ex0,2,3,5")

    Primitive.verbose = args.verbose
    Assembly.verbose = args.verbose
    if not args.verbose:
        import warnings
        warnings.filterwarnings("ignore")

    solver_options = SolverOptions(assembly_name=args.assembly_name, output_dir=args.output_dir,
        use_initial_guess=args.use_initial_guess, use_two_step=args.use_two_step, maxsiters=args.maxsiters,
        parameter_sweep=args.parameter_sweep, ninitialization=args.ninitialization, maxiters=args.maxiters)
    solver = BruteForceSolver([], [], [], [], [], options=solver_options)
    # set program options
    ps.set_program_name("design explorer")
    ps.set_print_prefix("MACGYVER")
    ps.set_up_dir("z_up")
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("free")
    ps.set_build_gui(False)
    ps.init() # the init() also initializes opengl context that glGenTextures() needs

    assembly = AssemblyUI(args.assembly_name)
    library_constructor = PartsLibraryG
    env_lib = env_library(library_constructor)
    lib = full_library(library_constructor)
    arglib = args.library
    ######### the code below sets a smaller library to pick from
    if arglib == "demo":
        lib = demo_larger_library(library_constructor) if args.use_larger_library else demo_library(library_constructor)
    elif arglib == "toothbrush":
        lib = clip_toothbrush_library(library_constructor)
    elif arglib == "clipcable":
        lib = clip_cable_larger_library(library_constructor) if args.use_larger_library else clip_cable_library(library_constructor)
    elif arglib == "soapbottle":
        lib = basket_v2_larger_library(library_constructor) if args.use_larger_library else basket_v2_library(library_constructor)
    elif arglib == "mughanger":
        lib = mug_hanger_larger_library(library_constructor) if args.use_larger_library else mug_hanger_library(library_constructor)
    elif arglib == "smallbirdfeeder":
        lib = small_bird_feeder_library(library_constructor)
    elif arglib == "papertowelroll":
        lib = paper_towel_holder_library(library_constructor)
    elif arglib == "diapercaddy":
        lib = backseat_diaper_caddy_library(library_constructor)
    elif arglib == "birdfeeder":
        lib = bird_feeder_study_library(library_constructor)
    elif arglib == "readingnook":
        lib = reading_nook_library(library_constructor)
    elif arglib == "cliplight":
        lib = clip_light_library(library_constructor)
    elif arglib != "default":
        lib_file_path = os.path.join("fabhacks", "data", "library", arglib + ".pkl")
        if os.path.exists(lib_file_path):
            parts = pickle.load(open(lib_file_path, "rb"))
            lib = library_constructor(parts)
        else:
            print("library name not found! please make sure the library file exists under fabhacks/data/library/")
            exit(0)

    env_lib_parts = [p["constructor"].__name__ for p in env_lib.parts]
    env_lib_parts_selected = env_lib_parts[0]
    env_lib_primitives = ["Edge", "Rod", "Surface", "Tube", "Ring", "Hook"]
    env_lib_prim_selected = env_lib_primitives[0]
    env_lib_setups = available_env_setups
    selected_setup = arglib if arglib in env_lib_setups else env_lib_setups[0]
    if arglib in env_lib_setups:
        assembly = env_setup(arglib)
    prim_instr_texts = {
        "Edge": "Examples include table edges.",
        "Rod": "Examples include curtain rods, hanging rods, etc.",
        "Surface": "Examples include table surfaces, walls, etc.",
        "Tube": "Examples include pipes.",
        "Ring": "Examples include screw-in holes.",
        "Hook": "Examples include ceiling hooks.",
    }
    prim_param_defaults = {
        "Edge": {"width": 100, "length": 200, "height": 1.5},
        "Rod": {"radius": 10, "length": 500},
        "Surface": {"width": 300, "length": 300},
        "Tube": {"inner_radius": 10, "thickness": 1, "length": 500},
        "Ring": {"arc_radius": 25, "thickness": 2.5},
        "Hook": {"arc_radius": 25, "arc_angle": 300, "thickness": 2.5},
    }
    env_frame = Frame()
    lib_parts = [p["constructor"].__name__ for p in lib.parts]
    lib_parts_selected = lib_parts[0]
    assembly_for_viewing = AssemblyUI(args.assembly_name + "_viewing")
    changed = False
    click_selected_name = None
    click_selected_primitive = None
    menu_selected_name = None
    menu_selected_primitive = None
    child_primitives = []
    selected_primitive = None
    click_pid = None
    click_p = None
    prev_p = None
    menu_pid = None
    menu_p = None

    environment_selecting = True if arglib == "default" else False
    filter_attachable_parts = True
    is_primitive_env = False
    curr_part = None
    start_added = False if arglib == "default" else True
    end_added = False if arglib == "default" else True
    new_part_attached = False
    status_text = ""
    param_status_text = ""
    end_attached = False

    part_textures, prim_textures = load_textures()

    undo_stack = deque(maxlen=5)
    redo_stack = deque(maxlen=5)

    def polyscope_callback():
        # note: declaring nonlocal due to python scoping rules
        nonlocal solver_options, changed, assembly_for_viewing, assembly, env_lib_parts_selected, lib_parts_selected,\
            env_lib_primitives, env_lib_prim_selected, prim_instr_texts, prim_param_defaults, env_frame, prev_p, \
            curr_part, click_selected_name, click_selected_primitive, is_primitive_env, click_pid, click_p, menu_pid, menu_p,\
            filter_attachable_parts, environment_selecting, start_added, end_added, env_lib_setups, selected_setup,\
            child_primitives, selected_primitive, part_textures, prim_textures, menu_selected_name, menu_selected_primitive,\
            undo_stack, redo_stack, new_part_attached, status_text, param_status_text, end_attached

        # note: entity (e.g. button) cannot be the same name in imgui
        psim.Begin("Assembly Design", True)
        psim.PushItemWidth(150)
        psim.SetWindowPos((10.0, 10.0))
        psim.SetWindowSize((360, 830))

        changed, environment_selecting = psim.Checkbox("Environment Setup", environment_selecting)
        show_help_marker("If this is checked, you can select environment parts and primitives to add to the scene. "
                        "When this is unchecked and you check it again, it will reset the assembly.")

        if changed and environment_selecting:
            # reset assembly
            assembly = AssemblyUI(args.assembly_name)
            start_added = False
            end_added = False

        changed = psim.BeginCombo("Select Preset Environment", selected_setup)
        if changed:
            for val in env_lib_setups:
                _, selected = psim.Selectable(val, selected_setup == val)
                if selected:
                    selected_setup = val
            if selected_setup != "default":
                assembly = env_setup(selected_setup)
                assembly_for_viewing = AssemblyUI()
                start_added = True
                end_added = True
                environment_selecting = False
                update_view(start_added, assembly_for_viewing, assembly, solver_options=solver_options)
            psim.EndCombo()

        # environment selection UI
        psim.PushItemWidth(80)
        if environment_selecting:
            changed, is_primitive_env = psim.Checkbox("Use Primitives", is_primitive_env)
            if is_primitive_env:
                changed = psim.BeginCombo("Environment Primitives", env_lib_prim_selected)
                if changed:
                    for val in env_lib_primitives:
                        _, selected = psim.Selectable(val, env_lib_prim_selected == val)
                        if selected:
                            env_lib_prim_selected = val
                    psim.EndCombo()
                psim.TextUnformatted(prim_instr_texts[env_lib_prim_selected])
                for param, default_val in prim_param_defaults[env_lib_prim_selected].items():
                    changed, val = psim.InputFloat(param, default_val)
                    if changed and val > 0 and val != default_val:
                        prim_param_defaults[env_lib_prim_selected][param] = val
                        curr_part = primitive_env(env_lib_prim_selected, prim_param_defaults[env_lib_prim_selected])
                        assembly_for_viewing = viewing_assembly(curr_part)
                        update_view(start_added, assembly_for_viewing, assembly, solver_options=solver_options)
                psim.SameLine()
                if psim.Button("Show Env Prim"):
                    curr_part = primitive_env(env_lib_prim_selected, prim_param_defaults[env_lib_prim_selected])
                    assembly_for_viewing = viewing_assembly(curr_part)
                    update_view(start_added, assembly_for_viewing, assembly, solver_options=solver_options)
            else:
                index = 0
                env_lib_parts_with_textures = [p for p in env_lib_parts if PartType[p].value in part_textures]
                for part in env_lib_parts_with_textures:
                    # TODO: fix for parampart
                    info = part_textures[PartType[part].value]
                    if psim.ImageButton(info["texture"], [info["width"], info["height"]]):
                        env_lib_parts_selected = part
                        curr_part = part_env(env_lib.get_part(part))
                        new_part_attached = False
                        assembly_for_viewing = viewing_assembly(curr_part)
                        update_view(start_added, assembly_for_viewing, assembly, solver_options=solver_options)
                    if index % 3 != 2 and index != len(env_lib_parts_with_textures) - 1:
                        psim.SameLine()
                    index += 1

            changed, env_frame.pos = psim.DragFloat3("Frame Position", env_frame.pos)
            if changed:
                curr_part.frame = env_frame
                update_view(start_added, assembly_for_viewing, assembly, reset=False, solver_options=solver_options)
            changed, env_frame.rot = psim.DragFloat3("Frame Rotation", env_frame.rot)
            if changed:
                curr_part.frame = env_frame
                update_view(start_added, assembly_for_viewing, assembly, reset=False, solver_options=solver_options)

            if curr_part is not None:
                if not start_added:
                    if psim.Button("Add Part as Starting Environment"):
                        curr_part.frame = Frame()
                        assembly.add(curr_part, copy.deepcopy(env_frame))
                        start_added = True
                        curr_part = None
                        env_frame = Frame()
                elif not end_added:
                    # child_primitives = curr_part.children_varnames
                    # selected_primitive = child_primitives[0]
                    # if len(child_primitives) > 1:
                    #     changed = psim.BeginCombo("Part Primitive to use", selected_primitive)
                    #     if changed:
                    #         for val in child_primitives:
                    #             _, selected = psim.Selectable(val, selected_primitive == val)
                    #             if selected:
                    #                 selected_primitive = val
                    #         psim.EndCombo()
                    if psim.Button("Add Part as Target Object"):
                        child_index = 0 #child_primitives.index(selected_primitive)
                        for c in curr_part.children:
                            c.set_reduced(True)
                        assembly.end_with(curr_part.children[child_index], copy.deepcopy(curr_part.children[child_index].get_global_frame()))
                        end_added = True
                        environment_selecting = False
                        curr_part = None
        else:
            changed, filter_attachable_parts = psim.Checkbox("Filter Parts that Attach to Assembly", filter_attachable_parts)
            if psim.Button("Reset View"):
                ps.reset_camera_to_home_view()
            psim.Separator()
            psim.TextUnformatted("Attachable Parts")

            index = 0
            for part in lib_parts:
                if PartType[part].value not in part_textures:
                    continue
                if not filter_attachable_parts or\
                    assembly.can_attach(solver, lib.get_part(part), assembly_conn=click_p):
                    info = part_textures[PartType[part].value]
                    if psim.ImageButton(info["texture"], [info["width"], info["height"]]):
                        param_status_text = ""
                        lib_parts_selected = part
                        curr_part = lib.get_part(part)
                        menu_selected_name = None
                        menu_selected_primitive = None
                        menu_p = None
                        new_part_attached = False
                    if index % 3 != 2 and index != len(lib_parts) - 1:
                        psim.SameLine()
                    if "Param" in lib_parts_selected and part == lib_parts_selected and curr_part is not None:
                        for p, v in curr_part.part_params.items():
                            changed, v = psim.InputFloat(p, v)
                            curr_part.part_params[p] = v
                            psim.SameLine()
                            if psim.Button("Update Parameter"):
                                curr_part.param_init()
                                curr_part.init_mesh_metrics(reload=True)
                                param_status_text = f"parametric {part} updated"
                        index += 1
                    index += 1
            psim.TextUnformatted(param_status_text)
        psim.PopItemWidth()
        psim.Separator()
        if len(undo_stack) > 0:
            if psim.Button("Undo"):
                status_text = ""
                redo_stack.append(copy.deepcopy(assembly))
                assembly = undo_stack.pop()
                end_attached = assembly.all_end_connected()
                update_view(start_added, assembly_for_viewing, assembly, reset=False, solver_options=solver_options)
        if len(redo_stack) > 0:
            psim.SameLine()
            if psim.Button("Redo"):
                status_text = ""
                undo_stack.append(copy.deepcopy(assembly))
                assembly = redo_stack.pop()
                end_attached = assembly.all_end_connected()
                update_view(start_added, assembly_for_viewing, assembly, reset=False, solver_options=solver_options)
        psim.PopItemWidth()
        psim.End()

        if not environment_selecting and curr_part is not None:
            psim.Begin("Part Connector Primitives", True)
            psim.SetWindowPos([400., 690.])
            psim.SetWindowSize([800., 150.])
            if not new_part_attached:
                for c in curr_part.children:
                    if not filter_attachable_parts or\
                        can_attach_to_clicked_primitive(click_selected_primitive, c, click_p, solver, assembly):
                        info = prim_textures[c.ohe.value]
                        if psim.ImageButton(info["texture"], [info["width"], info["height"]]):
                            status_text = ""
                            prev_selected_primitive = menu_selected_primitive
                            menu_selected_name = str(c)
                            menu_p = c
                            if ps.has_surface_mesh(menu_selected_name):
                                menu_selected_primitive = ps.get_surface_mesh(menu_selected_name)
                                menu_selected_primitive.set_transparency(1)
                                if prev_selected_primitive is not None:
                                    prev_selected_primitive.set_transparency(0.5)
                        psim.SameLine()
            if menu_p is not None and click_p is not None:
                if psim.Button("Attach to assembly"):
                    status_text = ""
                    undo_stack.append(copy.deepcopy(assembly))
                    redo_stack.clear()
                    new_curr_part = lib.get_part(lib_parts_selected, force_new=True)
                    assembly.attach(new_curr_part.children[menu_p.child_id], click_p)
                    update_view(start_added, assembly_for_viewing, assembly, reset=False, solver_options=solver_options)
                    menu_selected_primitive = None
                    menu_selected_name = None
                    menu_p = None
                    click_selected_primitive = None
                    click_selected_name = None
                    click_p = None
                    click_pid = None
                    new_part_attached = True
                show_help_marker("This works only when you have selected a primitive from the assembly"
                                " and a primitive from the part you want to attach from the menu below.")
            if click_p is not None and prev_p is not None:
                if assembly.can_connect(click_p, prev_p) and not (not end_attached and (click_p.parent.id in assembly.end_part_ids() or prev_p.parent.id in assembly.end_part_ids())):
                    if psim.Button("Connect within assembly (Click to check feasibility)"):
                        temp_assembly = copy.deepcopy(assembly)
                        status_text = ""
                        # provide info about whether it is a valid connect()
                        temp_assembly.connect(click_p, prev_p)
                        _, created_loops = temp_assembly.loops() # <prune
                        if temp_assembly.check_constraints_with_linear_program(created_loops):
                            if temp_assembly.check_constraints_feasibility(solver_options):
                                status_text = "feasible connect()"
                                undo_stack.append(copy.deepcopy(assembly))
                                redo_stack.clear()
                                assembly.connect(click_p, prev_p)
                                # print(temp_assembly.con_check_initial_guess)
                                # assembly.set_frames_from_x(temp_assembly.con_check_initial_guess)
                            else:
                                status_text = "infeasible connect()"
                        else:
                            status_text = "infeasible connect(); parts not long enough"# via linear program"
                        if status_text == "feasible connect()":
                            update_view(start_added, assembly_for_viewing, assembly, reset=False, solver_options=solver_options, after_connect=True)
                        click_selected_primitive = None
                        click_selected_name = None
                        click_p = None
                        click_pid = None
                    show_help_marker("This works only when you have clicked on two primitives from the assembly"
                                    " and neither of them are part of an end part of the environment setup.")
            if click_p is not None and prev_p is not None:
                if click_p.parent.id in assembly.end_part_ids() or prev_p.parent.id in assembly.end_part_ids()\
                    and click_p.parent.id != prev_p.parent.id:
                    if (click_p.parent.id in assembly.end_part_ids() and not assembly.end_connected(click_p.parent.id)) or\
                        (prev_p.parent.id in assembly.end_part_ids() and not assembly.end_connected(prev_p.parent.id)):
                        if psim.Button("Attach end to assembly"):
                            status_text = ""
                            undo_stack.append(copy.deepcopy(assembly))
                            redo_stack.clear()
                            if click_p.parent.id in assembly.end_part_ids():
                                assembly.attach(click_p, prev_p)
                            else:
                                assembly.attach(prev_p, click_p)
                            update_view(start_added, assembly_for_viewing, assembly, reset=False, solver_options=solver_options)
                            end_attached = True
                            menu_selected_primitive = None
                            menu_selected_name = None
                            menu_p = None
                            click_selected_primitive = None
                            click_selected_name = None
                            click_p = None
                            click_pid = None
                        show_help_marker("This works only when you have clicked on two primitives from the assembly"
                                        " and one of them is part of an end part of the environment setup.")
            psim.TextUnformatted(status_text)
            psim.End()

        # process clicks
        if ps.have_selection():
            name = ps.get_selection()[0]
            if click_selected_name != name:
                if "Primitive" in name:
                    prev_selected_primitive = click_selected_primitive
                    prev_p = click_p
                    click_selected_name = name
                    click_selected_primitive = ps.get_surface_mesh(name)
                    click_selected_primitive.set_transparency(1)
                    click_pid = int(click_selected_name.split("-")[0][9:])
                    click_p = assembly.primitive_graph.nodes[click_pid]["primitive"]
                    if prev_selected_primitive is not None:
                        prev_selected_primitive.set_transparency(0.75)
                    ps.look_at(ps.get_camera_world_position(), click_p.get_global_frame().pos)
        else:
            prev_selected_primitive = None
            prev_p = None
            click_selected_primitive = None
            click_selected_name = None
            click_p = None
            click_pid = None

        psim.Begin("Assembly Parameters Controls", True)
        psim.SetWindowPos([1080., 10.])
        psim.SetWindowSize([350., 600.])
        assembly.polyscope_callback(click_p, solver_options, environment_selecting)
        psim.End()

    # init and show the window
    if arglib != "default":
        assembly.visualize()
    ps.set_user_callback(polyscope_callback)
    ps.show()
    clear_textures(part_textures)
    clear_textures(prim_textures)
