import polyscope as ps
from polyscope import imgui as psim
import argparse

from ..assembly.assembly import Assembly
from ..assembly.assemblyUI import AssemblyUI
from ..assembly.frame import Frame
from ..parts.library import *
from ..parts.env import Environment
from ..primitives.primitive import Primitive

def part_env(part):
    return Environment({"part": part})

def viewing_assembly(env):
    ASSEMBLY = AssemblyUI()
    ASSEMBLY.add(env, Frame())
    return ASSEMBLY

def icon_generator():
    parser = argparse.ArgumentParser(description="Design explorer for home hacks.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="whether to enable full printing")
    parser.add_argument("--assembly_name", type=str, default="ASSEMBLY", metavar="N",
                        help="assembly name to store as files")
    args = parser.parse_args()

    Primitive.verbose = args.verbose
    Assembly.verbose = args.verbose
    if not args.verbose:
        import warnings
        warnings.filterwarnings("ignore")

    # set program options
    ps.set_program_name("design explorer")
    ps.set_print_prefix("MACGYVER")
    ps.set_up_dir("z_up")
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")

    library_constructor = PartsLibraryG
    env_lib = complete_library(library_constructor)

    env_lib_parts = [p["constructor"].__name__ for p in env_lib.parts]
    env_lib_parts_selected = env_lib_parts[0]
    env_frame = Frame()
    assembly_for_viewing = AssemblyUI(args.assembly_name + "_viewing")
    assembly = AssemblyUI(args.assembly_name)
    changed = False
    curr_selected_name = None
    curr_selected_primitive = None
    child_primitives = []
    selected_primitive = None
    primitive_meshes = []

    environment_selecting = True
    is_part_env = True
    curr_part = None
    start_added = False
    end_added = False

    def polyscope_callback():
        # note: declaring nonlocal due to python scoping rules
        nonlocal changed, assembly_for_viewing, assembly, env_lib_parts_selected, env_frame,\
            curr_part, curr_selected_name, curr_selected_primitive, is_part_env, primitive_meshes,\
            environment_selecting, start_added, end_added, child_primitives, selected_primitive

        # note: entity (e.g. button) cannot be the same name in imgui

        psim.PushItemWidth(200)

        changed = psim.BeginCombo("All Parts", env_lib_parts_selected)
        if changed:
            for val in env_lib_parts:
                _, selected = psim.Selectable(val, env_lib_parts_selected == val)
                if selected:
                    env_lib_parts_selected = val
            psim.EndCombo()
        psim.SameLine()
        if psim.Button("Show Part"):
            curr_part = part_env(env_lib.get_part(env_lib_parts_selected))
            prev_selected_primitive = None
            curr_selected_primitive = None
            assembly_for_viewing = viewing_assembly(curr_part)
            ps.remove_all_structures()
            primitive_meshes = assembly_for_viewing.visualize()
            ps.reset_camera_to_home_view()
        if psim.Button("Screenshot Custom"):
            if curr_part is not None:
                filename = env_lib_parts_selected.lower()
                if curr_selected_primitive is not None:
                    filename += str(selected_primitive.child_id + 1)
                filename += ".png"
                ps.screenshot(filename=filename, transparent_bg=False)
                for p in primitive_meshes:
                    p.set_enabled(True)
        psim.PopItemWidth()

        # process clicks
        if ps.have_selection():
            name = ps.get_selection()[0]
            if "Primitive" in name:
                if curr_selected_name != name:
                    for p in primitive_meshes:
                        p.set_enabled(False)
                    prev_selected_primitive = curr_selected_primitive
                    curr_selected_name = name
                    curr_selected_primitive = ps.get_surface_mesh(name)
                    curr_selected_primitive.set_transparency(1)
                    curr_selected_primitive.set_enabled(True)
                    pid = int(name.split("-")[0][9:])
                    selected_primitive = assembly_for_viewing.primitive_graph.nodes[pid]["primitive"]
                    if prev_selected_primitive is not None:
                        prev_selected_primitive.set_transparency(0.5)
        else:
            curr_selected_name = None
            curr_selected_primitive = None

    # init and show the window
    ps.init()
    ps.set_user_callback(polyscope_callback)
    ps.show()
