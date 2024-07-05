from .example_programs import *
from ..primitives.primitive import Primitive
from ..solver.brute_force_solver import SolverOptions
import polyscope as ps
import argparse

def select_demo(example):
    demo = None
    if example == 0:
        demo = demo_program_R() #1e-3
    elif example == 1:
        demo = clip_toothbrush_2con_R() #1e-6
    elif example == 2:
        demo = clip_cable_R() #1e-6
    elif example == 3:
        demo = bathroom_organizer_v2_R()
    elif example == 5:
        demo = mug_hanger3_R() #1e-3
    elif example == 8:
        demo = debug_connection_program()
    elif example == 11:
        # demo = bird_feeder()
        demo = bird_feeder_variation1()
    elif example == 12:
        demo = paper_towel_holder()
    elif example == 13:
        demo = backseat_diaper_caddy()
    elif example == 15:
        demo = bathcloth_basket2()
    elif example == 16:
        # demo = clip_light()
        demo = clip_light_526()
    elif example == 18:
        demo = clip_light_550()
    elif example == 19:
        demo = clip_light_459()
    elif example == 20:
        demo = clip_light_487()
    elif example == 17:
        demo = reading_nook()
    else:
        print("example no not available!")
        exit(0)
    return demo

def viewer():
    parser = argparse.ArgumentParser(description="Viewer for macgyverism assembly solving.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="whether to enable full printing")
    parser.add_argument("--show_spec", action="store_true", default=False,
                        help="whether to show the original configuration")
    parser.add_argument("--show_frames", action="store_true", default=False,
                        help="whether to show the connector frames for debugging")
    parser.add_argument("--view_part", action="store_true", default=False,
                        help="whether to view part")
    parser.add_argument("--output_dir", type=str, default=".", help="where to save the output files")
    parser.add_argument("--use_initial_guess", action="store_true", default=False,
                        help="whether to use frame propagation instead of running the solver")
    parser.add_argument("--use_two_step", action="store_true", default=False,
                        help="whether to use two-step optimization (physics) in the solver")
    parser.add_argument("--debug_print", action="store_true", default=False)
    parser.add_argument("--solve_with_flips", action="store_true", default=False,
                        help="whether to check all alignment flips in the solver")
    parser.add_argument("--multichain", action="store_true", default=False, help="whether to use multichain solve")
    parser.add_argument("--parameter_sweep", action="store_true", default=False,
                        help="whether to use all initial guesses from parameter sweep")
    parser.add_argument("--ninitialization", type=int, default=None,
                        help="number of initializations to try; only used when parameter_sweep is True")
    parser.add_argument("--random_initialization", action="store_true", default=False)
    parser.add_argument("--maxiters", type=int, default=1000, metavar="I",
                        help="max number of iterations for the solver")
    parser.add_argument("--maxsiters", type=int, default=1000, metavar="I",
                        help="max number of iterations for the simulation")
    parser.add_argument("--example", type=int, default=8, metavar="E",
                        help="the example to view (default: 8). Current list of examples"
                        " are: 0-demo, 1-toothbrush2(tbv0/1), 2-charger, 3-soapbottle,"
                        " 5-mughanger, 8-debug, 11-birdfeeder,"
                        " 12-papertowelroll, 13-diapercaddy, 14-userstudybirdfeeder,"
                        " 15-userstudybasket, 16-cliplight, 17-readingnook.")
    parser.add_argument("--save", type=str, default="", help="the pickle file to load")
    args = parser.parse_args()
    Primitive.verbose = args.verbose
    Assembly.verbose = args.verbose
    if not args.verbose:
        import warnings
        warnings.filterwarnings("ignore")

    solver_options = SolverOptions(assembly_name="ex"+str(args.example), output_dir=args.output_dir,
        use_initial_guess=args.use_initial_guess, use_two_step=args.use_two_step, maxiters=args.maxiters, maxsiters=args.maxsiters,
        parameter_sweep=args.parameter_sweep, ninitialization=args.ninitialization, random_init=args.random_initialization)

    # set program options
    ps.set_program_name("macgyvering")
    ps.set_print_prefix("MACGYVER")
    ps.set_up_dir("z_up")
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    # init and show the window
    ps.init()

    if args.view_part:
        # part viewing
        demo = part_viewing()
    else:
        demo = select_demo(args.example)
        if not args.show_spec:
            if args.use_initial_guess:
                print(demo.compute_initial_guess_frames(False))
            elif args.save != "":
                demo.load_pickle(args.save)
            elif args.solve_with_flips:
                print(demo.solve_with_flips(solver_options))
                if args.parameter_sweep:
                    print(demo.solve(solver_options))
            elif args.multichain:
                demo.set_initial_guess_for_multichains(solver_options)
            elif args.debug_print:
                print(demo.check_constraints_feasibility(solver_options))
                demo.write_constraints("ex"+str(args.example))
                import sys
                sys.exit(0)
            else:
                print(demo.check_constraints_feasibility(solver_options))
                print(demo.solve(solver_options))
                print(demo.check_constraints_feasibility(solver_options))
                # print(demo.check_end_objective())

    # visualize the program
    demo.visualize(args.show_frames, after_sim_or_reset=True)
    # print(demo.hash()) # print ascii graph
    # print(f"number of cycles: {len(demo.loops()[1])}")
    # print(f"number of parts: {demo.part_graph.order()}")
    # print("full program:")
    # print(demo.insert_part_construction_program())
    # demo.prepare_params()
    # print(f"number of params: {len(demo.initial_guess_params)}")
    ps.set_user_callback(demo.polyscope_callback)
    ps.show()
