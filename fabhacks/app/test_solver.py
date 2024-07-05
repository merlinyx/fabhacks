from .example_programs import *
from .viewer import select_demo
from ..solver.brute_force_solver import SolverOptions
import polyscope as ps

def test_con_check():
    with open("test_concheck_rest.txt", "w") as f, open("test_concheck_rest_full.txt", "w") as f2:
    # with open("test_concheck_ex12.txt", "w") as f:
        for example in [1, 2, 3, 5, 11, 12, 13, 15]:
        # for example in [12]:
            try:
                demo = select_demo(example)
                solver_options = SolverOptions(assembly_name="ex"+str(example), parameter_sweep=True, ninitialization=16)
                nsuccess = 0
                linprog_times = []
                concheck_times = []
                concheck_nTs = []
                total_times = []
                successes = []
                for i in range(10):
                    f2.write(f"test run {i}\n")
                    _, created_loops = demo.loops()
                    if len(created_loops) > 0:
                        linprog_success, linprog_time = demo.check_constraints_with_linear_program(created_loops)
                    else:
                        linprog_success, linprog_time = True, 0
                    if not linprog_success:
                        f2.write("rejected by linprog\n")
                    if example == 12:
                        f2.write(f"linprog success: {linprog_success}, linprog time: {linprog_time}\n")
                        linprog_success = True
                    feasible, nT, concheck_time = demo.check_constraints_feasibility(solver_options)
                    successes.append(feasible)
                    linprog_times.append(linprog_time)
                    concheck_times.append(concheck_time)
                    concheck_nTs.append(nT)
                    total_times.append(linprog_time + concheck_time)
                total_concheck_time = 0
                total_nT = 0
                total_total_time = 0
                for i in range(10):
                    if successes[i]:
                        nsuccess += 1
                        total_concheck_time += concheck_times[i]
                        total_nT += concheck_nTs[i]
                        total_total_time += total_times[i]
                f.write("////////////////////////////////////////////////////////////////////////////\n")
                f2.write("////////////////////////////////////////////////////////////////////////////\n")
                f.write(f"example {example} success rate: {nsuccess/10 * 100}%\n")
                f.write(f"average linprog time: {sum(linprog_times)/10}\n")
                if nsuccess != 10:
                    f2.write(f"average concheck time: {sum(concheck_times)/10}\n")
                    f2.write(f"average total time: {sum(total_times)/10}\n")
                    f2.write(f"average nT: {sum(concheck_nTs)/10}\n")
                if nsuccess > 0:
                    f.write(f"average concheck time (succeeded): {total_concheck_time/nsuccess}\n")
                    f.write(f"average total time (succeeded): {total_total_time/nsuccess}\n")
                    if total_nT > 0:
                        f.write(f"average each initial guess concheck time (succeeded): {total_concheck_time/total_nT}\n")
                        f.write(f"average number of initial guesses (succeeded): {total_nT/nsuccess}\n")
                f.write("////////////////////////////////////////////////////////////////////////////\n")
                f2.write("////////////////////////////////////////////////////////////////////////////\n")
            except Exception as e:
                f.write(f"example {example} failed\n")
                f.write(e)

def test_con_check_ex17():
    # with open("test_concheck_ex17.txt", "w") as f:
    with open("test_concheck_ex17_mc16.txt", "w") as f:
        for example in [17]:
            solver_options = SolverOptions(assembly_name="ex"+str(example), parameter_sweep=True, ninitialization=16, random_init=True)
            nsuccess = 0
            linprog_times = []
            concheck_times = []
            concheck_nTs = []
            total_times = []
            successes = []
            for i in range(10):
                try:
                    f.write(f"test run {i}\n")
                    demo = select_demo(example)
                    # demo.set_saved_initial_guess()
                    demo.set_initial_guess_for_multichains(SolverOptions(assembly_name="ex"+str(example)))
                    _, created_loops = demo.loops()
                    if len(created_loops) > 0:
                        linprog_success, linprog_time = demo.check_constraints_with_linear_program(created_loops)
                    else:
                        linprog_success, linprog_time = True, 0
                    if not linprog_success:
                        f.write("rejected by linprog\n")
                    feasible, nT, concheck_time = demo.check_constraints_feasibility(solver_options)
                    successes.append(feasible)
                    linprog_times.append(linprog_time)
                    concheck_times.append(concheck_time)
                    concheck_nTs.append(nT)
                    total_times.append(linprog_time + concheck_time)
                except Exception as e:
                    f.write(f"example {example} failed\n")
                    f.write(e)
            total_concheck_time = 0
            total_nT = 0
            total_total_time = 0
            for i in range(10):
                if successes[i]:
                    nsuccess += 1
                    total_concheck_time += concheck_times[i]
                    total_nT += concheck_nTs[i]
                    total_total_time += total_times[i]
            f.write("////////////////////////////////////////////////////////////////////////////\n")
            f.write(f"example {example} success rate: {nsuccess/10 * 100}%\n")
            f.write(f"average linprog time: {sum(linprog_times)/10}\n")
            if nsuccess != 10:
                f.write(f"average concheck time: {sum(concheck_times)/10}\n")
                f.write(f"average total time: {sum(total_times)/10}\n")
                f.write(f"average nT: {sum(concheck_nTs)/10}\n")
            if nsuccess > 0:
                f.write(f"average concheck time (succeeded): {total_concheck_time/nsuccess}\n")
                f.write(f"average total time (succeeded): {total_total_time/nsuccess}\n")
                if total_nT > 0:
                    f.write(f"average each initial guess concheck time (succeeded): {total_concheck_time/total_nT}\n")
                    f.write(f"average number of initial guesses (succeeded): {total_nT/nsuccess}\n")
            f.write("////////////////////////////////////////////////////////////////////////////\n")

def test_solver():
    # set program options
    ps.set_program_name("macgyvering")
    ps.set_print_prefix("MACGYVER")
    ps.set_up_dir("z_up")
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    # init and show the window
    ps.init()
    # with open("test_rest.txt", "w") as f:
    with open("test_rest_di.txt", "w") as f:
        for example in [0, 1, 2, 3, 5, 11, 12, 13, 15, 16, 18, 19, 20]:
            try:
                demo = select_demo(example)
                solver_options = SolverOptions(assembly_name="ex"+str(example), parameter_sweep=True, ninitialization=16)
                nsuccess = 0
                total_time_total = 0
                each_solve_times_total = []
                nsolves_total = 0
                for i in range(10):
                    f.write(f"test run {i}\n")
                    total_time, each_solve_times, solve_success = demo.solve(solver_options)
                    if solve_success:
                        demo.visualize(False, after_sim_or_reset=True)
                        ps.screenshot(filename=f"ex{example}_run{i}.png", transparent_bg=False)
                        ps.remove_all_structures()
                        nsuccess += 1
                        nsolves_total += len(each_solve_times)
                    total_time_total += total_time
                    each_solve_times_total.extend(each_solve_times)
                f.write("////////////////////////////////////////////////////////////////////////////\n")
                f.write(f"example {example} success rate {nsuccess/10 * 100}%\n")
                f.write(f"average total time {total_time_total/10}\n")
                if len(each_solve_times_total) > 0:
                    f.write(f"average each solve time {sum(each_solve_times_total)/len(each_solve_times_total)}\n")
                if nsuccess > 0:
                    f.write(f"average number of initial guesses {nsolves_total/nsuccess}\n")
                f.write("////////////////////////////////////////////////////////////////////////////\n")
            except Exception as e:
                f.write(f"example {example} failed\n")
                f.write(e)

def test_solver_ex17():
    # set program options
    ps.set_program_name("macgyvering")
    ps.set_print_prefix("MACGYVER")
    ps.set_up_dir("z_up")
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    # init and show the window
    ps.init()
    with open("ex17_test_mc.txt", "w") as f:
        for example in [17]:
            solver_options = SolverOptions(assembly_name="ex"+str(example))
            nsuccess = 0
            total_time_total = 0
            each_solve_times_total = []
            nsolves_total = 0
            for i in range(10):
                try:
                    f.write(f"test run {i}\n")
                    demo = select_demo(example)
                    # demo.set_saved_initial_guess()
                    demo.set_initial_guess_for_multichains(solver_options)
                    print(demo.check_constraints_feasibility(solver_options), flush=True)
                    total_time, each_solve_times, solve_success = demo.solve(solver_options)
                    print(total_time, each_solve_times, solve_success, flush=True)
                    if solve_success:
                        demo.visualize(False, after_sim_or_reset=True)
                        ps.screenshot(filename=f"ex{example}_run{i}.png", transparent_bg=False)
                        ps.remove_all_structures()
                        nsuccess += 1
                        nsolves_total += len(each_solve_times)
                    total_time_total += total_time
                    each_solve_times_total.extend(each_solve_times)
                except Exception as e:
                    f.write(f"example {example} failed\n")
                    f.write(e)
            f.write("////////////////////////////////////////////////////////////////////////////\n")
            f.write(f"example {example} success rate {nsuccess/10 * 100}%\n")
            f.write(f"average total time {total_time_total/10}\n")
            if len(each_solve_times_total) > 0:
                f.write(f"average each solve time {sum(each_solve_times_total)/len(each_solve_times_total)}\n")
            if nsuccess > 0:
                f.write(f"average number of initial guesses {nsolves_total/nsuccess}\n")
            f.write("////////////////////////////////////////////////////////////////////////////\n")
            f.flush()
