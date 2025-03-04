import sys
import os

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.gray_scott import init_grids, solve_gray_scott, plot_animation, last_frame_gray_scott


def script_gray_scott():

    #parameters setting 
    x_length = 100
    n_steps = 50
    total_time = 10000
    time_step_size = 0.1
    diffusion_coefficient_u =  0.16
    diffusion_coefficient_v =  0.08
    U_supply = 0.01
    k = 0.047
    chemical_simulated = "U"

    chemical_U, chemical_V = init_grids(total_time, time_step_size, n_steps)
    solve_gray_scott(chemical_U, chemical_V, total_time, time_step_size, x_length, n_steps, diffusion_coefficient_u, diffusion_coefficient_v, U_supply, k)
    plot_animation(chemical_U)
    last_frame_gray_scott(chemical_U, f"figures/{chemical_simulated}, Last frame for f = {U_supply} and k = {k}.png")

if __name__ == "__main__":
    script_gray_scott()