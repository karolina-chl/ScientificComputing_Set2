from src.gray_scott import init_grids, solve_gray_scott, plot_animation

def script_gray_scott():

    #parameters setting 
    x_length = 100
    y_length = 100
    n_steps = 50
    total_time = 1000
    time_step_size = 0.1
    diffusion_coefficient_u =  0.16
    diffusion_coefficient_v =  0.08
    U_supply = 0.035
    k = 0.060

    chemical_U, chemical_V = init_grids(total_time, time_step_size, x_length, y_length, n_steps)
    solve_gray_scott(chemical_U, chemical_V, total_time, time_step_size, x_length, n_steps, diffusion_coefficient_u, diffusion_coefficient_v, U_supply, k)
    plot_animation(chemical_U)


if __name__ == "__main__":
    script_gray_scott()