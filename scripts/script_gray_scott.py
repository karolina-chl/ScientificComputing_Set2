from src.gray_scott import init_grids, solve_gray_scott, plot_animation, last_frame_gray_scott_save
    
def animation(): 
    x_length = 100
    n_steps = 100
    total_time = 5000
    time_step_size = 1
    diffusion_coefficient_u =  0.16
    diffusion_coefficient_v =  0.08
    U_supply = 0.035
    k = 0.060
    noise_boundry = 0

    chemical_U, chemical_V = init_grids(total_time, time_step_size, n_steps)
    solve_gray_scott(chemical_U, chemical_V, total_time, time_step_size, x_length, n_steps, diffusion_coefficient_u, diffusion_coefficient_v, U_supply, k, noise_boundry)
    plot_animation(chemical_V) 

def script_gray_scott(total_time, noise_boundry, U_supply, k):
    x_length = 100
    n_steps = 100
    total_time = total_time
    time_step_size = 1
    diffusion_coefficient_u =  0.16
    diffusion_coefficient_v =  0.08
    U_supply = U_supply
    k = k
    noise_boundry = noise_boundry

    chemical_U, chemical_V = init_grids(total_time, time_step_size, n_steps)
    solve_gray_scott(chemical_U, chemical_V, total_time, time_step_size, x_length, n_steps, diffusion_coefficient_u, diffusion_coefficient_v, U_supply, k, noise_boundry)
    last_frame_gray_scott_save(chemical_V, f"results/gray_scott/ {total_time}, {noise_boundry}, f_{U_supply}, k_{k}.png")

if __name__ == "__main__":

    animation()

    for time in (1000,2000,5000):
        script_gray_scott(time, 0, 0.035, 0.060)

    parameters = [(0.03,0.055), (0.018, 0.051), (0.035, 0.060),(0.026, 0.051)]
    for U_supply, k in parameters: 
        script_gray_scott(5000, 0, U_supply, k)

    noise = [0.001, 0.01, 0.02, 0.1]     
    for noise_boundry in (noise):
        script_gray_scott(5000, noise_boundry, 0.035, 0.060)