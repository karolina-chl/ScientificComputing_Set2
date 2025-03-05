import numpy as np

from src.monte_carlo import monte_carlo_sim, plot_monte_carlo, animate_monte_carlo_sim
from src.utils import run_multiple_simulations, generate_heatmap, save_data

def scrpt_monte_carlo():
    grid_size = 101
    sticking_prob = 0.5
    save_plot = False
    data_file_name = "monte_carlo_data.npy"
    plot_file_name = "monte_carlo_experiment_1_0.png"

    seed_growth_grid_states, walker_final_states, walk_count = monte_carlo_sim(grid_size, sticking_prob)

    #save_data(monte_carlo_data, data_file_name)

    plot_monte_carlo(seed_growth_grid_states[walk_count-1], save_plot, plot_file_name)

    filename = "monte_carlo_experiment_anim.png"

    animate_monte_carlo_sim(seed_growth_grid_states, walker_final_states, grid_size, animation_speed=250)

def scrpt_monte_carlo_multiple_simulations():
    grid_size = 101
    sticking_prob = 1.0
    num_simulations = 100

    all_seed_growth_grids, mean_free_paths, walk_counts = run_multiple_simulations(grid_size, sticking_prob, num_simulations)

    #print("Mean free paths: ", mean_free_paths)
    #print("Walk counts: ", walk_counts)

    generate_heatmap(all_seed_growth_grids)
    
if __name__ == "__main__":
    scrpt_monte_carlo()
    #scrpt_monte_carlo_multiple_simulations()