import os
import numpy as np
import matplotlib.pyplot as plt

from src.monte_carlo import monte_carlo_sim, animate_monte_carlo_sim
from src.utils import generate_heatmap, plot_grid

def generate_3_by_1_heatmap(results, sticking_prob_str):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    plot_grid(results["successful_seed_growth_grid_states"][25], 
              growth= results["successful_seed_growth_grid_states"][25], 
              file=None, 
              make_cbar=False, 
              fig=fig, 
              ax=axs[0])
    
    plot_grid(results["successful_seed_growth_grid_states"][successful_walks//2], 
              growth= results["successful_seed_growth_grid_states"][successful_walks//2], 
              file=None, 
              make_cbar=False, 
              fig=fig, 
              ax=axs[1])
    
    plot_grid(results["successful_seed_growth_grid_states"][successful_walks - 1], 
              growth= results["successful_seed_growth_grid_states"][successful_walks - 1], 
              file=None, 
              make_cbar=False, 
              fig=fig, 
              ax=axs[2])

    plt.tight_layout()
    plt.savefig(os.path.join("results", "monte_carlo","successful_seed_growth_grid_states" + sticking_prob_str + ".png"))
    plt.show()


def generate_multiple_heatmaps(results, sticking_prob_str):
    generate_heatmap(results["successful_seed_growth_grid_states"], 
                     "Successful seed growth grid", 
                     "Age of Seed Growth",
                     save_plot=True,
                     file_path=os.path.join("results", "monte_carlo","heatmap_successful_seed_growth_grid_state" + sticking_prob_str + ".png"))
    generate_heatmap(results["seed_growth_grid_states"],
                     "Seed growth grid",
                     "Age of Seed Growth",
                     save_plot=True,
                     file_path=os.path.join("results", "monte_carlo","heatmap_seed_growth_grid_state" + sticking_prob_str + ".png"))
    generate_heatmap(results["walker_final_states"], 
                     "Final walker states", 
                     "Number of walkers",
                     save_plot=True,
                     file_path=os.path.join("results", "monte_carlo","heatmap_walker_final_states" + sticking_prob_str + ".png"))
    generate_heatmap(results["successful_walker_final_states"], 
                     "Successful walker final states", 
                     "Number of walkers",
                     save_plot=True,
                     file_path=os.path.join("results", "monte_carlo","heatmap_successful_walker_final_states" + sticking_prob_str + ".png"))

if __name__ == "__main__":
    grid_size = 101
    sticking_prob = 1.0
    max_walkers_per_sim = 250000
    sticking_prob_str = str(sticking_prob).replace(".", "_")

    """
    results are a dictionary with the following keys
    "seed_growth_grid_states": np.array
    "walker_final_states": np.array
    "successful_seed_growth_grid_states": np.array
    "successful_walker_final_states": np.array
    """
    results, walk_count, successful_walks = monte_carlo_sim(grid_size, 
                                                            sticking_prob, 
                                                            max_walkers=max_walkers_per_sim)
    
    animate_monte_carlo_sim(results["successful_seed_growth_grid_states"], 
                            results["successful_walker_final_states"], 
                            grid_size,
                            save_animation=False,
                            filename="monte_carlo_animation.mp4", 
                            animation_speed=500)
    
    generate_3_by_1_heatmap(results, sticking_prob_str)
        
    generate_multiple_heatmaps(results, sticking_prob_str)