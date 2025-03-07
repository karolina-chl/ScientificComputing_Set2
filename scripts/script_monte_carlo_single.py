import os
import numpy as np

from src.monte_carlo import monte_carlo_sim, animate_monte_carlo_sim
from src.utils import generate_heatmap, plot_grid

if __name__ == "__main__":
    grid_size = 101
    sticking_prob = 0.1
    max_walkers_per_sim = 250000
    sticking_prob_str = str(sticking_prob).replace(".", "_")

    results, walk_count, successful_walks = monte_carlo_sim(grid_size, 
                                                            sticking_prob, 
                                                            max_walkers=max_walkers_per_sim)
    
    print("Number of successful walks: ", successful_walks)
    print("Number of total walks: ", walk_count)
    print("Success rate: ", successful_walks / walk_count)

    animate_monte_carlo_sim(results["successful_seed_growth_grid_states"], 
                            results["successful_walker_final_states"], 
                            grid_size,
                            save_animation=False,
                            filename="monte_carlo_animation.mp4", 
                            animation_speed=500)

    plot_grid(results["successful_seed_growth_grid_states"][successful_walks - 1], 
              growth= results["successful_seed_growth_grid_states"][successful_walks - 1], 
              file=os.path.join("results", "monte_carlo","successful_seed_growth_grid_state" + sticking_prob_str + ".png"), 
              title=sticking_prob, 
              make_cbar=False, 
              fig=None, 
              ax=None)
        
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