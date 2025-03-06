import os
import numpy as np

from src.monte_carlo import monte_carlo_sim, plot_monte_carlo, animate_monte_carlo_sim, run_multiple_simulations
from src.utils import generate_heatmap, save_data, plot_histogram, plot_many_line, plot_variance


def scrpt_monte_carlo():
    grid_size = 101
    sticking_prob = 0.1
    max_walkers = 100000

    save_plot = False
    plot_file_name = "monte_carlo_experiment_1_0.png"

    results, walk_count, successful_walks= monte_carlo_sim(grid_size, sticking_prob, max_walkers=max_walkers)
    
    print("Number of successful walks: ", successful_walks)
    print("Number of total walks: ", walk_count)
    print("Success rate: ", successful_walks/walk_count)

    plot_monte_carlo(results["successful_seed_growth_grid_states"][successful_walks-1],
                     save_plot, 
                     plot_file_name)
    
    """
    if walk_count < max_walkers:
        plot_monte_carlo(results["seed_growth_grid_states"][walk_count-1],
                        save_plot, 
                        plot_file_name)
    """
    
    
    generate_heatmap(results["successful_seed_growth_grid_states"], 
                    "Successful seed growth grid", 
                    "Age of Seed Growth")
    generate_heatmap(results["seed_growth_grid_states"],
                    "Seed growth grid",
                    "Age of Seed Growth")
    generate_heatmap(results["walker_final_states"], 
                    "Final walker states", 
                    "Number of walkers")
    generate_heatmap(results["successful_walker_final_states"], 
                    "Successful walker final states", 
                    "Number of walkers")

    save_anim = False
    filename = "monte_carlo_experiment_animation_1_0.mp4"

    animate_monte_carlo_sim(results["successful_seed_growth_grid_states"], 
                            results["successful_walker_final_states"],
                            grid_size, 
                            save_anim, 
                            filename, 
                            animation_speed=200)

def scrpt_monte_carlo_multiple_simulations(grid_size, sticking_prob, num_simulations, save_names):
    final_seed_growth_states_save_file, \
    all_walk_counts_save_file, \
    all_successful_walks_save_file, \
    all_avg_walk_lengths_save_file, \
    all_avg_successful_walk_lengths_save_file, \
    all_growth_over_time_save_file = save_names

    (final_seed_growth_states, 
     all_walk_counts, 
     all_successful_walks, 
     all_avg_walk_lengths, 
     all_avg_successful_walk_lengths,
     all_growth_over_time) = run_multiple_simulations(grid_size, sticking_prob, num_simulations)
    
    save_data(final_seed_growth_states, final_seed_growth_states_save_file)
    save_data(all_walk_counts, all_walk_counts_save_file)
    save_data(all_successful_walks, all_successful_walks_save_file)
    save_data(all_avg_walk_lengths, all_avg_walk_lengths_save_file)
    save_data(all_avg_successful_walk_lengths, all_avg_successful_walk_lengths_save_file)
    save_data(all_growth_over_time, all_growth_over_time_save_file)

    print("All data saved successfully")

def scrpt_plot_monte_carlo_results(save_names):
    final_seed_growth_states_save_file, \
    all_walk_counts_save_file, \
    all_successful_walks_save_file, \
    all_avg_walk_lengths_save_file, \
    all_avg_successful_walk_lengths_save_file, \
    all_growth_over_time_save_file = save_names

    final_seed_growth_states = np.load(os.path.join("data", final_seed_growth_states_save_file))
    all_walk_counts = np.load(os.path.join("data", all_walk_counts_save_file))
    all_successful_walks = np.load(os.path.join("data", all_successful_walks_save_file))
    all_avg_walk_lengths = np.load(os.path.join("data", all_avg_walk_lengths_save_file))
    all_avg_successful_walk_lengths = np.load(os.path.join("data", all_avg_successful_walk_lengths_save_file))
    all_growth_over_time = np.load(os.path.join("data", all_growth_over_time_save_file))

    final_seed_growth_states_save_file = final_seed_growth_states_save_file.replace(".npy", ".png")
    all_walk_counts_save_file = all_walk_counts_save_file.replace(".npy", ".png")
    all_successful_walks_save_file = all_successful_walks_save_file.replace(".npy", ".png")
    all_avg_walk_lengths_save_file = all_avg_walk_lengths_save_file.replace(".npy", ".png")
    all_avg_successful_walk_lengths_save_file = all_avg_successful_walk_lengths_save_file.replace(".npy", ".png")
    all_growth_over_time_save_file = all_growth_over_time_save_file.replace(".npy", ".png")
    
    generate_heatmap(final_seed_growth_states, 
                    "Final seed growth states", 
                    "Frequency",
                    save_plot=True,
                    plot_file_name=final_seed_growth_states_save_file)
    
    plot_histogram(all_walk_counts, 
                   "Histogram of all walk counts", 
                   "Number of walks", 
                   "Frequency",
                   save_plot=True,
                   plot_file_name=all_walk_counts_save_file)
    plot_histogram(all_successful_walks, 
                   "Histogram of all successful walks", 
                   "Number of successful walks", 
                   "Frequency",
                   save_plot=True,
                   plot_file_name=all_successful_walks_save_file)
    plot_histogram(all_avg_walk_lengths, 
                   "Histogram of all average walk lengths", 
                   "Average walk length", 
                   "Frequency",
                   save_plot=True,
                   plot_file_name=all_avg_walk_lengths_save_file)  
    plot_histogram(all_avg_successful_walk_lengths, 
                   "Histogram of all average successful walk lengths", 
                   "Average successful walk length", 
                   "Frequency",
                   save_plot=True,
                   plot_file_name=all_avg_successful_walk_lengths_save_file)
    
    plot_variance(all_growth_over_time,
                    "Growth over time",
                    "Time",
                    "Growth")
    plot_many_line(all_growth_over_time,
                     "Growth over time",
                     "Time",
                     "Growth")
    

if __name__ == "__main__":
    grid_size = 101
    sticking_prob = 1.0
    num_simulations = 1000

    sticking_prob_str = str(sticking_prob).replace(".", "_")

    final_seed_growth_states_save_file = "final_seed_growth_states_" + sticking_prob_str + "_sp.npy"
    all_walk_counts_save_file = "all_walk_counts_" + sticking_prob_str + "_sp.npy"
    all_successful_walks_save_file = "all_successful_walks_" + sticking_prob_str + "_sp.npy"
    all_avg_walk_lengths_save_file = "all_avg_walk_lengths_" + sticking_prob_str + "_sp.npy"
    all_avg_successful_walk_lengths_save_file = "all_avg_successful_walk_lengths_" + sticking_prob_str + "_sp.npy"
    all_growth_over_time_save_file = "all_growth_over_time_" + sticking_prob_str + "_sp.npy"

    save_names = [final_seed_growth_states_save_file, 
                  all_walk_counts_save_file, 
                  all_successful_walks_save_file, 
                  all_avg_walk_lengths_save_file, 
                  all_avg_successful_walk_lengths_save_file, 
                  all_growth_over_time_save_file]

    #scrpt_monte_carlo()
    #scrpt_monte_carlo_multiple_simulations(grid_size, sticking_prob, num_simulations, save_names)
    scrpt_plot_monte_carlo_results(save_names)