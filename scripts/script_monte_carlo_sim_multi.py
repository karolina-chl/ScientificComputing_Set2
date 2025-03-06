import os
import numpy as np

from src.monte_carlo import run_multiple_simulations
from src.utils import save_data

def generate_save_names(sticking_prob_str):
    return [
        f"final_seed_growth_states_{sticking_prob_str}_sp.npy",
        f"all_walk_counts_{sticking_prob_str}_sp.npy",
        f"all_successful_walks_{sticking_prob_str}_sp.npy",
        f"all_avg_walk_lengths_{sticking_prob_str}_sp.npy",
        f"all_avg_successful_walk_lengths_{sticking_prob_str}_sp.npy",
        f"all_growth_over_time_{sticking_prob_str}_sp.npy"
    ]

def load_data_files(save_names):
    return [np.load(os.path.join("data", name)) for name in save_names]

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

def run_multiple_sticking_probs(grid_size, sticking_prob_array, num_simulations):
    for sticking_prob in sticking_prob_array:
        sticking_prob_str = str(sticking_prob).replace(".", "_")
        save_names = generate_save_names(sticking_prob_str)
        scrpt_monte_carlo_multiple_simulations(grid_size, sticking_prob, num_simulations, save_names)

if __name__ == "__main__":
    grid_size = 101
    sticking_prob_array = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    num_simulations = 20

    run_multiple_sticking_probs(grid_size, sticking_prob_array, num_simulations)