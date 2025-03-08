import os
import numpy as np

from src.utils import (
    load_data, 
    generate_heatmap, 
    plot_histogram, 
    flat_histogram, 
    plot_cross_section, 
    plot_deviation, 
    plot_single_y_slice_density, 
    flat_histogram_multiple, 
    plot_cross_section_and_deviation_multiple, 
    plot_y_slice_density_multiple
)

def generate_save_names(sticking_prob_str):
    return [
        f"final_seed_growth_states_{sticking_prob_str}_sp.npy",
        f"all_walk_counts_{sticking_prob_str}_sp.npy",
        f"all_successful_walks_{sticking_prob_str}_sp.npy",
        f"all_avg_walk_lengths_{sticking_prob_str}_sp.npy",
        f"all_avg_successful_walk_lengths_{sticking_prob_str}_sp.npy"
    ]

def load_data_files(save_names):
    return [np.load(os.path.join("data", name)) for name in save_names]

def plot_monte_carlo_sp_single(sticking_prob):
    sticking_prob_str = str(sticking_prob).replace(".", "_")
    save_names = generate_save_names(sticking_prob_str)

    final_seed_growth_states, all_walk_counts, all_successful_walks, all_avg_walk_lengths, all_avg_successful_walk_lengths = load_data_files(save_names)

    generate_heatmap(final_seed_growth_states, 
                     "Final seed growth states", 
                     "Frequency", 
                     save_plot=True, 
                     file_path=os.path.join("results", "monte_carlo", "heatmap_" + save_names[0].replace(".npy", ".png")))
    flat_histogram(final_seed_growth_states, 
                   "Flat histogram of final seed growth states", 
                   "Cell Occupation Probability", 
                   "Frequency of grid cells", 
                   save_plot=True, 
                   file_path=os.path.join("results", "monte_carlo", "histogram_" + save_names[0].replace(".npy", ".png")))
    plot_cross_section(final_seed_growth_states, 
                       save_plot=True, 
                       file_path=os.path.join("results", "monte_carlo", "cross_section_" + save_names[0].replace(".npy", ".png")))
    plot_single_y_slice_density(final_seed_growth_states,
                                save_plot=True, 
                                file_path=os.path.join("results", "monte_carlo", "y_slice_density_" + save_names[0].replace(".npy", ".png")))
    plot_deviation(final_seed_growth_states, 
                   save_plot=True, 
                   file_path=os.path.join("results", "monte_carlo", "deviation_" + save_names[0].replace(".npy", ".png")))
    plot_histogram(all_walk_counts, 
                   "Histogram of all walk counts", 
                   "Number of walks", 
                   "Frequency", 
                   save_plot=True, 
                   file_path=os.path.join("results", "monte_carlo", "histogram_" + save_names[1].replace(".npy", ".png")))
    plot_histogram(all_successful_walks, 
                   "Histogram of all successful walks", 
                   "Number of successful walks", 
                   "Frequency", 
                   save_plot=True, 
                   file_path=os.path.join("results", "monte_carlo", "histogram_" + save_names[2].replace(".npy", ".png")))
    plot_histogram(all_avg_walk_lengths, 
                   "Histogram of all average walk lengths", 
                   "Average walk length", 
                   "Frequency", 
                   save_plot=True, 
                   file_path=os.path.join("results", "monte_carlo", "histogram_" + save_names[3].replace(".npy", ".png")))
    plot_histogram(all_avg_successful_walk_lengths, 
                   "Histogram of all average successful walk lengths", 
                   "Average successful walk length", 
                   "Frequency", 
                   save_plot=True, 
                   file_path=os.path.join("results", "monte_carlo", "histogram_" + save_names[4].replace(".npy", ".png")))

def plot_mont_carlo_sp_range(sticking_prob_array):
    data_array = [load_data("final_seed_growth_states_" + str(str(sticking_prob).replace(".", "_")) + "_sp.npy") for sticking_prob in sticking_prob_array]

    

    plot_cross_section_and_deviation_multiple(sticking_prob_array,
                                              data_array,
                                              save_plot=True, 
                                              file_path=os.path.join("results", "monte_carlo", "cross_section_and_deviation_all.png"))
    
    flat_histogram_multiple(sticking_prob_array,
                            data_array,
                            "Flat histogram of final seed growth states", 
                            "Frequency of grid cells", 
                            "Frequency",
                            save_plot=True, 
                            file_path=os.path.join("results", "monte_carlo", "histogram_all.png"))
    
    plot_y_slice_density_multiple(sticking_prob_array,
                                  data_array,
                                  save_plot=True, 
                                  file_path=os.path.join("results", "monte_carlo", "y_slice_density_all.png"))

if __name__ == "__main__":

    """
    Plot single sticking probability results
    """
    sticking_prob = 1.0

    plot_monte_carlo_sp_single(sticking_prob)

    """
    Plot nultiple sticking probability results
    """
    sticking_prob_array = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    plot_mont_carlo_sp_range(sticking_prob_array)