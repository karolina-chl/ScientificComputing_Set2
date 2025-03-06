import os
import numpy as np

from src.utils import generate_heatmap, plot_histogram, plot_many_line, plot_variance, flat_histogram, plot_many_runs_experiment, flat_histogram_multiple, plot_many_runs_experiment_multiple

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

def scrpt_plot_monte_carlo_results(sticking_prob):
    sticking_prob_str = str(sticking_prob).replace(".", "_")
    save_names = generate_save_names(sticking_prob_str)

    final_seed_growth_states, all_walk_counts, all_successful_walks, all_avg_walk_lengths, all_avg_successful_walk_lengths, all_growth_over_time = load_data_files(save_names)

    generate_heatmap(final_seed_growth_states, "Final seed growth states", "Frequency", save_plot=True, plot_file_name="heatmap_" + save_names[0].replace(".npy", ".png"))
    flat_histogram(final_seed_growth_states, "Flat histogram of final seed growth states", "Final seed growth states", "Frequency", save_plot=True, plot_file_name="histogram_" + save_names[0].replace(".npy", ".png"))
    plot_many_runs_experiment(final_seed_growth_states, save_plot=True, plot_file_name="many_runs_" + save_names[0].replace(".npy", ".png"))
    plot_histogram(all_walk_counts, "Histogram of all walk counts", "Number of walks", "Frequency", save_plot=True, plot_file_name="histogram_" + save_names[1].replace(".npy", ".png"))
    plot_histogram(all_successful_walks, "Histogram of all successful walks", "Number of successful walks", "Frequency", save_plot=True, plot_file_name="histogram_" + save_names[2].replace(".npy", ".png"))
    plot_histogram(all_avg_walk_lengths, "Histogram of all average walk lengths", "Average walk length", "Frequency", save_plot=True, plot_file_name="histogram_" + save_names[3].replace(".npy", ".png"))
    plot_histogram(all_avg_successful_walk_lengths, "Histogram of all average successful walk lengths", "Average successful walk length", "Frequency", save_plot=True, plot_file_name="histogram_" + save_names[4].replace(".npy", ".png"))
    plot_variance(all_growth_over_time, "Growth over time", "Time", "Growth", save_plot=True, plot_file_name="variance_" + save_names[5].replace(".npy", ".png"))
    plot_many_line(all_growth_over_time, "Growth over time", "Time", "Growth", save_plot=True, plot_file_name="lines_" + save_names[5].replace(".npy", ".png"))

if __name__ == "__main__":

    sticking_prob = 1.0
    
    scrpt_plot_monte_carlo_results(sticking_prob)

    plot_many_runs_experiment_multiple([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                                       "final_seed_growth_states_",
                                       save_plot=True, 
                                       plot_file_name="y_cross_sections_all.png")

    flat_histogram_multiple([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                            "Flat histogram of final seed growth states", 
                            "Final seed growth states", 
                            "Frequency",
                            "final_seed_growth_states_",
                            save_plot=True, 
                            plot_file_name="flat_histogram_multiple.png")