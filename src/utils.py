import os
import matplotlib.pyplot as plt
import numpy as np

from src.monte_carlo import monte_carlo_sim, plot_monte_carlo, animate_monte_carlo_sim

def generate_heatmap(all_seed_growth_grids, 
                     title, 
                     colorbar_label, 
                     ylabel="Y", 
                     xlabel="X", 
                     save_plot=False, 
                     plot_file_name="heatmap.png"):
    summed_grid = np.sum(all_seed_growth_grids, axis=0)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(summed_grid, cmap='hot', interpolation='nearest')
    plt.colorbar(label=colorbar_label)
    #plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    if save_plot:
        os.makedirs("results", exist_ok=True)
        save_location = os.path.join("results", plot_file_name)
        plt.savefig(save_location)

    plt.show()

def plot_histogram(data, 
                   title, 
                   xlabel, 
                   ylabel, 
                   save_plot=False, 
                   plot_file_name="histogram.png"):

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=50)
    #plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    
    if save_plot:
        os.makedirs("results", exist_ok=True)
        save_location = os.path.join("results", plot_file_name)
        plt.savefig(save_location)

    plt.show()

def plot_many_line(data_array, 
                   title, 
                   xlabel, 
                   ylabel, 
                   save_plot=False, 
                   plot_file_name="multi_line_plot.png"):
    plt.figure(figsize=(8, 4))
    
    for data in data_array:
        plt.plot(data, color='grey', alpha=0.5)
    
    average_data = np.nanmean(data_array, axis=0)
    plt.plot(average_data, color='blue', linewidth=2, label='Average')
    
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    
    if save_plot:
        os.makedirs("results", exist_ok=True)
        save_location = os.path.join("results", plot_file_name)
        plt.savefig(save_location)

    plt.show()

def plot_variance(data,
                  title, 
                  xlabel, 
                  ylabel, 
                  save_plot=False, 
                  plot_file_name="errorbar_plot.png"):
    mean_growth = np.nanmean(data, axis=0)
    std_dev = np.nanstd(data, axis=0)
    confidence = 1.96 * std_dev / np.sqrt(data.shape[0])

    iterations = np.arange(data.shape[1])
    
    plt.figure(figsize=(8, 4))
    plt.plot(iterations, mean_growth, label="Mean Growth Rate", color="blue")
    plt.fill_between(iterations, mean_growth - confidence, mean_growth + confidence, 
                     color="blue", alpha=0.3, label="95% CI")
    plt.errorbar(range(len(mean_growth)), mean_growth, yerr=std_dev, alpha=0.5)
    #plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    
    if save_plot:
        os.makedirs("results", exist_ok=True)
        save_location = os.path.join("results", plot_file_name)
        plt.savefig(save_location)

    plt.show()

def save_data(data, filename):
    os.makedirs("data", exist_ok=True)
    
    # Ensure filename is a string
    if isinstance(filename, list):
        filename = "_".join(map(str, filename))
    
    file_location = os.path.join("data", filename)
    
    if os.path.exists(file_location):
        existing_data = np.load(file_location)
        data = np.concatenate((existing_data, data))
    
    np.save(file_location, data)

