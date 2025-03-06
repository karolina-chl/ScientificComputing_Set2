import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.cm import get_cmap

def plot_boolean_grid(grid, save_plot, filename):
    """
    Plots the final seed growth grid.

    Parameters
    ----------
    grid : np.ndarray
        Single grid of 0s and 1s.
    save_plot : bool
        Whether to save the plot.
    filename : str
        The filename to save the plot

    Returns
    -------
    None
    """
    os.makedirs("results", exist_ok=True)
    file_location = os.path.join("results", filename)

    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap="gray_r", origin="upper", extent=[0, 1, 0, 1])
    plt.axis("off")

    plt.tight_layout() 

    if save_plot:
        plt.savefig(file_location, dpi=300)
        print(f"Plot saved as {file_location}")
    else:
        plt.show()

def generate_heatmap(all_seed_growth_grids, 
                     title, 
                     colorbar_label, 
                     ylabel="Y", 
                     xlabel="X", 
                     save_plot=False, 
                     plot_file_name="heatmap.png"):
    summed_grid = np.sum(all_seed_growth_grids, axis=0)
    normalized_grid = summed_grid / all_seed_growth_grids.shape[0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_grid, cmap='hot', interpolation='nearest')
    plt.colorbar(label=colorbar_label)
    #plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    plt.tight_layout() 

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

    plt.tight_layout() 
    
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

    plt.tight_layout() 
    
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

    plt.tight_layout() 
    
    if save_plot:
        os.makedirs("results", exist_ok=True)
        save_location = os.path.join("results", plot_file_name)
        plt.savefig(save_location)

    plt.show()


def plot_many_runs_experiment(data,
                            save_plot=False, 
                            plot_file_name="flat_histogram.png"):
    num_runs, _, grid_size = data.shape

    sum_grid = np.sum(data, axis=0)
    ys = np.linspace(0,1,grid_size)
    xs = ys.copy()
    center = xs[grid_size//2]
    xdiff = np.abs(xs - center)

    num_cells_per_cross = np.sum(sum_grid, axis=1) / num_runs
    
    
    mabs = np.sum(xdiff[None, :]*sum_grid/num_runs, axis=1)
    mean = np.sum(xs[None, :]*sum_grid/np.sum(sum_grid, axis=1)[:,None], axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(ys, mabs, label='$|x-<x>|$')
    plt.plot(ys, mean, label='<x>')
    plt.xlabel('y (cross section)')
    plt.ylabel('x (center of mass)')
    plt.legend()

    plt.tight_layout() 

    if save_plot:
        plot_file_name = "deviation_" + plot_file_name
        os.makedirs("results", exist_ok=True)
        save_location = os.path.join("results", plot_file_name)
        plt.savefig(save_location)

    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.plot(ys, num_cells_per_cross, label='$N_y$')
    plt.xlabel('y (cross section)', fontsize=16)
    plt.ylabel('Number of cells', fontsize=16)
    plt.legend()

    plt.tight_layout() 

    if save_plot:
        plot_file_name = "cross_section_" + plot_file_name
        os.makedirs("results", exist_ok=True)
        save_location = os.path.join("results", plot_file_name)
        plt.savefig(save_location)

    plt.show()

def plot_many_runs_experiment_multiple(sticking_prob_array,
                            load_file_name,
                            save_plot=False, 
                            plot_file_name="flat_histogram.png"):
    
    plt.figure(figsize=(8, 4))
    cmap = get_cmap('winter')  # Use a colormap with easily distinguishable colors

    for i, sticking_prob in enumerate(sticking_prob_array):
        sticking_prob_str = str(sticking_prob).replace(".", "_")
        data = load_data(load_file_name + str(sticking_prob_str) + "_sp.npy")
        num_runs, _, grid_size = data.shape

        sum_grid = np.sum(data, axis=0)
        ys = np.linspace(0,1,grid_size)
        xs = ys.copy()
        center = xs[grid_size//2]
        xdiff = np.abs(xs - center)

        num_cells_per_cross = np.sum(sum_grid, axis=1) / num_runs
        plt.plot(ys, num_cells_per_cross, label=str(sticking_prob), color=cmap(i/len(sticking_prob_array)))

    plt.xlabel('y (cross section)', fontsize=16)
    plt.ylabel('Number of cells', fontsize=16)
    plt.legend()

    plt.tight_layout() 

    if save_plot:
        plot_file_name = "cross_section_" + plot_file_name
        os.makedirs("results", exist_ok=True)
        save_location = os.path.join("results", plot_file_name)
        plt.savefig(save_location)
    
    plt.show()
    
def flat_histogram(data,
                  title, 
                  xlabel, 
                  ylabel, 
                  save_plot=False, 
                  plot_file_name="flat_histogram.png"):
    num_runs = data.shape[0]
    sum_grid = np.sum(data, axis=0).reshape([-1]) / num_runs
    hist, bins = np.histogram(sum_grid, np.linspace(0,1,20))
    
    plt.figure(figsize=(8, 4))
    plt.bar(bins[:-1], hist, width=0.05)
    #plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.yscale('log')

    plt.tight_layout() 

    if save_plot:
        os.makedirs("results", exist_ok=True)
        save_location = os.path.join("results", plot_file_name)
        plt.savefig(save_location)
    
    plt.show()

def flat_histogram_multiple(sticking_prob_array,
                            title, 
                            xlabel, 
                            ylabel, 
                            load_file_name,
                            save_plot=False, 
                            plot_file_name="flat_histogram.png"):
    
    plt.figure(figsize=(8, 4))

    for sticking_prob in sticking_prob_array:
        sticking_prob_str = str(sticking_prob).replace(".", "_")
        data = load_data(load_file_name + str(sticking_prob_str) + "_sp.npy")
        num_runs = data.shape[0]
        sum_grid = np.sum(data, axis=0).reshape([-1]) / num_runs
        hist, bins = np.histogram(sum_grid, np.linspace(0,1,20))
        plt.bar(bins[:-1], hist, width=0.05, alpha=0.5, label=str(sticking_prob))

    #plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.yscale('log')

    plt.tight_layout() 

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

def load_data(filename):
    file_location = os.path.join("data", filename)
    return np.load(file_location)