import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.cm import get_cmap


"""
Plots Should be universal
"""
def plot_grid(c, growth=None, file=None, title='', make_cbar=True, fig=None, ax=None ):
    
    #plt.tight_layout()
    
    grid_size = c.shape[-1]
    return_ax = True
    if fig is None:
        fig, ax = plt.subplots()
        return_ax=False
    heatmap = ax.imshow(c, cmap="hot", extent=[0, 1, 0, 1])
    if make_cbar:
        cbar = plt.colorbar(heatmap)
        cbar.set_label("Concentration")
    if growth is not None:      
        heatmap = ax.imshow(growth, alpha=growth, cmap='tab20b',  extent=[0, 1, 0, 1])
    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)
    ax.set_title(title)

    
    if file is not None:
        plt.savefig(file, dpi=600)

    if return_ax:
        return fig, ax
    plt.show()

def generate_heatmap(all_seed_growth_grids, 
                     title, 
                     colorbar_label, 
                     ylabel="Y", 
                     xlabel="X", 
                     save_plot=False, 
                     file_path="heatmap.png"):
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
        plt.savefig(file_path)

    plt.show()

def plot_histogram(data, 
                   title, 
                   xlabel, 
                   ylabel, 
                   save_plot=False, 
                   file_path="histogram.png"):

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=50)
    #plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    plt.tight_layout() 
    
    if save_plot:
        plt.savefig(file_path)

    plt.show()

def plot_cross_section(data,
                        save_plot=False, 
                        file_path="flat_histogram.png"):
    num_runs, _, grid_size = data.shape

    sum_grid = np.sum(data, axis=0)
    ys = np.linspace(0,1,grid_size)
    xs = ys.copy()
    center = xs[grid_size//2]
    xdiff = np.abs(xs - center)
   
    mabs = np.sum(xdiff[None, :]*sum_grid/num_runs, axis=1)
    mean = np.sum(xs[None, :]*sum_grid/np.sum(sum_grid, axis=1)[:,None], axis=1)
    
    plt.figure(figsize=(4,8))
    plt.plot(mabs, ys, label='$|x-<x>|$')
    plt.plot(mean, ys, label='<x>')
    plt.xlabel('y (cross section)')
    plt.ylabel('x (center of mass)')
    plt.legend()

    plt.tight_layout() 

    if save_plot:
        plt.savefig(file_path)

    plt.show()
    
def plot_deviation(data,
                    save_plot=False, 
                    file_path="flat_histogram.png"):
    num_runs, _, grid_size = data.shape

    sum_grid = np.sum(data, axis=0)
    ys = np.linspace(0,1,grid_size)
    xs = ys.copy()
    center = xs[grid_size//2]
    xdiff = np.abs(xs - center)

    num_cells_per_cross = np.sum(sum_grid, axis=1) / num_runs

    plt.figure(figsize=(4,8))
    plt.plot(num_cells_per_cross, ys, label='$N_y$')
    plt.ylabel('y (cross section)', fontsize=16)
    plt.xlabel('Number of cells', fontsize=16)
    plt.legend()

    plt.tight_layout() 

    if save_plot:
        plt.savefig(file_path)

    plt.show()

def flat_histogram(data,
                  title, 
                  xlabel, 
                  ylabel, 
                  save_plot=False, 
                  file_path="flat_histogram.png"):
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
        plt.savefig(file_path)
    
    plt.show()


def plot_cross_section_and_deviation_multiple(parameter_array,
                                              array_of_arrays,
                                              save_plot=False, 
                                              file_path="y_cross_section_multi.png"):
    plt.figure(figsize=(4,8))
    cmap = get_cmap('magma')  # Use a colormap with easily distinguishable colors

    for parameter, data in zip(parameter_array, array_of_arrays):
        num_runs, _, grid_size = data.shape

        sum_grid = np.sum(data, axis=0)
        ys = np.linspace(0,1,grid_size)
        xs = ys.copy()
        center = xs[grid_size//2]
        xdiff = np.abs(xs - center)

        num_cells_per_cross = np.sum(sum_grid, axis=1) / num_runs
        plt.plot(num_cells_per_cross, ys, label=str(parameter), color=cmap(parameter/max(parameter_array)))
    
    plt.ylabel('y (cross section)', fontsize=16)
    plt.xlabel('Number of cells', fontsize=16)
    plt.legend()

    plt.tight_layout() 

    if save_plot:
        plt.savefig(file_path)
    
    plt.show()

def flat_histogram_multiple(parameter_array,
                            array_of_arrays,
                            title, 
                            xlabel, 
                            ylabel, 
                            save_plot=False, 
                            file_path="flat_histogram_multi.png"):
    
    plt.figure(figsize=(8, 4))

    for parameter, data in zip(parameter_array, array_of_arrays):
        num_runs = data.shape[0]
        sum_grid = np.sum(data, axis=0).reshape([-1]) / num_runs
        hist, bins = np.histogram(sum_grid, np.linspace(0,1,20))
        plt.bar(bins[:-1], hist, width=0.05, alpha=0.5, label=str(parameter))

    #plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.yscale('log')

    plt.tight_layout() 

    if save_plot:
        plt.savefig(file_path)
    
    plt.show()

def plot_single_y_slice_density(data, save_plot=False, file_path="y_slice_density.png"):
    num_runs, _, grid_size = data.shape

    sum_grid = np.sum(data, axis=0)
    ys = np.linspace(0, 1, grid_size)
    xs = ys.copy()
    center = xs[grid_size // 2]
    xdiff = np.abs(xs - center)

    mean_diff_from_center = np.sum(xdiff[None, :] * sum_grid, axis=1) / np.sum(sum_grid, axis=1)

    plt.figure(figsize=(4, 8))
    plt.plot(mean_diff_from_center, ys, label='Mean Difference from Center', color='blue')
    plt.xlabel('Mean Difference from Center', fontsize=16)
    plt.ylabel('y (cross section)', fontsize=16)
    plt.legend()

    plt.tight_layout()

    if save_plot:
        plt.savefig(file_path)

    plt.show()

def plot_y_slice_density_multiple(parameter_array, array_of_arrays, save_plot=False, file_path="y_slice_density_multi.png"):
    plt.figure(figsize=(4, 8))
    cmap = get_cmap('magma') 

    for parameter, data in zip(parameter_array, array_of_arrays):
        num_runs, _, grid_size = data.shape

        sum_grid = np.sum(data, axis=0)
        ys = np.linspace(0, 1, grid_size)
        xs = ys.copy()
        center = xs[grid_size // 2]
        xdiff = np.abs(xs - center)

        mean_diff_from_center = np.sum(xdiff[None, :] * sum_grid, axis=1) / np.sum(sum_grid, axis=1)

        plt.plot(mean_diff_from_center, ys, label=str(parameter), color=cmap(parameter/max(parameter_array)))

    plt.xlabel('Mean Difference from Center', fontsize=16)
    plt.ylabel('y (cross section)', fontsize=16)
    plt.legend()

    plt.tight_layout()

    if save_plot:
        plt.savefig(file_path)

    plt.show()


"""
Save/Load npy files
"""
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