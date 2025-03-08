import os
import numpy as np
import matplotlib.pyplot as plt

from src.dla_fin_diff import *
from src.utils import *

def many_runs_experiment(num_runs = 10, eta =2, omega = 1.85):
    """
    simulates a number of runs of the dla model, saving the final growth after reaching the top
    params:
        num_runs:   num_runs
        eta:        dla model parameter
        omega:      finite difference solver parameter
        
    returns:
        final grids saved as a numpy array
    """  
    np.random.seed(43)
    set_numba_seed(np.random.randint(1000000000))
    grid_size = 100
    initial_cond = np.zeros([grid_size, grid_size])
    initial_cond[-2, grid_size//2] = 1

    final_grids = np.zeros([num_runs, grid_size, grid_size])
    
    for run in range(num_runs):
        g, c , num_iter, total_sor_iter= dla_growth(eta, omega, initial_cond, growth_steps=10000)
        print(total_sor_iter)
        final_grids[run] = g[num_iter]
        
    np.save(os.path.join('data', 'many_runs_eta_{}'.format(eta)), final_grids)
    
    

def plot_many_runs_experiment(file, skip_ends=1):
    """
    plot the histogram of cell occupancy for a timeseries of dla runs with one constant set of parameters
    params:
        file:   location where the list of grids is stored
        skip_ends: ignore the first / last n rows of the grid
        
        
    returns:
        histogram of cell occupancy
        mean / mean abs difference from centerline plot
        mean number of cells in each row plot
    """  
    grids = np.load(file)
    mean_abs_diff(grids)
    plt.show()
    # plot_grid(np.mean(grids, axis=0), file=file.replace('data', 'results/diffusion_limited_aggregation').replace('npy', 'png'), title=r'$\eta = 2$')
    num_runs, _, grid_size = grids.shape
    sum_grid = np.sum(grids, axis=0)
    ys = np.linspace(0,1,grid_size)
    xs = ys.copy()
    center = xs[grid_size//2]
    print(center)
    # xdiff = np.abs(xs - center)
    x_ind = np.array(range(grid_size))
    xdiff = np.abs(x_ind - grid_size//2)
    
    plt.imshow(sum_grid/num_runs)
    plt.show()
    
    num_cells_per_cross = np.sum(sum_grid, axis=1) / num_runs
    # plt.plot(ys, xdiff)
    # mabs = np.sum(xdiff[None, :]*sum_grid/num_runs, axis=1) / num_cells_per_cross[:,None]
    # plt.imshow(xdiff[None, :]*sum_grid)
    # mean = np.sum(xs[None, :]*sum_grid/np.sum(sum_grid, axis=1)[:,None], axis=1)
    # plt.plot(ys[skip_ends:-skip_ends], mabs[skip_ends:-skip_ends], label='$|x-<x>|$')
    # plt.plot(ys[skip_ends:-skip_ends], mean[skip_ends:-skip_ends], label='<x>')
    # plt.legend()
    # plt.show()
    mabs = mean_abs_diff(grids)
    plt.plot(ys[:-skip_ends], num_cells_per_cross[:-skip_ends], label='$N_y$')
    plt.plot(ys[:-skip_ends], mabs[:-skip_ends], label=r'$\langle|x-x_c|\rangle$')
    plt.legend()
    plt.show()
    
        
if __name__ == '__main__':
    #many_runs_experiment(100, 1, 1.8)
    plot_many_runs_experiment(os.path.join('data', 'many_runs_eta_1.npy'))
    
    
    flat_histogram(np.load(os.path.join('data', 'many_runs_eta_4.0.npy')), 
                   'Flat histogram of final seed growth states', 
                   'Cell Occupation Probability', 
                   'Frequency of grid cells', 
                   save_plot=True, 
                   file_path=os.path.join('results', 'diffusion_limited_aggregation', 'histogram_many_runs_eta_1.png'))
    
    
    #many_runs_experiment(100, 0.5, 1.8)
    plot_many_runs_experiment(os.path.join('data', 'many_runs_eta_0.5.npy'))
    
    
    #run the experiments from the shell file
    # eta = float(sys.argv[1])
    # many_runs_experiment(1000,eta, 1.85)
    
    # visualize the data
    # plot_many_runs_experiment(os.path.join('data', 'many_runs_eta_0.0.npy'))
    # plot_many_runs_experiment(os.path.join('data', 'many_runs_eta_0.125.npy'))
    # plot_many_runs_experiment(os.path.join('data', 'many_runs_eta_0.5.npy'))
    # plot_many_runs_experiment(os.path.join('data', 'many_runs_eta_1.0.npy'))
    # plot_many_runs_experiment(os.path.join('data', 'many_runs_eta_2.0.npy'))
    
    
    #analyze data of the many runs experiment
    etas = [0., 0.125, 0.5, 1., 2., 4.]

    array_of_arrays = [np.load(os.path.join('data', 'many_runs_eta_{}.npy'.format(eta))) for eta in etas]   

    plot_cross_section_and_deviation_multiple(etas,
                                              array_of_arrays,
                                              parameter_name=r'$\eta$',
                                              save_plot=True, 
                                              file_path=os.path.join('results', 'diffusion_limited_aggregation', 'cross_section_and_deviation_all.png'))