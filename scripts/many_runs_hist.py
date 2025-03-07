
import sys
sys.path.append('../src')

from dla_fin_diff import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt


def many_runs_experiment(num_runs = 10, eta =2, omega = 1.85):
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
        
    np.save('../data/many_runs_eta_{}'.format(eta), final_grids)
    # plot_grid(c[-1], g[-1])


    # plot_animation(c[:num_iter], g[:num_iter])
    
def mean_abs_diff(grids):
    num_runs, _, grid_size = grids.shape
    x_ind = np.array(range(grid_size))
    xdiff = np.abs(x_ind - grid_size//2)
    diff_grids = xdiff[None, None, :] * grids
    sum_abs_diff = np.sum(diff_grids, axis=-1)
    Ny = np.sum(grids, axis=-1)
    
    mean_abs_diff = np.mean(sum_abs_diff / Ny, axis=0)
    
    return mean_abs_diff
    

def plot_many_runs_experiment(file, skip_ends=1):
    grids = np.load(file)
    mean_abs_diff(grids)
    plt.show()
    # plot_grid(np.mean(grids, axis=0), file=file.replace('data', 'results').replace('npy', 'png'), title=r'$\eta = 2$')
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
    # print(mean)
    # print(msqd)
    # plt.plot(ys[skip_ends:-skip_ends], mabs[skip_ends:-skip_ends], label='$|x-<x>|$')
    # plt.plot(ys[skip_ends:-skip_ends], mean[skip_ends:-skip_ends], label='<x>')
    # plt.legend()
    # plt.show()
    mabs = mean_abs_diff(grids)
    plt.plot(ys[:-skip_ends], num_cells_per_cross[:-skip_ends], label='$N_y$')
    plt.plot(ys[:-skip_ends], mabs[:-skip_ends], label=r'$\langle|x-x_c|\rangle$')
    plt.legend()
    plt.show()
    
def flat_histogram(file):    
    grids = np.load(file)
    num_runs, _, grid_size = grids.shape
    sum_grid = np.sum(grids, axis=0).reshape([-1]) / num_runs
    hist, bins = np.histogram(sum_grid, np.linspace(0,1,20))
    
    print(np.sum(grids) / num_runs)
    plt.bar(bins[:-1], hist, width=0.05)
    plt.yscale('log')
    
    plt.show()
        
if __name__ == '__main__':
    # many_runs_experiment(100, 1, 1.8)
    # plot_many_runs_experiment('../data/many_runs_eta_1.npy')
    
    # flat_histogram('../data/many_runs_eta_4.npy')
    
    plot_many_runs_experiment('../data/many_runs_eta_0.0.npy')
    plot_many_runs_experiment('../data/many_runs_eta_0.5.npy')
    plot_many_runs_experiment('../data/many_runs_eta_1.0.npy')
    plot_many_runs_experiment('../data/many_runs_eta_2.0.npy')
    plot_many_runs_experiment('../data/many_runs_eta_4.0.npy')
    
    # many_runs_experiment(100, 0.5, 1.8)
    # plot_many_runs_experiment('../data/many_runs_eta_0.5.npy')
    
    # eta = float(sys.argv[1])
    # many_runs_experiment(1000,eta, 1.85)