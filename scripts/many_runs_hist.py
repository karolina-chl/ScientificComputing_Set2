
import sys
sys.path.append('../src')

from dla_fin_diff import *
from utils import *
import numpy as np


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

def plot_many_runs_experiment(file):
    grids = np.load(file)
    plot_grid(np.mean(grids, axis=0), file=file.replace('data', 'plots').replace('npy', 'png'), title='$\eta = 2$')
        
if __name__ == '__main__':
    # many_runs_experiment(100, 1, 1.8)
    plot_many_runs_experiment('../data/many_runs_eta_1.npy')