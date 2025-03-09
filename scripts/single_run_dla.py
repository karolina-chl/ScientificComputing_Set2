import os
from src.dla_fin_diff import *
from src.utils import *
import numpy as np
import matplotlib.pyplot as plt

def plot_single_run_with_eta(etas):
    """
    simulates and plots the growth and nutrient concentration after reaching the top for 3 different parameter choices of eta
    params:
        etas: length-3 array containing 3 eta values
        
    returns:
        plot of the growth cluster / concentration
    """  
    
    fig, axs = plt.subplots(1,3,sharey=True, figsize=[15,5]) 
    for i, eta in enumerate(etas):
        np.random.seed(42)
        set_numba_seed(np.random.randint(1000000000))
        grid_size = 100
        initial_cond = np.zeros([grid_size, grid_size])
        initial_cond[-2, grid_size//2] = 1
        omega = 1.85
        
        g, c , num_iter, total_sor_iter= dla_growth(eta, omega, initial_cond, growth_steps=10000)
            
        plot_grid(c[num_iter-1], g[num_iter-1],make_cbar=False, title=r'$\eta={}$'.format(eta), fig=fig, ax=axs[i])

        print(total_sor_iter)       
        
    fig.tight_layout()
    plt.savefig(os.path.join('results', 'diffusion_limited_aggregation', 'run_with_eta_{}.png'.format('_'.join(map(str, etas)))), dpi=600)
    plt.show()


def main():
    plot_single_run_with_eta([0.5, 1, 2])
        
        
if __name__ == '__main__':
    main()
