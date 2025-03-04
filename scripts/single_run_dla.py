
import sys
sys.path.append('../src')

from dla_fin_diff import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def plot_single_run_with_eta(etas):
    
    fig, axs = plt.subplots(1,3,sharey=True, figsize=[15,5]) 
    for i, eta in enumerate(etas):
        np.random.seed(42)
        set_numba_seed(np.random.randint(1000000000))
        grid_size = 100
        initial_cond = np.zeros([grid_size, grid_size])
        initial_cond[-2, grid_size//2] = 1
        # plot_grid(initial_cond)
                        
        # eta =2
        omega = 1.85
        
        g, c , num_iter, total_sor_iter= dla_growth(eta, omega, initial_cond, growth_steps=10000)
            
        plot_grid(c[num_iter-1], g[num_iter-1],make_cbar=False, title=r'$\eta={}$'.format(eta), fig=fig, ax=axs[i])

        print(total_sor_iter)       

        # plot_animation(c[:num_iter], g[:num_iter])
    fig.tight_layout()
    plt.savefig('../plots/run_with_eta_{}.png', dpi=600)
    plt.show()
        
        
if __name__ == '__main__':
    plot_single_run_with_eta([0.5, 1, 2])