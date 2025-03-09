import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils import *

def compare_seed_growth_states(eta, p_s, DLA_array, monte_carlo_array):
    DLA_grid = DLA_array[2]
    MC_grid = monte_carlo_array[0]

    fig, axs = plt.subplots(1,2,sharey=True, figsize=[8,4]) 
    plot_grid(DLA_grid, DLA_grid, make_cbar=False, title='DLA', fig=fig, ax=axs[0])
    plot_grid(MC_grid, MC_grid, make_cbar=False, title='Monte Carlo', fig=fig, ax=axs[1])

    fig.tight_layout()
    plt.savefig(os.path.join('results', 'compare_seed_states_DLA_MC'), dpi=600)
    plt.show()
    

def compare_DLA_and_monte_carlo_cross_sections(eta, p_s, DLA_array, monte_carlo_array):
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8,8))

    DLA_num_runs, _, DLA_grid_size = DLA_array.shape
    DLA_sum_grid = np.sum(DLA_array, axis=0)
    DLA_ys = np.linspace(1, 0, DLA_grid_size)
    DLA_mabs = mean_abs_diff(DLA_array)
    DLA_num_cells_per_cross = np.sum(DLA_sum_grid, axis=1) / DLA_num_runs

    MC_num_runs, _, MC_grid_size = monte_carlo_array.shape
    MC_sum_grid = np.sum(monte_carlo_array, axis=0)
    MC_ys = np.linspace(1, 0, MC_grid_size)
    MC_mabs = mean_abs_diff(monte_carlo_array)
    MC_num_cells_per_cross = np.sum(MC_sum_grid, axis=1) / MC_num_runs

    axs[0].plot(DLA_num_cells_per_cross, DLA_ys, label='$\eta =$ ' + str(eta))
    axs[1].plot(DLA_mabs, DLA_ys, label='$\eta =$ ' + str(eta))

    axs[0].plot(MC_num_cells_per_cross, MC_ys, label='$p_s =$ ' + str(p_s))
    axs[1].plot(MC_mabs, MC_ys, label='$p_s =$ ' + str(p_s))    
    
    axs[0].set_ylabel('y', fontsize=16)
    axs[0].grid()
    axs[1].grid()
    axs[0].set_xlabel(r'$\langle N_y \rangle$', fontsize=16)
    axs[1].set_xlabel(r'$\langle |x - x_c|\rangle$', fontsize=16)
    axs[0].legend(loc=4)

    plt.tight_layout() 

    plt.savefig(os.path.join('results', 'cross_section_comparison_eta_{}_p_s_{}.png'.format(eta, p_s)), dpi=600)

    plt.show()

def main():
    etas = [0., 0.125, 0.5, 1., 2., 4.]
    sticking_prob_array = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    array_of_DLA_arrays = [np.load(os.path.join('data', 'many_runs_eta_{}.npy'.format(eta))) for eta in etas]  
    array_of_monte_carlo_arrays = [load_data("final_seed_growth_states_" + str(str(sticking_prob).replace(".", "_")) + "_sp.npy") for sticking_prob in sticking_prob_array]

    """
    eta = 1.0
    p_s = 0.2
    """
    eta = 1.0
    p_s = 0.2
    DLA_array = np.load(os.path.join('data', 'many_runs_eta_{}.npy'.format(eta)))
    monte_carlo_array = load_data("final_seed_growth_states_" + str(str(p_s).replace(".", "_")) + "_sp.npy")

    compare_DLA_and_monte_carlo_cross_sections(eta, p_s, DLA_array, monte_carlo_array)

    compare_seed_growth_states(eta, p_s, DLA_array, monte_carlo_array)



if __name__ == "__main__":
    main()