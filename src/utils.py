import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import get_cmap
import numpy as np

def plot_grid(c, growth=None, file=None, title='', make_cbar=True, fig=None, ax=None ):
    
    # plt.tight_layout()
    
    
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    
    if file is not None:
        plt.savefig(file, dpi=600)

    if return_ax:
        return fig, ax
    plt.show()
    
    

def plot_animation(c, g = None, frame_steps=1):
    num_steps = c.shape[0]
    fig, ax = plt.subplots()
    heatmap = ax.imshow(c[0], cmap="hot", extent=[0, 1, 0, 1])
    if g is not None:
        growthmap = ax.imshow(g[0], alpha=g[0], cmap='tab20b',  extent=[0, 1, 0, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Equilibrium Diffusion")

    cbar = plt.colorbar(heatmap)
    cbar.set_label("Concentration")

    def update(frame):
        heatmap.set_array(c[frame]) 
        if g is not None:
            growthmap.set_array(g[frame])
            growthmap.set_alpha(g[frame])
        ax.set_title(f"Equilibrium Diffusion (frame = {frame})")
        return heatmap,

    ani = animation.FuncAnimation(fig, update, frames=range(0, num_steps, frame_steps ), interval=50, blit=False)
    plt.show()
    
    
    
def mean_abs_diff(grids):
    num_runs, _, grid_size = grids.shape
    x_ind = np.array(range(grid_size))
    xdiff = np.abs(x_ind - grid_size//2)
    diff_grids = xdiff[None, None, :] * grids
    sum_abs_diff = np.sum(diff_grids, axis=-1)
    Ny = np.sum(grids, axis=-1)
    
    mean_abs_diff = np.mean(sum_abs_diff / Ny, axis=0)
    mean_abs_diff /= grid_size
    
    return mean_abs_diff

"""
    Should work for arbitrary grids, hopefully default params are still working with MC
"""
def plot_cross_section_and_deviation_multiple(parameter_array,
                                            load_file_name,
                                            comma_separator='_',
                                            load_file_ending='_sp.npy',
                                            parameter_name=r'$p_c$',
                                            save_plot=False, 
                                            plot_file = "results/monte_carlo/cross_section_plot.png"):
    
    fig, axs = plt.subplots(1, 2,sharey=True, figsize=(8,8))
    cmap = get_cmap('magma')  # Use a colormap with easily distinguishable colors
    
    

    for i, parameter in enumerate(parameter_array):
        sticking_prob_str = str(parameter).replace(".", comma_separator)
        data = np.load(load_file_name + str(sticking_prob_str) + load_file_ending)
        num_runs, _, grid_size = data.shape

        sum_grid = np.sum(data, axis=0)
        ys = np.linspace(1,0,grid_size)
        # xs = ys.copy()
        # center = xs[grid_size//2]
        
        mabs = mean_abs_diff(data)

        num_cells_per_cross = np.sum(sum_grid, axis=1) / num_runs
        axs[0].plot(num_cells_per_cross, ys, label=str(parameter), color=cmap(i/len(parameter_array)))
        axs[1].plot(mabs, ys, label=str(parameter), color=cmap(i/len(parameter_array)))
    
    axs[0].set_ylabel('y', fontsize=16)
    axs[0].grid()
    axs[1].grid()
    axs[0].set_xlabel(r'$\langle N_y \rangle$', fontsize=16)
    axs[1].set_xlabel(r'$\langle |x - x_c|\rangle$', fontsize=16)
    axs[0].legend(title=parameter_name, loc=4)

    plt.tight_layout() 

    if save_plot:
        plt.savefig(plot_file, dpi=600)
    
    plt.show()