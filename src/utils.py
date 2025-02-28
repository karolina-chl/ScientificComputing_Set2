import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_grid(c, growth=None):
    grid_size = c.shape[-1]
    fig, ax = plt.subplots()
    heatmap = ax.imshow(c, cmap="hot", extent=[0, 1, 0, 1])
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Concentration")
    if growth is not None:      
        heatmap = ax.imshow(growth, alpha=growth, cmap='tab20b',  extent=[0, 1, 0, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Equilibrium Diffusion")



    
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