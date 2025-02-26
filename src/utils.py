import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_grid(c, growth=None):
    grid_size = c.shape[-1]
    fig, ax = plt.subplots()
    heatmap = ax.imshow(c, cmap="hot", extent=[0, 1, 0, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Equilibrium Diffusion")

    cbar = plt.colorbar(heatmap)
    cbar.set_label("Concentration")


    
    plt.show()
    
    

def plot_animation(grids, frame_steps=1):
    num_steps = grids.shape[0]
    fig, ax = plt.subplots()
    heatmap = ax.imshow(grids, cmap="hot", extent=[0, 1, 0, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Equilibrium Diffusion")

    cbar = plt.colorbar(heatmap)
    cbar.set_label("Concentration")

    def update(frame):
        heatmap.set_array(grids[frame]) 
        ax.set_title(f"Equilibrium Diffusion (frame = {frame})")
        return heatmap,

    ani = animation.FuncAnimation(fig, update, frames=range(0, num_steps, frame_steps ), interval=50, blit=False)
    plt.show()