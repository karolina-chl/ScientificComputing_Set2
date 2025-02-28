import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


x_length = 100
y_length = 100
n_steps = 50
total_time = 1000
time_step_size = 0.1
diffusion_coefficient_u =  0.16
diffusion_coefficient_v =  0.08
U_supply = 0.035
k = 0.060
V_decay = U_supply + k

def init_grids(total_time, time_step_size, n_steps):
    # Step size
    time_step_num = int(total_time/time_step_size)
    
    # # Initiallizing the grid  
    # x_dimension = np.linspace(0, x_length, n_steps) 
    # y_dimension = np.linspace(0, y_length, n_steps)

    chemical_U = np.zeros((time_step_num, n_steps, n_steps))
    chemical_V = np.zeros((time_step_num, n_steps, n_steps))

    # Boundry conditions for U and V
    chemical_U[0,0, :] = 1.0
    chemical_U[0,-1, :] = 0.0
    chemical_V[0,0, :] = 0.0
    chemical_V[0,-1, :] = 0.0

    # Initial conditions for U and V 
    chemical_U[0,:,:] = 0.5
    chemical_V[0,23:28,23:28] = 0.25

    return chemical_U, chemical_V

    
def solve_gray_scott(chemical_U, chemical_V, total_time, time_step_size):
    # Solve 
    x_step_size = x_length/n_steps
    time_step_num = int(total_time/time_step_size)

    stability_value_u = 4*time_step_size*diffusion_coefficient_u/x_step_size**2
    stability_value_v = 4*time_step_size*diffusion_coefficient_v/x_step_size**2
    if (stability_value_u > 1):
        print(f"{stability_value_u }.Stability issue with chemical U, the solution is not stable. Choose different values for U")
    elif (stability_value_v > 1): 
        print(f"{stability_value_v}.Stability issue with chemical V, the solution is not stable. Choose different values for V")
    else: 
        for time in range(time_step_num-1): 
            chemical_U_new = chemical_U[time].copy()
            chemical_V_new = chemical_V[time].copy()
            for rows in range(1, n_steps -1):
                for columns in range(n_steps):
                    #changes in chemical U
                    difussion_component_U = ((diffusion_coefficient_u/(x_step_size**2))*(
                    chemical_U[time, rows + 1, columns] + chemical_U[time, rows -1, columns]
                    + chemical_U[time, rows, (columns + 1)%n_steps] + chemical_U[time, rows, (columns -1)%n_steps] 
                    - 4*chemical_U[time, rows, columns])
                    )

                    reaction_component_U = (
                    - chemical_U[time, rows, columns]*chemical_V[time, rows, columns]**2 
                    + U_supply * (1-chemical_U[time, rows, columns])
                    )
                    
                    chemical_U_new[rows, columns] += time_step_size*(difussion_component_U + reaction_component_U)

                    #changes in chamical V 
                    difussion_component_V = ((diffusion_coefficient_v/(x_step_size**2))*(
                    chemical_V[time, rows + 1, columns] + chemical_V[time, rows -1, columns]
                    + chemical_V[time, rows, (columns + 1)%n_steps] + chemical_V[time, rows, (columns -1)%n_steps] 
                    - 4*chemical_V[time, rows, columns])
                    )

                    reaction_component_V = (
                    chemical_U[time, rows, columns]*chemical_V[time, rows, columns]**2 
                    - (V_decay)* (chemical_V[time, rows, columns])
                    )

                    chemical_V_new[rows, columns] += time_step_size*(difussion_component_V + reaction_component_V)

            chemical_U[time + 1] = chemical_U_new
            chemical_V[time + 1] = chemical_V_new

            # Top and Bottom Boundaries
            chemical_U[time + 1,0, :] = 1.0
            chemical_U[time + 1,-1, :] = 0.0

            chemical_V[time + 1,0, :] = 0.0
            chemical_V[time + 1,-1, :] = 0.0


def plot_animation(c, g = None, frame_steps=1):
    num_steps = c.shape[0]
    print(num_steps)
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

    animation.FuncAnimation(fig, update, frames=range(0, num_steps, frame_steps ), interval=1, blit=False)
    plt.show()

