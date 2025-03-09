import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit


def init_grids(total_time, time_step_size, n_steps):
    """
    Initiallizes two grids, one for chemical U and one for chemical V, both have identical size. 
    As a initial condition, U grid is set to 0.5 everywhere, V grid has a small square equal to 0.25 in the middle. 
    Boundry conditions are implemented in the grid. 

    Parameters
    ----------
    total_time : int
        Total time of the simulation.
    time_step_size : int
        The size of time step size.
    n_steps : int
        The size of the grid. 
        
    Returns
    -------
    chemical_U : np.ndarray
        The grid for U chemical, with implemented initial conditions and boundry conditions. 
    
    chemical_V : np.ndarray
        The grid for V chemical, with implemented initial conditions and boundry conditions.
    """
    # Step size
    time_step_num = int(total_time/time_step_size)

    chemical_U = np.zeros((time_step_num, n_steps, n_steps))
    chemical_V = np.zeros((time_step_num, n_steps, n_steps))

    # Initial conditions for U and V 
    chemical_U[0,:,:] = 0.5
    chemical_V[0,45:55,45:55] = 0.25

    return chemical_U, chemical_V


@njit   
def solve_gray_scott(chemical_U, chemical_V, total_time, time_step_size, x_length, n_steps, diffusion_coefficient_u, diffusion_coefficient_v, U_supply, k, noise_boundry):
    """
    Simulates gray_scott model in two dimensions. Both horizontally and vertically the grids have periodic boundries. Noise can be introduced into the model.  

    Parameters
    ----------
    chemical_U : np.ndarray
        The grid for U chemical, with implemented initial conditions and boundry conditions. 
    chemical_V : np.ndarray
        The grid for V chemical, with implemented initial conditions and boundry conditions.
    total_time : int
        Total time of the simulation.
    time_step_size : int
        The size of time step size.
    x_length : int
        The max value of X. 
    n_steps : int
        The size of the grid. 
    diffusion_coefficient_u: float
        Diffusion coefficient for chemical U. 
    diffusion_coefficient_v: float 
        Diffusion coefficient for chemical V. 
    U_supply : float
        The rate at which U is supplied. 
    k : float 
        k parameter. The sum (f + k) controls the rate at which chemical V decays. 
    noise_boundry : float
        The boudry of the uniform distribution from which the noise comes form. If set to zero, no noise is introduced into the model. 


    Returns
    -------
    None
    """
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
            for rows in range(n_steps):
                for columns in range(n_steps):
                    #changes in chemical U
                    difussion_component_U = ((diffusion_coefficient_u/(x_step_size**2))*(
                    chemical_U[time, (rows + 1)%n_steps, columns] + chemical_U[time, (rows -1)%n_steps, columns]
                    + chemical_U[time, rows, (columns + 1)%n_steps] + chemical_U[time, rows, (columns -1)%n_steps] 
                    - 4*chemical_U[time, rows, columns])
                    )

                    reaction_component_U = (
                    - chemical_U[time, rows, columns]*chemical_V[time, rows, columns]**2 
                    + U_supply * (1-chemical_U[time, rows, columns])
                    )
                    noise = np.random.uniform(-noise_boundry,noise_boundry,1)
                    noise_U = noise.item()

                    chemical_U_new[rows, columns] = chemical_U_new[rows, columns] + time_step_size*(difussion_component_U + reaction_component_U + noise_U)

                    #changes in chemical V 
                    difussion_component_V = ((diffusion_coefficient_v/(x_step_size**2))*(
                    chemical_V[time, (rows + 1)%n_steps, columns] + chemical_V[time, (rows -1)%n_steps, columns]
                    + chemical_V[time, rows, (columns + 1)%n_steps] + chemical_V[time, rows, (columns -1)%n_steps] 
                    - 4*chemical_V[time, rows, columns])
                    )

                    reaction_component_V = (
                    chemical_U[time, rows, columns]*chemical_V[time, rows, columns]**2 
                    - (U_supply + k)* (chemical_V[time, rows, columns])
                    )
                    noise = np.random.uniform(-noise_boundry,noise_boundry,1)
                    noise_V = noise.item()

                    chemical_V_new[rows, columns] = chemical_V_new[rows, columns] + time_step_size*(difussion_component_V + reaction_component_V + noise_V)

            chemical_U[time + 1] = chemical_U_new
            chemical_V[time + 1] = chemical_V_new

def plot_animation(c):
    """
    Plots animation for gray scott simulation.   

    Parameters
    ----------
    c : np.ndarray
        The final gray scott concentration grid.
    -------
    None
    """
    frame_steps=1
    num_steps = c.shape[0]
    fig, ax = plt.subplots()
    heatmap = ax.imshow(c[0], cmap="hot", extent=[0, 1, 0, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    cbar = plt.colorbar(heatmap)
    cbar.set_label("Concentration")

    def update(frame):
        heatmap.set_array(c[frame]) 
        ax.set_title(f"Gray Scott (frame = {frame})")
        return heatmap,

    anim = animation.FuncAnimation(fig, update, frames=range(0, num_steps, frame_steps ), interval=1, blit=False)
    plt.show()

    return anim


def last_frame_gray_scott_save(c, filename):
    """
    Returns and saves the last frame of the Gray-Scott simulation in high quality.

    Parameters
    ----------
    c : np.ndarray
        The final Gray-Scott concentration grid.
    filename : str, optional
        The name of the output image file.
    dpi : int, optional
        Dots per inch for high-resolution output.

    Returns
    -------
    np.ndarray
        The last frame of the simulation.
    """
    last_frame = c[-1]  # Get the last time step

    fig, ax = plt.subplots(figsize = (12,10))
    image = ax.imshow(last_frame, cmap="hot", extent=[0, 1, 0, 1])
    ax.set_xlabel("X", fontsize = 35)
    plt.xticks([0,1],fontsize=35)
    ax.set_ylabel("Y", fontsize = 35)
    plt.yticks([0,1], fontsize=35)

    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=35)
    cbar.set_label("Concentration", fontsize = 35)
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    return last_frame

