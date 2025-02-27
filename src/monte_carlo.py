import os
import numpy as np
import matplotlib.pyplot as plt
import random

from numba import njit
from matplotlib.animation import FuncAnimation


DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def initialize_grid(grid_size):
    """
    Initializes the grid with a seed in the center of the bottom row.

    Parameters
    ----------
    grid_size : int
        The size of the grid.
    
    Returns
    -------
    seed_growth_grid : np.ndarray
        The initialized grid.
    """
    seed_growth_grid = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    seed_growth_grid[grid_size - 1, center] = 1
    
    return seed_growth_grid

def initialize_walker(grid_size):
    """
    Initializes the walker at a random x position in the top row.

    Parameters
    ----------
    grid_size : int
        The size of the grid.

    Returns
    -------
    x : int
        The x position of the walker.
    y : int
        The y position of the walker. 0 in this case
    """
    return np.random.randint(0, grid_size), 0

def random_walk(x, y, grid_size, seed_growth_grid):
    """
    Moves the walker in a random direction.
    
    Parameters
    ----------
    x : int
        The initial x position of the walker.
    y : int
        The initial y position of the walker.
    grid_size : int
        The size of the grid.
    seed_growth_grid : np.ndarray
        The seed growth grid.

    Returns
    -------
    x : int
        The new x position of the walker.
    y : int
        The new y position of the walker.
    """
    available_directions = []

    for x_direction, y_direction in DIRECTIONS:
        # Checks possible x and y values for the next step
        possible_x, possible_y = (x + x_direction) % grid_size, y + y_direction

        if (0 <= possible_y):
            available_directions.append((x_direction, y_direction))
            
        elif (0 <= possible_y < grid_size # Check if the y value is within the grid
            and not seed_growth_grid[possible_y, possible_x]): # Check if the next step is not an active growth site

            available_directions.append((x_direction, y_direction))

    # If there are no available directions, return the current position
    if not available_directions:
        return x, y

    # Randomly choose a direction from the available directions
    x_change, y_change = random.choice(available_directions)

    return (x + x_change) % grid_size, y + y_change

def stick_or_walk(x, y, sticking_prob, seed_growth_grid, grid_size):
    """
    Determines if the walker sticks to the seed growth or continues walking.
    If the walker sticks, the seed growth grid is updated.
    If the walker continues walking, the seed growth grid remains unchanged.

    Parameters
    ----------
    x : int
        The x position of the walker.
    y : int
        The y position of the walker.
    sticking_prob : float
        The probability of the walker sticking to the seed growth.
    seed_growth_grid : np.ndarray
        The seed growth grid.
    grid_size : int
        The size of the grid.

    Returns
    -------
    bool
        True if the walker sticks to the seed growth, False otherwise.
    """
    for x_direction, y_direction in DIRECTIONS:
        new_x = (x + x_direction) % grid_size
        new_y = y + y_direction

        if (0 <= new_y < seed_growth_grid.shape[0] # Check if the y value is within the grid
            and seed_growth_grid[new_y, new_x]): # Check if the next step is an active growth site

            if np.random.random() < sticking_prob:
                seed_growth_grid[y, x] = 1
                return True
    
    return False

def monte_carlo_sim(grid_size, num_walkers, sticking_prob):
    """
    Simulates the growth of a seed crystal using a Monte Carlo random walk method.
    Seed starts at the center of the bottom row.
    Walkers start at random x positions in the top row.
    Walkers move randomly until they stick to the seed growth or reach the bottom row.

    Parameters
    ----------
    grid_size : int
        The size of the grid.
    num_walkers : int
        The number of walkers to simulate.
    sticking_prob : float
        The probability of the walker sticking to the seed growth.
    
    Returns
    -------
    seed_growth_grid : np.ndarray
        The final seed growth grid.
    """
    seed_growth_grid = initialize_grid(grid_size)

    for _ in range(num_walkers):
        x, y = initialize_walker(grid_size)

        while True:
            x_new, y_new = random_walk(x, y, grid_size, seed_growth_grid)

            if x_new == x and y_new == y:
                break

            if y_new >= grid_size:
                break

            x, y = x_new, y_new

            if stick_or_walk(x, y, sticking_prob, seed_growth_grid, grid_size):
                break

    return seed_growth_grid

def animate_monte_carlo_sim(grid_size, num_walkers, sticking_prob, animation_speed=500):
    """
    Animates the growth of a seed crystal using a Monte Carlo random walk method.
    Seed starts at the center of the bottom row.
    Walkers start at random x positions in the top row.
    Walkers move randomly until they stick to the seed growth or reach the bottom row.

    Parameters
    ----------
    grid_size : int
        The size of the grid.
    num_walkers : int
        The number of walkers to simulate.
    sticking_prob : float
        The probability of the walker sticking to the seed growth.
    animation_speed : int, optional
        The speed of the animation in milliseconds, by default 500.
    
    Returns
    -------
    seed_growth_grid : np.ndarray
        The final seed growth grid.
    """
    seed_growth_grid = initialize_grid(grid_size)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'grey', 'orange', 'red', 'green', 'blue'])
    norm = plt.cm.colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 7], cmap.N)
    
    img = ax.imshow(seed_growth_grid, cmap=cmap, norm=norm, origin="upper", extent=[0, 1, 0, 1])

    def update(frame):
        nonlocal seed_growth_grid

        grid_copy = seed_growth_grid.copy()

        x, y = initialize_walker(grid_size)

        starting_x = x
        starting_y = y

        grid_copy[starting_y, starting_x] = 6

        while True:
            x_new, y_new = random_walk(x, y, grid_size, seed_growth_grid)

            if x_new == x and y_new == y:
                grid_copy[y_new, x_new] = 3
                break

            if y_new >= grid_size:
                grid_copy[y, x] = 4
                break

            x, y = x_new, y_new

            if stick_or_walk(x, y, sticking_prob, seed_growth_grid, grid_size):
                grid_copy[y_new, x_new] = 5
                break

            grid_copy[y_new, x_new] = 2
            grid_copy[starting_y, starting_x] = 6

        img.set_array(grid_copy)

        return [img]

    ani = FuncAnimation(fig, update, frames=num_walkers, repeat=False, interval=animation_speed)
    plt.show()

    return seed_growth_grid

def plot_monte_carlo(grid, save_plot, filename):
    """
    Plots the final seed growth grid.

    Parameters
    ----------
    grid : np.ndarray
        The final seed growth grid.
    save_plot : bool
        Whether to save the plot.
    filename : str
        The filename to save the plot

    Returns
    -------
    None
    """
    os.makedirs("results", exist_ok=True)
    file_location = os.path.join("results", filename)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray_r", origin="upper", extent=[0, 1, 0, 1])
    plt.axis("off")

    if save_plot:
        plt.savefig(file_location, dpi=300)
        print(f"Plot saved as {file_location}")
    else:
        plt.show()