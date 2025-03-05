import os
import numpy as np
import matplotlib.pyplot as plt
import random

from numba import njit
from enum import IntEnum
from matplotlib.animation import FuncAnimation

class EnumCellTypes(IntEnum):
    EMPTY_WHITE = 0
    GROWTH_BLACK = 1
    WALK_PATH_GREY = 2
    WALK_FAIL_ORANGE = 3
    WALK_BOUNDARY_RED = 4
    NEW_GROWTH_GREEN = 5
    WALK_START_BLUE = 6

@njit
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
    seed_growth_grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    center = grid_size // 2
    seed_growth_grid[grid_size - 1, center] = 1 # EnumCellTypes.GROWTH_BLACK
    
    return seed_growth_grid

@njit
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

@njit
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
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    available_directions = []

    for x_direction, y_direction in DIRECTIONS:
        # Checks possible x and y values for the next step
        possible_x, possible_y = (x + x_direction) % grid_size, y + y_direction

        # Make the out-of-bounds cells available to randomly select
        if (possible_y >= grid_size or possible_y < 0):
            available_directions.append((x_direction, y_direction))
        # Make only the non-seed growth cells available to randomly select
        elif (0 <= possible_y < grid_size 
            and not seed_growth_grid[possible_y, possible_x]):

            available_directions.append((x_direction, y_direction))

    # If there are no available directions, return the current position
    if not available_directions:
        return x, y

    # Randomly choose a direction from the available directions
    idx = np.random.randint(len(available_directions))
    x_change, y_change = available_directions[idx]

    return (x + x_change) % grid_size, y + y_change

@njit
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
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for x_direction, y_direction in DIRECTIONS:
        new_x = (x + x_direction) % grid_size
        new_y = y + y_direction

        if (0 <= new_y < grid_size # Check if the y value is within the grid
            and seed_growth_grid[new_y, new_x]): # Check if the next step is an active growth site

            if np.random.random() < sticking_prob:
                seed_growth_grid[y, x] = 1 # EnumCellTypes.GROWTH_BLACK
                return True
    
    return False

@njit
def monte_carlo_single_walk(seed_growth_grid, grid_size, sticking_prob):
    """
    Simulates a single walker moving randomly until it sticks to the seed growth, 
    or reaches the bottom/top row, or the walker fails to move.

    Parameters
    ----------
    seed_growth_grid : np.ndarray
        The seed growth grid.
    grid_size : int
        The size of the grid.
    sticking_prob : float
        The probability of the walker sticking to the seed growth.
    
    Returns
    -------
    grid_copy : np.ndarray
        The grid after the walker has moved.
    walk_length : int
        The number of steps the walker took before sticking or failing.
    """
    grid_copy = seed_growth_grid.copy()

    x, y = initialize_walker(grid_size)

    starting_x = x
    starting_y = y

    walk_length = 0

    while True:
        x_new, y_new = random_walk(x, y, grid_size, seed_growth_grid)

        if x_new == x and y_new == y:
            grid_copy[y_new, x_new] = 3 # EnumCellTypes.WALK_FAIL_ORANGE
            break

        if y_new >= grid_size or y_new < 0:
            grid_copy[y, x] = 4 # EnumCellTypes.WALK_BOUNDARY_RED
            break

        x, y = x_new, y_new

        if stick_or_walk(x, y, sticking_prob, seed_growth_grid, grid_size):
            grid_copy[y_new, x_new] = 5 # EnumCellTypes.NEW_GROWTH_GREEN
            break

        grid_copy[y_new, x_new] = 2 # EnumCellTypes.WALK_PATH_GREY

        walk_length += 1
    
    grid_copy[starting_y, starting_x] = 6 # EnumCellTypes.WALK_START_BLUE

    return grid_copy, walk_length

def monte_carlo_sim(grid_size, sticking_prob, max_walkers=1000000):
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
    monte_carlo_data : list
        A list of tuples containing the seed growth grid, the final grid, 
        the starting x and y positions, and the walk length after each walk.
    """
    seed_growth_grid = initialize_grid(grid_size)

    walk_count = 0

    seed_growth_grid_states = np.zeros((max_walkers, grid_size, grid_size), dtype=np.int8)
    walker_final_states = np.zeros((max_walkers, grid_size, grid_size), dtype=np.int8)

    for i in range(max_walkers):
        if np.any(seed_growth_grid[0]):
            print("Seed growth has reached the top row after {} walkers.".format(walk_count))
            break

        else:
            walker_final_states[i], walk_length = monte_carlo_single_walk(seed_growth_grid, grid_size, sticking_prob)
            seed_growth_grid_states[i] = seed_growth_grid.copy()
            walk_count += 1

    return seed_growth_grid_states, walker_final_states, walk_count

def animate_monte_carlo_sim(seed_growth_grid_states, walker_final_states, grid_size, animation_speed=500):
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
    num_walks = 0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'grey', 'orange', 'red', 'green', 'blue'])
    norm = plt.cm.colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 7], cmap.N)
    
    img = ax.imshow(seed_growth_grid_states[0], cmap=cmap, norm=norm, origin="upper", extent=[0, 1, 0, 1])

    def update(frame):

        combined_grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        combined_grid += seed_growth_grid_states[frame]
        combined_grid += walker_final_states[frame]

        img.set_array(combined_grid)
        ax.set_title(f"Walker {frame}")

        if np.all(seed_growth_grid_states[frame] == 0):
            ani.event_source.stop()
            num_walks = frame

        return [img]

    ani = FuncAnimation(fig, update, frames=len(seed_growth_grid_states), repeat=False, interval=animation_speed)
    plt.show()

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
