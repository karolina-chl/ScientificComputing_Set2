import os
import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit

DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def initialize_grid(grid_size):
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    grid[grid_size - 1, grid_size // 2] = 1
    
    return grid

def create_walker(grid_size):
    return np.random.randint(0, grid_size), 0

def random_walk(x, y, grid_size, grid):
    available_directions = []

    for x_direction, y_direction in DIRECTIONS:
        # Checks possible x and y values for the next step
        possible_x, possible_y = (x + x_direction) % grid_size, y + y_direction

        # If the next step is within the grid, 
        # and is not an active growth site
        if 0 <= possible_y < grid_size and not grid[possible_y, possible_x]: 
            # Add the direction to the available directions
            available_directions.append((x_direction, y_direction))

    if not available_directions:
        return x, y

    x_change, y_change = random.choice(available_directions)
    return (x + x_change) % grid_size, y + y_change 

def stick_or_walk(x, y, sticking_prob, grid, grid_size):
    for x_direction, y_direction in DIRECTIONS:
        new_x, new_y = (x + x_direction) % grid_size, y + y_direction

        if 0 <= new_y < grid.shape[0] and 0 <= new_x < grid.shape[1] and grid[new_y, new_x]:
            if np.random.random() < sticking_prob:
                grid[y, x] = 1
                return True
    
    return False

def monte_carlo_sim(grid_size, num_walkers, sticking_prob):
    grid = initialize_grid(grid_size)

    for _ in range(num_walkers):
        x, y = create_walker(grid_size)

        while True:
            x_new, y_new = random_walk(x, y, grid_size, grid)

            if x_new == x and y_new == y:
                break

            x, y = x_new, y_new

            if y >= grid_size:
                break

            if stick_or_walk(x, y, sticking_prob, grid, grid_size):
                break

    return grid

def plot_monte_carlo(grid, save_plot, filename):
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