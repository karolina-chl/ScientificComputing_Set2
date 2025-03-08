import os
import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from enum import IntEnum
from matplotlib.animation import FuncAnimation
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    successful_walk : bool
        True if the walker stuck to the seed growth, False
        otherwise.
    stop_type : str
        The reason the walker stopped
    """
    grid_copy = np.zeros((grid_size, grid_size), dtype=np.int8)
    #seed_growth_grid.copy()

    x, y = initialize_walker(grid_size)

    starting_x = x
    starting_y = y

    walk_length = 0
    successful_walk = False
    stop_type = None

    while True:
        x_new, y_new = random_walk(x, y, grid_size, seed_growth_grid)

        if x_new == x and y_new == y:
            grid_copy[y_new, x_new] = 3 # EnumCellTypes.WALK_FAIL_ORANGE
            stop_type = "fail"
            break

        if y_new >= grid_size or y_new < 0:
            grid_copy[y, x] = 4 # EnumCellTypes.WALK_BOUNDARY_RED
            stop_type = "boundary"
            break

        x, y = x_new, y_new

        if stick_or_walk(x, y, sticking_prob, seed_growth_grid, grid_size):
            grid_copy[y_new, x_new] = 5 # EnumCellTypes.NEW_GROWTH_GREEN
            successful_walk = True
            stop_type = "stick"
            break

        grid_copy[y_new, x_new] = 2 # EnumCellTypes.WALK_PATH_GREY

        walk_length += 1
    
    grid_copy[starting_y, starting_x] = 6 # EnumCellTypes.WALK_START_BLUE

    return grid_copy, walk_length, successful_walk, stop_type

def monte_carlo_sim(grid_size, sticking_prob, max_walkers=100000):
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
    results : dict
        A dictionary containing the seed growth grid states after each walk,
        the final walker states after each walk, and the walk length statistics.
    walk_count : int
        The number of walkers simulated.
    successful_walks : int
        The number of walkers that successfully stuck to the seed growth
    """
    seed_growth_grid = initialize_grid(grid_size)

    walk_count = 0
    successful_walks = 0

    results = {
        "seed_growth_grid_states": np.zeros((max_walkers, grid_size, grid_size), dtype=np.int8),
        "walker_final_states": np.zeros((max_walkers, grid_size, grid_size), dtype=np.int8),
        "walk_length_stats": np.zeros(max_walkers, dtype=np.int32),
        "successful_seed_growth_grid_states": np.zeros((max_walkers, grid_size, grid_size), dtype=np.int8),
        "successful_walker_final_states": np.zeros((max_walkers, grid_size, grid_size), dtype=np.int8),
        "successful_walk_length_stats": np.zeros(max_walkers, dtype=np.int32),
        "stop_types": np.zeros(max_walkers, dtype=str)
    }

    while True:
        if successful_walks >= max_walkers:
            print("Ran out of storage space for successful walkers.")
            break

        if walk_count == max_walkers:
            print("Ran out of storage space for all walkers.")
            print("Only storing successful walkers.")

        if np.any(seed_growth_grid[0]):
            print("Seed growth has reached the top row after {} walkers.".format(walk_count))
            break
        else:
            (walker_final_state_single, 
             walk_length, 
             successful_walk, 
             results["stop_types"]) = monte_carlo_single_walk(seed_growth_grid, grid_size, sticking_prob)

            if successful_walk:
                # Stores only successful sticks
                # If the max_walkers is reached, the successful sticks continue being stored
                # until the the number of successful sticks reaches the max_walkers
                results["successful_seed_growth_grid_states"][successful_walks] = seed_growth_grid.copy()
                results["successful_walker_final_states"][successful_walks] = walker_final_state_single
                results["successful_walk_length_stats"][successful_walks] = walk_length
                successful_walks += 1
            elif walk_count < max_walkers:
                # Stores all successful and unsuccessful sticks
                # If the max_walkers is reached, the unsuccessful sticks stop being stored
                results["seed_growth_grid_states"][walk_count] = seed_growth_grid.copy()
                results["walker_final_states"][walk_count] = walker_final_state_single
                results["walk_length_stats"][walk_count] = walk_length


            walk_count += 1

    return results, walk_count, successful_walks

def monte_carlo_sim_final_state_only(grid_size, sticking_prob, iterations_to_save = 25000):
    """
    Simulates the growth of a seed crystal using a Monte Carlo random walk method.
    Seed starts at the center of the bottom row.
    Walkers start at random x positions in the top row.
    Walkers move randomly until they stick to the seed growth or reach the bottom row.
    This function saves statistical data for the final state of of the simulation.

    Parameters
    ----------
    grid_size : int
        The size of the grid.
    sticking_prob : float
        The probability of the walker sticking to the seed growth.
    iterations_to_save : int
        The number of iterations to save the growth over time.

    Returns
    -------
    seed_growth_grid : np.ndarray
        The final seed growth grid.
    walk_count : int
        The number of walkers simulated.
    successful_walk_count : int
        The number of walkers that successfully stuck to the seed growth.
    avg_walk_length : float
        The average walk length for all walkers.
    avg_successful_walk_length : float
        The average successful walk length for all successful walkers.
    growth_over_time : np.ndarray
        The growth over time for the simulation.
    """
    seed_growth_grid = initialize_grid(grid_size)

    walk_count = 0
    successful_walk_count = 0
    failed_walks = 0
    boundary_walks = 0

    walk_length_sum = 0
    successful_walk_length_sum = 0

    growth_over_time = np.full(iterations_to_save, np.nan)

    growth_over_time[0] = np.sum(seed_growth_grid)

    while True:
        if np.any(seed_growth_grid[0]):
            print("Seed growth has reached the top row after {} walkers.".format(walk_count))
            break
        else:
            (_, 
             walk_length, 
             successful_walk, 
             stop_type) = monte_carlo_single_walk(seed_growth_grid, grid_size, sticking_prob)

            if successful_walk:
                successful_walk_count += 1
                successful_walk_length_sum += walk_length
            
            if stop_type == "fail":
                failed_walks += 1
            elif stop_type == "boundary":
                boundary_walks += 1

            walk_count += 1
            walk_length_sum += walk_length

            if walk_count % 10 == 0:
                growth_over_time[walk_count//10] = np.sum(seed_growth_grid)

    avg_walk_length = walk_length_sum / walk_count
    avg_successful_walk_length = successful_walk_length_sum / successful_walk_count

    return seed_growth_grid, walk_count, successful_walk_count, avg_walk_length, avg_successful_walk_length, growth_over_time

def animate_monte_carlo_sim(seed_growth_grid_states, 
                            walker_final_states, 
                            grid_size,
                            save_animation=False,
                            filename="monte_carlo_animation.mp4", 
                            animation_speed=500):
    """
    Animates the growth of a seed crystal using a Monte Carlo random walk method.
    Seed starts at the center of the bottom row.
    Walkers start at random x positions in the top row.
    Walkers move randomly until they stick to the seed growth or reach the bottom row.

    Parameters
    ----------
    seed_growth_grid_states : np.ndarray
        The seed growth grid states after each walk.
    walker_final_states : np.ndarray
        The final walker states after each walk.
    grid_size : int
        The size of the grid.
    save_animation : bool
        Whether to save the animation.
    filename : str
        The filename to save the animation.
    animation_speed : int
        The speed of the animation in milliseconds.
    
    Returns
    -------
    None
    """

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

        return [img]

    ani = FuncAnimation(fig, update, frames=len(seed_growth_grid_states), repeat=False, interval=animation_speed)
    
    if save_animation:
        os.makedirs("results", exist_ok=True)
        filename = os.path.join("results", filename)
        ani.save(filename, writer='ffmpeg', fps=1000/animation_speed)
    else:
        plt.show()


def run_multiple_simulations(grid_size, 
                             sticking_prob, 
                             num_simulations, 
                             iterations_to_save=25000):
    """
    Runs multiple Monte Carlo simulations.

    Parameters
    ----------
    grid_size : int
        The size of the grid.
    sticking_prob : float
        The probability of the walker sticking to the seed growth.
    num_simulations : int
        The number of simulations to run.
    iterations_to_save : int
        The number of iterations to save the growth over time.
    
    Returns
    -------
    final_seed_growth_states : list
        A list of the final seed growth states for each simulation.
    all_walk_counts : list
        A list of the number of walkers simulated for each simulation.
    all_successful_walks : list
        A list of the number of successful walkers for each simulation.
    all_avg_walk_lengths : list
        A list of the average walk lengths for each simulation.
    all_avg_successful_walk_lengths : list
        A list of the average successful walk lengths for each simulation.
    all_growth_over_time : list
        A list of the growth over time for each simulation.
    """
    final_seed_growth_states = np.zeros((num_simulations, grid_size, grid_size), dtype=np.int8)
    all_walk_counts = np.zeros(num_simulations, dtype=np.int32)
    all_successful_walks = np.zeros(num_simulations, dtype=np.int32)
    all_avg_walk_lengths = np.zeros(num_simulations, dtype=np.float32)
    all_avg_successful_walk_lengths = np.zeros(num_simulations, dtype=np.float32)
    all_growth_over_time = np.zeros((num_simulations, iterations_to_save), dtype=np.float32)


    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(monte_carlo_sim_final_state_only, grid_size, sticking_prob) for _ in range(num_simulations)]
        for i, future in enumerate(as_completed(futures)):
            sim_result = future.result()
            print(f"Simulation {i+1} complete.")
            final_seed_growth_states[i] = sim_result[0]
            all_walk_counts[i] = sim_result[1]
            all_successful_walks[i] = sim_result[2]
            all_avg_walk_lengths[i] = sim_result[3]
            all_avg_successful_walk_lengths[i] = sim_result[4]
            all_growth_over_time[i] = sim_result[5]

    return (final_seed_growth_states, 
            all_walk_counts, 
            all_successful_walks, 
            all_avg_walk_lengths, 
            all_avg_successful_walk_lengths, 
            all_growth_over_time)