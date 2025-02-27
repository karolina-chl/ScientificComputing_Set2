import numpy as np

from src.monte_carlo import monte_carlo_sim, plot_monte_carlo, animate_monte_carlo_sim

def scrpt_monte_carlo():
    grid_size = 101
    num_walkers = 2500
    sticking_prob = 1.0
    save_plot = True
    filename = "monte_carlo_experiment.png"

    grid = monte_carlo_sim(grid_size, num_walkers, sticking_prob)

    plot_monte_carlo(grid, save_plot, filename)

    grid = animate_monte_carlo_sim(grid_size, num_walkers, sticking_prob, animation_speed=500)

    
if __name__ == "__main__":
    scrpt_monte_carlo()