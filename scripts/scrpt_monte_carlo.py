import numpy as np

from src.monte_carlo import monte_carlo_sim, plot_monte_carlo

def scrpt_monte_carlo():
    grid_size = 100
    num_walkers = 1000
    sticking_prob = 1.0
    save_plot = True
    filename = "monte_carlo_experiment.png"

    grid = monte_carlo_sim(grid_size, num_walkers, sticking_prob)

    plot_monte_carlo(grid, save_plot, filename)

    
if __name__ == "__main__":
    scrpt_monte_carlo()