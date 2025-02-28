import numpy as np

from src.monte_carlo import monte_carlo_sim, plot_monte_carlo, animate_monte_carlo_sim

def scrpt_monte_carlo():
    grid_size = 101
    sticking_prob = 0.5
    save_plot = False
    filename = "monte_carlo_experiment_1_0_500000.png"

    seed_growth_grid, stuck_positions, growth_over_time, mean_free_paths, walk_count = monte_carlo_sim(grid_size, sticking_prob)

    plot_monte_carlo(seed_growth_grid, save_plot, filename)

    filename = "monte_carlo_experiment_anim.png"

    seed_growth_grid, stuck_positions, growth_over_time, mean_free_pathsgrid, walk_count = animate_monte_carlo_sim(grid_size, sticking_prob, animation_speed=250)

    plot_monte_carlo(seed_growth_grid, save_plot, filename)
    
if __name__ == "__main__":
    scrpt_monte_carlo()