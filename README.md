# Scientific Computing Set 2

Victoria Peterson - 15476758

Paul Jungnickel - 15716554

Karolina Chłopicka - 15716546

## Table of Contents

1. [Overview](#overview)
2. [Usage and Installation](#usage-and-installation)
3. [Implementation](#implementation)
4. [Contributing](#contributing)
5. [License](#license)

## Overview

- **DLA**: Diffusion Limited Aggregation simulations and analysis.
- **Monte Carlo**: Monte Carlo random walk simulations and analysis.
- **Gray-Scott**: Gray-Scott model simulations and analysis. To generate and save all the plots, run the `scripts/script_gray_scott.py` file. The animation will appear, and all images will be saved to the results folder.

## Usage and Installation

To run the simulations and generate plots, follow these steps:

1. Clone the repository:

    ```sh
    gh repo clone karolina-chl/ScientificComputing_Set2
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**: Ensure you have all the required dependencies installed. You can install them using the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

4. **Initialize Directories as Packages**: This step will ensure that all function imports from different directories are recognized by initializing the directories as packages in the virtual environment. This step may be unnecessary if your IDE automatically established the environmental variable `PYTHON_PATH` for your imports:

    ```sh
    pip install -e .
    ```

5. **Run Simulations**: Use the provided scripts to run the simulations. For example, to run a single Monte Carlo simulation, modify and run:

    ```sh
    python scripts/script_monte_carlo_single.py
    ```

6. **Generate Plots**: Use the provided utility functions to generate plots from the simulation results. For example:

    ```python
    from src.utils import generate_heatmap, plot_histogram

    # Example usage
    generate_heatmap(data, "Title", "Colorbar Label", save_plot=True, plot_file_name="heatmap.png")
    plot_histogram(data, "Title", "X Label", "Y Label", save_plot=True, plot_file_name="histogram.png")
    ```

## Implementation

The Diffusion Limited Aggregation, Monte Carlo Random Walk, and Gray-Scott models are implemented using Python and consist of several modules to handle different aspects of the simulation and analysis. Below is an overview of the main components:

### Source Code

- `src/finite_difference.py`: Contains the implementation of the finite difference method for time-independent diffusion.
- `src/dla_fin_diff.py`: Contains the implementation of the Diffusion Limited Aggregation model using finite difference methods.
- `src/gray_scott.py`: Contains the implementation of the Gray-Scott model.
- `src/monte_carlo.py`: Contains the implementation of the Monte Carlo random walk simulation.
- `src/utils.py`: Contains utility functions for plotting and saving data.

### Scripts

- `scripts/script_gray_scott.py`: Script to run Gray-Scott simulations and generate plots.
- `scripts/script_monte_carlo_single.py`: Script to run a single Monte Carlo simulation with variable parameters.
- `scripts/script_monte_carlo_sim_multi.py`: Script to run multiple Monte Carlo simulations and save the results with variable parameters.
- `scripts/script_monte_carlo_plot_multi.py`: Script to read Monte Carlo Simulation data output from `script_monte_carlo_sim_multi.py` and generate plots.
- `scripts/single_run_dla.py`: Script to run a single Diffusion Limited Aggregation simulation.
- `scripts/optimal_omega.py`: Script to find the optimal omega for the Diffusion Limited Aggregation model.
- `scripts/many_runs_hist.py`: Script to run multiple Diffusion Limited Aggregation simulations and generate histograms.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.