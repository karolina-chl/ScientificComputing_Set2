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

5. **Run Simulations**: You can run the simulations using either script entry points, general function usage, or by directly modifying and running the source file functions.

    - **Using Script Entry Points**: Use the provided script entry points to run different simulations depending on parameters defined in their script files:

        ```sh
        gray_scott # Runs a the Gray-Scott model and outputs the result
        single_run_dla # Runs the single general DLA model and outputs a multi-step growth/diffusion plot
        optimal_omega # Runs an optimal Omega calculation for the DLA model and outputs the result
        many_runs_hist # Runs many DLA simulations and outputs the result
        monte_carlo_multi # Runs a specified number of Monte Carlo Random Walk DLA simulations for a range of sticking probabilities
        script_monte_carlo_single # Runs a single Monte Carlo Random Walk DLA simulation and outputs the results
        ```

    - **Using General Function Usage**: Alternatively, you can run the scripts directly. For example, to run a single Monte Carlo simulation, modify and run:

        ```sh
        python scripts/script_monte_carlo_single.py
        ```

    - **Directly Modifying and Running Source File Functions**: You can also directly modify and run the functions in the source files. For example, to run a single Monte Carlo simulation, modify and run:

        ```python
        from src.monte_carlo import run_single_simulation

        # Modify parameters as needed
        run_single_simulation(parameters)
        ```

6. **Generate Plots**: Some of the scripts separate simulation runs and plotting, and thus you can generate plots using either script entry points, general function usage, or by directly modifying and running the source file functions.

    - **Using Script Entry Points**: Use the provided script entry points to generate plots depending on parameters defined in their script files:

        ```sh
        monte_carlo_plot_multi # Plots a range of statistics for results saved from script_monte_carlo_sim_multi simulations
        compare_DLA_MC # Plots comparisons between the general DLA and Monte Carlo methods for specified parameters
        ```

    - **Using General Function Usage**: Alternatively, you can run the scripts directly. For example, to generate a number of pre-built plots, modify and run:

        ```sh
        python scripts/script_monte_carlo_plot_multi.py
        ```

    - **Directly Modifying and Running Source File Functions**: You can also directly modify and run the functions in the source files. For example, to generate a heatmap, modify and run:

        ```python
        from src.utils import generate_heatmap

        # Modify parameters as needed
        generate_heatmap(data, "Title", "Colorbar Label", save_plot=True, plot_file_name="heatmap.png")
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
- `scripts/script_compare_DLA.py`: Script to plot data comparisons between the general DLA model and the Monte Carlo Model for specified parameters

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.