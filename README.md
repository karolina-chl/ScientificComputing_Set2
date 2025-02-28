# Scientific Computing Set 2
#### Victoria Peterson - 15476758, Paul Jungnickel - 15716554, Karolina Chłopicka - 15716546

## Table of Contents

1. [Usage and Installation](#usage-and-installation)
2. [Implementation](#implementation)
3. [Contributing](#contributing)
4. [License](#license)


## Usage and Installation
To run the simulations and generate plots, follow these steps:

1. Clone the repository:
```sh
gh repo clone karolina-chl/ScientificComputing_Set2
```

2. Create and activate a virtual environment
``` sh
python -m venv venv
venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**: Ensure you have all the required dependencies installed. You can install them using the `requirements.txt` file:
```sh
pip install -r requirements.txt
```

4. **Initialize Directories as Packages**: This step will ensure that all function imports from different directories are recognized by initializing the directories as packages in the virtual envirnment. This step may be unnecessary if your IDE automatically established the environmental variable `PYTHON_PATH` for your imports.
``` sh
pip install -e .
```

4. **Run Simulations**: Use the provided scripts to run the simulations. For example, to run a simulation using the GUI:
```sh
python src/main_run_gui.py
```

5. **Generate Plots**: Use the [PlotGenerator](http://_vscodecontentref_/1) class to generate plots from the simulation results. For example:
```python
from ForestFirePercolation.src.plots.plot_generator import PlotGenerator

generator = PlotGenerator("path/to/your/data.csv")
generator.generate_base_experiment_plots()
```


## Implementation
The Diffusion Limted Aggregation, Monte Carlo Random Walk, and Gray-Scott models are implemented using Python and consists of several modules to handle different aspects of the simulation and analysis. Below is an overview of the main components:



## Contributing
If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
