import unittest
import numpy as np

from unittest.mock import patch, call, MagicMock
from src.monte_carlo import initialize_grid

class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        self.testParameter = 1
        pass

    def test_initialize_grid(self):
        grid_size = 101
        seed_growth_grid = np.zeros((grid_size, grid_size), dtype=int)
        center = grid_size // 2
        seed_growth_grid[grid_size - 1, center] = 1.0

        empty_grid = initialize_grid(grid_size)


        self.assertEqual(seed_growth_grid, empty_grid)

