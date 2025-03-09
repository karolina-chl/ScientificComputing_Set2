import unittest
import numpy as np
from numba import njit
import os 

from src.dla_fin_diff import neighbors_grid, set_numba_seed, grow_g, dla_growth

class TestDLAGrowth(unittest.TestCase):
    
    def setUp(self):
        """Set up a small test grid for controlled testing"""
        self.grid_size = 5
        self.test_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.test_grid[2, 2] = 1 
    
    def test_neighbors_grid(self):
        """Test if the neighbors of a single live cell are correctly identified."""
        expected_output = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        expected_output[1, 2] = 1  # Top neighbor
        expected_output[3, 2] = 1  # Bottom neighbor
        expected_output[2, 1] = 1  # Left neighbor
        expected_output[2, 3] = 1  # Right neighbor
        
        result = neighbors_grid(self.test_grid)
        np.testing.assert_array_equal(result, expected_output)

    def test_grow_g(self):
        """Test if a single cell grows correctly given a probability grid."""
        g = self.test_grid.copy()
        p_g = np.zeros_like(g, dtype=float)
        p_g[1, 2] = 1.0  
        neighbors = neighbors_grid(g)
        
        reached_top = grow_g(g, p_g, neighbors)
        
        self.assertEqual(g[1, 2], 1)  
        self.assertFalse(reached_top)  
    
    def test_dla_growth_early_stop(self):
        """Test if the growth process stops when the top row is reached."""
        initial_condition = np.zeros((self.grid_size, self.grid_size), dtype=int)
        initial_condition[-2, 2] = 1  # Start one row above the bottom
        gs = 50
        
        g, c, t, total_sor_iter = dla_growth(
            eta=1, omega=1.5, initial_condition=initial_condition, growth_steps=gs  # Increased steps
        )
        
        assert isinstance(g, np.ndarray)
        assert g.shape[0] == gs

        # Ensure the process stopped within the allowed steps
        self.assertLessEqual(t, 500, "Simulation took too many steps to reach the top")


if __name__ == '__main__':
    unittest.main()


