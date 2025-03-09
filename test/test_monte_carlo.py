import unittest
import numpy as np

from unittest.mock import patch, call, MagicMock
from src.monte_carlo import *
from src.dla_fin_diff import neighbors_grid

class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        self.grid_size=10
        self.sticking_prob = 1
    
    def test_final_shape_is_connected(self):
        #assume the each live cell has a live neighbor
        neighbor_coords = [(0,1),(0,-1),(1,0),(-1,0)]
        sim_result = monte_carlo_sim_final_state_only(self.grid_size, self.sticking_prob)
        final_grid = sim_result[0]
        for row in range(1, self.grid_size-1):
            for col in range(self.grid_size):
                if final_grid[row, col]==EnumCellTypes.GROWTH_BLACK:
                    has_live_neighbor=False
                    for dy, dx in neighbor_coords:
                        has_live_neighbor = has_live_neighbor or final_grid[row+dy, (col+dx)%self.grid_size] == EnumCellTypes.GROWTH_BLACK
                    self.assertTrue(has_live_neighbor)

    def test_grow_by_one(self):
        # test that the cluster grows by one in each timestep
        results, _, cluster_size = monte_carlo_sim(self.grid_size, self.sticking_prob)
        grids = results['successful_seed_growth_grid_states']
        for t in range(cluster_size-1):
            num_new_growths = np.sum(grids[t+1]) - np.sum(grids[t])
            self.assertTrue(num_new_growths == 1)
            
            
    def test_animate_mc_sim(self):
        # test the animation function on a small grid
        results, _, cluster_size = monte_carlo_sim(self.grid_size, self.sticking_prob)
        animate_monte_carlo_sim(results["seed_growth_grid_states"],
                                results["walker_final_states"], 
                                self.grid_size, 
                                save_animation=False, 
                                filename="monte_carlo_animation_all.mp4", 
                                animation_speed=10)
            
if __name__ == '__main__':
    unittest.main()