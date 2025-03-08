
import unittest
import numpy as np

from src.gray_scott import init_grids, solve_gray_scott, plot_animation, last_frame_gray_scott_save

class TestGrayScott(unittest.TestCase):

    def test_init_grids(self):
        total_time = 100
        time_step_size = 1
        n_steps = 100
        chemical_U, chemical_V = init_grids(total_time, time_step_size, n_steps)
        
        self.assertEqual(chemical_U.shape, (total_time, n_steps, n_steps))
        self.assertEqual(chemical_V.shape, (total_time, n_steps, n_steps))
        self.assertTrue(np.all(chemical_U[0,:,:] == 0.5))
        self.assertTrue(np.all(chemical_V[0,45:55,45:55] == 0.25))

    def test_solve_gray_scott(self):
        total_time = 10
        time_step_size = 1
        x_length = 1
        n_steps = 10
        diffusion_coefficient_u = 0.1
        diffusion_coefficient_v = 0.05
        U_supply = 0.04
        k = 0.06
        noise_boundry = 0.01

        chemical_U, chemical_V = init_grids(total_time, time_step_size, n_steps)
        solve_gray_scott(chemical_U, chemical_V, total_time, time_step_size, x_length, n_steps, diffusion_coefficient_u, diffusion_coefficient_v, U_supply, k, noise_boundry)
        
        self.assertEqual(chemical_U.shape, (total_time, n_steps, n_steps))
        self.assertEqual(chemical_V.shape, (total_time, n_steps, n_steps))

    def test_plot_animation(self):
        total_time = 10
        time_step_size = 1
        n_steps = 10
        chemical_U, chemical_V = init_grids(total_time, time_step_size, n_steps)
        anim = plot_animation(chemical_U)
        
        self.assertIsNotNone(anim)

    def test_last_frame_gray_scott_save(self):
        total_time = 10
        time_step_size = 1
        n_steps = 10
        chemical_U, chemical_V = init_grids(total_time, time_step_size, n_steps)
        filename = "test_output.png"
        last_frame = last_frame_gray_scott_save(chemical_U, filename)
        
        self.assertEqual(last_frame.shape, (n_steps, n_steps))
        self.assertTrue(np.all(last_frame == chemical_U[-1]))

if __name__ == '__main__':
    unittest.main()