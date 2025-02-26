import numpy as np
from finite_difference import *
from utils import *


def neighbors_grid(g):
    grid_size = g.shape[-1]
    
    top = np.roll(g, 1, axis=0)
    down = np.roll(g, -1, axis=0)
    left = np.roll(g, 1, axis=1)
    right = np.roll(g, -1, axis=1)
    neighbors_grid = (top+down+left+right) > 0
    return np.maximum(0, neighbors_grid - g)


@njit
def grow_g(g, p_g):    
    x = np.random.uniform(0,1)
    p=0
    for i, row in enumerate(g):
        for j, el in enumerate(row):
            p += p_g[i,j]
            if p>x:
                # grow in this cell
                g[i,j] = 1
                return


def dla_growth(eta, omega, initial_condition, growth_steps=1000, diffusion_tolerance=1e-9):
    
    grid_size = initial_condition.shape[0]
    c = np.zeros([growth_steps, grid_size, grid_size])
    g = np.zeros_like(c)
    g[0] = initial_condition
    c[0], _,_ = SOR_top_down(c[0], omega, tolerance=diffusion_tolerance, mask=1-g[0])
    for t in range(0, growth_steps-1):
        
        
        c[t+1], _,_ = SOR_top_down(c[t].copy(), omega, tolerance=diffusion_tolerance, mask=1-g[t])
        
        g[t+1] = g[t]
        neighbors = neighbors_grid(g[t+1])
        p_g = neighbors * c[t+1]**eta
        p_g = p_g / np.sum(p_g)
        
        # plot_grid(c[t])
        
        grow_g(g[t+1], p_g)
    return g, c
        
    
        
if __name__ == '__main__':
    grid_size = 100
    initial_cond = np.zeros([grid_size, grid_size])
    initial_cond[-2, grid_size//2] = 1
    # plot_grid(initial_cond)
                    
    eta =1
    omega = 1.87
    
    g, c = dla_growth(eta, omega, initial_cond, growth_steps=1000)
        
    plot_grid(g[-1])
        
        
    