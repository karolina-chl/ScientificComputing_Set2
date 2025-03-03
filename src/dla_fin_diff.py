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
def grow_g(g, p_g, neighbors):    
    neighbor_coords = [(0,1),(0,-1),(1,0),(-1,0)]
    grid_size = g.shape[0]
    x = np.random.uniform(0,1)
    p=0
    for i in range(grid_size):
        for j in range(grid_size):
            p += p_g[i,j]
            if p>x:
                # grow in this cell
                g[i,j] = 1
                #update the neighbors grid
                neighbors[i,j] = 0
                for dy, dx in neighbor_coords:
                    if 0<= i+dy < grid_size:
                        neighbors[i+dy, (j+dx) % grid_size] = 1 - g[i+dy, (j+dx) % grid_size]
                return


def dla_growth(eta, omega, initial_condition, growth_steps=1000, diffusion_tolerance=1e-9):
    
    grid_size = initial_condition.shape[0]
    c = np.zeros([growth_steps, grid_size, grid_size])
    g = np.zeros_like(c)
    g[0] = initial_condition
    neighbors = neighbors_grid(g[0])
    c[0], _,_ = SOR_top_down(c[0], omega, tolerance=diffusion_tolerance, mask=1-g[0])
    for t in range(0, growth_steps-1):
        
        if(t%(growth_steps//100)==0):
            print('.', end='', flush=True)
        
        
        c[t+1], _,_ = SOR_top_down(c[t].copy(), omega, tolerance=diffusion_tolerance, mask=1-g[t])
        
        g[t+1] = g[t]
        p_g = neighbors * np.maximum(1e-9, c[t+1])**eta
        # p_g[np.isnan(p_g)]=0
        # print(p_g)
        print(t, np.sum(p_g), np.max(p_g))
        p_g = p_g / np.sum(p_g)
        
        # plot_grid(c[t])
        
        reached_top = grow_g(g[t+1], p_g, neighbors)
        if reached_top:
            break
        # plot_grid(neighbors)
    print('.')
    
    return g, c
        
    
# def interval_bisection_omega(f, xlim=[0,2], tol=1e-3):
#     while xlim[1] - xlim[0] > tol:
#         dx
#         center = (xlim[1] + xlim[0]) /2
#         f_p = f(center + dx)
        
if __name__ == '__main__':
    np.random.seed(42)
    grid_size = 50
    initial_cond = np.zeros([grid_size, grid_size])
    initial_cond[-2, grid_size//2] = 1
    # plot_grid(initial_cond)
                    
    eta =0.5
    omega = 1.87
    
    g, c = dla_growth(eta, omega, initial_cond, growth_steps=1000)
        
    # plot_grid(c[-1], g[-1])
    plot_animation(c, g)
        
        
    