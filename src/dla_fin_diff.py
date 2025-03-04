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
def set_numba_seed(seed):
    np.random.seed(seed)

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
                return i==0
    assert(False)


def dla_growth(eta, omega, initial_condition, growth_steps=1000, diffusion_tolerance=1e-9):
    
    grid_size = initial_condition.shape[0]
    c = np.zeros([growth_steps, grid_size, grid_size])
    g = np.zeros_like(c)
    g[0] = initial_condition
    neighbors = neighbors_grid(g[0])
    c[0], sor_iter,_ = SOR_top_down(c[0], omega, tolerance=diffusion_tolerance, mask=1-g[0])
    total_sor_iter = sor_iter
    for t in range(0, growth_steps-1):
        
        if(t%(growth_steps//100)==0):
            print('.', end='', flush=True)
        
        
        c[t+1], sor_iter, sor_tol = SOR_top_down(c[t].copy(), omega, tolerance=diffusion_tolerance, mask=1-g[t])
        total_sor_iter += sor_iter
        # with high omega we sometimes see negative / very small concentrations
        c[t+1][c[t+1] < diffusion_tolerance] = 0
        
        g[t+1] = g[t]
        p_g = neighbors *  c[t+1]**eta            
        p_g = p_g / np.sum(p_g)
        
        # plot_grid(c[t])
        
        reached_top = grow_g(g[t+1], p_g, neighbors)
        if reached_top:
            break
        # plot_grid(neighbors)
    print('.')
    
    return g, c, t, total_sor_iter


if __name__ == '__main__':
    np.random.seed(42)
    set_numba_seed(np.random.randint(1000000000))
    grid_size = 100
    initial_cond = np.zeros([grid_size, grid_size])
    initial_cond[-2, grid_size//2] = 1
    # plot_grid(initial_cond)
                    
    eta =2
    omega = 1.85
    
    g, c , num_iter, total_sor_iter= dla_growth(eta, omega, initial_cond, growth_steps=10000)
        
    plot_grid(c[num_iter-1], g[num_iter-1], file='../plots/run_with_eta_{}.png'.format(eta), title=r'$\eta={}$'.format(eta), make_cbar=True)
    print(total_sor_iter)
    

    plot_animation(c[:num_iter], g[:num_iter])
        
        
    