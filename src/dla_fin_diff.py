import numpy as np
from finite_difference import *
from utils import *

def neighbors_grid(g):
    """
    given a boolean grid of live/ dead cells, compute a boolean grid where all neighboring points of live cells are highlighted.
    Uses 4 views of the array shifted by one position, so should be fairly fast even without jit.
    params:
        g:      grid of live cells at one timestep [grid_size x grid_size]
        
    returns:
        neighbors: grid of cells that are neighbors of the live cell cluster [grid_size x grid_size]
    """        
    top = np.roll(g, 1, axis=0)
    down = np.roll(g, -1, axis=0)
    left = np.roll(g, 1, axis=1)
    right = np.roll(g, -1, axis=1)
    neighbors_grid = (top+down+left+right) > 0
    return np.maximum(0, neighbors_grid - g)

@njit
def set_numba_seed(seed):
    """
    To set the seed for calls to np.random inside numba jit, the seed also needs to be set inside a jit function
    """
    np.random.seed(seed)

@njit
def grow_g(g, p_g, neighbors):
    """
    given a grid of probabilities of choosing a cell, choose the next cell to activate and update the 
    grid of live cells and their neighbors
    params:
        g:      grid of live cells at one timestep [grid_size x grid_size]
        p_g:    likelihood of choosing a cell [grid_size x grid_size]
        neighbors: mask of neighbors of the current live cells
        omega:              parameter of SOR
        initial_condition:  grid of initial live cells [grid_size x grid_size], live=1
        growth_steps:       number of cells to grow / number of DLA iterations
        diffusion_tolerance:stop SOR when changes between iterations are smaller than tolerance
        verbose:            decide if progress bar should be printed to stdout
        
    returns:
        g:      grid of live cells at each timestep [growth_steps x grid_size x grid_size]
        c:      grid of nutrient concentration at each timestep [growth_steps x grid_size x grid_size]
        t:      last growth timestep when top is reached
        total_sor_iter:  total number of finite difference timesteps of the simulation

    """    
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


def dla_growth(eta, omega, initial_condition, growth_steps=1000, diffusion_tolerance=1e-4, adaptive_SOR=True, verbose=True):    
    """Diffusion Limited Aggregation model with a uniform source at top and sink at the bottom
    The nutrient concentration is computed using the finite difference Successive over-relaxation (SOR) method
    The simulation is always stopped when the top row is reached
    params:
        eta:                probability of choosing growth cell scales with c**eta
        omega:              parameter of SOR
        initial_condition:  grid of initial live cells [grid_size x grid_size], live=1
        growth_steps:       number of cells to grow / number of DLA iterations
        diffusion_tolerance:stop SOR when changes between iterations are smaller than tolerance
        adaptive_SOR:       decide if omega is automatically reduced, if False, SOR can become unstable
        verbose:            decide if progress bar should be printed to stdout
        
    returns:
        g:      grid of live cells at each timestep [growth_steps x grid_size x grid_size]
        c:      grid of nutrient concentration at each timestep [growth_steps x grid_size x grid_size]
        t:      last growth timestep when top is reached
        total_sor_iter:  total number of finite difference timesteps of the simulation

    """
    grid_size = initial_condition.shape[0]
    c = np.zeros([growth_steps, grid_size, grid_size])
    g = np.zeros_like(c)
    g[0] = initial_condition
    neighbors = neighbors_grid(g[0])
    basic_gradient = np.linspace(1,0,grid_size)
    c[0] = basic_gradient[:, None]
    c[0], sor_iter,_ = SOR_top_down(c[0], omega, tolerance=diffusion_tolerance, mask=1-g[0])
    total_sor_iter = sor_iter
    for t in range(0, growth_steps-1):
        
        if verbose and (t%(growth_steps//100)==0):
            print('.', end='', flush=True)
        
        c[t+1], sor_iter, sor_tol = SOR_top_down(c[t].copy(), omega, tolerance=diffusion_tolerance, mask=1-g[t], adaptive=adaptive_SOR)
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
    if verbose:
        print('.')
    
    return g, c, t, total_sor_iter

