
import sys
sys.path.append('../src')

from dla_fin_diff import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar





def optimal_omega(eta, num_runs_for_mean=20, grid_size=100):
    
    def mean_num_sor_iterations(omega):
        print(omega)
        initial_cond = np.zeros([grid_size, grid_size])
        initial_cond[-2, grid_size//2] = 1
        total_sor_iter =0
        for i in range(num_runs_for_mean):
            print('.', end='', flush=True)            
            g, c , num_iter, sor_iter= dla_growth(eta, omega, initial_cond, growth_steps=10000, diffusion_tolerance=1e-5, verbose=False)
            total_sor_iter += sor_iter
        print(total_sor_iter / num_runs_for_mean)        
        return total_sor_iter / num_runs_for_mean
        
    
    res = minimize_scalar(mean_num_sor_iterations, bracket=[1,2],bounds=[1,2], tol=1e-3)  
    return res
      

if __name__ == '__main__':
    etas = np.linspace(0,4,50)
    w_min = np.zeros_like(etas)
    for i, eta in enumerate(etas):
        res = optimal_omega(eta, grid_size=20)
        w_min[i] = res['x']
    np.savez('../data/opt_omega_vs_eta.npz')
    
    
    plt.plot(etas, w_min)
    plt.show()