
import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

from src.dla_fin_diff import *
from src.utils import *
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
      
      
def count_non_converged(eta, omega_0, num_runs=100, grid_size=100, adaptive_SOR = False):
    
    
    initial_cond = np.zeros([grid_size, grid_size])
    initial_cond[-2, grid_size//2] = 1
    sor_iters =np.zeros(num_runs)
    non_converged=0
    for i in range(num_runs):    
        try:        
            g, c , num_iter, sor_iter= dla_growth(eta, omega_0, initial_cond, growth_steps=10000, diffusion_tolerance=1e-5, adaptive_SOR=adaptive_SOR, verbose=False)
            sor_iters[i] =sor_iter
        except:
            non_converged +=1
    mean_sor_iter = np.sum(sor_iters) / (num_runs-non_converged) if non_converged < num_runs else np.inf
    std_sor_iter = np.std(sor_iters[sor_iters>0])
    print(omega_0, (num_runs-non_converged), non_converged, mean_sor_iter, std_sor_iter)
    return non_converged, mean_sor_iter, std_sor_iter

def non_conv_experiment(eta, file=os.path.join('data', 'opt_omega')):
    print(eta)
    
    np.random.seed(42)
    set_numba_seed(np.random.randint(1000000000))
    
    num_runs = 100
    omegas = [1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
    non_conv = np.zeros([len(omegas), 2])
    mean_iters = np.zeros([len(omegas), 2])
    std_iters = np.zeros([len(omegas), 2])
    for i in range(len(omegas)):
        omega_0 = omegas[i]
        print('adaptive')
        non_converged, mean_sor_iter, std_sor_iter= count_non_converged(eta, omega_0, num_runs=num_runs, adaptive_SOR=True)
        non_conv[i, 0] = non_converged
        mean_iters[i, 0] = mean_sor_iter
        std_iters[i, 0] = mean_sor_iter
        print('non-adaptive')
        non_converged, mean_sor_iter, std_sor_iter = count_non_converged(eta, omega_0, num_runs=num_runs, adaptive_SOR=False)
        non_conv[i, 1] = non_converged
        mean_iters[i, 1] = mean_sor_iter
        std_iters[i, 1] = mean_sor_iter
        
    np.savez((file + str(eta)), omegas, non_conv, mean_iters, std_iters)
        
def plot_non_conv(file=os.path.join('data', 'opt_omega.npz'), num_runs=20):
    arrs  = np.load(file)
    titles = ['adaptive', 'non-adaptive']
    omegas, non_conv, mean_iters, std_iters = [arrs['arr_{}'.format(i)] for i in range(4)]
    print(non_conv)
    for i, title in enumerate(titles):
        err = 1.96 * std_iters[:,i] / np.sqrt(num_runs)
        print(err)
        plt.errorbar(omegas, mean_iters[:,i], yerr=err, label=title, linestyle='', marker='o')
    plt.legend()
    plt.show()
        
def parallel_non_conv(etas):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_f = [executor.submit(non_conv_experiment, eta, os.path.join('data', 'opt_omega')) for eta in etas]

if __name__ == '__main__':
    
    # non_conv_experiment()
    etas = [0, 0.125, 0.5, 1, 2, 8]
    parallel_non_conv(etas)
    plot_non_conv()
    etas = np.linspace(0,4,50)
    w_min = np.zeros_like(etas)
    for i, eta in enumerate(etas):
        res = optimal_omega(eta, grid_size=20)
        w_min[i] = res['x']
    np.savez(os.path.join('data', 'opt_omega_vs_eta.npz'))
    
    
    plt.plot(etas, w_min)
    plt.show()