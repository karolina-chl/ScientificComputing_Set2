import numpy as np
from numba import jit, njit, prange


@njit
def SOR_top_down(c,omega, max_steps=100000, mask=None, tolerance= None, adaptive=True):
    """SOR finite difference method for time-independent diffusion
    
    params:
        c:          grid of concentration values shape [grid_size x grid_size]
        omega:      parameter of SOR
        mask:       grid of sinks shape [grid_size x grid_size]
        mask:       grid of sinks shape [grid_size x grid_size]
        tolerance:  stop when changes between iterations are smaller than tolerance
        adaptive:   specify whether omega should be adjusted when the method starts to become unstable

    returns:
        c:      modified grid c
        t:      last timestep
        tol:    change from previous-to-last iteration to last iteration

    """
    # top down flow boundary
    c[0] = 1
    c[-1] = 0
    width, height = c.shape
    if mask is None:
        mask = np.ones_like(c)
    eps = 1e100
    c_old = c.copy()
    for t in range(0, max_steps - 1):
        # Top and Bottom Boundaries
        c[0] = c_old[0]
        c[-1] = c_old[-1]
        for i in range(1, height -1):
            c[i, -1] = c_old[i, -1] # to avoid adding sink on left side
            for j in range(width):
                if mask[i,j] ==0:
                    c[i,j] = 0
                    continue
                c0 = c_old[i, j]
                c1 = c_old[i+1, j]
                c2 = c[i-1, j]
                c3 = c[i, (j-1)% width] 
                c4 = c_old[i, (j+1)% width ]
                c[i, j] = omega / 4.0 * (c1 + c2 + c3+ c4) + (1-omega) * c0 
                if np.isnan(c[i,j]):
                    print(t, i,j)
                    assert False, 'SOR became unstable, please try a lower omega'

        if tolerance is not None:
            eps_prev = eps
            eps = np.max(np.abs(c - c_old))
            if adaptive and eps > eps_prev:
                omega = omega-0.01
                continue
                # print('reduced omega to ', omega)
            if eps < tolerance:
                break
        c, c_old = c_old, c

    return c, t, eps




