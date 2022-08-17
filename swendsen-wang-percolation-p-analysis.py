import numpy as np
import matplotlib.pyplot as plt
import random
import numba
from numba import jit , prange , config, njit, threading_layer
from functions import initgrid
from functions import critical
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
from functions import check_percolation_periodic
from functions import decision
from scipy.signal import savgol_filter
import scipy
from numpy import savetxt
from math import exp
from scipy.stats import norm
import time
start_time = time.time()



@jit(nopython=True,cache = True ,locals={'p_start_x': numba.float64 , 'p_start_y': numba.float64} )
def swendsen_wang_chain_magnetisation2(grid,p_y,p_fix_x, delta_p,res, periodic = True):


    p_start_x = p_fix_x
    p_start_y = p_y

    T_values = [0.]
    label, bondx, bondy, label_ids = fill_bonds_identify_clusters(p_start_x, p_start_y, 0, grid, periodic)
    grid_flipped = 1 - grid
    label_flipped, bondx_flipped, bondy_flipped, label_ids_flipped = fill_bonds_identify_clusters(p_start_x, p_start_y, 0, grid_flipped,
                                                                                                  periodic)

    for steps in range(0,res):
        increase_p_1 = check_percolation_periodic(label, bondx, bondy, True , False)
        increase_p_2 = check_percolation_periodic(label_flipped,bondx_flipped,bondy_flipped,True, False)
        increase_p_3 = check_percolation_periodic(label, bondx, bondy, False, True)
        increase_p_4 = check_percolation_periodic(label_flipped, bondx_flipped, bondy_flipped,False, True)

        if (increase_p_1 == True) or (increase_p_2 == True) or (increase_p_3 == True) or (increase_p_4 == True) :


            p_start_y = p_start_y - delta_p


        else:

            p_start_y = p_start_y + delta_p


        if p_start_y >= 1:
            p_start_y = 1
        elif p_start_y <= 0:
            p_start_y = 0
        else:
            pass


        for id_1 in label_ids:
            if decision(0.5):
                for xx in range(0, len(grid)):
                    for yy in range(0, len(grid)):
                        if label[yy][xx] == id_1:
                            grid[yy][xx] = 1 - grid[yy][xx]

        for id_2 in label_ids_flipped:
            if decision(0.5):
                for xx in range(0, len(grid)):
                    for yy in range(0, len(grid)):
                        if label_flipped[yy][xx] == id_2:
                            grid[yy][xx] = 1 - grid[yy][xx]

        label, bondx, bondy, label_ids = fill_bonds_identify_clusters(p_start_x, p_start_y, 0, grid, periodic)
        grid_flipped = 1 - grid
        label_flipped, bondx_flipped, bondy_flipped, label_ids_flipped = fill_bonds_identify_clusters(p_start_x, p_start_y,
                                                                                                      0, grid_flipped,
                                                                                                      periodic)
        T_values.append(p_start_y)

        #print(steps)
    #T_c =-2 / np.log(1 - p_start_x)


    return p_start_y , T_values[1:]

@jit(nopython=True,parallel = True )
def data_gen(N):
    p_fix = np.arange(0,1,0.005)
    #factor_fix = np.asarray([1000])
    p_var =  [[0.]] * len(p_fix)
    a = 0.+0.
    j = 1
    for i in prange(0,len(p_fix)):
        grid = initgrid(N,0.5)
        p_y = p_fix[i]
        a,b = swendsen_wang_chain_magnetisation2(grid,p_y,p_y, 0.01,1000, periodic = True)

        a,b = swendsen_wang_chain_magnetisation2(grid,a,p_y, 1/(20*N**2),1000, periodic = True)

        p_var[i] = b
        progression = len(p_var)
        for j in range(0,len(p_fix)):
            if p_var[int(j)] == [0.]:
                progression = progression - 1
        print(progression/len(p_var))



   # T_temp = np.asarray(T_a)

    return  p_var , p_fix

p_var_10,p_fix_10 = data_gen(10)
p_var = np.array([])
for data in p_var_10:
    p_var=np.append(p_var,norm.fit(np.asarray(data)))
p_fix = np.asarray(p_fix_10)
savetxt('data/p_var_10.csv', p_var, delimiter=',')
savetxt('data/p_fix_10.csv', p_fix, delimiter=',')

print("Threading layer chosen: %s" % threading_layer())

p_var_20,p_fix_20 = data_gen(15)
p_var = np.array([])
for data in p_var_20:
    p_var=np.append(p_var,norm.fit(np.asarray(data)))
p_fix = np.asarray(p_fix_20)
savetxt('data/p_var_20.csv', p_var, delimiter=',')
savetxt('data/p_fix_20.csv', p_fix, delimiter=',')

p_var_30,p_fix_30 = data_gen(20)
p_var = np.array([])
for data in p_var_30:
    p_var=np.append(p_var,norm.fit(np.asarray(data)))
p_fix = np.asarray(p_fix_30)
savetxt('data/p_var_30.csv', p_var, delimiter=',')
savetxt('data/p_fix_30.csv', p_fix, delimiter=',')

p_var_40,p_fix_40 = data_gen(25)
p_var = np.array([])
for data in p_var_40:
    p_var=np.append(p_var,norm.fit(np.asarray(data)))
p_fix = np.asarray(p_fix_40)
savetxt('data/p_var_40.csv', p_var, delimiter=',')
savetxt('data/p_fix_40.csv', p_fix, delimiter=',')

p_var_50,p_fix_50 = data_gen(30)
p_var = np.array([])
for data in p_var_50:
    p_var=np.append(p_var,norm.fit(np.asarray(data)))
p_fix = np.asarray(p_fix_50)
savetxt('data/p_var_50.csv', p_var, delimiter=',')
savetxt('data/p_fix_50.csv', p_fix, delimiter=',')

print("--- %s seconds ---" % (time.time() - start_time))
#436s