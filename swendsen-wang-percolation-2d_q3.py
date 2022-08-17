import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit , prange, set_num_threads
from functions import initgrid
from functions import critical
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
from functions import fill_bonds_identify_clusters_check_periodic_perco
from functions import decision
from scipy.signal import savgol_filter
import scipy
from numba import jit , prange , config, njit, threading_layer
import numba
from numpy import savetxt
from math import exp
from math import log

import time
start_time = time.time()

@jit(nopython=True,cache = False ,locals={'p_start_x': numba.float64 , 'p_start_y': numba.float64, 'p_start_d': numba.float64, 'T': numba.float64} , fastmath = True)
def swendsen_wang_chain_magnetisation2(grid_0,p_fix,p_var, delta_p,res, periodic = True):
    p_start_d = 0.5

    N = len(grid_0)
    p_start_x = p_var
    p_start_y = p_fix

    T_values = [0.]
    p_values = [0.]

    grid_1 = grid_0 - 1
    grid_2 = grid_0 + 1

    label_0,label_ids_0 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_0, periodic)
    label_1,label_ids_1 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_1, periodic)
    label_2,label_ids_2 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_2, periodic)



    #grid_flipped = 1 - grid
    #label_flipped, bondx_flipped, bondy_flipped,bondd_flipped, label_ids_flipped = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_flipped,
     #                                                                                             periodic)

    for steps in range(0,res):
        if check_percolation_nonperiodic(label_0,True , False) == True:
            p_start_x = p_start_x - delta_p
        elif check_percolation_nonperiodic(label_0,False , True) == True:
            p_start_x = p_start_x - delta_p
        elif check_percolation_nonperiodic(label_1,True , False) == True:
            p_start_x = p_start_x - delta_p
        elif check_percolation_nonperiodic(label_1,False , True) == True:
            p_start_x = p_start_x - delta_p
        elif check_percolation_nonperiodic(label_2,True , False) == True:
            p_start_x = p_start_x - delta_p
        elif check_percolation_nonperiodic(label_2,False , True) == True:
            p_start_x = p_start_x - delta_p
        else:
            p_start_x = p_start_x + delta_p


        if p_start_x >= 1:
            p_start_x = 1.
        elif p_start_x <= 0:
            p_start_x = 0.
        else:
            pass


        for id_1 in label_ids_0:
            if decision(1/3):
                pass
            elif decision(1/2):
                for xx in range(0, N):
                    for yy in range(0, N):
                        if label_0[yy][xx] == id_1:
                            grid_0[yy][xx] = 0
            else:
                for xx in range(0, N):
                    for yy in range(0, N):
                        if label_0[yy][xx] == id_1:
                            grid_0[yy][xx] = 2

        for id_1 in label_ids_1:
            if decision(1/3):
                pass
            elif decision(1/2):
                for xx in range(0, N):
                    for yy in range(0, N):
                        if label_1[yy][xx] == id_1:
                            grid_0[yy][xx] = 0
            else:
                for xx in range(0, N):
                    for yy in range(0, N):
                        if label_1[yy][xx] == id_1:
                            grid_0[yy][xx] = 1

        for id_1 in label_ids_2:
            if decision(1/3):
                pass
            elif decision(1/2):
                for xx in range(0, N):
                    for yy in range(0, N):
                        if label_2[yy][xx] == id_1:
                            grid_0[yy][xx] = 1
            else:
                for xx in range(0, N):
                    for yy in range(0, N):
                        if label_2[yy][xx] == id_1:
                            grid_0[yy][xx] = 2




        grid_1 = grid_0 - 1
        grid_2 = grid_0 + 1

        label_0, label_ids_0 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d,
                                                                                       grid_0, periodic)
        label_1, label_ids_1 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d,
                                                                                       grid_1, periodic)
        label_2, label_ids_2 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d,
                                                                                       grid_2, periodic)

        p_values.append(p_start_x)
        #print(steps)
    #T_c =-2 / np.log(1 - p_start_x)


    return p_start_x , p_values[1:]

@jit(nopython=True,parallel = True )
def data_gen(N):
    p_fix = np.arange(0.0,1.02,0.02)
    p_var =  [0.] * len(p_fix)
    p_err = [0.]* len(p_fix)
    a = 0.+0.
    j = 1
    for i in prange(0,len(p_fix)):
        grid = initgrid(N,0.5)
        p_fixed = p_fix[i]
        a,b = swendsen_wang_chain_magnetisation2(grid,p_fixed,p_fixed, 0.001,10000, periodic = True)

        a,b = swendsen_wang_chain_magnetisation2(grid,p_fixed,a, 1/(20*N**2),100000, periodic = True)

        p_var[i] = np.mean(np.asarray(b))
        print(p_var[i])
        p_err[i] = np.std(np.asarray(b))/np.sqrt(len(b))
        progression = len(p_var)



   # T_temp = np.asarray(T_a)

    return  p_var, p_err , p_fix


p_var_10,p_err_10,p_fix_10 = data_gen(8)
p_var = np.asarray(p_var_10)
p_fix = np.asarray(p_fix_10)
savetxt('data/p_var_10_p_d05.csv', p_var, delimiter=',')
savetxt('data/p_err_10_p_d05.csv', p_err_10, delimiter=',')
savetxt('data/p_fix_10_p_d05.csv', p_fix, delimiter=',')
print('1')
p_var_10,p_err_10,p_fix_10 = data_gen(16)
p_var = np.asarray(p_var_10)
p_fix = np.asarray(p_fix_10)
savetxt('data/p_var_20_p_d05.csv', p_var, delimiter=',')
savetxt('data/p_err_20_p_d05.csv', p_err_10, delimiter=',')
savetxt('data/p_fix_20_p_d05.csv', p_fix, delimiter=',')
print('1')
p_var_10,p_err_10,p_fix_10 = data_gen(32)
p_var = np.asarray(p_var_10)
p_fix = np.asarray(p_fix_10)
savetxt('data/p_var_30_p_d05.csv', p_var, delimiter=',')
savetxt('data/p_err_30_p_d05.csv', p_err_10, delimiter=',')
savetxt('data/p_fix_30_p_d05.csv', p_fix, delimiter=',')
print('1')
p_var_10,p_err_10,p_fix_10 = data_gen(64)
p_var = np.asarray(p_var_10)
p_fix = np.asarray(p_fix_10)
savetxt('data/p_var_40_p_d05.csv', p_var, delimiter=',')
savetxt('data/p_err_40_p_d05.csv', p_err_10, delimiter=',')
savetxt('data/p_fix_40_p_d05.csv', p_fix, delimiter=',')


print("--- %s seconds ---" % (time.time() - start_time))
