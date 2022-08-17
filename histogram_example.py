import numpy as np
import random
import numba
from numba import jit , prange , config, njit, threading_layer
from functions import initgrid
from functions import critical
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
from functions import fill_bonds_identify_clusters_check_periodic_perco
from functions import decision
from scipy.signal import savgol_filter
import scipy
from numpy import savetxt
from math import exp
from scipy.stats import norm
import time
start_time = time.time()
from math import log
from uncertainties import ufloat
from uncertainties import unumpy
from scipy.optimize import curve_fit
import uncertainties.umath as umath
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.size': 30})

start_time = time.time()

@jit(nopython=True,cache = False ,locals={'p_start_x': numba.float64 , 'p_start_y': numba.float64, 'p_start_d': numba.float64, 'T': numba.float64} , fastmath = True)
def swendsen_wang_chain_magnetisation2(grid,J_x , J_y , J_d , p_start , delta_p,res, periodic = True):

    T = -2*J_x * 1/(log(1-p_start))
    p_start_x = p_start
    p_start_y = 1 - exp(-2*(1/T)* J_y)
    p_start_d = 1 - exp(-2*(1/T)* J_d)

    T_values = [0.]
    label, label_ids, percolated_2= fill_bonds_identify_clusters_check_periodic_perco(p_start_x, p_start_y, p_start_d, grid)
    grid_flipped = 1 - grid
    label_flipped, label_ids_flipped, percolated_1 = fill_bonds_identify_clusters_check_periodic_perco(p_start_x, p_start_y, p_start_d, grid_flipped)

    for steps in range(0,res):


        if (percolated_2 == True) or (percolated_1 == True):


            p_start_x = p_start_x - delta_p


        else:

            p_start_x = p_start_x + delta_p


        if p_start_x >= 1:
            p_start_x = 1
        elif p_start_x <= 0:
            p_start_x = 0
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

        T = -2*J_x * 1/(log(1-p_start_x))
        p_start_y = 1 - exp(-2 * (1 / T) * J_y)
        p_start_d = 1 - exp(-2 * (1 / T) * J_d)

        label, label_ids, percolated_2 = fill_bonds_identify_clusters_check_periodic_perco(p_start_x, p_start_y,
                                                                                           p_start_d, grid)
        grid_flipped = 1 - grid
        label_flipped, label_ids_flipped, percolated_1 = fill_bonds_identify_clusters_check_periodic_perco(p_start_x,
                                                                                                           p_start_y,
                                                                                                           p_start_d,
                                                                                                           grid_flipped)

        T_values.append(T)

        #print(steps)
    #T_c =-2 / np.log(1 - p_start_x)


    return p_start_x , T_values[1:] , grid

@jit(nopython=True )
def data_gen(N, J_x , J_y , J_d, steps):



    grid = initgrid(N,0.5)

    a,b , grid= swendsen_wang_chain_magnetisation2(grid,J_x , J_y , J_d,0.5,0.1,100, periodic = True)

    a,b ,grid = swendsen_wang_chain_magnetisation2(grid,J_x , J_y , J_d,a,1/(40*(N**2)),steps, periodic = True)

    return b






@jit(nopython=True , fastmath = True)
def data_gen2(N , J_x , J_y , J_d):
    probs = [0.] * len(N)
    err_probs = [0.] * len(N)
    for i in range(0,len(N)):





        b = data_gen(N[i],J_x , J_y , J_d, 100000)
        b = np.asarray(b)







    return b


#N = np.asarray([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120])
#N = np.asarray([10,20,30,40,50,60,70,80,90,100])
#N = np.asarray([80])
N = np.asarray([32])
#N = np.asarray([64])


J_x = 1
J_y = 1
J_d = 0

#numba.set_num_threads(4)
temps  = data_gen2(N , J_x , J_y , J_d)

mean_T = np.mean(temps)
std_T = np.std(temps)
T = ufloat(mean_T,std_T)
print(1- umath.exp(-2*1/(T)))


np.savetxt('./data/hist_data_iso.csv',temps,delimiter=",")
'''
#N = np.asarray([64])


J_x = 1
J_y = 1
J_d = 0

#numba.set_num_threads(4)
temps  = data_gen2(N , J_x , J_y , J_d)




np.savetxt('./data/hist_data_aniso.csv',temps,delimiter=",")

print("--- %s seconds ---" % (time.time() - start_time))
'''