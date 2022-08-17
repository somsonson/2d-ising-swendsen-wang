import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit , prange
from functions import initgrid
from functions import  fill_bonds_identify_clusters_check_periodic_perco
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
from numpy import savetxt
from math import log
import timeit
import numba as numba
from numba_plyfit import fit_poly
from functions  import  fill_bonds_identify_clusters_check_periodic_perco_exit_if_percolated

import time
start_time = time.time()

plt.rcParams.update({'font.size': 30})

@jit(nopython=True, cache= True,parallel = True )
def simulatesquarelatticepercolation_topo(sample_prob, systemsize,iterations ):
    percolation_prob = []
    for probability in sample_prob:
        percolated_counter = 0
        for iteration in prange(0,iterations):

            grid = initgrid(systemsize,1)
            #a , b , percolated = fill_bonds_identify_clusters_check_periodic_perco(probability, probability, 0.3, grid)
            percolated = fill_bonds_identify_clusters_check_periodic_perco_exit_if_percolated(probability, probability, 0.3, grid)
            if percolated == True:
                percolated_counter = percolated_counter + 1
        percolation_prob.append(percolated_counter/iterations)
    return sample_prob, percolation_prob


sample_prob, percolation_prob_10 = simulatesquarelatticepercolation_topo(np.arange(0.3,0.7,0.005) , 16 , 10000)
print('1')
sample_prob, percolation_prob_20 = simulatesquarelatticepercolation_topo(np.arange(0.3,0.7,0.005) , 32 , 10000)
print('1')
sample_prob, percolation_prob_30 = simulatesquarelatticepercolation_topo(np.arange(0.3,0.7,0.005) , 64 , 10000)
print('1')
sample_prob, percolation_prob_40 = simulatesquarelatticepercolation_topo(np.arange(0.3,0.7,0.005) , 128 , 10000)
print('1')

percolation_probs = [sample_prob,percolation_prob_10,percolation_prob_20,percolation_prob_30,percolation_prob_40]
np.savetxt('./data/perco_topo.csv', percolation_probs,delimiter=",")


@jit(nopython=True, cache= True,parallel = True )
def simulatesquarelatticepercolation_non_periodic(sample_prob, systemsize,iterations ):
    percolation_prob = []
    for probability in sample_prob:
        percolated_counter = 0
        for iteration in prange(0, iterations):

            grid = initgrid(systemsize, 1)
            label, all_label_ids = fill_bonds_identify_clusters(probability, probability, 0, grid, False)
            percolated_1 = check_percolation_nonperiodic(label, False, True)
            percolated_2 = check_percolation_nonperiodic(label, True,False)
            if percolated_1 == True or percolated_2 == True:
                percolated_counter = percolated_counter + 1
        percolation_prob.append(percolated_counter / iterations)
    return sample_prob, percolation_prob


sample_prob, percolation_prob_10 = simulatesquarelatticepercolation_non_periodic(np.arange(0.3,0.7,0.005) , 16 , 10000)
sample_prob, percolation_prob_20 = simulatesquarelatticepercolation_non_periodic(np.arange(0.3,0.7,0.005) , 32 , 10000)
sample_prob, percolation_prob_30 = simulatesquarelatticepercolation_non_periodic(np.arange(0.3,0.7,0.005) , 64 , 10000)
sample_prob, percolation_prob_40 = simulatesquarelatticepercolation_non_periodic(np.arange(0.3,0.7,0.005) , 128 , 10000)

percolation_probs = [sample_prob,percolation_prob_10,percolation_prob_20,percolation_prob_30,percolation_prob_40]
np.savetxt('./data/perco_site_x_y_nonperiodic.csv', percolation_probs,delimiter=",")


@jit(nopython=True, cache= True,parallel = True )
def simulatesquarelatticepercolation_ext(sample_prob, systemsize,iterations ):
    percolation_prob = []
    for probability in sample_prob:
        percolated_counter = 0
        for iteration in prange(0, iterations):

            grid = initgrid(systemsize, 1)
            label, all_label_ids = fill_bonds_identify_clusters(probability, probability, 0, grid, True)
            percolated_1 = check_percolation_nonperiodic(label, False, True)
            percolated_2 = check_percolation_nonperiodic(label, True,False)
            if percolated_1 == True or percolated_2 == True:
                percolated_counter = percolated_counter + 1
        percolation_prob.append(percolated_counter / iterations)
    return sample_prob, percolation_prob


sample_prob, percolation_prob_10 = simulatesquarelatticepercolation_ext(np.arange(0.2,0.6,0.005) , 16 , 10000)
sample_prob, percolation_prob_20 = simulatesquarelatticepercolation_ext(np.arange(0.2,0.6,0.005) , 32 , 10000)
sample_prob, percolation_prob_30 = simulatesquarelatticepercolation_ext(np.arange(0.2,0.6,0.005) , 64 , 10000)
sample_prob, percolation_prob_40 = simulatesquarelatticepercolation_ext(np.arange(0.2,0.6,0.005) , 128 , 10000)

percolation_probs = [sample_prob,percolation_prob_10,percolation_prob_20,percolation_prob_30,percolation_prob_40]
np.savetxt('./data/perco_ext.csv', percolation_probs,delimiter=",")

print("--- %s seconds ---" % (time.time() - start_time))