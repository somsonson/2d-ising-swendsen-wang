import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit , prange
from functions import initgrid
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
from functions import decision
from scipy.signal import savgol_filter
import scipy
from numpy import savetxt
from math import exp



@jit(nopython=True )
def swendsen_wang_chain_magnetisation(grid,p_x , p_y, thermalisation_steps=10000 , measuring_steps = 500, periodic = True):
################################ Thermalisation #############################
    for iteration in range(0,thermalisation_steps):
        #print(iteration)
        label, label_ids = fill_bonds_identify_clusters(p_x, p_y, 0, grid, periodic)

        grid_flipped = 1 - grid
        label_flipped , label_ids_flipped = fill_bonds_identify_clusters(p_x, p_y, 0, grid_flipped, periodic)

        for id_1 in label_ids:
            if decision(0.5):
                for xx in range(0,len(grid)):
                   for yy in range(0,len(grid)):
                       if label[yy][xx] == id_1:
                           grid[yy][xx] = 1 - grid[yy][xx]

        for id_2 in label_ids_flipped:
            if decision(0.5):
                for xx in range(0,len(grid)):
                   for yy in range(0,len(grid)):
                       if label_flipped[yy][xx] == id_2:
                           grid[yy][xx] = 1 - grid[yy][xx]

################################ calculating values of system and flip  #############################

    absolute_magnetisation = (np.array([0.]))[1:]
    energies_of_system = (np.array([0.]))[1:]
    for i in range(0,measuring_steps):
        magnetisation = 0.
        energy_of_system = 0.
        for xx in range(0, len(grid)):
            for yy in range(0, len(grid)):
                magnetisation = magnetisation + (2*grid[yy][xx] - 1)/ (len(grid)**2)
                energy_of_system = energy_of_system + (-1)*( (2*grid[yy][xx] - 1) * (2*grid[yy][(xx-1) % len(grid)] - 1) +
                    (2 * grid[(yy-1) % len(grid)][xx] - 1)*(2*grid[yy][xx] - 1))

        label,  label_ids = fill_bonds_identify_clusters(p_x, p_y, 0, grid, periodic)
        grid_flipped = 1 - grid
        label_flipped,  label_ids_flipped = fill_bonds_identify_clusters(p_x, p_y, 0, grid_flipped,
                                                                                                      periodic)

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
        absolute_magnetisation = np.append(absolute_magnetisation , abs(magnetisation))
        energies_of_system = np.append(energies_of_system , energy_of_system)


    return np.mean(absolute_magnetisation**2)  ,np.std(absolute_magnetisation)/np.sqrt(measuring_steps-1), np.mean(energies_of_system),np.std(energies_of_system)/np.sqrt(measuring_steps-1), grid


@jit(nopython=True, parallel = True, fastmath=True)
def parallell():
    size = 8
    resolution = 19
    start_inteval = 1
    end_interval = 10.5
    J_x = 1
    J_y = 1
    temp = [0.] * resolution
    avg_magnet =  [0.] * resolution
    avg_energy =  [0.] * resolution
    std_magnet = [0.] * resolution
    std_energy = [0] * resolution
    #progress = 0
    for n in prange(0,resolution):
        #progress = progress + 1
        #print(progress)

        i = np.arange(start_inteval,end_interval,(end_interval - start_inteval)/resolution)[n]
        p_x = 1 - exp(-2*J_x/i)
        p_y = 1 - exp(-2*J_y/i)
        average_energy = 0.
        average_magnet = 0.
        
        grid = initgrid(size,0.5)
        chain_values = swendsen_wang_chain_magnetisation(grid,p_x,p_y, measuring_steps= 100000)

        temp[n] = i
        avg_magnet[n] = chain_values[0]
        std_magnet[n] = chain_values[1]
        avg_energy[n] = chain_values[2] / size**2
        std_energy[n] = chain_values[3] / size**2
        #
    return temp , avg_magnet , std_magnet,avg_energy , std_energy


temp, avg_magnet, std_magnet, avg_energy, std_energy = parallell()


savetxt('data/temp.csv',temp, delimiter=',')
savetxt('data/avg_energy.csv',avg_energy, delimiter=',')
savetxt('data/std_energy.csv',std_energy, delimiter=',')
savetxt('data/avg_magn.csv',avg_magnet, delimiter=',')
savetxt('data/std_magn.csv',std_magnet, delimiter=',')




