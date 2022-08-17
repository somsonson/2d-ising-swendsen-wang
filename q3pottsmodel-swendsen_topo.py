import numpy as np
import numba
from numba import jit , prange , config, njit, threading_layer
from functions import initgridq3
from functions import fill_bonds_identify_clusters
from functions import decision
from functions import  fill_bonds_identify_clusters_check_periodic_perco
from math import exp
from math import log
from uncertainties import ufloat
import uncertainties.umath as umath
import time
import os
start_time = time.time()

@jit(nopython=True,locals={'p_start_x': numba.float64 , 'p_start_y': numba.float64, 'p_start_d': numba.float64, 'T': numba.float64} , fastmath = True)
def swendsen_wang_chain_magnetisation2(grid_0,J_x , J_y , J_d , p_start , delta_p,res, periodic = True):
    N = len(grid_0)
    T = -2*J_x * 1/(log(1-p_start))
    p_start_x = p_start
    p_start_y = 1 - exp(-2*(1/T)* J_y)
    p_start_d = 1 - exp(-2*(1/T)* J_d)

    T_values = [0.]
    p_values = [0.]

    grid_1 = grid_0 - 1
    grid_2 = grid_0 + 1

    percolated_1 = False
    percolated_2 = False
    label_0, label_ids_0, percolated_0 = fill_bonds_identify_clusters_check_periodic_perco(p_start_x, p_start_y, p_start_d,
                                                                                       grid_0)
    if percolated_0 == True:
        label_1, label_ids_1 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_1,
                                                                        True)
        label_2, label_ids_2 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_2,
                                                            True)
    else:
        label_1, label_ids_1, percolated_1 = fill_bonds_identify_clusters_check_periodic_perco(p_start_x, p_start_y,
                                                                                           p_start_d,
                                                                                           grid_1)
        if percolated_1 == True:
            label_2, label_ids_2 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_2,
                                                                True)

        else:
            label_2, label_ids_2, percolated_2 = fill_bonds_identify_clusters_check_periodic_perco(p_start_x, p_start_y,
                                                                                                   p_start_d,
                                                                                                   grid_2)




    #grid_flipped = 1 - grid
    #label_flipped, bondx_flipped, bondy_flipped,bondd_flipped, label_ids_flipped = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_flipped,
     #                                                                                             periodic)

    for steps in range(0,res):
        if (percolated_1 == True) or (percolated_2 == True) or (percolated_0 == True):

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

        T = -2*J_x * 1/(log(1-p_start_x))
        p_start_y = 1 - exp(-2 * (1 / T) * J_y)
        p_start_d = 1 - exp(-2 * (1 / T) * J_d)


        grid_1 = grid_0 - 1
        grid_2 = grid_0 + 1

        percolated_1 = False
        percolated_2 = False
        label_0, label_ids_0, percolated_0 = fill_bonds_identify_clusters_check_periodic_perco(p_start_x, p_start_y,
                                                                                               p_start_d,
                                                                                               grid_0)
        if percolated_0 == True:
            label_1, label_ids_1 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_1,
                                                                True)
            label_2, label_ids_2 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_2,
                                                                True)
        else:
            label_1, label_ids_1, percolated_1 = fill_bonds_identify_clusters_check_periodic_perco(p_start_x, p_start_y,
                                                                                                   p_start_d,
                                                                                                   grid_1)
            if percolated_1 == True:
                label_2, label_ids_2 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_2,
                                                                    True)

            else:
                label_2, label_ids_2, percolated_2 = fill_bonds_identify_clusters_check_periodic_perco(p_start_x,
                                                                                                       p_start_y,
                                                                                                       p_start_d,
                                                                                                       grid_2)
        T_values.append(T)
        p_values.append(p_start_x)
        #print(steps)
    #T_c =-2 / np.log(1 - p_start_x)


    return p_start_x , T_values[1:] , grid_0 , p_values[1:]

@jit(nopython=True )
def data_gen(N, J_x , J_y , J_d, steps):

    grid = initgridq3(N)
    a,b,grid , p_values = swendsen_wang_chain_magnetisation2(grid,J_x , J_y , J_d,0.5,0.001,10000, periodic = True)
    a,b,grid , p_values = swendsen_wang_chain_magnetisation2(grid,J_x , J_y , J_d,a,1/(20*(N**2)),steps, periodic = True)

    return b , p_values






@jit(nopython=True, parallel = True , fastmath = True)
def data_gen2(N , J_x , J_y , J_d):
    probs = [0.] * len(N)
    err_probs = [0.] * len(N)
    for i in range(0,len(N)):

        probs_temp = [0.] * 10
        probs_merged = np.array([0.])
        probs_merged = probs_merged[1:]


        for j in prange(0,10):


            b , p_values= data_gen(N[i],J_x , J_y , J_d, 100000)
            b = np.asarray(b)

            probs_temp[j] = np.mean(np.asarray(p_values))
            #plt.hist(p)
            #plt.show()
            #print(np.mean(np.asarray(p_values)))

            #print( np.mean(np.asarray(b)))



        probs[i] =  np.mean(np.asarray(probs_temp))
        err_probs[i] = np.std(np.asarray(probs_temp))/np.sqrt(float(len(probs_temp)-1))

        print(i)
    return probs , err_probs


#N = np.asarray([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120])
#N = np.asarray([10,20,30,40,50,60,70,80,90,100])
#N = np.asarray([80])
N = np.asarray([16])
#N = np.asarray([20,25,30,35,40,45,50,55,60,65,70,75])

T = []
T_err = []




J_x = 1
J_y = 1
J_d = 0

probs , probs_err = data_gen2(N , J_x , J_y , J_d)






for i in range(0,len(probs)):
    p = ufloat(probs[i] , probs_err[i])
    T_dist = -2*J_x/umath.log(1-p)
    print(T_dist)

    T.append(T_dist.nominal_value)
    T_err.append(T_dist.std_dev )

print(T)
print(T_err)
np.savetxt("./Data_q3/q3temps" + str(J_x) + str(J_y) + str(J_d) + str(N)+".csv",T , delimiter= ',')
np.savetxt("./Data_q3/q3tempserr" + str(J_x) + str(J_y) + str(J_d) + str(N)+".csv",T_err , delimiter= ',')


#np.savetxt("./data/q3temps" + str(J_x)+ str(J_y)+str(J_d)+ ".csv",T , delimiter= ',')
#np.savetxt("./data/q3tempserr + " + str(J_x)+ str(J_y)+str(J_d)+ ".csv",T_err , delimiter= ',')
#np.savetxt("./data/q3systemsizes + " + str(J_x)+ str(J_y)+str(J_d)+".csv",N , delimiter= ',')




print("--- %s seconds ---" % (time.time() - start_time))