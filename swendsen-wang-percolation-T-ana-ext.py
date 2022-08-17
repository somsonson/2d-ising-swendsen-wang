import numpy as np
import random
import numba
from numba import jit , prange , config, njit, threading_layer
from functions import initgrid
from functions import critical
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
#from functions import check_percolation_periodic
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
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


@jit(nopython=True,cache = False ,locals={'p_start_x': numba.float64 , 'p_start_y': numba.float64, 'p_start_d': numba.float64, 'T': numba.float64, 'delta_p':  numba.float64} )
def swendsen_wang_chain_magnetisation2(grid,J_x , J_y , J_d , p_start , delta_p,res, periodic = True):

    T = -2*J_x * 1/(log(1-p_start))
    p_start_x = p_start
    p_start_y = 1 - exp(-2*(1/T)* J_y)
    p_start_d = 1 - exp(-2*(1/T)* J_d)

    T_values = [0.]
    label,label_ids = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid, periodic)
    grid_flipped = 1 - grid
    label_flipped, label_ids_flipped = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_flipped,
                                                                                                  periodic)

    for steps in range(0,res):
        #print(steps)
        if check_percolation_nonperiodic(label, True , False) == True:
            p_start_x = p_start_x - delta_p
        elif check_percolation_nonperiodic(label_flipped,True, False) == True:
            p_start_x = p_start_x - delta_p
        elif check_percolation_nonperiodic(label, False, True) == True:
            p_start_x = p_start_x - delta_p
        elif check_percolation_nonperiodic(label_flipped,False, True) == True:
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

        label,label_ids = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid, periodic)
        grid_flipped = 1 - grid
        label_flipped, label_ids_flipped = fill_bonds_identify_clusters(p_start_x, p_start_y,
                                                                                                      p_start_d, grid_flipped,
                                                                                                      periodic)
        T_values.append(T)

        #print(steps)
    #T_c =-2 / np.log(1 - p_start_x)


    return p_start_x , T_values[1:] , grid

@jit(nopython=True )
def data_gen(N, J_x , J_y , J_d, steps):



    grid = initgrid(N,0.5)

    a,b , grid = swendsen_wang_chain_magnetisation2(grid,J_x , J_y , J_d,0.5,0.001,10000, periodic = True)

    a,b, grid = swendsen_wang_chain_magnetisation2(grid,J_x , J_y , J_d,a,1/(20*(N**2)),steps, periodic = True)

    return b






@jit(nopython=True, parallel = True )
def data_gen2(N , J_x , J_y , J_d):
    probs = [0.] * len(N)
    err_probs = [0.] * len(N)
    for i in range(0,len(N)):

        probs_temp = [0.] * 10
        probs_merged = np.array([0.])
        probs_merged = probs_merged[1:]


        for j in prange(0,10):


            b = data_gen(N[i],J_x , J_y , J_d,100000)
            b = np.asarray(b)

            p = 1 - np.exp(-2*(1/b)*J_x)
            probs_temp[j] = np.mean(p)



        probs[i] =  np.mean(np.asarray(probs_temp))
        err_probs[i] = np.std(np.asarray(probs_temp))/np.sqrt(float(len(probs_temp)))

        print(i)
    return probs , err_probs


#N = np.asarray([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120])
#N = np.asarray([10,20,30,40,50,60,70,80,90,100])
N = np.asarray([48,56,64,72,80,88,96,104,112,120,128])
#N = np.asarray([40,42,46,48,50,52,54,56,58,60,62,64,66,68])

J_x = 1
J_y = 1
J_d = 0


probs , probs_err = data_gen2(N , J_x , J_y , J_d)



T = []
T_err = []
T2 = []
T_err2 = []

for i in range(0,len(probs)):
    p = ufloat(probs[i] , probs_err[i])
    T_dist = -2*J_x/umath.log(1-p)
    print(T_dist)
    #n,bins,patches = plt.hist(T_dist, 1000,align='mid')
    #plt.show()
    #plt.cla()


    #n,bins,patches = plt.hist(p, 1000,align='mid')
    #plt.show()
    #plt.cla()




    #print(len(p),'asdasd')
    #print(p_mean_std)
    #print(T_mu)
    #print(1/critical(J_x,J_y,J_d))
    #T.append(-2*J_x/np.log(1-np.mean(p)))
    T.append(T_dist.nominal_value)
    T_err.append(T_dist.std_dev )
    T2.append(T_dist.nominal_value)
    T_err2.append(T_dist.std_dev)



np.savetxt("./data/temps_ext" + str(J_x)+ str(J_y)+str(J_d)+ ".csv",T , delimiter= ',')
np.savetxt("./data/tempserr_ext" + str(J_x)+ str(J_y)+str(J_d)+ ".csv",T_err , delimiter= ',')
np.savetxt("./data/systemsizes_ext" + str(J_x)+ str(J_y)+str(J_d)+".csv",N , delimiter= ',')


def temp():

    #plt.errorbar(N**-1. , T , yerr= T_err, fmt = 'o', markersize= 1, color = 'g')
    #plt.errorbar(N**-1. , T2 , yerr= T_err2, fmt = 'o', markersize= 1, color = 'r')


    def func(x, a, b,c):
        return a * (x **(-1.)) + b + c * (x **(-2.))


    popt, pcov = curve_fit(func,N, T, sigma=T_err)
    popt2, pcov2 = curve_fit(func,N, T2, sigma=T_err2)

    print(popt)
    print(popt2)

    print(np.sqrt(pcov[1][1]))
    print(np.sqrt(pcov2[1][1]))


    n_std = np.abs((1/critical(J_x,J_y,J_d)- popt[1])/np.sqrt(pcov[1][1]))
    n_std2 = np.abs((1/critical(J_x,J_y,J_d)- popt2[1])/np.sqrt(pcov2[1][1]))

    #plt.plot(np.append(N**(-1.),0),func(np.append(N, 1000000000000000000000),*popt), label = 'crit temp T =' + str(round(popt[1],5))+ '+/-' + str(round(np.sqrt(pcov[1][1]),5)))
    #plt.plot(np.append(N**(-1.),0),func(np.append(N, 1000000000000000000000),*popt2), label = 'T-space crit temp T =' + str(round(popt2[1],5))+ '+/-' + str(round(np.sqrt(pcov2[1][1]),5)))

    #plt.hlines([1/critical(J_x,J_y,J_d)[0]], 0 , 0.05, label = 'theoretical value' + str(1/critical(J_x,J_y,J_d)))
    #plt.xlabel('L^-1')
    #plt.ylabel('T')
    #plt.title('Temperatures for sizes N =' + str(N) + " and J_x = "+ str(J_x) + " J_y =" + str(J_y)+ " J_d = " + str(J_d)+ "\n n-std-deviation" + str(n_std) + "\n " + str(n_std2))
    #plt.legend()
    #plt.savefig("plots/1swendsen_perco_"+ str(N) + " and J_x = "+ str(J_x) + " J_y =" + str(J_y)+ " J_d = " + str(J_d) +".pdf" )
    #plt.show()

    #plt.cla()

    #plt.errorbar(N**-1. , T -func(N,*popt), yerr= T_err, fmt = 'o', markersize= 1)
    #plt.errorbar(N**-1. , T2 -func(N,*popt2), yerr= T_err2, fmt = 'o', markersize= 1)

    #plt.hlines(0,min(N**(-1.)), max(N**(-1.)))
    #plt.savefig("plots/1swendsen_perco_residue_"+ str(N) + " and J_x = "+ str(J_x) + " J_y =" + str(J_y)+ " J_d = " + str(J_d) +".pdf" )

    #plt.show()
