import numpy as np
from numpy import random
import numba
from numba import jit , prange , config, njit, threading_layer
from functions import initgrid
from functions import critical
from functions import initgridq3
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
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
from matplotlib import pyplot as plt


import time
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

    label_0, bondx_0, bondy_0, bondd_0,label_ids_0 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_0, periodic)
    label_1, bondx_1, bondy_1, bondd_1,label_ids_1 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_1, periodic)
    label_2, bondx_2, bondy_2, bondd_2,label_ids_2 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_2, periodic)



    #grid_flipped = 1 - grid
    #label_flipped, bondx_flipped, bondy_flipped,bondd_flipped, label_ids_flipped = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d, grid_flipped,
     #                                                                                             periodic)

    for steps in range(0,res):
        increase_p_1 = check_percolation_nonperiodic(label_0,True , False)
        increase_p_2 = check_percolation_nonperiodic(label_0,False , True)
        increase_p_3 = check_percolation_nonperiodic(label_1, True, False)
        increase_p_4 = check_percolation_nonperiodic(label_1, False, True)
        increase_p_5 = check_percolation_nonperiodic(label_2, True, False)
        increase_p_6 = check_percolation_nonperiodic(label_2, False, True)

        if (increase_p_1 == True) or (increase_p_2 == True) or (increase_p_3 == True) or (increase_p_4 == True) or (increase_p_5 == True) or (increase_p_6 == True):


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

        label_0, bondx_0, bondy_0, bondd_0, label_ids_0 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d,
                                                                                       grid_0, periodic)
        label_1, bondx_1, bondy_1, bondd_1, label_ids_1 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d,
                                                                                       grid_1, periodic)
        label_2, bondx_2, bondy_2, bondd_2, label_ids_2 = fill_bonds_identify_clusters(p_start_x, p_start_y, p_start_d,
                                                                                       grid_2, periodic)
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

        probs_temp = [0.] * 9
        probs_merged = np.array([0.])
        probs_merged = probs_merged[1:]


        for j in prange(0,9):


            b , p_values= data_gen(N[i],J_x , J_y , J_d, 50000)
            b = np.asarray(b)

            probs_temp[j] = np.mean(np.asarray(p_values))
            #plt.hist(p)
            #plt.show()

            print( np.mean(np.asarray(b)))



        probs[i] =  np.mean(np.asarray(probs_temp))
        err_probs[i] = np.std(np.asarray(probs_temp))/np.sqrt(float(len(probs_temp)))

        print(i)
    return probs , err_probs


#N = np.asarray([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120])
#N = np.asarray([10,20,30,40,50,60,70,80,90,100])
#N = np.asarray([80])
#N = np.asarray([40,42,46,48,50,52,54,56,58,60,62,64,66,68])
N = np.asarray([64])


J_x = 1/2
J_y = 1/2
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



np.savetxt("./data/q3temps" + str(J_x)+ str(J_y)+str(J_d)+ ".csv",T , delimiter= ',')
np.savetxt("./data/q3tempserr + " + str(J_x)+ str(J_y)+str(J_d)+ ".csv",T_err , delimiter= ',')
np.savetxt("./data/q3systemsizes + " + str(J_x)+ str(J_y)+str(J_d)+".csv",N , delimiter= ',')


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

print("--- %s seconds ---" % (time.time() - start_time))