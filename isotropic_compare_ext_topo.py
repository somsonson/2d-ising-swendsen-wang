import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit , prange
from functions import initgrid
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
#from functions import check_percolation_periodic
from numpy import savetxt
from math import log
import timeit
from scipy.optimize import curve_fit
from functions import critical
from numpy import genfromtxt
from scipy.stats import binned_statistic
from scipy.signal import find_peaks
from math import sinh
from math import  log
import scipy.optimize
from numpy import log
from numpy import sinh
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.stats import norm
import matplotlib.mlab as mlab
from uncertainties import ufloat
import uncertainties
from uncertainties import unumpy as unp
from praktikum import analyse
plt.rcParams.update({'font.size': 22})

plt.locator_params(axis = 'x', nbins = 3)
plt.locator_params(axis = 'y', nbins = 3 )


k =2
#N = np.asarray([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120])
#N = np.asarray([10,20,30,40,50,60,70,80,90,100])
#N = np.asarray([16,24,32,40,48,56,64])[k:]
#N = np.asarray([40,42,46,48,50,52,54,56,58,60,62,64,66,68])

N = np.asarray([48,56,64,72,80,88,96,104,112,120,128])[k:]



J_x = 1
J_y = 1
J_d = 0

print(1/critical(J_x,J_y,J_d)[0])

T = genfromtxt('Data/temps110.csv', delimiter=',')[k:]
T_err = genfromtxt('Data/tempserr + 110.csv', delimiter=',')[k:] * np.sqrt(10 / 9)


fitsss = np.polyfit(N**(-1.) , T,1, cov = True)
print(fitsss)




plt.errorbar(N**-1. , T , yerr= T_err, fmt = 'o', markersize= 5, color = 'blue', label = ' $T_{topological} $')
def func2(x, a, b):
    return a * (x **(-1.)) + b
popt2, pcov2 = curve_fit(func2,N, T, sigma=T_err)
plt.plot(np.append(N**(-1.),0),func2(np.append(N, 1000000000000000000000),*popt2), color = 'blue')
print(popt2)
T = genfromtxt('Data/temps_ext110.csv', delimiter=',')[k:]
T_err = genfromtxt('Data/tempserr_ext110.csv', delimiter=',')[k:] * np.sqrt(10 / 9)
N = np.asarray([48,56,64,72,80,88,96,104,112,120,128])[k:]
plt.errorbar(N**-1. , T , yerr= T_err, fmt = 'o', markersize= 5, color = 'g', label = '$T_{extension} $')
def func2(x, a, b):
    return a * (x **(-1.)) + b
popt, pcov = curve_fit(func2,N, T, sigma=T_err)
print(popt)
plt.plot(np.append(N**(-1.),0),func2(np.append(N, 1000000000000000000000),*popt), color = 'g')
plt.title('T_c-T_topo = ' + str((popt2[1] - 1/critical(J_x,J_y,J_d)[0])/np.sqrt(pcov2[1][1])) + 'ASD'+str((popt[1] - 1/critical(J_x,J_y,J_d)[0])/np.sqrt(pcov[1][1]))  )


plt.hlines([1/critical(J_x,J_y,J_d)[0]], 0 , 0.03, label = 'Theoretical value $T_c$' , color = 'r')
plt.xlabel('$L^{-1}$')
plt.ylabel('$T$')
plt.legend()
plt.xlim(0,0.018)
plt.ylim(2.21,2.4)
plt.locator_params(axis = 'x', nbins = 3)
plt.locator_params(axis = 'y', nbins = 3 )
#plt.show()
plt.cla()




T = genfromtxt('Data/temps110.csv', delimiter=',')[k:]
T_err = genfromtxt('Data/tempserr + 110.csv', delimiter=',')[k:] * np.sqrt(10 / 9)
N = np.asarray([48,56,64,72,80,88,96,104,112,120,128])[k:]

def func2(x, a, b):
    return a * (x **(-1.)) + b
popt2, pcov2 = curve_fit(func2,N, T, sigma=T_err)
print(popt2)

plt.errorbar(N**-1. , (T -func2(N,*popt2))*1000, yerr= T_err*1000, fmt = 'o', markersize= 1)#, label = '$T_c(L) - T_{fit}$')
plt.hlines(0,min(N**(-1.)), max(N**(-1.)), color = 'r')
print(str(sum(((T-(func2(N, *popt2)))/T_err)**2)/(len(T)-len(popt2)) ))
plt.xlabel('$L^{-1}$')
plt.ylabel('$T$ in $10^{-3}$')
#plt.legend()
plt.tight_layout()
plt.locator_params(axis = 'x', nbins = 3)
plt.locator_params(axis = 'y', nbins = 3 )
plt.show()



T = genfromtxt('Data/temps_ext110.csv', delimiter=',')[k:]
T_err = genfromtxt('Data/tempserr_ext110.csv', delimiter=',')[k:] * np.sqrt(10 / 9)
N = np.asarray([48,56,64,72,80,88,96,104,112,120,128])[k:]

def func2(x, a, b):
    return a * (x **(-1.)) + b
popt2, pcov2 = curve_fit(func2,N, T, sigma=T_err)


plt.errorbar(N**-1. , (T -func2(N,*popt2))*1000, yerr= T_err*1000, fmt = 'o', markersize= 1)#, label = '$T_c(L) - T_{fit}$')
plt.hlines(0,min(N**(-1.)), max(N**(-1.)), color = 'r')
print(str(sum(((T-(func2(N, *popt2)))/T_err)**2)/(len(T)-len(popt2)) ))
plt.xlabel('$L^{-1}$')
plt.ylabel('$T$ in $10^{-3}$')
#plt.legend()
plt.tight_layout()
plt.locator_params(axis = 'x', nbins = 3)
plt.locator_params(axis = 'y', nbins = 3 )
plt.show()