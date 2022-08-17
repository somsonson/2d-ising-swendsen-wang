import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit , prange
from functions import initgrid
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
from numpy import savetxt
from math import log
import timeit
from scipy.optimize import curve_fit
from functions import criticalq3
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
plt.rcParams.update({'font.size': 22})

k =0
#N = np.asarray([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120])
#N = np.asarray([10,20,30,40,50,60,70,80,90,100])
#N = np.asarray([16,24,32,40,48,56,64])[k:]
#N = np.asarray([40,42,46,48,50,52,54,56,58,60,62,64,66,68])
N = np.asarray([64,72,80,88,96,104,112,120,128])[k:]
#N = np.asarray([20,25,30,35,40,45,50,55])[k:]
#N = np.asarray([20,25,30,35,40,45,50,55,60,65,70,75])[k:]

J_x = float(1.0)
J_y = float(1.0)
J_d = float(0.0)

T = []
T_err = []

for i in N:
    T.append(genfromtxt('data/q3temps'+str(J_x)+str(J_y)+str(J_d)+str([i])+'.csv', delimiter=','))
    T_err.append(genfromtxt('data/q3tempserr'+str(J_x)+str(J_y)+str(J_d)+str([i])+'.csv', delimiter=',')* np.sqrt(10/9))


plt.errorbar(N**(-6/5) , T , yerr= T_err, fmt = 'o', markersize= 3, color = 'r')


#def func(x, a, b,c ):
#    return a * (x **(-6/5)) + b + c * ((x **(-6/5)))**2

def func2(x, a,b,c):
    return a * (x **(-6/5)) + b+ c * ((x **(-6/5)))**2

#popt, pcov = curve_fit(func,N, T, sigma=T_err)
popt2, pcov2 = curve_fit(func2,N, T, sigma=T_err)#, p0 =[0,0,-6/5])


print(popt2,np.sqrt(pcov2[1][1]))
print(criticalq3(J_x,J_y,J_d))
#print(np.sqrt(pcov[1][1]))


#n_std = np.abs((1/(log(1 + np.sqrt(3)))- popt[1])/np.sqrt(pcov[1][1]))
n_std2 = np.abs((criticalq3(J_x,J_y,J_d)- popt2[1])/np.sqrt(pcov2[1][1]))

#plt.plot(np.append(N**(-6/5),0),func(np.append(N, 1000000000000000000000),*popt), label = 'crit temp T =' + str(round(popt[1],8))+ '+/-' + str(round(np.sqrt(pcov[1][1]),8)))
plt.plot(np.append(N**(-6/5),0),func2(np.append(N, 1000000000000000000000),*popt2), label = 'crit temp T1 =' + str(round(popt2[1],8))+ '+/-' + str(round(np.sqrt(pcov2[1][1]),8)))

plt.hlines(criticalq3(J_x,J_y,J_d), 0 , 0.01, label = 'theoretical value' + str(2/(log(1 + np.sqrt(3)))))
plt.xlabel('L^-1')
plt.ylabel('T')
plt.title('Temperatures for sizes N =' + str(N) + " and J_x = "+ str(J_x) + " J_y =" + str(J_y)+ " J_d = " + str(J_d)+ "\n n-std-deviation" + str(n_std2) )
plt.legend()
plt.savefig("plots/11swendsen_perco_topo_q3"+ str(N) + " and J_x = "+ str(J_x) + " J_y =" + str(J_y)+ " J_d = " + str(J_d) +".pdf" )
#plt.show()

plt.cla()
#plt.title(   str(n_std2) +"chiq/dof" +str(sum(((T-(func2(N, *popt2)))/T_err)**2)/(len(T)-len(popt2)) ))
plt.locator_params(axis = 'x', nbins = 3)
plt.locator_params(axis = 'y', nbins = 3)
#plt.errorbar(N**(-6/5)+0.001 , T -func(N,*popt), yerr= T_err, fmt = 'o', markersize= 1)
plt.errorbar(N**(-6/5) , (T -func2(N,*popt2))*1000, yerr= np.asarray(T_err)*1000, fmt = 'o', markersize= 1)
print(str(sum(((T-(func2(N, *popt2)))/T_err)**2)/(len(T)-len(popt2)) ))


plt.hlines(0,min(N**(-6/5)), max(N**(-6/5)), color='r')
plt.savefig("plots/11swendsen_perco_residue_topo_q3"+ str(N) + " and J_x = "+ str(J_x) + " J_y =" + str(J_y)+ " J_d = " + str(J_d) +".pdf" )
plt.xlabel('$L^{-1}$')
plt.ylabel('$T$ in $10^{-3}$')
plt.tight_layout()
plt.show()
