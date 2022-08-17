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
from numpy import genfromtxt



p_crit_periodic_x_perc = genfromtxt('Data_temp/p_crit_periodic_x_perc.csv', delimiter=',')[2:]
p_crit_non_periodic_x_perc = genfromtxt('Data_temp/p_crit_non_periodic_x_perc.csv', delimiter=',')[2:]
p_crit_periodic_x_perc_03 = genfromtxt('Data_temp/p_crit_periodic_x_perc_03.csv', delimiter=',')[2:]
p_crit_non_periodic_x_perc_03 = genfromtxt('Data_temp/p_crit_non_periodic_x_perc_03.csv', delimiter=',')[2:]
N = genfromtxt('Data_temp/N_values.csv', delimiter=',')[2:]



plt.axhline(y=0.0, color='r', linestyle='-')
plt.ylabel('$L^{-3/4}$')
plt.xlabel('p')
plt.xlim(0.4,0.6)

def func(x, a, b):
	return a * x + b



popt, pcov = curve_fit(func, p_crit_periodic_x_perc,N**(-3/4))
popt2, pcov2 = curve_fit(func, p_crit_non_periodic_x_perc,N**(-3/4))
popt3, pcov3 = curve_fit(func, p_crit_periodic_x_perc_03,N**(-3/4))
popt4, pcov4 = curve_fit(func, p_crit_non_periodic_x_perc_03,N**(-3/4))



print(-popt[1]/popt[0])
print(-popt2[1]/popt2[0])
print(-popt3[1]/popt3[0])
print(-popt4[1]/popt4[0])

plt.plot(np.append(p_crit_periodic_x_perc, 0.499),func(np.append(p_crit_periodic_x_perc, 0.499),popt[0],popt[1]),label = 'x-perc non periodic, p=0.5, result = 0.4997')
plt.plot(np.append(p_crit_non_periodic_x_perc, 0.4997),func(np.append(p_crit_non_periodic_x_perc, 0.4997),popt2[0],popt2[1]),label = 'x-perc periodic, p=0.5 result = 0.4970')
plt.plot(np.append(p_crit_periodic_x_perc_03, 0.5),func(np.append(p_crit_periodic_x_perc_03, 0.5),popt3[0],popt3[1]),label = 'x-perc periodic, p=0.3 result = 0.4970')
plt.plot(np.append(p_crit_non_periodic_x_perc_03, 0.5),func(np.append(p_crit_non_periodic_x_perc_03, 0.5),popt4[0],popt4[1]),label = 'x-perc non-periodic, p=0.5 result = 0.499')

#plt.plot(np.append(p3, 0.51),func(np.append(p3, 0.51),popt3[0],popt3[1]),label = 'x-perc periodic, p=0.3, result = 0.4999')

#plt.scatter(p2,N2**(-3/4))
#plt.scatter(p3,N3**(-3/4))
plt.scatter(p_crit_periodic_x_perc,N**(-3/4))
plt.scatter(p_crit_non_periodic_x_perc,N**(-3/4))
plt.scatter(p_crit_periodic_x_perc_03,N**(-3/4))
plt.scatter(p_crit_non_periodic_x_perc_03,N**(-3/4))

plt.legend()
plt.show()
plt.savefig('square')