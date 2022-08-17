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
from scipy.optimize import fsolve

plt.rcParams.update({'font.size': 28})
plt.locator_params(axis = 'x', nbins = 4)
plt.locator_params(axis = 'y', nbins = 4)




p_fix_10 = genfromtxt('data/p_fix_10.csv', delimiter=',')
p_var_calculated_10 = genfromtxt('data/p_var_10.csv', delimiter=',')
err_p_var_calculated_10 = genfromtxt('data/p_err_10.csv', delimiter=',')




print(len(p_var_calculated_10))
print(len(err_p_var_calculated_10))
print(len(p_fix_10))

p_fix_20 = genfromtxt('data/p_fix_20.csv', delimiter=',')
p_var_calculated_20 = genfromtxt('data/p_var_20.csv', delimiter=',')
err_p_var_calculated_20 = genfromtxt('data/p_err_20.csv', delimiter=',')



p_fix_30 = genfromtxt('data/p_fix_30.csv', delimiter=',')
p_var_calculated_30 = genfromtxt('data/p_var_30.csv', delimiter=',')
err_p_var_calculated_30 = genfromtxt('data/p_err_30.csv', delimiter=',')


p_fix_40 = genfromtxt('data/p_fix_40.csv', delimiter=',')
p_var_calculated_40 = genfromtxt('data/p_var_40.csv', delimiter=',')
err_p_var_calculated_40 = genfromtxt('data/p_err_40.csv', delimiter=',')









#p_fix_50 = genfromtxt('./Data/p_fix_50.csv', delimiter=',')
#p_var_50 = genfromtxt('./Data/p_var_50.csv', delimiter=',')


#p_var_calculated_50 = []
#err_p_var_calculated_50 = []

#p_var_calculated_50 = p_var_50[::2]
#err_p_var_calculated_50 = p_var_50[1::2]





plt.errorbar( p_fix_10, p_var_calculated_10,yerr = err_p_var_calculated_10,fmt = 'o', markersize = 3,label='$L=8$')
plt.errorbar( p_fix_20, p_var_calculated_20,yerr = err_p_var_calculated_20,fmt = 'o',markersize = 3, label= '$L=16$')
plt.errorbar( p_fix_30, p_var_calculated_30,yerr = err_p_var_calculated_30,fmt = 'o',markersize = 3, label = '$L=32$')
plt.errorbar(p_fix_40, p_var_calculated_40,yerr = err_p_var_calculated_40,fmt = 'o',markersize = 3, label = '$L=64$')
#plt.errorbar(p_fix_50, p_var_calculated_50,yerr = err_p_var_calculated_50,fmt = 'o',markersize = 2, label = 'N=5 0')

#plt.errorbar(p_fix_50, p_var_calculated_50,yerr = err_p_var_calculated_50,fmt = 'o',markersize = 2)

plt.tight_layout()


def f(p_c):
    return 1-np.exp(-np.arcsinh(((1)/(np.sinh(-np.log(1-p_c))))))



p_c = np.arange(0.001,1.001+0.01,0.01)
#plt.plot(p_c, [f(i) for i in p_c], label = 'theoretical values')


plt.xlabel('$p_x$')
plt.ylabel('$p_y$')

T = np.append((1/np.arange(0.001,1.101,0.001)),np.arange(0.01,1,0.001)[::-1])

def p(J,T):
    return 1-np.exp(-2*(1/T)*J)


for i in range(1,2):
    plt.plot(p(i*10,T),p(1,T),label= "$J_x =$" + str(i*10) + "$\quad J_y = 1$" )

for i in range(1,3):
    plt.plot(p(1,T),p(i,T),label= "$J_x =1$"  + "$\quad J_y = $" + str(i))




def critical_p(p_x, p_d):
    def f(B):
        return sinh(- np.log(1-p_x)) * sinh(- np.log(1-B)) + sinh(- np.log(1-p_d)) * (
                sinh(- np.log(1-p_x)) + sinh(- np.log(1-B)) )- 1

    c = fsolve(f, 1-p_x, maxfev=400000000)
    return c


y = [critical_p(p_c[i], 0.0) for i in range(0,len(p_c))]
for i in range(0,len(y)):
    if y[i] < 0:
        y[i] = 0
plt.plot(p_c , y, label = 'Theoretical solution')



###############################################
# q = 3 case

def critical_p_q3(p_x, p_d):
    def f(p_y):
        return (1/(1-p_x) - 1)*(1/(1-p_y) - 1)*(1/(1-p_d) - 1) + (1/(1-p_x) - 1)*(1/(1-p_y) - 1) + (1/(1-p_x) - 1)*(1/(1-p_d) - 1) + (1/(1-p_y) - 1)*(1/(1-p_d) - 1) -3

    c = fsolve(f, 1-p_x, maxfev=400000000)
    return c

y = [critical_p_q3(p_c[i], 0.0) for i in range(0,len(p_c))]
for i in range(0,len(y)):
    if y[i] < 0:
        y[i] = 0
#plt.plot(p_c , y, label = 'Theoretical solution')










plt.axis('square')
plt.legend(loc = 'upper right',fontsize=16.5)

plt.show()
plt.cla()


