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


#N = np.asarray([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120])
#N = np.asarray([10,20,30,40,50,60,70,80,90,100])
#N = np.asarray([16,24,32,40,48,56,64])[k:]
#N = np.asarray([40,42,46,48,50,52,54,56,58,60,62,64,66,68])

N = 50


J_x = 1/2
J_y = 1/2
J_d = 0

T = genfromtxt('Data_temp/q3temps_J.csv', delimiter=',')
T_err = genfromtxt('Data_temp/q3tempserr_J.csv', delimiter=',')

plt.errorbar(np.arange(1,10), T , yerr = T_err , fmt = 'o')
plt.show()