import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
from scipy.optimize import curve_fit
import math
plt.rcParams.update({'font.size': 28})

my_data = genfromtxt('Data_temp/hist_data_iso.csv', delimiter=',')
#my_data = genfromtxt('./data/hist_data_aniso.csv', delimiter=',')


fig, ax1 = plt.subplots()

ax1.hist(my_data,bins=50, range= (2.2,2.35))
#ax1.hist(my_data,bins=81 ,  range= (11.5,14.0))
ax1.set_yscale('log')
ax1.set_xlabel('$T$')
ax1.set_ylabel('$\#$ ')
plt.tight_layout()
y_vals = ax1.get_yticks()
#ax1.set_yticklabels(['{:3.0f}'.format(x / 1000) for x in y_vals])

plt.show()
