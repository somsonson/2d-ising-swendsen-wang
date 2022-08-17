from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit, prange
from functions import initgrid
from functions import critical
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
from functions import decision
from scipy.signal import savgol_filter
import scipy
from numpy import savetxt
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
from PIL import Image
from numpy import sinh
plt.rcParams.update({'font.size': 25})

def M(T):
    N=1
    if T< 1 / critical(1, 1, 0)[0]:
        return N*(1-sinh(2/T)**(-4))**(1/8)
    else:
        return 0

data = np.asarray([1, -1.99714, 7.34105e-05,
1.5, -1.95107, 0.000300177,
2 ,-1.74489 ,0.00119636,
2.5, -1.2189 ,0.00253109,
3 ,-0.842481, 0.00107065,
3.5, -0.665117, 0.000754289,
4, -0.557586, 0.000609268,
4.5, -0.483793, 0.000775787,
5, -0.428631, 0.000511742,
5.5 ,-0.384788, 0.000603265,
6 ,-0.349588 ,0.000629367,
6.5, -0.3205, 0.000503484,
7, -0.296854, 0.00058267,
7.5, -0.274962 ,0.000456312,
8, -0.255631, 0.000324044,
8.5, -0.239997, 0.000397869,
9 ,-0.226759, 0.000381908,
9.5, -0.215103, 0.000397888,
10, -0.203263, 0.000366705])

energy_data = np.asarray([])
temp_data = np.asarray([])
energy_data_std = np.asarray([])

for i in range(0,len(data)):
    if i % 3  == 1:
        energy_data= np.append(energy_data,data[i])
    if i % 3 == 0:
        temp_data= np.append(temp_data,data[i])
    if i % 3 == 2:
        energy_data_std= np.append(energy_data_std,data[i])

avg_energy = genfromtxt('Data_temp/avg_energy.csv', delimiter=',')
std_energy = genfromtxt('Data_temp/std_energy.csv', delimiter=',')
avg_magnet = genfromtxt('Data_temp/avg_magn.csv', delimiter=',')
std_magn = genfromtxt('Data_temp/std_magn.csv', delimiter=',')

temp = genfromtxt('Data_temp/temp.csv', delimiter=',')

C_v = np.gradient(avg_energy, temp)
# dydx = scipy.signal.savgol_filter(energy, window_length=11, polyorder=10, deriv=1)




''' 

fig, ax = plt.subplots(2)
ax[0].errorbar(temp, avg_energy, yerr=std_energy, fmt='.k', markersize=5, elinewidth=1, capsize=0, label='data')
ax[0].vlines(1 / critical(1, 1, 0)[0], min(avg_energy), max(avg_energy), label='$T_c$', color='r')
ax[0].tick_params(left = True, right = False , labelleft = True ,
                labelbottom = False, bottom = True)

#ax[0].hlines(0, min(temp), max(temp))
#ax[0].errorbar(temp_data,energy_data, yerr= energy_data_std,fmt='.k', markersize=7, elinewidth=1, capsize=0, label='given-data', color = 'r')
ax[1].set_xlabel('$T$ ')
ax[0].set_ylabel('$E(T)$ ')
ax[1].scatter(temp, C_v, marker='o', s=5, color = 'black' )
ax[1].set_ylabel('$dE/dT$')
ax[1].vlines(1 / critical(1, 1, 0)[0], min(C_v), max(C_v), label=' $T_c$ ', color='r')
ax[0].legend()
ax[0].set_box_aspect(1/2)
ax[1].set_box_aspect(1/2)
#ax[0].set_xlim((min(temp),max(temp)))

#ax[0].set_ylim(-2.1,0)
#ax[0].set_xscale('log')
#ax[1].set_xscale('log')
ax[0].legend(loc = "upper left")
plt.tight_layout()
plt.savefig('swendsen-wang-energy_low.pdf')
#lt.show()


plt.errorbar(temp, avg_magnet, yerr=std_magn, fmt='.k', markersize=5, elinewidth=1, capsize=0, label='data')
plt.xlabel('$T$')

plt.ylabel('$<|m|>$')
#plt.hlines(0, min(temp), max(temp))
plt.ylim(0,1)
plt.vlines(1 / critical(1, 10, 0)[0], min(avg_magnet), max(avg_magnet), label='$T_c$', color='r')
plt.legend()
plt.tight_layout()
#plt.plot(temp , [M(i) for i in temp])
plt.savefig('swendsen-wang-magnetisation_250_2.pdf')
plt.cla()
#plt.show()

#print(energy_data)
#print(avg_energy/64)
#print((energy_data-(avg_energy/64))/energy_data)
'''

fig, ax2 = plt.subplots(2,1)
#ax2[1].locator_params(axis = 'x', nbins = 3)
#ax2[1].locator_params(axis = 'y', nbins = 3)
ax2[0].errorbar(temp , avg_energy , yerr=std_energy , fmt='.k', markersize=7, elinewidth=1, capsize=0, label='Swendsen-Wang algorithm')
#ax2[0].vlines(1 / critical(1, 1, 0)[0], min(avg_energy), max(avg_energy), label='theoretical estimation', color='r')
ax2[0].hlines(0, min(temp), max(temp))
ax2[0].errorbar(temp_data ,energy_data , yerr= energy_data_std ,fmt='.k', markersize=7, elinewidth=1, capsize=0, label='Metropolis update algorithm', color = 'r')
ax2[0].set_ylabel('$E(T)/64$  ')
ax2[0].set_xlabel('$T$ ')

ax2[0].legend()
ax2[0].set_ylim((-3,0))
print(avg_energy[0::2]/64)
print(energy_data)
print(avg_energy/64)
ax2[1].scatter(temp_data ,100* (avg_energy -energy_data )/energy_data , color = 'r')
ax2[1].set_ylabel('Relative deviation \n in $\%$')
ax2[1].set_xlabel('$T$ ')
ax2[1].hlines(0, min(temp), max(temp), color = 'black')
ax2[1].set_ylim([-1, 1])
#plt.tight_layout()
plt.savefig('swendsen-wang-energy-comp.pdf')
plt.show()

print(np.equal(temp_data, temp))
print(temp_data,temp)

#print(temp_data_2)
