import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 30})
my_data = genfromtxt('Data_temp/perco_topo.csv', delimiter=',')

sample_prob = my_data[0]

err_my_data = np.sqrt((my_data[1]  * (1-my_data[1])**2 + (1-my_data[1]) * my_data[1]**2)/10000)
plt.errorbar(sample_prob, my_data[1],yerr = err_my_data, label = '$L = 16$')
err_my_data = np.sqrt((my_data[2]  * (1-my_data[2])**2 + (1-my_data[2]) * my_data[2]**2)/10000)
plt.errorbar(sample_prob, my_data[2],yerr = err_my_data, label = '$L = 32$')
err_my_data = np.sqrt((my_data[3]  * (1-my_data[3])**2 + (1-my_data[3]) * my_data[3]**2)/10000)
plt.errorbar(sample_prob, my_data[3],yerr = err_my_data, label = '$L = 64$')
err_my_data = np.sqrt((my_data[4]  * (1-my_data[4])**2 + (1-my_data[4]) * my_data[4]**2)/10000)
plt.errorbar(sample_prob, my_data[4],yerr = err_my_data, label = '$L = 128$')

sample_prob = np.arange(0.3,0.501,0.01)
percolation_prob = [0.]*len(sample_prob)
plt.plot(sample_prob, percolation_prob, label = 'Thermodynamic limit', color = 'g')

sample_prob = np.arange(0.5,0.7,0.01)
percolation_prob = [1.]*len(sample_prob)
plt.plot(sample_prob, percolation_prob, color = 'g')


plt.vlines(0.5, 0,1 , color = 'g' , linestyles= 'dashed')
plt.xlabel('connecting probability $p$')
plt.ylabel('percolated systems / total systems')


plt.legend()
plt.show()
plt.cla()

my_data = genfromtxt('Data_temp/perco_site_x_y_nonperiodic.csv', delimiter=',')

sample_prob = my_data[0]

err_my_data = np.sqrt((my_data[1]  * (1-my_data[1])**2 + (1-my_data[1]) * my_data[1]**2)/10000)
plt.errorbar(sample_prob, my_data[1],yerr = err_my_data, label = '$L = 16$')
err_my_data = np.sqrt((my_data[2]  * (1-my_data[2])**2 + (1-my_data[2]) * my_data[2]**2)/10000)
plt.errorbar(sample_prob, my_data[2],yerr = err_my_data, label = '$L = 32$')
err_my_data = np.sqrt((my_data[3]  * (1-my_data[3])**2 + (1-my_data[3]) * my_data[3]**2)/10000)
plt.errorbar(sample_prob, my_data[3],yerr = err_my_data, label = '$L = 64$')
err_my_data = np.sqrt((my_data[4]  * (1-my_data[4])**2 + (1-my_data[4]) * my_data[4]**2)/10000)
plt.errorbar(sample_prob, my_data[4],yerr = err_my_data, label = '$L = 128$')

sample_prob = np.arange(0.4,0.501,0.01)
percolation_prob = [0.]*len(sample_prob)
plt.plot(sample_prob, percolation_prob, label = 'Thermodynamic limit', color = 'g')

sample_prob = np.arange(0.5,0.8,0.01)
percolation_prob = [1.]*len(sample_prob)
plt.plot(sample_prob, percolation_prob, color = 'g')


plt.vlines(0.5, 0,1 , color = 'g' , linestyles= 'dashed')
plt.xlabel('connecting probability $p$')
plt.ylabel('percolated systems / total systems')


plt.legend()
plt.show()
plt.cla()



my_data = genfromtxt('Data_temp/perco_ext.csv', delimiter=',')

sample_prob = my_data[0]

err_my_data = np.sqrt((my_data[1]  * (1-my_data[1])**2 + (1-my_data[1]) * my_data[1]**2)/10000)
plt.errorbar(sample_prob, my_data[1],yerr = err_my_data, label = '$L = 16$')
err_my_data = np.sqrt((my_data[2]  * (1-my_data[2])**2 + (1-my_data[2]) * my_data[2]**2)/10000)
plt.errorbar(sample_prob, my_data[2],yerr = err_my_data, label = '$L = 32$')
err_my_data = np.sqrt((my_data[3]  * (1-my_data[3])**2 + (1-my_data[3]) * my_data[3]**2)/10000)
plt.errorbar(sample_prob, my_data[3],yerr = err_my_data, label = '$L = 64$')
err_my_data = np.sqrt((my_data[4]  * (1-my_data[4])**2 + (1-my_data[4]) * my_data[4]**2)/10000)
plt.errorbar(sample_prob, my_data[4],yerr = err_my_data, label = '$L = 128$')

sample_prob = np.arange(0.2,0.501,0.01)
percolation_prob = [0.]*len(sample_prob)
plt.plot(sample_prob, percolation_prob, label = 'Thermodynamic limit', color = 'g')

sample_prob = np.arange(0.5,0.6,0.01)
percolation_prob = [1.]*len(sample_prob)
plt.plot(sample_prob, percolation_prob, color = 'g')


plt.vlines(0.5, 0,1 , color = 'g' , linestyles= 'dashed')
plt.xlabel('connecting probability $p$')
plt.ylabel('percolated systems / total systems')


plt.legend()
plt.show()
plt.cla()





########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

my_data = genfromtxt('Data_temp/perco_topo.csv', delimiter=',')
err = []
sample_prob = my_data[0]
id_1 = (np.abs(my_data[1] - 0.5)). argmin()
err.append(np.sqrt((my_data[1][id_1]  * (1-my_data[1][id_1])**2 + (1-my_data[1][id_1]) * my_data[1][id_1]**2)/10000))
id_2 = (np.abs(my_data[2] - 0.5)). argmin()
err.append(np.sqrt((my_data[2][id_2]  * (1-my_data[2][id_2])**2 + (1-my_data[2][id_2]) * my_data[2][id_2]**2)/10000))
id_3 = (np.abs(my_data[3] - 0.5)). argmin()
err.append(np.sqrt((my_data[3][id_3]  * (1-my_data[3][id_3])**2 + (1-my_data[3][id_3]) * my_data[3][id_3]**2)/10000))
id_4 = (np.abs(my_data[4] - 0.5)). argmin()
err.append(np.sqrt((my_data[4][id_4]  * (1-my_data[4][id_4])**2 + (1-my_data[4][id_4]) * my_data[4][id_4]**2)/10000))
err = [0.01] * 4
err = np.asarray(err)/np.sqrt(12)
scaling_probs = [sample_prob[id_1],sample_prob[id_2],sample_prob[id_3],sample_prob[id_4]]
systemsizes = np.array([16,32,64,128])

plt.errorbar(systemsizes**(-3/4), scaling_probs, yerr= err  , fmt = 'o', color = 'r', label ='$p_c(L)$ for percolation- \n probability $p_p  = 0.5$ ')

def func2(x, a,b ):
    return a * (x **(-3/4)) + b


popt2, pcov2 = curve_fit(func2,systemsizes, scaling_probs, sigma=err)
print(popt2, np.sqrt(pcov2[0][0]), np.sqrt(pcov2[1][1]))
plt.plot(np.append(systemsizes**(-3/4),0),func2(np.append(systemsizes,10000000000000),*popt2), color = 'r')
print("chiq_0.5", sum(((scaling_probs-(func2(systemsizes, *popt2)))/err)**2)/(len(scaling_probs)-len(popt2)))
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
my_data = genfromtxt('Data_temp/perco_topo.csv', delimiter=',')
err = []
sample_prob = my_data[0]
id_1 = (np.abs(my_data[1] - 0.15)). argmin()
err.append(np.sqrt((my_data[1][id_1]  * (1-my_data[1][id_1])**2 + (1-my_data[1][id_1]) * my_data[1][id_1]**2)/10000))
id_2 = (np.abs(my_data[2] - 0.15)). argmin()
err.append(np.sqrt((my_data[2][id_2]  * (1-my_data[2][id_2])**2 + (1-my_data[2][id_2]) * my_data[2][id_2]**2)/10000))
id_3 = (np.abs(my_data[3] - 0.15)). argmin()
err.append(np.sqrt((my_data[3][id_3]  * (1-my_data[3][id_3])**2 + (1-my_data[3][id_3]) * my_data[3][id_3]**2)/10000))
id_4 = (np.abs(my_data[4] - 0.15)). argmin()
err.append(np.sqrt((my_data[4][id_4]  * (1-my_data[4][id_4])**2 + (1-my_data[4][id_4]) * my_data[4][id_4]**2)/10000))
scaling_probs = [sample_prob[id_1],sample_prob[id_2],sample_prob[id_3],sample_prob[id_4]]
systemsizes = np.array([16,32,64,128])
err = [0.01] * 4
err = np.asarray(err)/np.sqrt(12)
plt.errorbar(systemsizes**(-3/4), scaling_probs, yerr= err  , fmt = 'o', color = 'g', label = '$p_c(L)$ for percolation- \n probability $p_p = 0.3$ ')

def func2(x, a,b ):
    return a * (x **(-3/4)) + b


popt2, pcov2 = curve_fit(func2,systemsizes, scaling_probs, sigma=err)
print(popt2, np.sqrt(pcov2[0][0]), np.sqrt(pcov2[1][1]))
plt.plot(np.append(systemsizes**(-3/4),0),func2(np.append(systemsizes,10000000000000),*popt2), color = 'g')
print("chiq_0.3", sum(((scaling_probs-(func2(systemsizes, *popt2)))/err)**2)/(len(scaling_probs)-len(popt2)))

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
''' 
my_data = genfromtxt('./data/perco_ext.csv', delimiter=',')
err = []
sample_prob = my_data[0]
id_1 = (np.abs(my_data[1] - 0.5)). argmin()
id_2 = (np.abs(my_data[2] - 0.5)). argmin()
id_3 = (np.abs(my_data[3] - 0.5)). argmin()
id_4 = (np.abs(my_data[4] - 0.5)). argmin()
err = [0.01] * 4
err = np.asarray(err)/np.sqrt(12)
scaling_probs = [sample_prob[id_1],sample_prob[id_2],sample_prob[id_3],sample_prob[id_4]]
systemsizes = np.array([16,32,64,128])

plt.errorbar(systemsizes**(-3/4), scaling_probs, yerr= err  , fmt = 'o', color = 'b', label ='$p_c(L)$ for percolation- \n probability $p_p$ = 0.5 ')

def func2(x, a,b ):
    return a * (x **(-3/4)) + b


popt2, pcov2 = curve_fit(func2,systemsizes, scaling_probs, sigma=err)
print(popt2, np.sqrt(pcov2[0][0]), np.sqrt(pcov2[1][1]))
plt.plot(np.append(systemsizes**(-3/4),0),func2(np.append(systemsizes,10000000000000),*popt2), color = 'r')
print("chiq_0.5", sum(((scaling_probs-(func2(systemsizes, *popt2)))/err)**2)/(len(scaling_probs)-len(popt2)))
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

my_data = genfromtxt('./data/perco_ext.csv', delimiter=',')
err = []
sample_prob = my_data[0]
id_1 = (np.abs(my_data[1] - 0.15)). argmin()
id_2 = (np.abs(my_data[2] - 0.15)). argmin()
id_3 = (np.abs(my_data[3] - 0.15)). argmin()
id_4 = (np.abs(my_data[4] - 0.15)). argmin()
scaling_probs = [sample_prob[id_1],sample_prob[id_2],sample_prob[id_3],sample_prob[id_4]]
systemsizes = np.array([16,32,64,128])
err = [0.01] * 4
err = np.asarray(err)/np.sqrt(12)
plt.errorbar(systemsizes**(-3/4), scaling_probs, yerr= err  , fmt = 'o', color = 'orange', label = '$p_c(L)$ for percolation- \n probability $p_p$ = 0.3 ')

def func2(x, a,b ):
    return a * (x **(-3/4)) + b


popt2, pcov2 = curve_fit(func2,systemsizes, scaling_probs, sigma=err)
print(popt2, np.sqrt(pcov2[0][0]), np.sqrt(pcov2[1][1]))
plt.plot(np.append(systemsizes**(-3/4),0),func2(np.append(systemsizes,10000000000000),*popt2), color = 'g')
print("chiq_0.3", sum(((scaling_probs-(func2(systemsizes, *popt2)))/err)**2)/(len(scaling_probs)-len(popt2)))

'''



plt.xlabel('$L^{-3/4}$')
plt.ylabel('$p_c(L)$')
plt.legend()
plt.xlim(0,max(systemsizes**(-3/4))+0.02)
plt.ylim(0.41,0.55)
plt.locator_params(axis = 'x', nbins = 5)
plt.locator_params(axis = 'y', nbins = 5 )
plt.show()