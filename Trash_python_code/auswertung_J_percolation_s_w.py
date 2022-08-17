import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit , prange
from functions import initgrid
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
from functions import check_percolation_periodic
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


def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y



p_fix_10 = genfromtxt('data/p_fix_10.csv', delimiter=',')
p_var_10 = genfromtxt('data/T_val_10.csv', delimiter=',')

p_var_calculated_10 = []
err_p_var_calculated_10 = []
#print(min(T_a[4]), max(T_a[7]))
for p_data in p_var_10:
    n,bins,patches = plt.hist(p_data, 100,facecolor='green',align='mid')
    # plt.show()
    #p_var_calculated_10.append(bins[np.where(n == max(n))][0])
    (mu,sigma) = norm.fit(p_data)
    p_var_calculated_10.append(mu)
    err_p_var_calculated_10.append(sigma)
    print(mu)
    plt.show()
    plt.cla()


p_fix_20 = genfromtxt('data/p_fix_20.csv', delimiter=',')
p_var_20 = genfromtxt('data/T_val_20.csv', delimiter=',')

p_var_calculated_20 = []
err_p_var_calculated_20 = []
#print(min(T_a[4]), max(T_a[7]))
for p_data in p_var_20:
    n,bins,patches = plt.hist(p_data, 100,facecolor='green',align='mid')
    # plt.show()
    #p_var_calculated_10.append(bins[np.where(n == max(n))][0])
    (mu,sigma) = norm.fit(p_data)
    p_var_calculated_20.append(mu)
    err_p_var_calculated_20.append(sigma)
    print(mu)
    #plt.show()
    plt.cla()

plt.cla()


p_fix_30 = genfromtxt('data/p_fix_30.csv', delimiter=',')
p_var_30 = genfromtxt('data/T_val_30.csv', delimiter=',')

p_var_calculated_30 = []
err_p_var_calculated_30 = []
#print(min(T_a[4]), max(T_a[7]))
for p_data in p_var_30:
    n,bins,patches = plt.hist(p_data, 100,facecolor='green',align='mid')
    # plt.show()
    #p_var_calculated_10.append(bins[np.where(n == max(n))][0])
    (mu,sigma) = norm.fit(p_data)
    p_var_calculated_30.append(mu)
    err_p_var_calculated_30.append(sigma)
    print(mu)
    #plt.show()
    plt.cla()



p_fix_40 = genfromtxt('data/p_fix_40.csv', delimiter=',')
p_var_40 = genfromtxt('data/T_val_40.csv', delimiter=',')

p_var_calculated_40 = []
err_p_var_calculated_40 = []

#print(min(T_a[4]), max(T_a[7]))
for p_data in p_var_40:
    n,bins,patches = plt.hist(p_data, 100,facecolor='green',align='mid')
    # plt.show()
    #p_var_calculated_10.append(bins[np.where(n == max(n))][0])
    (mu,sigma) = norm.fit(p_data)
    p_var_calculated_40.append(mu)
    err_p_var_calculated_40.append(sigma)
    print(mu)
    #plt.show()
    plt.cla()




p_fix_50 = genfromtxt('data/p_fix_50.csv', delimiter=',')
p_var_50 = genfromtxt('data/T_val_50.csv', delimiter=',')


p_var_calculated_50 = []
err_p_var_calculated_50 = []

for p_data in p_var_50:
    n,bins,patches = plt.hist(p_data, 100,facecolor='green',align='mid')
    # plt.show()
    #p_var_calculated_10.append(bins[np.where(n == max(n))][0])
    (mu,sigma) = norm.fit(p_data)
    p_var_calculated_50.append(mu)
    err_p_var_calculated_50.append(sigma)
    print(mu)
    plt.show()
    plt.cla()



plt.errorbar(p_var_calculated_10 / p_fix_10, p_var_calculated_10,yerr = err_p_var_calculated_10,fmt = 'o', markersize = 2,label='N=10')
plt.errorbar(p_var_calculated_20 / p_fix_20, p_var_calculated_20,yerr = err_p_var_calculated_20,fmt = 'o',markersize = 2, label= 'N=20')
plt.errorbar(p_var_calculated_30 / p_fix_30, p_var_calculated_30,yerr = err_p_var_calculated_30,fmt = 'o',markersize = 2, label = 'N=30')
plt.errorbar(p_var_calculated_40 / p_fix_40, p_var_calculated_40,yerr = err_p_var_calculated_40,fmt = 'o',markersize = 2, label = 'N=40')
#plt.errorbar(p_var_calculated_50 / p_fix_50, p_var_calculated_50,yerr = err_p_var_calculated_50,fmt = 'o',markersize = 2)
plt.legend()
plt.title('critiacal probabilities')
plt.tight_layout()


def f(p_c):
    return 1-np.exp(-np.arcsinh(((1)/(np.sinh(-np.log(1-p_c))))))

def f2(p_c):
    return 1-np.exp(-np.arcsinh(((1)/(np.sinh(-np.log(1-p_c)))))) - p_c * factor

p_c = np.arange(0.0,1+0.04,0.04)




plt.plot(p_c, [f(i) for i in p_c], label = 'theoretical values')




plt.xlabel('p_x')
plt.ylabel('p_y')
plt.show()
plt.cla()

N=np.asarray([10.,20.,40.,80.])

def func(x, a, b,c):
    return a * x ** (c) + b

gw = []

for t in range(0,len(p_fix_10)):
    T_1 = -2/(np.log(1-np.asarray([p_var_calculated_10[t],p_var_calculated_20[t],p_var_calculated_30[t],p_var_calculated_40[t]])))
    #err_p_var_calculated = [err_p_var_calculated_10[t],err_p_var_calculated_20[t],err_p_var_calculated_30[t],err_p_var_calculated_40[t]]
    popt, pcov = curve_fit(func, N, T_1, p0=[0., 1., -1])
    print(popt)
    gw = [popt[1]]
    factor = p_fix_50[t]
    #sol = [f(popt[1]/factor)]
    plt.scatter(N**(-1),T_1, color = 'b' )
    plt.scatter([0.],gw,color='r')
    #plt.scatter([0.],sol,color='g')
    plt.show()
    plt.cla()












def bin():
    def func(x, a, b,c):
        return a * x**(-c) + b


    N = [10,20,30,40,50]
    y_1 = [p_var_calculated_10[0],p_var_calculated_20[0],p_var_calculated_30[0],p_var_calculated_40[0],p_var_calculated_50[0]]
    y_2 = [p_var_calculated_10[1],p_var_calculated_20[1],p_var_calculated_30[1],p_var_calculated_40[1],p_var_calculated_50[1]]
    y_3 = [p_var_calculated_10[2],p_var_calculated_20[2],p_var_calculated_30[2],p_var_calculated_40[2],p_var_calculated_50[2]]
    y_4 = [p_var_calculated_10[3],p_var_calculated_20[3],p_var_calculated_30[3],p_var_calculated_40[3],p_var_calculated_50[3]]
    y_5 = [p_var_calculated_10[4],p_var_calculated_20[4],p_var_calculated_30[4],p_var_calculated_40[4],p_var_calculated_50[4]]
    y_6 = [p_var_calculated_10[5],p_var_calculated_20[5],p_var_calculated_30[5],p_var_calculated_40[5],p_var_calculated_50[5]]
    y_7 = [p_var_calculated_10[6],p_var_calculated_20[6],p_var_calculated_30[6],p_var_calculated_40[6],p_var_calculated_50[6]]
    y_8 = [p_var_calculated_10[7],p_var_calculated_20[7],p_var_calculated_30[7],p_var_calculated_40[7],p_var_calculated_50[7]]
    y_9 = [p_var_calculated_10[8],p_var_calculated_20[8],p_var_calculated_30[8],p_var_calculated_40[8],p_var_calculated_50[8]]
    y_10 =[p_var_calculated_10[9],p_var_calculated_20[9],p_var_calculated_30[9],p_var_calculated_40[9],p_var_calculated_50[9]]
    y_11 =[p_var_calculated_10[10],p_var_calculated_20[10],p_var_calculated_30[10],p_var_calculated_40[10],p_var_calculated_50[10]]
    y_12 =[p_var_calculated_10[11],p_var_calculated_20[11],p_var_calculated_30[11],p_var_calculated_40[11],p_var_calculated_50[11]]
    y_13 =[p_var_calculated_10[12],p_var_calculated_20[12],p_var_calculated_30[12],p_var_calculated_40[12],p_var_calculated_50[12]]
    y_14 =[p_var_calculated_10[13],p_var_calculated_20[13],p_var_calculated_30[13],p_var_calculated_40[13],p_var_calculated_50[13]]
    y_15 =[p_var_calculated_10[14],p_var_calculated_20[14],p_var_calculated_30[14],p_var_calculated_40[14],p_var_calculated_50[14]]
    y_16 =[p_var_calculated_10[15],p_var_calculated_20[15],p_var_calculated_30[15],p_var_calculated_40[15],p_var_calculated_50[15]]
    y_17 =[p_var_calculated_10[16],p_var_calculated_20[16],p_var_calculated_30[16],p_var_calculated_40[16],p_var_calculated_50[16]]
    y_18 =[p_var_calculated_10[17],p_var_calculated_20[17],p_var_calculated_30[17],p_var_calculated_40[17],p_var_calculated_50[17]]
    y_19 =[p_var_calculated_10[18],p_var_calculated_20[18],p_var_calculated_30[18],p_var_calculated_40[18],p_var_calculated_50[18]]
    y_20 =[p_var_calculated_10[19],p_var_calculated_20[19],p_var_calculated_30[19],p_var_calculated_40[19],p_var_calculated_50[19]]
    #y_21 =[p_var_calculated_10[20],p_var_calculated_20[20],p_var_calculated_30[20],p_var_calculated_40[20],p_var_calculated_50[20]]

    gw = []

    popt, pcov = curve_fit(func,N, y_1)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_2)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_3)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_4)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_5)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_6)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_7)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_8)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_9)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_10)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_11)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func, N, y_12)
    print(popt)
    gw.append(popt[1])


    popt, pcov = curve_fit(func, N, y_12)
    print(popt)
    gw.append(popt[1])


    popt, pcov = curve_fit(func,N, y_13)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_14)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_15)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_16)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_17)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_18)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_19)
    print(popt)
    gw.append(popt[1])

    popt, pcov = curve_fit(func,N, y_20)
    print(popt)
    gw.append(popt[1])


    plt.xlim(-0.4,1.1)
    plt.ylim(-0.4,1.1)

    plt.plot(gw / (1/np.arange(0.001,1.101,0.1)),gw, color = 'r')
    plt.plot(p_fix_10, [f(i) for i in p_fix_10])
    plt.show()