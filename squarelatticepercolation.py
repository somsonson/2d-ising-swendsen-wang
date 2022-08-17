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
import numba as numba
from numba_plyfit import fit_poly

start = timeit.default_timer()

@jit(nopython=True , parallel = True)
def squarelatticepercolationplot(percolation_x,percolation_y,N_s, periodic , x_and_y, averaging, start_inteval = 0., end_interval = 1. , resolution = 100):

    occupation_prob = np.zeros(resolution+1, dtype = numba.float64)#[0.] * (resolution+1)
    perculation_prob = np.zeros(resolution+1, dtype = numba.float64)# [0.] * (resolution+1)
    std_perculation_prob_sq =  np.zeros(resolution+1, dtype = numba.float64)
    if x_and_y== True:
        for i in prange(0,resolution+1):
            p = np.arange(start_inteval, end_interval + (end_interval - start_inteval) / resolution, (end_interval - start_inteval) / resolution)[i]
            occupation_prob[i] = p
            n_percolated_grids = 0
            if periodic == False:
                for j in prange(0,averaging):
                    grid = np.zeros((N_s,N_s))+1.
                    label , bondx , bondy,all_label_ids = fill_bonds_identify_clusters(p, p, 0, grid, False)
                    #total_filled_bonds.append(n_bonds)
                    #if check_percolation(label , bondx , bondy) == True:
                    #    proby1 = proby1 +1
                    if check_percolation_nonperiodic(label, percolation_x , percolation_y) == True:
                        n_percolated_grids = n_percolated_grids +1
                    else:
                        pass
                #prob1.append(proby1 / 40)
                perculation_prob[i] = (n_percolated_grids / averaging)
                std_perculation_prob_sq = 1 / averaging *( n_percolated_grids*(1-(n_percolated_grids / averaging))**2 + (averaging-n_percolated_grids)*(0-(n_percolated_grids / averaging))**2)
            else:
                for j in prange(0,averaging):
                    grid = np.zeros((N_s,N_s))+1.
                    label , bondx , bondy,all_label_ids = fill_bonds_identify_clusters(p, p, 0, grid, True)
                    #total_filled_bonds.append(n_bonds)
                    #if check_percolation(label , bondx , bondy) == True:
                    #    proby1 = proby1 +1
                    if check_percolation_periodic(label, bondx , bondy,percolation_x,percolation_y) == True:
                        n_percolated_grids = n_percolated_grids +1
                    else:
                        pass
                #prob1.append(proby1 / 40)
                perculation_prob[i] = (n_percolated_grids / averaging)
                std_perculation_prob_sq = 1 / averaging *( n_percolated_grids*(1-(n_percolated_grids / averaging))**2 + (averaging-n_percolated_grids)*(0-(n_percolated_grids / averaging))**2)
    else:
        for i in prange(0,resolution+1):
            p = np.arange(start_inteval, end_interval + (end_interval - start_inteval) / resolution, (end_interval - start_inteval) / resolution)[i]
            occupation_prob[i] = p
            n_percolated_grids = 0
            if periodic == False:
                for j in prange(0,averaging):
                    grid = np.zeros((N_s,N_s))+1
                    label , bondx , bondy,all_label_ids = fill_bonds_identify_clusters(p, p, 0, grid, False)
                    #total_filled_bonds.append(n_bonds)
                    #if check_percolation(label , bondx , bondy) == True:
                    #    proby1 = proby1 +1
                    if check_percolation_nonperiodic(label, percolation_x , False) == True or check_percolation_nonperiodic(label, False , percolation_y) == True :
                        n_percolated_grids = n_percolated_grids +1
                    else:
                        pass
                #prob1.append(proby1 / 40)
                perculation_prob[i] = (n_percolated_grids / averaging)
                std_perculation_prob_sq = 1 / averaging *( n_percolated_grids*(1-(n_percolated_grids / averaging))**2 + (averaging-n_percolated_grids)*(0-(n_percolated_grids / averaging))**2)
            else:
                for j in prange(0,averaging):
                    grid = np.zeros((N_s,N_s))+1.
                    label , bondx , bondy,all_label_ids = fill_bonds_identify_clusters(p, p, 0, grid, True)
                    #total_filled_bonds.append(n_bonds)
                    #if check_percolation(label , bondx , bondy) == True:
                    #    proby1 = proby1 +1
                    if check_percolation_periodic(label, bondx , bondy,False,percolation_y) == True or check_percolation_periodic(label, bondx , bondy,percolation_x, False) == True:
                        n_percolated_grids = n_percolated_grids +1
                    else:
                        pass
                #prob1.append(proby1 / 40)
                perculation_prob[i] = (n_percolated_grids / averaging)
                std_perculation_prob_sq = 1 / averaging *( n_percolated_grids*(1-(n_percolated_grids / averaging))**2 + (averaging-n_percolated_grids)*(0-(n_percolated_grids / averaging))**2)


    return  perculation_prob, occupation_prob

@jit(nopython=True, fastmath= True, locals={'middle_id': numba.int64})
def resolve_crit_occupation_prob2(N_s, accuracy , percolation_x,percolation_y, periodic , x_and_y,right=1,left=1):
    left_bound_interval = 0.4
    right_bound_interval = 0.6
    averaging = 100
    i = 0

    idss = 0
    while i < accuracy:
        #print(i)
        monotone = False
        perculation_prob, occupation_prob = squarelatticepercolationplot(percolation_x,
                                                                         percolation_y, N_s, periodic, x_and_y,
                                                                         averaging,
                                                                         start_inteval=left_bound_interval,
                                                                         end_interval=right_bound_interval, resolution=10)
        approx_middle_count = 0.
        mean = 1.
        middle_id = int(0)
        print(i)
        while monotone == False:
            for j in range(1,11):
                if perculation_prob[j-1] <= perculation_prob[j] and j == 10:
                    monotone = True
                elif perculation_prob[j-1] > perculation_prob[j]:
                    break
                else:
                    pass

            perculation_prob_addition, occupation_prob_addition = squarelatticepercolationplot( percolation_x,
                                                                             percolation_y,N_s, periodic, x_and_y,
                                                                             averaging,
                                                                             start_inteval=left_bound_interval,
                                                                             end_interval=right_bound_interval,
                                                                             resolution=10)




            for g in prange(0,len(perculation_prob)):
                perculation_prob[g] = (perculation_prob[g]*(averaging * mean) + (averaging)*perculation_prob_addition[g])/(averaging * (mean+1))
            mean = mean +1
            #print(middle_id)
            #print(perculation_prob)
            if middle_id == np.abs(np.asarray(perculation_prob) - 0.5).argmin():
               # print('ka')
                approx_middle_count = approx_middle_count + 1
            else:
                approx_middle_count = 0
            if approx_middle_count == 100.:
                monotone = True
            middle_id = np.abs(np.asarray(perculation_prob) - 0.5).argmin()
            #print(approx_middle_count)
            #print(middle_id)


        print("occ", occupation_prob)
        print("per",perculation_prob)

        id = (np.abs(np.asarray(perculation_prob) - 0.5)).argmin()
        perculation_prob_temp = np.delete(perculation_prob,id)
        id2 = (np.abs(np.asarray(perculation_prob_temp) - 0.5)).argmin()
        interval_middle = min(occupation_prob[id],occupation_prob[id2])+abs(occupation_prob[id] - occupation_prob[id2])/2
        if (id != 0 and id != (len(perculation_prob)-1)):
            right_bound_interval = interval_middle + right*(occupation_prob[1] - occupation_prob[0])
            left_bound_interval = interval_middle  - left* (occupation_prob[1] - occupation_prob[0])
            i = i+1
        elif i == accuracy-1:
            right_bound_interval = interval_middle + right*(occupation_prob[1] - occupation_prob[0])
            left_bound_interval = interval_middle  - left* (occupation_prob[1] - occupation_prob[0])

            i = i+1
        elif id == 0:
            right_bound_interval = interval_middle + 3*(occupation_prob[1] - occupation_prob[0])
            left_bound_interval = interval_middle  -  7*(occupation_prob[1] - occupation_prob[0])

        elif id == len(perculation_prob)-1:
            right_bound_interval = interval_middle + 7 *(occupation_prob[1] - occupation_prob[0])
            left_bound_interval = interval_middle - 3* (occupation_prob[1] - occupation_prob[0])


    return interval_middle , right_bound_interval , left_bound_interval

@jit(nopython=True, fastmath= True, locals={'middle_id': numba.int64}, cache= True)
def resolve_crit_occupation_prob(N_s, accuracy , percolation_x,percolation_y, periodic , x_and_y,averaging ,left_bound_interval = 0.45,right_bound_interval = 0.55):

    i = 0
    while i < accuracy:

        perculation_prob, occupation_prob = squarelatticepercolationplot(percolation_x,
                                                                         percolation_y, N_s, periodic, x_and_y,
                                                                         averaging,
                                                                         start_inteval=left_bound_interval,
                                                                         end_interval=right_bound_interval, resolution=10)

        print(i)
        a_0, a_1 = fit_poly(occupation_prob, perculation_prob, deg=1)

        counts = 1
        while True:
            #print('##################')
            averaging_succeded = False
            percualtion_prob_append, occupation_prob_append = squarelatticepercolationplot(percolation_x,
                                                                                           percolation_y, N_s, periodic,
                                                                                           x_and_y,
                                                                                           averaging,
                                                                                           start_inteval=left_bound_interval,
                                                                                           end_interval=right_bound_interval,
                                                                                           resolution=10)

            perculation_prob = (perculation_prob*counts + percualtion_prob_append)/(counts+1)
            counts = counts + 1
            a_0_before = a_0
            a_1_before = a_1
            a_0, a_1 = fit_poly(occupation_prob, perculation_prob, deg=1)

            if abs((a_0-a_0_before)/a_0) < 0.01 and abs((a_1-a_1_before)/a_1) < 0.0001:
                break



        #print("occ", occupation_prob)
        #print("per",perculation_prob_averaged)


        interval_middle = (0.3-a_1)/a_0
        right_bound_interval = interval_middle + 1 * (occupation_prob[1] - occupation_prob[0])
        left_bound_interval = interval_middle - 1 * (occupation_prob[1] - occupation_prob[0])
        i = i+1

    return interval_middle , right_bound_interval , left_bound_interval


#print(squarelatticepercolationplot(True,False, 40, False , True, 1000, start_inteval = 0.5, end_interval = 0.6 , resolution = 1))



p_crit_non_periodic_x_perc = []
L = [10,20,30,40,50,60]

print('N=10')
a,b,c = resolve_crit_occupation_prob(10,4, True , False , False , True, 1000 , left_bound_interval=0.49, right_bound_interval= 0.52)
print(a)
p_crit_non_periodic_x_perc.append(a)

print('N=20')
a,b,c = resolve_crit_occupation_prob(20,4, True , False , False , True, 1000 , left_bound_interval=0.49, right_bound_interval= 0.52)
print(a)
p_crit_non_periodic_x_perc.append(a)

print('N=30')
a,b,c = resolve_crit_occupation_prob(30,4, True , False , False , True, 1000 , left_bound_interval=0.49, right_bound_interval= 0.52)
print(a)
p_crit_non_periodic_x_perc.append(a)

print('N=40')
a,b,c = resolve_crit_occupation_prob(40,4, True , False , False , True, 1000 , left_bound_interval=0.49, right_bound_interval= 0.52)
print(a)
p_crit_non_periodic_x_perc.append(a)

print('N=50')
a,b,c = resolve_crit_occupation_prob(50,4, True , False , False , True, 1000 , left_bound_interval=0.49, right_bound_interval= 0.52)
print(a)
p_crit_non_periodic_x_perc.append(a)

print('N=60')
a,b,c = resolve_crit_occupation_prob(60,4, True , False , False , True, 1000 , left_bound_interval=0.49, right_bound_interval= 0.52)
print(a)
p_crit_non_periodic_x_perc.append(a)

p_crit_periodic_x_perc = []

print('N=10')
a,b,c = resolve_crit_occupation_prob(10,4, True , False , True , True, 1000  , left_bound_interval=0.46, right_bound_interval= 0.5)
p_crit_periodic_x_perc.append(a)
print('N=20')
a,b,c = resolve_crit_occupation_prob(20,4, True , False , True, True, 1000  , left_bound_interval=0.46, right_bound_interval= 0.5)
p_crit_periodic_x_perc.append(a)
print('N=30')
a,b,c = resolve_crit_occupation_prob(30,4, True , False , True, True, 1000  , left_bound_interval=0.46, right_bound_interval= 0.5)
p_crit_periodic_x_perc.append(a)
print('N=40')
a,b,c = resolve_crit_occupation_prob(40,4, True , False , True , True, 1000  , left_bound_interval=0.46, right_bound_interval= 0.5)
p_crit_periodic_x_perc.append(a)
print('N=50')
a,b,c = resolve_crit_occupation_prob(50,4, True , False , True , True, 1000  , left_bound_interval=0.46, right_bound_interval= 0.5)
p_crit_periodic_x_perc.append(a)
print('N=60')
a,b,c = resolve_crit_occupation_prob(60,4, True , False , True , True, 1000  , left_bound_interval=0.46, right_bound_interval= 0.5)
p_crit_periodic_x_perc.append(a)





savetxt('data/p_crit_periodic_x_perc_03.csv',p_crit_periodic_x_perc, delimiter=',')
savetxt('data/p_crit_non_periodic_x_perc_03.csv',p_crit_non_periodic_x_perc, delimiter=',')
savetxt('data/N_values_03.csv',L, delimiter=',')



def trash():
    prob1, param_p1 = squarelatticepercolationplot(True , False , 10, False, True, 10000)
    prob2, param_p2 = squarelatticepercolationplot(True , False , 20, False, True, 10000)
    prob3, param_p3 = squarelatticepercolationplot(True , False , 25, False, True, 10000)


    plt.plot( param_p1 , prob1, label = 'N=10, x-perc, non-periodic')
    plt.plot( param_p2 , prob2,label = 'N=20, x-perc, non-periodic')
    plt.plot( param_p3 , prob3,label = 'N=25, x-perc, non-periodic')


    plt.hlines(0.5, 0 , 1, color='r')
    plt.vlines(0.5, 0 , 1, color='r')
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('rel. häufigkeit perkolation')
    plt.tight_layout()
    plt.savefig("squarelatticepercolation1.pdf")
    plt.cla()


    prob3, param_p3 = squarelatticepercolationplot(True , True , 10, False , True, 10000)
    prob4, param_p4 = squarelatticepercolationplot(True , True , 20, False,  True, 10000)
    prob5, param_p5 = squarelatticepercolationplot(True , True , 25, False,  True, 10000)


    plt.plot( param_p3 , prob3,label = 'N=10, x-and-y-perc, non-periodic')
    plt.plot( param_p4 , prob4,label = 'N=20, x-and-y-perc, non-periodic')
    plt.plot( param_p5 , prob5,label = 'N=25, x-and-y-perc, non-periodic')


    plt.hlines(0.5, 0 , 1, color='r')
    plt.vlines(0.5, 0 , 1, color='r')
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('rel. häufigkeit perkolation')
    plt.tight_layout()
    plt.savefig("squarelatticepercolation2.pdf")
    plt.cla()

    prob9, param_p9 = squarelatticepercolationplot(True , True , 10, False , False, 10000)
    prob10, param_p10 = squarelatticepercolationplot(True , True , 20, False,  False, 10000)
    prob11, param_p11 = squarelatticepercolationplot(True , True , 25, False,  False, 10000)


    plt.plot( param_p9 , prob9,label = 'N=10, x-or-y-perc, non-periodic')
    plt.plot( param_p10 , prob10,label = 'N=20, x-or-y-perc, non-periodic')
    plt.plot( param_p11 , prob11,label = 'N=20, x-or-y-perc, non-periodic')


    plt.hlines(0.5, 0 , 1, color='r')
    plt.vlines(0.5, 0 , 1, color='r')
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('rel. häufigkeit perkolation')
    plt.tight_layout()
    plt.savefig("squarelatticepercolation3.pdf")
    plt.cla()

    prob5, param_p5 = squarelatticepercolationplot(True , True , 10,True , True, 10000)
    prob6, param_p6 = squarelatticepercolationplot(True , True , 20, True, True, 10000)
    prob7, param_p7 = squarelatticepercolationplot(True , True , 25, True, True, 10000)


    plt.plot( param_p5 , prob5,label = 'N=10, x-and-y-perc, periodic')
    plt.plot( param_p6 , prob6,label = 'N=20, x-and-y-perc, periodic')
    plt.plot( param_p7 , prob7,label = 'N=25, x-and-y-perc, periodic')


    plt.hlines(0.5, 0 , 1, color='r')
    plt.vlines(0.5, 0 , 1, color='r')
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('rel. häufigkeit perkolation')
    plt.tight_layout()
    plt.savefig("squarelatticepercolation4.pdf")
    plt.cla()

    prob5, param_p5 = squarelatticepercolationplot(True , True , 10,True , False, 10000)
    prob6, param_p6 = squarelatticepercolationplot(True , True , 20, True, False, 10000)
    prob7, param_p7 = squarelatticepercolationplot(True , True , 25, True, False, 10000)


    plt.plot( param_p5 , prob5,label = 'N=10, x-or-y-perc, periodic')
    plt.plot( param_p6 , prob6,label = 'N=20, x-or-y-perc, periodic')
    plt.plot( param_p7 , prob7,label = 'N=25, x-or-y-perc, periodic')


    plt.hlines(0.5, 0 , 1, color='r')
    plt.vlines(0.5, 0 , 1, color='r')
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('rel. häufigkeit perkolation')
    plt.tight_layout()
    plt.savefig("squarelatticepercolation5.pdf")
    plt.cla()

    prob7, param_p7 = squarelatticepercolationplot(True , False , 10,True , True, 10000)
    prob8, param_p8 = squarelatticepercolationplot(True , False , 20, True, True, 10000)
    prob9, param_p9 = squarelatticepercolationplot(True , False , 25, True, True, 10000)


    plt.plot( param_p7 , prob7,label = 'N=10, x-perc, periodic')
    plt.plot( param_p8 , prob8,label = 'N=20, x-perc, periodic')
    plt.plot( param_p9 , prob9,label = 'N=25, x-perc, periodic')


    plt.hlines(0.5, 0 , 1, color='r')
    plt.vlines(0.5, 0 , 1, color='r')
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('rel. häufigkeit perkolation')
    plt.tight_layout()
    plt.savefig("squarelatticepercolation6.pdf")
    plt.cla()


    placing_prob = []
    percolation_prob = []

    for i in range(0,5):
        prob, param_p = squarelatticepercolationplot(True, False, 20+ i*2, False, True, 100, start_inteval=0.4 , end_interval= 0.6)
        placing_prob.append(param_p)
        percolation_prob.append(prob)
        plt.plot(placing_prob[i], percolation_prob[i], label='N=' + str(i + 10) + 'x-and-y-percolation')
        for j in range(1,len(placing_prob[i])):
            if percolation_prob[i][j-1] < percolation_prob[i][j] and j == (len(placing_prob[i])-1):
                print('monotone', i)

    plt.xlabel('p')
    plt.ylabel('rel. häufigkeit perkolation')
    plt.hlines(0.5, 0 , 1, color='r')
    plt.vlines(0.5, 0 , 1, color='r')
    plt.legend()
    plt.tight_layout()
    plt.savefig("squarelatticepercolation7.pdf")

    plt.show()

    savetxt('data/squarelattice.csv',np.transpose(np.column_stack((placing_prob,percolation_prob))), delimiter=',')

