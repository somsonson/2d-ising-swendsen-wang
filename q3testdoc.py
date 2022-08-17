import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit , prange
from functions import initgrid
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
from functions import check_percolation_periodic
from functions import decision
from scipy.signal import savgol_filter
import scipy
from numpy import savetxt
from functions import initgridq3

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

#grid = np.array([
#    [0, 0, 1],[0,1,0],[1,0,0]])

grid_0 = np.transpose(np.array([
    [0,0,1,0],
    [1,0,0,2],
    [0,1,1,0],
    [2,2,0,2]
]))


#bondx = np.array([[1, 1, 1, 1, 1, 1],
#                  [0, 0, 0, 0, 0, 0, ],
#                  [0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0]])

#bondy = np.array([[1, 0, 0, 0, 0, 0],
#                  [1, 0, 0, 0, 0, 0, ],
#                  [1, 0, 0, 0, 0, 0],
#                  [1, 0, 0, 0, 0, 0],
#                  [1, 0, 0, 0, 0, 0],
#                  [1, 0, 0, 0, 0, 0]])

#grid = initgrid(7,0.5)
periodic = True
#grid_0 = initgridq3(4)

grid_1 = grid_0 - 1
grid_2 = grid_0 + 1

label_0, bondx_0, bondy_0, bondd_0, label_ids_0 = fill_bonds_identify_clusters(1,1,1, grid_0,
                                                                               periodic)
label_1, bondx_1, bondy_1, bondd_1, label_ids_1 = fill_bonds_identify_clusters(1,1,1, grid_1,
                                                                               periodic)
label_2, bondx_2, bondy_2, bondd_2, label_ids_2 = fill_bonds_identify_clusters(1,1,1, grid_2,
                                                                               periodic)

#print(check_percolation(label, bondx, bondy))

fig, ax2 = plt.subplots(2, 3)
ax2[0][0].imshow(label_0)
ax2[0][1].imshow(grid_0)
ax2[1][0].imshow(label_1)
ax2[1][1].imshow(label_2)

for i in range(0, label_0.shape[0]):
    for j in range(0, label_0.shape[1]):
        c = label_0[j, i]
        g = grid_0[j, i]
        ax2[0][0].text(i, j, str(c), va='center', ha='center')
        ax2[0][1].text(i, j, str(g), va='center', ha='center')


#print(all_label_ids)

unique_labels = []
for x in range(0,3):
    for y in range(0,3):
        if label_0[x][y] not in unique_labels:
            unique_labels.append(label_0[x][y])

unique_labels = selection_sort(unique_labels)

print(unique_labels)
print('################')
#print(all_label_ids_0)


#print( check_percolation_periodic(label_0,bondx_0 ,bondy_0,bondd_0,True , False),
#         check_percolation_periodic(label_0,bondx_0 ,bondy_0,bondd_0,False , True),
#         check_percolation_periodic(label_1,bondx_1 ,bondy_1,bondd_1,True , False),
#       check_percolation_periodic(label_1,bondx_1 ,bondy_1,bondd_1,False , True),
#          check_percolation_periodic(label_2,bondx_2 ,bondy_2,bondd_2,True , False),
#          check_percolation_periodic(label_2,bondx_2 ,bondy_2,bondd_2,False , True))



print( check_percolation_nonperiodic(label_0,True , False),
         check_percolation_nonperiodic(label_0,False , True),
         check_percolation_nonperiodic(label_1, True, False),
       check_percolation_nonperiodic(label_1, False, True),
          check_percolation_nonperiodic(label_2, True, False),
          check_percolation_nonperiodic(label_2, False, True))



plt.show()