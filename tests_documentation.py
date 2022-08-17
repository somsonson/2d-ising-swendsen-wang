import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit , prange
from functions import initgrid
from functions import fill_bonds_identify_clusters
from functions import check_percolation_nonperiodic
from functions import fill_bonds_identify_clusters_check_periodic_perco
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

grid = (np.array([
    [1 ,1,1],

    [1 ,1,1],
   [ 1 ,1,1]

]))


#grid =np.array([[1,0,0],[1,1,0],[0,1,0]])

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
#label, all_label_ids , percolated = fill_bonds_identify_clusters_check_periodic_perco(1, 1, 0, grid)
#print(percolated)

for i in range(100000):
    print(i)
    grid = np.array(initgrid(10,0.5),dtype=int)

    try:
        label_1, all_label_ids , percolated = fill_bonds_identify_clusters_check_periodic_perco(1, 1, 1, grid)
        label, all_label_ids =fill_bonds_identify_clusters(1, 1, 1,
                                                                            grid, True)
    except RecursionError:
        plt.imshow(grid)
        plt.show()

    if ((check_percolation_nonperiodic(label, False, True)== False) and (check_percolation_nonperiodic(label, True, False) == False)) and (percolated == True):
        print(check_percolation_nonperiodic(label, False, True))
        print(check_percolation_nonperiodic(label,True, False))
        print(percolated)
        plt.imshow(grid)
        plt.show()

''' 
    if np.array_equal(label , label_1) == False:
        print('no')
        fig, ax2 = plt.subplots(2, 2)
        ax2[0][0].imshow(label)
        ax2[0][1].imshow(label_1)
        ax2[1][1].imshow(grid)

        for i in range(0, label.shape[0]):
            for j in range(0, label.shape[1]):
                c = label[j, i]
                g = label_1[j, i]
                # a = bondx[j,i]
                # b = bondy[j, i]
                ax2[0][0].text(i, j, str(c), va='center', ha='center')
                ax2[0][1].text(i, j, str(g), va='center', ha='center')
        plt.show()
'''





grid = initgrid(10,1)
label, all_label_ids = fill_bonds_identify_clusters(0.9, 0.9, 0, grid, False)
percolated_1 = check_percolation_nonperiodic(label, False, True)
percolated_2 = check_percolation_nonperiodic(label, True,False)

fig, ax2 = plt.subplots(2, 2)
ax2[0][0].imshow(label)
ax2[0][1].imshow(grid)
#ax2[1][0].imshow(bondx)
#ax2[1][1].imshow(bondy)
for i in range(0, label.shape[0]):
  for j in range(0, label.shape[1]):
      c = label[j, i]
      g = grid[j, i]
     # a = bondx[j,i]
     # b = bondy[j, i]
      ax2[0][0].text(i, j, str(c), va='center', ha='center')
      ax2[0][1].text(i, j, str(g), va='center', ha='center')


print(percolated_2)
print(percolated_1)

plt.show()








'''

#print(label)
#print(check_percolation(label, bondx, bondy))

fig, ax2 = plt.subplots(2, 2)
ax2[0][0].imshow(label)
ax2[0][1].imshow(grid)
#ax2[1][0].imshow(bondx)
#ax2[1][1].imshow(bondy)
for i in range(0, label.shape[0]):
  for j in range(0, label.shape[1]):
      c = label[j, i]
      g = grid[j, i]
     # a = bondx[j,i]
     # b = bondy[j, i]
      ax2[0][0].text(i, j, str(c), va='center', ha='center')
      ax2[0][1].text(i, j, str(g), va='center', ha='center')
  #    ax2[1][1].text(i, j, str(a), va='center', ha='center')
   #   ax2[1][0].text(i, j, str(b), va='center', ha='center')


#print(all_label_ids)

unique_labels = []
for x in range(0,3):
  for y in range(0,3):
      if label[x][y] not in unique_labels:
          unique_labels.append(label[x][y])

unique_labels = selection_sort(unique_labels)

print(unique_labels)
print('################')
print(all_label_ids)


print(check_percolation_nonperiodic(label, True ,False))
print(check_percolation_nonperiodic(label ,False,True))
#print(check_percolation_periodic(label, bondx, bondy, bondd, True ,False))
#print(check_percolation_periodic(label, bondx, bondy, bondd, False,True))
print(percolated)
plt.show()
'''