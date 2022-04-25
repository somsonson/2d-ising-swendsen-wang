import numpy as np
import matplotlib.pyplot as plt
import random
from functions import decision
from functions import initgrid
from functions import fill_bonds_identify_clusters
from functions import flip_cluster
from numba import jit
N_s = 10
'''

grid , x , y  = initgrid(N_s)
# now check for clusters in the grid using Hoshen–Kopelman_algorithm

#grid = np.array([[1, 0, 1, 1, 1, 0],
# [1, 0, 0, 1, 1, 0],
# [1 ,0 ,1 ,0 ,1 ,1],
# [1, 0, 1, 1, 1, 1],
# [1, 1, 0, 1, 0, 1],
# [1, 0, 1, 0, 0, 1]])

#grid = np.array([[1, 0, 1, 0, 0, 1],
# [1, 0, 1, 0, 0, 0,],
# [1, 0, 0, 1, 1, 0],
# [0, 0, 0, 1, 0, 0],
# [1, 1, 1, 0, 1, 1],
# [1, 1, 1, 1, 0, 0]])

label_complete = fill_bonds_identify_clusters(0.5, grid)

newgrid = flip_cluster(grid, label_complete)


unique_labels = np.unique(label_complete)


#print(unique_labels[0] ,  label_complete , np.where(  label_complete == unique_labels[0]) )

#print(np.where(  label_complete == unique_labels[0])[0] , np.where(  label_complete == unique_labels[0])[1])
for x in unique_labels:
    if len(np.where(  label_complete == x)[0]) == N_s or len(np.where(  label_complete == x)[1]) == N_s:
        print("perculation achieved")
        
'''

@jit(nopython=True)
def swenden_wang(p, N_s , N_max ):
    grid, x, y = initgrid(N_s)
    grid_init = grid
    for i in range(0,N_max):
        label_complete = fill_bonds_identify_clusters(p, grid)

        grid = flip_cluster(grid, label_complete)

        #unique_labels = np.delete(np.unique(label_complete),0)
        unique_labels = np.unique(grid )
        if  0.8*N_s*N_s < np.count_nonzero(grid == 0) or 0.8*N_s*N_s < np.count_nonzero(grid == 1):
            return i, grid, grid_init

        #print(counts)

        # print(unique_labels[0] ,  label_complete , np.where(  label_complete == unique_labels[0]) )

        # print(np.where(  label_complete == unique_labels[0])[0] , np.where(  label_complete == unique_labels[0])[1])
        #for xx in unique_labels:
        #    #print(np.where(grid == xx)[0],np.where(grid == xx)[1] )
        #    #print(len(set(np.where(grid == xx)[0])))
        #    if len(set(np.where(grid == xx)[0])) >= np.rint(0.99*N_s): #and len(set(np.where(label_complete == xx)[1])) >= N_s:
        #            #print(len(np.where(label_complete == xx)[0]) ,len(np.where(label_complete == xx)[1]) )
        #            return i , grid , grid_init
        print(i)
    return N_max , grid , grid_init


fig, ax = plt.subplots(2,1)
a,b,c = swenden_wang(0.1,100,100)
ax[0].imshow(b)
ax[1].imshow(c)
plt.show()


'''

prob_of_perculation = np.array([])
tries = 0
for i in range(8,15):
    #print(i)
    for j in range(0,10):
        print(j)
        tries =tries + 1/swenden_wang( i / 15, 100 , 100 )[0]
    prob_of_perculation = np.append(prob_of_perculation, tries/10)
    tries = 0

#print(length)
fig,ax = plt.subplots()
ax.plot(np.arange(len(prob_of_perculation)) , prob_of_perculation)
plt.savefig("bendikt.pdf")


#print(swenden_wang(0.9, 20 , 1000 ))

'''

'''

fig, ax = plt.subplots(2,1)
ax[0].imshow(label_complete)
ax[1].imshow(grid)
for i in range(0, label_complete.shape[0]):
    for j in range(0, label_complete.shape[1]):
        c = label_complete[j,i]
        g = grid[j,i]
        ax[0].text(i, j, str(c), va='center', ha='center')
        ax[1].text(i, j, str(g), va='center', ha='center')

plt.show()


fig, ax2 = plt.subplots(2,1)
ax2[0].imshow(newgrid)
ax2[1].imshow(grid)
for i in range(0, newgrid.shape[0]):
    for j in range(0, newgrid.shape[1]):
        c = newgrid[j,i]
        g = grid[j,i]
        ax2[0].text(i, j, str(c), va='center', ha='center')
        ax2[1].text(i, j, str(g), va='center', ha='center')

plt.show()

print(newgrid + grid)
print(newgrid)

'''