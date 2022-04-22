import numpy as np
import matplotlib.pyplot as plt
import random
from functions import decision
from functions import initgrid
from functions import fill_bonds_identify_clusters
from functions import flip_cluster
N_s = 6



grid , x , y  = initgrid(N_s)
# now check for clusters in the grid using Hoshen–Kopelman_algorithm

#grid = np.array([[1, 0, 1, 1, 1, 0],
 #[1, 0, 0, 1, 1, 0],
# [1 ,0 ,1 ,0 ,1 ,1],
 #[1, 0, 1, 1, 1, 1],
 #[1, 1, 0, 1, 0, 1],
 #[1, 0, 1, 0, 0, 1]])

label_complete = fill_bonds_identify_clusters(1, grid)

newgrid = flip_cluster(grid, label_complete)












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

'''
newgrid = np.zeros((N_s, N_s))

flipped_values = np.array([])

for xx in x:
    for yy in y:
        if label_complete[xx][yy] == 0.:
            newgrid[xx][yy] = grid[xx][yy]
        elif label_complete[xx][yy] > 0. and random.random() < 0.5 and np.all(label_complete[xx][yy] != flipped_values):
            flipped_values = np.append(flipped_values ,label_complete[xx][yy] )
            for xxx in x:
                for yyy in y:
                    if label_complete[xxx][yyy] == label_complete[xx][yy]:
                        newgrid[xxx][yyy] = 0
        elif label_complete[xx][yy] < 0. and random.random() < 0.5 and np.all(label_complete[xx][yy] != flipped_values):
            flipped_values = np.append(flipped_values, label_complete[xx][yy])
            for xxx in x:
                for yyy in y:
                    if label_complete[xxx][yyy] == label_complete[xx][yy]:
                        newgrid[xxx][yyy] = 1
        elif np.all(label_complete[xx][yy] != flipped_values):
            flipped_values = np.append(flipped_values, label_complete[xx][yy])
            for xxx in x:
                for yyy in y:
                    if label_complete[xxx][yyy] == label_complete[xx][yy]:
                        newgrid[xxx][yyy] = grid[xx][yy]
        else:
            pass

'''

