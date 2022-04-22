import numpy as np
import matplotlib.pyplot as plt
import random

# first define the 2-d grid with random spins

sigma = np.asarray([1, 0])
N_s = 10
x = np.linspace(0, N_s - 1, N_s).astype(int)
y = np.linspace(0, N_s - 1, N_s).astype(int)

grid = np.array([x for yy in range(0, len(y))])

for xx in x:
    for yy in y:
        grid[xx][yy] = random.choices(sigma, k=1, weights=[50, 50])[0]

# now check for clusters in the grid using Hoshen–Kopelman_algorithm

print(grid)

label_id = 1
label = np.zeros((N_s, N_s))
label_neg = np.zeros((N_s, N_s))
for xx in np.append(x, 0):
    for yy in np.append(y, 0):
        if grid[xx][yy] == 1.:
            # use modulo here to fullfill the boundary conditon
            left = grid[(xx - 1) % N_s, yy]
            above = grid[xx, (yy - 1) % N_s]
            label_left = label[(xx - 1) % N_s, yy]
            label_above = label[xx, (yy - 1) % N_s]

            if left == 0. and above == 0. and label[xx, yy] == 0:
                label[xx, yy] = label_id
                label_id = label_id + 1
            elif left != 0. and above == 0.:
                if label[(xx - 1) % N_s, yy] == 0:
                    label[xx, yy] = label_id
                    label[(xx - 1) % N_s, yy] = label_id


                elif label[xx, yy] != 0:
                    for xxx in x:
                        for yyy in y:
                            if label[xx, yy] == label[xxx, yyy]:
                                label[xxx, yyy] = label[(xx - 1) % N_s, yy]
                else:
                    label[xx, yy] = label[(xx - 1) % N_s, yy]
                label_id = label_id + 1


            elif above != 0. and left == 0.:
                if label[xx, (yy - 1) % N_s] == 0:
                    label[xx, yy] = label_id
                    label[xx, (yy - 1) % N_s] = label_id
                    label_id = label_id + 1

                elif label[xx, yy] != 0:
                    for xxx in x:
                        for yyy in y:
                            if label[xx, yy] == label[xxx, yyy]:
                                label[xxx, yyy] = label[xx, (yy - 1) % N_s]

                else:
                    label[xx, yy] = label[xx, (yy - 1) % N_s]
                label_id = label_id + 1
            elif above != 0. and left != 0.:
                label[xx, yy] = label_id
                for xxx in x:
                    for yyy in y:
                        if label[xxx, yyy] == label_left and label_left != 0:
                            label[xxx, yyy] = label_id
                        if label[xxx, yyy] == label_above and label_above != 0:
                            label[xxx, yyy] = label_id
                label_id = label_id + 1
            else:
                pass

                # if label[xx,yy] == label[(xx+1)%N_s,yy] and label[(xx+1)%N_s,yy] != 0:
                #    label[xx, yy] = label_id
                # if label[xx,yy] == label[xx,(yy+1)%N_s] and label[xx,(yy+1)%N_s] != 0:
                #    label[xx, yy] = label_id

print(grid)
print(label)

label_pos = label

grid = grid +1

# now check for clusters in the grid using Hoshen–Kopelman_algorithm

print(grid)

label_id = 1
label = np.zeros((N_s, N_s))
label_neg = np.zeros((N_s, N_s))
for xx in np.append(x, 0):
    for yy in np.append(y, 0):
        if grid[xx][yy] == 1.:
            # use modulo here to fullfill the boundary conditon
            left = grid[(xx - 1) % N_s, yy]
            above = grid[xx, (yy - 1) % N_s]
            label_left = label[(xx - 1) % N_s, yy]
            label_above = label[xx, (yy - 1) % N_s]

            if left == 0. and above == 0. and label[xx, yy] == 0:
                label[xx, yy] = label_id
                label_id = label_id + 1
            elif left != 0. and above == 0.:
                if label[(xx - 1) % N_s, yy] == 0:
                    label[xx, yy] = label_id
                    label[(xx - 1) % N_s, yy] = label_id


                elif label[xx, yy] != 0:
                    for xxx in x:
                        for yyy in y:
                            if label[xx, yy] == label[xxx, yyy]:
                                label[xxx, yyy] = label[(xx - 1) % N_s, yy]
                else:
                    label[xx, yy] = label[(xx - 1) % N_s, yy]
                label_id = label_id + 1


            elif above != 0. and left == 0.:

                if label[xx, (yy - 1) % N_s] == 0:
                    label[xx, yy] = label_id
                    label[xx, (yy - 1) % N_s] = label_id
                    label_id = label_id + 1

                elif label[xx, yy] != 0:
                    for xxx in x:
                        for yyy in y:
                            if label[xx, yy] == label[xxx, yyy]:
                                label[xxx, yyy] = label[xx, (yy - 1) % N_s]




                else:
                    label[xx, yy] = label[xx, (yy - 1) % N_s]
                label_id = label_id + 1
            elif above != 0. and left != 0.:
                label[xx, yy] = label_id
                for xxx in x:
                    for yyy in y:
                        if label[xxx, yyy] == label_left and label_left != 0:
                            label[xxx, yyy] = label_id
                        if label[xxx, yyy] == label_above and label_above != 0:
                            label[xxx, yyy] = label_id
                label_id = label_id + 1
            else:
                pass

                # if label[xx,yy] == label[(xx+1)%N_s,yy] and label[(xx+1)%N_s,yy] != 0:
                #    label[xx, yy] = label_id
                # if label[xx,yy] == label[xx,(yy+1)%N_s] and label[xx,(yy+1)%N_s] != 0:
                #    label[xx, yy] = label_id

print(grid-1)
#print(label)
print(label)
print(label_pos)




label_complete = label_pos - label

print(label_complete)


plt.imshow(label_complete)
plt.colorbar()
plt.show()