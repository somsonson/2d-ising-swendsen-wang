import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit

@jit(nopython=True)
def decision(probability):
    return random.random() < probability

@jit(nopython=True)
def initgrid(N_s):
    #initializes square grid with N_s x N_s dimension
    sigma = np.asarray([1, 0])
    x = np.arange(N_s)
    y = np.arange(N_s)

    grid = np.zeros((N_s,N_s))

    for xx in x:
        for yy in y:
            if decision(0.5) == True:
                grid[xx][yy] =1
            else:
                grid[xx][yy] =0
    return grid ,x , y

@jit(nopython=True)
def fill_bonds_identify_clusters(p , grid):
    # this function takes a grid including and casts bonds with probability p, then it identifies all clusters
    N_s = len(grid)
    x = np.arange(N_s)
    y = np.arange(N_s)
    prob = p
    label_id = 1
    last = 0
    label = np.zeros((N_s, N_s))
    label_neg = np.zeros((N_s, N_s))
    for xx in np.append(x, len(x)):
        last = 0
        for yy in np.append(y, len(y)):
            #print(xx)
            if xx == len(x) or yy == len(y):
                last = 1
            xx = xx % N_s
            yy = yy % N_s
            if (grid[xx][yy] == 1. and decision(prob)) or (grid[xx][yy] == 1. and last == 1):
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

    #print(grid)
    #print(label)

    label_pos = label

    grid = 1 - grid

    # now check for clusters in the grid using Hoshen–Kopelman_algorithm

    #print(grid)

    label_id = 1
    label = np.zeros((N_s, N_s))
    label_neg = np.zeros((N_s, N_s))
    for xx in np.append(x, 0):
        for yy in np.append(y, 0):
            if grid[xx][yy] == 1. and decision(prob):
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


    #print(grid - 1)
    grid = grid - 1
    # print(label)
    #print(label)
    #print(label_pos)

    label_complete = label_pos - label
    return label_complete


@jit(nopython=True)
def flip_cluster(grid, label_complete):
    N_s = len(grid)
    x = np.arange(N_s)
    y = np.arange(N_s)
    newgrid = np.zeros((N_s, N_s))

    flipped_values = np.array([0.5])

    for xx in x:
        for yy in y:
            if label_complete[xx][yy] == 0.:
                newgrid[xx][yy] = grid[xx][yy]
            elif label_complete[xx][yy] > 0. and random.random() < 0.5 and np.all(
                    label_complete[xx][yy] != flipped_values):
                flipped_values = np.append(flipped_values, label_complete[xx][yy])
                for xxx in x:
                    for yyy in y:
                        if label_complete[xxx][yyy] == label_complete[xx][yy]:
                            newgrid[xxx][yyy] = 0
            elif label_complete[xx][yy] < 0. and random.random() < 0.5 and np.all(
                    label_complete[xx][yy] != flipped_values):
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
    return newgrid