import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numba import jit
from scipy.optimize import fsolve
from math import sinh

@jit(nopython=True, cache= True)
def decision(probability):
    return random.rand() < probability


@jit(nopython=True , cache= True)
def initgrid(N_s, p):
    # initializes square grid with N_s x N_s dimension with probability p

    grid = np.zeros((N_s, N_s))

    for xx in range(0,N_s):
        for yy in range(0,N_s):
            if decision(p):
                grid[xx][yy] = 1
            else:
                grid[xx][yy] = 0
    return grid

@jit(nopython=True , cache= True)
def initgridq3(N_s):
    # initializes square grid with N_s x N_s dimension with probability p

    grid = np.zeros((N_s, N_s))

    for xx in range(0,N_s):
        for yy in range(0,N_s):
            grid[xx][yy] = random.randint(0,3)
    return grid


@jit(nopython=True, cache= True)
def fill_bonds_identify_clusters(prob_x, prob_y, prob_d, grid, periodic):
    if periodic == True:
        var = 0
    else:
        var = 1
    N_s = len(grid)
    x = np.arange(N_s)
    y = np.arange(N_s)
    label = np.zeros((N_s, N_s))
    bondx = np.zeros((N_s, N_s))
    bondy = np.zeros((N_s, N_s))
    bondd = np.zeros((N_s, N_s))
    label_id = 1
    all_label_ids = []
    # x-richtung
    for xx in x[var:]:
        for yy in y[var:]:
            if grid[yy][xx] == 1 and grid[yy][(xx - 1) % N_s] == 1 and decision(prob_x):
                bondx[yy][xx] = 1
                if label[yy][xx] == 0 and label[yy][(xx - 1) % N_s] == 0:
                    label[yy][xx] = label_id
                    label[yy][(xx - 1) % N_s] = label_id
                    all_label_ids.append(label_id)
                elif label[yy][xx] != 0 and label[yy][(xx - 1) % N_s] == 0:
                    label[yy][xx] = label[yy][xx]
                    label[yy][(xx - 1) % N_s] = label[yy][xx]
                else:
                    label[yy][xx] = label[yy][(xx - 1) % N_s]
                    label[yy][(xx - 1) % N_s] = label[yy][(xx - 1) % N_s]

            label_id = label_id + 1
    # y-richtung
    for xx in x[var:]:
        for yy in y[var:]:
            if grid[yy][xx] == 1 and grid[(yy - 1) % N_s][xx] == 1 and decision(prob_y):
                bondy[yy][xx] = 1
                if label[yy][xx] == 0 and label[(yy - 1) % N_s][xx] == 0:
                    label[yy][xx] = label_id
                    label[(yy - 1) % N_s][xx] = label_id
                    all_label_ids.append(label_id)
                elif label[yy][xx] != 0 and label[(yy - 1) % N_s][xx] == 0:
                    label[(yy - 1) % N_s][xx] = label[yy][xx]
                elif label[yy][xx] == 0 and label[(yy - 1) % N_s][xx] != 0:
                    label[yy][xx] = label[(yy - 1) % N_s][xx]
                elif label[yy][xx] != 0 and label[(yy - 1) % N_s][xx] != 0 and label[(yy - 1) % N_s][xx] != label[yy][
                    xx]:
                    merge_label = label[(yy - 1) % N_s][xx]
                    all_label_ids.remove(merge_label)
                    for xxx in x:
                        for yyy in y:
                            if label[yyy][xxx] == merge_label:
                                label[yyy][xxx] = label[yy][xx]
                else:
                    pass
            label_id = label_id + 1



# d1-richtung
    for xx in x:
        for yy in y:
            if grid[yy][xx] == 1 and grid[(yy - 1) % N_s][(xx - 1) % N_s] == 1 and decision(prob_d):
                bondd[yy][xx] = 1
                if label[yy][xx] == 0 and label[(yy - 1) % N_s][(xx - 1) % N_s] == 0:
                    label[yy][xx] = label_id
                    label[(yy - 1) % N_s][(xx - 1) % N_s] = label_id
                    all_label_ids.append(label_id)
                elif label[yy][xx] != 0 and label[(yy - 1) % N_s][(xx - 1) % N_s] == 0:
                    label[(yy - 1) % N_s][(xx - 1) % N_s] = label[yy][xx]
                elif label[yy][xx] == 0 and label[(yy - 1) % N_s][(xx - 1) % N_s] != 0:
                    label[yy][xx] = label[(yy - 1) % N_s][(xx - 1) % N_s]
                elif label[yy][xx] != 0 and label[(yy - 1) % N_s][(xx - 1) % N_s] != 0 and label[(yy - 1) % N_s][(xx - 1) % N_s] != label[yy][xx]:
                    merge_label = label[(yy - 1) % N_s][(xx - 1) % N_s]
                    all_label_ids.remove(merge_label)
                    for xxx in x:
                        for yyy in y:
                            if label[yyy][xxx] == merge_label:
                                label[yyy][xxx] = label[yy][xx]
                else:
                    pass
            label_id = label_id + 1


    return label, bondx, bondy, bondd, all_label_ids


@jit(nopython=True, cache= False)
def check_percolation_periodic(label, bondx, bondy, bondd, perculation_x , perculation_y ):
    N = len(label)
    ids_to_be_checked = np.array([0.])
    label_transposed = np.transpose(label)
    bondx = np.transpose(bondx)
    ids_to_be_checked_if_connected = np.array([0.])
    ids_to_be_checked_if_connected = ids_to_be_checked_if_connected[1:]
    bondd_transposed = np.transpose(bondd)
    if perculation_x == True and perculation_y == True:
        for i in np.arange(N):
            if label[0][i] not in ids_to_be_checked:
                ids_to_be_checked = np.append(ids_to_be_checked, label[0][i])
            elif label[i][0] not in ids_to_be_checked:
                ids_to_be_checked = np.append(ids_to_be_checked, label[i][0])
            else:
                pass
        ids_to_be_checked = ids_to_be_checked[1:]
        for ids in ids_to_be_checked:
            for i in np.arange(N):
                if ids in label[i] and ids in label_transposed[i]:
                    pass
                else:
                    break
                if i == N - 1:
                    ids_to_be_checked_if_connected = np.append(ids_to_be_checked_if_connected, ids)


        for ids in ids_to_be_checked_if_connected:

            for i in np.arange(N):
                row1 = np.where(label[i] == ids)[0]
                row2 = np.where(label[(i + 1) % N] == ids)[0]
                #print(row1 , row2 , "rows")
                #print(label, "label")
                column1 = np.where(label_transposed[i] == ids)[0]
                column2 = np.where(label_transposed[(i + 1) % N] == ids)[0]

                check_bond_column_x = np.intersect1d(column1, column2)

                #print(check_bond_column_x)
                column1 = (column1 + 1) % N
                check_bond_column_d = np.intersect1d(column1, column2)
                #print(check_bond_column_d)

                check_bond_row_y = np.intersect1d(row1, row2)
                row1 = (row1 + 1) % N
                check_bond_row_d = np.intersect1d(row1, row2)

                if (1 not in bondx[(i + 1) % N][check_bond_column_x] and 1 not in bondd_transposed[(i + 1) % N][check_bond_column_d]) or (1 not in bondy[(i + 1) % N][check_bond_row_y]and 1 not in bondd[(i + 1) % N][check_bond_row_d]):
                    break
                if i == N - 1:
                    return True

    elif perculation_x == True:
        for i in np.arange(N):
            if label[i][0] not in ids_to_be_checked:
                ids_to_be_checked = np.append(ids_to_be_checked, label[i][0])
            else:
                pass
        ids_to_be_checked = ids_to_be_checked[1:]
        #print(ids_to_be_checked)
        for ids in ids_to_be_checked:
            for i in np.arange(N):
                if ids in label_transposed[i]:
                    pass
                else:
                    break
                if i == N - 1:
                    ids_to_be_checked_if_connected = np.append(ids_to_be_checked_if_connected, ids)
        #print(ids_to_be_checked_if_connected)


        for ids in ids_to_be_checked_if_connected:

            for i in np.arange(N):

                column1 = np.where(label_transposed[i] == ids)[0]
                column2 = np.where(label_transposed[(i + 1) % N] == ids)[0]

                check_bond_column_x = np.intersect1d(column1, column2)

                column1 = (column1 + 1) % N
                check_bond_column_d = np.intersect1d(column1, column2)


                if (1 not in bondx[(i + 1) % N][check_bond_column_x]) and (1 not in bondd_transposed[(i +1) % N][check_bond_column_d]):
                    break
                if i == N - 1:
                    return True

    else:
        for i in np.arange(N):
            if label[0][i] not in ids_to_be_checked:
                ids_to_be_checked = np.append(ids_to_be_checked, label[0][i])
            else:
                pass
        ids_to_be_checked = ids_to_be_checked[1:]
        for ids in ids_to_be_checked:
            for i in np.arange(N):
                if ids in label[i]:
                    pass
                else:
                    break
                if i == N - 1:
                    ids_to_be_checked_if_connected = np.append(ids_to_be_checked_if_connected, ids)


        for ids in ids_to_be_checked_if_connected:

            for i in np.arange(N):
                row1 = np.where(label[i] == ids)[0]
                row2 = np.where(label[(i + 1) % N] == ids)[0]



                check_bond_row_y = np.intersect1d(row1, row2)
                row1 = (row1 + 1) % N
                check_bond_row_d = np.intersect1d(row1, row2)

                if 1 not in bondy[(i + 1) % N][check_bond_row_y]and 1 not in bondd[(i + 1) % N][check_bond_row_d]:
                    break
                if i == N - 1:
                    ########################
                    row1_0 = np.where(label[0] == ids)[0]
                    row2_0 = np.where(label[(0 + 1) % N] == ids)[0]

                    check_bond_row_y_0 = np.intersect1d(row1_0, row2_0)
                    row1_0 = (row1_0 + 1) % N
                    check_bond_row_d_0 = np.intersect1d(row1_0, row2_0)

                    check_bond_row_d_0 = (check_bond_row_d_0 -1)  % N

                    if 0 in check_bond_row_d or 0 in check_bond_row_y:
                        A = True
                    else:
                        A = False

                    if 0 in check_bond_row_y_0 or 0 in check_bond_row_d_0:
                        B = True
                    else:
                        B = False

                    if A == True and B == True:
                        return True

                    for xx in range(1 , N):
                        if not bondy[i][xx] == 1:
                            A = False
                            B = False

                        if xx in check_bond_row_d or xx in check_bond_row_y:
                            A = True

                        if xx in check_bond_row_y_0 or xx in check_bond_row_d_0:
                            B = True

                        if A == True and B == True:
                            return True



    return False


@jit(nopython=True , cache= True)
def check_percolation_nonperiodic(label, x, y):
    N = len(label)
    ids_to_be_checked = np.array([0.])
    label_transposed = np.transpose(label)
    for i in np.arange(N):
        if label[0][i] not in ids_to_be_checked:
            ids_to_be_checked = np.append(ids_to_be_checked, label[0][i])
        elif label[i][0] not in ids_to_be_checked:
            ids_to_be_checked = np.append(ids_to_be_checked, label[i][0])
        else:
            pass
    ids_to_be_checked = ids_to_be_checked[ids_to_be_checked!=0]
    if x == True and y == True:
        for ids in ids_to_be_checked:
            for i in np.arange(N):
                if ids in label[i] and ids in label_transposed[i]:
                    pass
                else:
                    break
                if i == N-1:
                    return True
    elif x == True:
        for ids in ids_to_be_checked:
            for i in np.arange(N):
                if ids in label_transposed[i]:
                    pass
                else:
                    break
                if i == N-1:
                    return True
    elif y == True:
        for ids in ids_to_be_checked:
            for i in np.arange(N):
                if ids in label[i]:
                    pass
                else:
                    break
                if i == N-1:
                    return True

    return False

def critical(J_x,J_y,J_d):

    def f(B):
        return sinh(2 * J_x * (B)) *sinh(2 * J_y * (B)) + sinh(2 * J_d * (B)) * (
                sinh(2 * J_x * (B)) + sinh(2 * J_y * (B))) -1

    B = fsolve(f,2, maxfev = 400000000)
    return B



