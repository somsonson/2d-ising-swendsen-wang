import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numba import jit
from scipy.optimize import fsolve
from math import sinh


@jit(nopython=True, cache=False)
def decision(probability):
    return random.rand() < probability


@jit(nopython=True, cache=False)
def initgrid(N_s, p):
    # initializes square grid with N_s x N_s dimension with probability p

    grid = np.zeros((N_s, N_s))

    for xx in range(0, N_s):
        for yy in range(0, N_s):
            if decision(p):
                grid[xx][yy] = 1
            else:
                grid[xx][yy] = 0
    return grid


@jit(nopython=True, cache=False)
def initgridq3(N_s):
    # initializes square grid with N_s x N_s dimension with probability p

    grid = np.zeros((N_s, N_s))

    for xx in range(0, N_s):
        for yy in range(0, N_s):
            grid[xx][yy] = random.randint(0, 3)
    return grid


@jit(nopython=True, cache=False)
def fill_bonds_identify_clusters(prob_x, prob_y, prob_d, grid, periodic):
    if periodic == True:
        var = 0
    else:
        var = 1
    N_s = len(grid)
    x = np.arange(N_s)
    y = np.arange(N_s)
    label = np.zeros((N_s, N_s), dtype=np.int64)
    label_id = 1
    all_label_ids = []
    # x-richtung
    for xx in x[var:]:
        x_minus = (xx - 1) % N_s
        for yy in y:
            if grid[yy][xx] == 1 and grid[yy][x_minus] == 1 and decision(prob_x):
                if label[yy][xx] == 0 and label[yy][x_minus] == 0:
                    label[yy][xx] = label_id
                    label[yy][x_minus] = label_id
                    all_label_ids.append(label_id)
                elif label[yy][xx] != 0 and label[yy][x_minus] == 0:
                    label[yy][x_minus] = label[yy][xx]
                elif label[yy][xx] == 0 and label[yy][x_minus] != 0:
                    label[yy][xx] = label[yy][x_minus]
                elif label[yy][xx] != label[yy][x_minus]:
                    merge_label = label[yy][x_minus]
                    all_label_ids.remove(merge_label)
                    for xxx in x:
                        if label[yy][xxx] == merge_label:
                            label[yy][xxx] = label[yy][xx]

            label_id = label_id + 1
    # y-richtung
    for xx in x[var:]:
        for yy in y:
            y_minus = (yy - 1) % N_s
            if grid[yy][xx] == 1 and grid[y_minus][xx] == 1 and decision(prob_y):
                if label[yy][xx] == 0 and label[y_minus][xx] == 0:
                    label[yy][xx] = label_id
                    label[y_minus][xx] = label_id
                    all_label_ids.append(label_id)
                elif label[yy][xx] != 0 and label[y_minus][xx] == 0:
                    label[y_minus][xx] = label[yy][xx]
                elif label[yy][xx] == 0 and label[y_minus][xx] != 0:
                    label[yy][xx] = label[y_minus][xx]
                elif label[yy][xx] != 0 and label[y_minus][xx] != 0 and label[y_minus][xx] != label[yy][
                    xx]:
                    merge_label = label[y_minus][xx]
                    all_label_ids.remove(merge_label)
                    for xxx in x:
                        for yyy in y:
                            if label[yyy][xxx] == merge_label:
                                label[yyy][xxx] = label[yy][xx]
                else:
                    pass
            label_id = label_id + 1

    # d1-richtung
    for xx in x[var:]:
        x_minus = (xx - 1) % N_s
        for yy in y[var:]:
            y_minus = (yy - 1) % N_s
            if grid[yy][xx] == 1 and grid[y_minus][x_minus] == 1 and decision(prob_d):
                if label[yy][xx] == 0 and label[y_minus][x_minus] == 0:
                    label[yy][xx] = label_id
                    label[y_minus][x_minus] = label_id
                    all_label_ids.append(label_id)
                elif label[yy][xx] != 0 and label[y_minus][x_minus] == 0:
                    label[y_minus][x_minus] = label[yy][xx]
                elif label[yy][xx] == 0 and label[y_minus][x_minus] != 0:
                    label[yy][xx] = label[y_minus][x_minus]
                elif label[yy][xx] != 0 and label[y_minus][x_minus] != 0 and label[y_minus][
                    x_minus] != label[yy][xx]:
                    merge_label = label[y_minus][x_minus]
                    all_label_ids.remove(merge_label)
                    for xxx in x:
                        for yyy in y:
                            if label[yyy][xxx] == merge_label:
                                label[yyy][xxx] = label[yy][xx]
                else:
                    pass
            label_id = label_id + 1

    return label, all_label_ids


@jit(nopython=True, cache=True)
def loop_intersection(lst1, lst2):
    result = []
    for element1 in lst1:
        for element2 in lst2:
            if element1 == element2:
                result.append(element1)
    return result


@jit(nopython=True, cache=True)
def check_percolation_nonperiodic(label, x, y):
    N = len(label)
    ids_to_be_checked_x = np.array([0.])
    ids_to_be_checked_y = np.array([0.])
    label_transposed = np.transpose(label)
    for i in np.arange(N):
        if label[0][i] not in ids_to_be_checked_y:
            ids_to_be_checked_y = np.append(ids_to_be_checked_y, label[0][i])
        if label[i][0] not in ids_to_be_checked_x:
            ids_to_be_checked_x = np.append(ids_to_be_checked_x, label[i][0])

    ids_to_be_checked_x = ids_to_be_checked_x[1:]
    ids_to_be_checked_y = ids_to_be_checked_y[1:]
    ids_to_be_checked = np.intersect1d(ids_to_be_checked_x, ids_to_be_checked_y)
    if x == True and y == True:
        for ids in ids_to_be_checked:
            for i in np.arange(N):
                if ids in label[i] and ids in label_transposed[i]:
                    pass
                else:
                    break
                if i == N - 1:
                    return True
    elif x == True:
        for ids in ids_to_be_checked_x:
            for i in np.arange(N):
                if ids in label_transposed[i]:
                    pass
                else:
                    break
                if i == N - 1:
                    return True
    elif y == True:
        for ids in ids_to_be_checked_y:
            for i in np.arange(N):
                if ids in label[i]:
                    pass
                else:
                    break
                if i == N - 1:
                    return True

    return False


@jit(nopython=True, cache=False, fastmath=True)
def fill_bonds_identify_clusters_check_periodic_perco(prob_x, prob_y, prob_d, grid):
    N_s = len(grid)
    x = np.arange(N_s)
    y = np.arange(N_s)
    label = np.zeros((N_s, N_s), dtype=np.int64)
    roots = np.zeros((N_s, N_s, 2), dtype=np.int64)
    sizes = []
    label_id = 1
    all_label_ids = []
    percolated = False
    # x-richtung
    for xx in x:
        x_minus = (xx - 1) % N_s
        for yy in y:
            if grid[yy][xx] == 1 and grid[yy][x_minus] == 1 and decision(prob_x):
                if label[yy][xx] == 0 and label[yy][x_minus] == 0:
                    label[yy][xx] = label_id
                    label[yy][x_minus] = label_id
                    all_label_ids.append(label_id)
                    roots[yy][x_minus] = np.array([0, 1])
                    sizes.append(2)
                    # print(roots,"a")
                elif label[yy][xx] != 0 and label[yy][x_minus] == 0:
                    label[yy][x_minus] = label[yy][xx]
                    roots[yy][x_minus] = np.array([0, 1])
                    sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + 1
                    # print(roots,"b")
                elif label[yy][xx] == 0 and label[yy][x_minus] != 0:
                    label[yy][xx] = label[yy][x_minus]
                    roots[yy][xx] = np.array([0, -1])
                    sizes[all_label_ids.index(label[yy][x_minus])] = sizes[all_label_ids.index(
                        label[yy][x_minus])] + 1
                    # print(roots,"c")
                elif label[yy][xx] != label[yy][x_minus]:
                    if sizes[all_label_ids.index(label[yy][xx])] > sizes[
                        all_label_ids.index(label[yy][x_minus])]:
                        sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + sizes[
                            all_label_ids.index(label[yy][x_minus])]
                        del sizes[all_label_ids.index(label[yy][x_minus])]
                        merge_label = label[yy][x_minus]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            if label[yy][xxx] == merge_label:
                                label[yy][xxx] = label[yy][xx]

                        root_to_be_changed, roots = find_root(roots, yy, x_minus)
                        root_to_change_to, roots = find_root(roots, yy, xx)
                        # root_to_be_changed = root_to_be_changed - np.array([yy ,(xx - 1)])
                        # root_to_change_to = root_to_change_to - np.array([yy ,xx])
                        roots[(yy + root_to_be_changed[0]) % N_s][
                            (x_minus + root_to_be_changed[
                                1]) % N_s] = root_to_change_to - root_to_be_changed + np.array([0, +1])
                    else:
                        sizes[all_label_ids.index(label[yy][x_minus])] = sizes[all_label_ids.index(
                            label[yy][x_minus])] + sizes[all_label_ids.index(label[yy][xx])]
                        del sizes[all_label_ids.index(label[yy][xx])]
                        merge_label = label[yy][xx]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            if label[yy][xxx] == merge_label:
                                label[yy][xxx] = label[yy][x_minus]
                        root_to_be_changed, roots = find_root(roots, yy, xx)
                        root_to_change_to, roots = find_root(roots, yy, x_minus)
                        # root_to_be_changed = root_to_be_changed - np.array([yy ,xx])
                        # root_to_change_to = root_to_change_to - np.array([yy ,x_minus])
                        roots[(yy + root_to_be_changed[0]) % N_s][
                            (xx + root_to_be_changed[1]) % N_s] = root_to_change_to - root_to_be_changed + np.array(
                            [0, -1])

                    # print(roots,"d")

                else:
                    root_to_be_changed, roots = find_root(roots, yy, x_minus)
                    root_to_change_to, roots = find_root(roots, yy, xx)
                    # print(root_to_change_to)
                    # print(root_to_be_changed)
                    # print(roots,"e")
                    # print(yy,xx)
                    # print(np.abs(roots[yy][ x_minus] - roots[yy][xx]))
                    # print(roots[yy][ x_minus] ,roots[yy][xx] )
                    if not (np.array_equal(np.abs(roots[yy][xx] - roots[yy][x_minus]),
                                           np.array([0, 1])) or np.array_equal(
                            np.abs(roots[yy][xx] - roots[yy][x_minus]), np.array([1, 0]))):
                        percolated = True

            label_id = label_id + 1
    # print(roots)
    # y-richtung
    # print(roots,"HERELOL")

    for xx in x:
        for yy in y:
            y_minus = (yy - 1) % N_s
            if grid[yy][xx] == 1 and grid[y_minus][xx] == 1 and decision(prob_y):
                if label[yy][xx] == 0 and label[y_minus][xx] == 0:
                    label[yy][xx] = label_id
                    label[y_minus][xx] = label_id
                    all_label_ids.append(label_id)
                    roots[y_minus][xx] = np.array([1, 0])
                    sizes.append(2)
                elif label[yy][xx] != 0 and label[y_minus][xx] == 0:
                    label[y_minus][xx] = label[yy][xx]
                    roots[y_minus][xx] = np.array([1, 0])
                    sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + 1
                elif label[yy][xx] == 0 and label[y_minus][xx] != 0:
                    label[yy][xx] = label[y_minus][xx]
                    roots[yy][xx] = np.array([-1, 0])
                    sizes[all_label_ids.index(label[y_minus][xx])] = sizes[all_label_ids.index(
                        label[y_minus][xx])] + 1
                elif label[y_minus][xx] != label[yy][xx]:
                    # print(all_label_ids,label[yy][xx],label[(yy - 1)%N_s][xx])
                    # print(sizes)
                    if sizes[all_label_ids.index(label[yy][xx])] > sizes[
                        all_label_ids.index(label[y_minus][xx])]:
                        sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + sizes[
                            all_label_ids.index(label[y_minus][xx])]
                        del sizes[all_label_ids.index(label[y_minus][xx])]

                        merge_label = label[y_minus][xx]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            for yyy in y:
                                if label[yyy][xxx] == merge_label:
                                    label[yyy][xxx] = label[yy][xx]
                        root_to_be_changed, roots = find_root(roots, y_minus, xx)
                        root_to_change_to, roots = find_root(roots, yy, xx)
                        # root_to_be_changed = root_to_be_changed - np.array([y_minus ,xx])
                        # root_to_change_to = root_to_change_to - np.array([yy ,xx])
                        roots[(y_minus + root_to_be_changed[0]) % N_s][
                            (xx + root_to_be_changed[1]) % N_s] = root_to_change_to + np.array(
                            [1, 0]) - root_to_be_changed

                    else:
                        sizes[
                            all_label_ids.index(label[y_minus][xx])] = sizes[
                                                                                  all_label_ids.index(label[yy][xx])] + \
                                                                              sizes[
                                                                                  all_label_ids.index(
                                                                                      label[y_minus][xx])]
                        del sizes[all_label_ids.index(label[yy][xx])]

                        merge_label = label[yy][xx]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            for yyy in y:
                                if label[yyy][xxx] == merge_label:
                                    label[yyy][xxx] = label[y_minus][xx]
                        root_to_be_changed, roots = find_root(roots, yy, xx)
                        root_to_change_to, roots = find_root(roots, y_minus, xx)
                        # root_to_be_changed = root_to_be_changed - np.array([yy, xx])
                        # root_to_change_to = root_to_change_to - np.array([y_minus, xx])
                        # print(root_to_change_to, "HAHAHA ROOT TO CHANGE TO")
                        roots[(yy + root_to_be_changed[0]) % N_s][
                            (xx + root_to_be_changed[1]) % N_s] = root_to_change_to + np.array(
                            [-1, 0]) - root_to_be_changed
                else:
                    # print(roots,"e before root change")

                    root_to_be_changed, roots = find_root(roots, y_minus, xx)
                    root_to_change_to, roots = find_root(roots, yy, xx)
                    # print(root_to_be_changed)
                    # print(root_to_change_to)
                    # print(roots)
                    # print(root_to_change_to)
                    # print(root_to_be_changed)
                    # print(roots,"e")
                    # print(yy,xx)
                    # print(np.abs(roots[yy][ (xx - 1) % N_s] - roots[yy][xx]))
                    # print(roots[yy][ (xx - 1) % N_s] ,roots[yy][xx] )
                    # print(roots[y_minus][xx], roots[yy][xx])
                    # print(np.array([yy,xx]) + roots[yy][xx], "ga" , np.array([(yy-1)%N_s,xx]) + roots[(yy-1)%N_s][xx])
                    # print(yy,xx)
                    # print(label)
                    # print(sizes)
                    diff = np.abs(roots[y_minus][xx] - roots[yy][xx])
                    # print(diff, diff[0] , diff[1] , percolated)
                    if diff[0] == 1. and diff[1] == 0.:
                        pass
                    elif diff[0] == 0. and diff[1] == 1.:
                        pass

                    else:
                        # print("da hanb ich dich")
                        percolated = True

            label_id = label_id + 1

    # d1-richtung
    for xx in x:
        x_minus = (xx - 1) % N_s
        for yy in y:
            y_minus = (yy - 1) % N_s
            if grid[yy][xx] == 1 and grid[y_minus][x_minus] == 1 and decision(prob_d):
                if label[yy][xx] == 0 and label[y_minus][x_minus] == 0:
                    label[yy][xx] = label_id
                    label[y_minus][x_minus] = label_id
                    all_label_ids.append(label_id)
                    roots[y_minus][x_minus] = np.array([1, 1])
                    sizes.append(2)
                elif label[yy][xx] != 0 and label[y_minus][x_minus] == 0:
                    label[y_minus][x_minus] = label[yy][xx]
                    roots[y_minus][x_minus] = np.array([1, 1])
                    sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + 1
                elif label[yy][xx] == 0 and label[y_minus][x_minus] != 0:
                    label[yy][xx] = label[y_minus][x_minus]
                    roots[yy][xx] = np.array([-1, -1])
                    sizes[all_label_ids.index(label[y_minus][x_minus])] = sizes[all_label_ids.index(
                        label[y_minus][x_minus])] + 1
                elif label[yy][xx] != 0 and label[y_minus][x_minus] != 0 and label[y_minus][
                    x_minus] != label[yy][
                    xx]:
                    if sizes[all_label_ids.index(label[yy][xx])] > sizes[
                        all_label_ids.index(label[y_minus][x_minus])]:
                        sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + sizes[
                            all_label_ids.index(label[y_minus][x_minus])]
                        del sizes[all_label_ids.index(label[y_minus][x_minus])]

                        merge_label = label[y_minus][x_minus]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            for yyy in y:
                                if label[yyy][xxx] == merge_label:
                                    label[yyy][xxx] = label[yy][xx]

                        root_to_be_changed, roots = find_root(roots, y_minus, x_minus)
                        root_to_change_to, roots = find_root(roots, yy, xx)
                        # root_to_be_changed = root_to_be_changed - np.array([y_minus ,xx])
                        # root_to_change_to = root_to_change_to - np.array([yy ,xx])
                        roots[(y_minus + root_to_be_changed[0]) % N_s][
                            (x_minus + root_to_be_changed[1]) % N_s] = root_to_change_to + np.array(
                            [1, 1]) - root_to_be_changed

                    else:
                        sizes[
                            all_label_ids.index(label[y_minus][x_minus])] = sizes[all_label_ids.index(
                            label[yy][xx])] + sizes[
                                                                                              all_label_ids.index(
                                                                                                  label[y_minus][
                                                                                                      x_minus])]
                        del sizes[all_label_ids.index(label[yy][xx])]

                        merge_label = label[yy][xx]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            for yyy in y:
                                if label[yyy][xxx] == merge_label:
                                    label[yyy][xxx] = label[y_minus][x_minus]
                        root_to_be_changed, roots = find_root(roots, yy, xx)
                        root_to_change_to, roots = find_root(roots, y_minus, x_minus)
                        # root_to_be_changed = root_to_be_changed - np.array([yy, xx])
                        # root_to_change_to = root_to_change_to - np.array([y_minus, xx])
                        # print(root_to_change_to, "HAHAHA ROOT TO CHANGE TO")
                        roots[(yy + root_to_be_changed[0]) % N_s][
                            (xx + root_to_be_changed[1]) % N_s] = root_to_change_to + np.array(
                            [-1, -1]) - root_to_be_changed
                else:
                    # print(roots,"e before root change")

                    root_to_be_changed, roots = find_root(roots, y_minus, x_minus)
                    root_to_change_to, roots = find_root(roots, yy, xx)
                    # print(root_to_be_changed)
                    # print(root_to_change_to)
                    # print(roots)
                    # print(root_to_change_to)
                    # print(root_to_be_changed)
                    # print(roots,"e")
                    # print(yy,xx)
                    # print(np.abs(roots[yy][ x_minus] - roots[yy][xx]))
                    # print(roots[yy][ x_minus] ,roots[yy][xx] )
                    # print(roots[y_minus][xx], roots[yy][xx])
                    # print(np.array([yy,xx]) + roots[yy][xx], "ga" , np.array([(yy-1)%N_s,xx]) + roots[(yy-1)%N_s][xx])
                    # print(yy,xx)
                    # print(label)
                    # print(sizes)
                    diff = (roots[y_minus][x_minus] - roots[yy][xx])
                    # print(diff, diff[0] , diff[1] , percolated)
                    if diff[0] == 1. and diff[1] == 1.:
                        pass
                    elif diff[0] == -1. and diff[1] ==  -1.:
                        pass

                    else:
                        # print("da hanb ich dich")
                        percolated = True

            label_id = label_id + 1

    # print("asd",sizes,"brerr")
    # print(all_label_ids)
    # print(roots)
    return label, all_label_ids, percolated


@jit(nopython=True, cache=True, fastmath=True)
def fill_bonds_identify_clusters_check_periodic_perco_exit_if_percolated(prob_x, prob_y, prob_d, grid):
    N_s = len(grid)
    x = np.arange(N_s)
    y = np.arange(N_s)
    label = np.zeros((N_s, N_s), dtype=np.int64)
    roots = np.zeros((N_s, N_s, 2), dtype=np.int64)
    sizes = []
    label_id = 1
    all_label_ids = []
    # x-richtung
    for xx in x:
        for yy in y:
            if grid[yy][xx] == 1 and grid[yy][(xx - 1) % N_s] == 1 and decision(prob_x):
                if label[yy][xx] == 0 and label[yy][(xx - 1) % N_s] == 0:
                    label[yy][xx] = label_id
                    label[yy][(xx - 1) % N_s] = label_id
                    all_label_ids.append(label_id)
                    roots[yy][(xx - 1) % N_s] = np.array([0, 1])
                    sizes.append(2)
                    # print(roots,"a")
                elif label[yy][xx] != 0 and label[yy][(xx - 1) % N_s] == 0:
                    label[yy][(xx - 1) % N_s] = label[yy][xx]
                    roots[yy][(xx - 1) % N_s] = np.array([0, 1])
                    sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + 1
                    # print(roots,"b")
                elif label[yy][xx] == 0 and label[yy][(xx - 1) % N_s] != 0:
                    label[yy][xx] = label[yy][(xx - 1) % N_s]
                    roots[yy][xx] = np.array([0, -1])
                    sizes[all_label_ids.index(label[yy][(xx - 1) % N_s])] = sizes[all_label_ids.index(
                        label[yy][(xx - 1) % N_s])] + 1
                    # print(roots,"c")
                elif label[yy][xx] != label[yy][(xx - 1) % N_s]:
                    if sizes[all_label_ids.index(label[yy][xx])] > sizes[
                        all_label_ids.index(label[yy][(xx - 1) % N_s])]:
                        sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + sizes[
                            all_label_ids.index(label[yy][(xx - 1) % N_s])]
                        del sizes[all_label_ids.index(label[yy][(xx - 1) % N_s])]
                        merge_label = label[yy][(xx - 1) % N_s]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            if label[yy][xxx] == merge_label:
                                label[yy][xxx] = label[yy][xx]

                        root_to_be_changed, roots = find_root(roots, yy, (xx - 1) % N_s)
                        root_to_change_to, roots = find_root(roots, yy, xx)
                        # root_to_be_changed = root_to_be_changed - np.array([yy ,(xx - 1)])
                        # root_to_change_to = root_to_change_to - np.array([yy ,xx])
                        roots[(yy + root_to_be_changed[0]) % N_s][
                            ((xx - 1) % N_s + root_to_be_changed[
                                1]) % N_s] = root_to_change_to - root_to_be_changed + np.array([0, +1])
                    else:
                        sizes[all_label_ids.index(label[yy][(xx - 1) % N_s])] = sizes[all_label_ids.index(
                            label[yy][(xx - 1) % N_s])] + sizes[all_label_ids.index(label[yy][xx])]
                        del sizes[all_label_ids.index(label[yy][xx])]
                        merge_label = label[yy][xx]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            if label[yy][xxx] == merge_label:
                                label[yy][xxx] = label[yy][(xx - 1) % N_s]
                        root_to_be_changed, roots = find_root(roots, yy, xx)
                        root_to_change_to, roots = find_root(roots, yy, (xx - 1) % N_s)
                        # root_to_be_changed = root_to_be_changed - np.array([yy ,xx])
                        # root_to_change_to = root_to_change_to - np.array([yy ,(xx - 1) % N_s])
                        roots[(yy + root_to_be_changed[0]) % N_s][
                            (xx + root_to_be_changed[1]) % N_s] = root_to_change_to - root_to_be_changed + np.array(
                            [0, -1])

                    # print(roots,"d")

                else:
                    root_to_be_changed, roots = find_root(roots, yy, (xx - 1) % N_s)
                    root_to_change_to, roots = find_root(roots, yy, xx)
                    # print(root_to_change_to)
                    # print(root_to_be_changed)
                    # print(roots,"e")
                    # print(yy,xx)
                    # print(np.abs(roots[yy][ (xx - 1) % N_s] - roots[yy][xx]))
                    # print(roots[yy][ (xx - 1) % N_s] ,roots[yy][xx] )
                    if not (np.array_equal(np.abs(roots[yy][xx] - roots[yy][(xx - 1) % N_s]),
                                           np.array([0, 1])) or np.array_equal(
                        np.abs(roots[yy][xx] - roots[yy][(xx - 1) % N_s]), np.array([1, 0]))):
                        return True

            label_id = label_id + 1
    # print(roots)
    # y-richtung
    # print(roots,"HERELOL")

    for xx in x:
        for yy in y:
            if grid[yy][xx] == 1 and grid[(yy - 1) % N_s][xx] == 1 and decision(prob_y):
                if label[yy][xx] == 0 and label[(yy - 1) % N_s][xx] == 0:
                    label[yy][xx] = label_id
                    label[(yy - 1) % N_s][xx] = label_id
                    all_label_ids.append(label_id)
                    roots[(yy - 1) % N_s][xx] = np.array([1, 0])
                    sizes.append(2)
                elif label[yy][xx] != 0 and label[(yy - 1) % N_s][xx] == 0:
                    label[(yy - 1) % N_s][xx] = label[yy][xx]
                    roots[(yy - 1) % N_s][xx] = np.array([1, 0])
                    sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + 1
                elif label[yy][xx] == 0 and label[(yy - 1) % N_s][xx] != 0:
                    label[yy][xx] = label[(yy - 1) % N_s][xx]
                    roots[yy][xx] = np.array([-1, 0])
                    sizes[all_label_ids.index(label[(yy - 1) % N_s][xx])] = sizes[all_label_ids.index(
                        label[(yy - 1) % N_s][xx])] + 1
                elif label[(yy - 1) % N_s][xx] != label[yy][xx]:
                    # print(all_label_ids,label[yy][xx],label[(yy - 1)%N_s][xx])
                    # print(sizes)
                    if sizes[all_label_ids.index(label[yy][xx])] > sizes[
                        all_label_ids.index(label[(yy - 1) % N_s][xx])]:
                        sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + sizes[
                            all_label_ids.index(label[(yy - 1) % N_s][xx])]
                        del sizes[all_label_ids.index(label[(yy - 1) % N_s][xx])]

                        merge_label = label[(yy - 1) % N_s][xx]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            for yyy in y:
                                if label[yyy][xxx] == merge_label:
                                    label[yyy][xxx] = label[yy][xx]
                        root_to_be_changed, roots = find_root(roots, (yy - 1) % N_s, xx)
                        root_to_change_to, roots = find_root(roots, yy, xx)
                        # root_to_be_changed = root_to_be_changed - np.array([(yy - 1) % N_s ,xx])
                        # root_to_change_to = root_to_change_to - np.array([yy ,xx])
                        roots[((yy - 1) % N_s + root_to_be_changed[0]) % N_s][
                            (xx + root_to_be_changed[1]) % N_s] = root_to_change_to + np.array(
                            [1, 0]) - root_to_be_changed

                    else:
                        sizes[
                            all_label_ids.index(label[(yy - 1) % N_s][xx])] = sizes[
                                                                                  all_label_ids.index(label[yy][xx])] + \
                                                                              sizes[
                                                                                  all_label_ids.index(
                                                                                      label[(yy - 1) % N_s][xx])]
                        del sizes[all_label_ids.index(label[yy][xx])]

                        merge_label = label[yy][xx]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            for yyy in y:
                                if label[yyy][xxx] == merge_label:
                                    label[yyy][xxx] = label[(yy - 1) % N_s][xx]
                        root_to_be_changed, roots = find_root(roots, yy, xx)
                        root_to_change_to, roots = find_root(roots, (yy - 1) % N_s, xx)
                        # root_to_be_changed = root_to_be_changed - np.array([yy, xx])
                        # root_to_change_to = root_to_change_to - np.array([(yy - 1) % N_s, xx])
                        # print(root_to_change_to, "HAHAHA ROOT TO CHANGE TO")
                        roots[(yy + root_to_be_changed[0]) % N_s][
                            (xx + root_to_be_changed[1]) % N_s] = root_to_change_to + np.array(
                            [-1, 0]) - root_to_be_changed
                else:
                    # print(roots,"e before root change")

                    root_to_be_changed, roots = find_root(roots, (yy - 1) % N_s, xx)
                    root_to_change_to, roots = find_root(roots, yy, xx)
                    # print(root_to_be_changed)
                    # print(root_to_change_to)
                    # print(roots)
                    # print(root_to_change_to)
                    # print(root_to_be_changed)
                    # print(roots,"e")
                    # print(yy,xx)
                    # print(np.abs(roots[yy][ (xx - 1) % N_s] - roots[yy][xx]))
                    # print(roots[yy][ (xx - 1) % N_s] ,roots[yy][xx] )
                    # print(roots[(yy - 1) % N_s][xx], roots[yy][xx])
                    # print(np.array([yy,xx]) + roots[yy][xx], "ga" , np.array([(yy-1)%N_s,xx]) + roots[(yy-1)%N_s][xx])
                    # print(yy,xx)
                    # print(label)
                    # print(sizes)
                    diff = np.abs(roots[(yy - 1) % N_s][xx] - roots[yy][xx])
                    # print(diff, diff[0] , diff[1] , percolated)
                    if diff[0] == 1. and diff[1] == 0.:
                        pass
                    elif diff[0] == 0. and diff[1] == 1.:
                        pass

                    else:
                        # print("da hanb ich dich")
                        return True

            label_id = label_id + 1

    # d1-richtung
    for xx in x:
        for yy in y:
            if grid[yy][xx] == 1 and grid[(yy - 1) % N_s][(xx - 1) % N_s] == 1 and decision(prob_d):
                if label[yy][xx] == 0 and label[(yy - 1) % N_s][(xx - 1) % N_s] == 0:
                    label[yy][xx] = label_id
                    label[(yy - 1) % N_s][(xx - 1) % N_s] = label_id
                    all_label_ids.append(label_id)
                    roots[(yy - 1) % N_s][(xx - 1) % N_s] = np.array([1, 1])
                    sizes.append(2)
                elif label[yy][xx] != 0 and label[(yy - 1) % N_s][(xx - 1) % N_s] == 0:
                    label[(yy - 1) % N_s][(xx - 1) % N_s] = label[yy][xx]
                    roots[(yy - 1) % N_s][(xx - 1) % N_s] = np.array([1, 1])
                    sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + 1
                elif label[yy][xx] == 0 and label[(yy - 1) % N_s][(xx - 1) % N_s] != 0:
                    label[yy][xx] = label[(yy - 1) % N_s][(xx - 1) % N_s]
                    roots[yy][xx] = np.array([-1, -1])
                    sizes[all_label_ids.index(label[(yy - 1) % N_s][(xx - 1) % N_s])] = sizes[all_label_ids.index(
                        label[(yy - 1) % N_s][(xx - 1) % N_s])] + 1
                elif label[yy][xx] != 0 and label[(yy - 1) % N_s][(xx - 1) % N_s] != 0 and label[(yy - 1) % N_s][
                    (xx - 1) % N_s] != label[yy][
                    xx]:
                    if sizes[all_label_ids.index(label[yy][xx])] > sizes[
                        all_label_ids.index(label[(yy - 1) % N_s][(xx - 1) % N_s])]:
                        sizes[all_label_ids.index(label[yy][xx])] = sizes[all_label_ids.index(label[yy][xx])] + sizes[
                            all_label_ids.index(label[(yy - 1) % N_s][(xx - 1) % N_s])]
                        del sizes[all_label_ids.index(label[(yy - 1) % N_s][(xx - 1) % N_s])]

                        merge_label = label[(yy - 1) % N_s][(xx - 1) % N_s]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            for yyy in y:
                                if label[yyy][xxx] == merge_label:
                                    label[yyy][xxx] = label[yy][xx]

                        root_to_be_changed, roots = find_root(roots, (yy - 1) % N_s, (xx - 1) % N_s)
                        root_to_change_to, roots = find_root(roots, yy, xx)
                        # root_to_be_changed = root_to_be_changed - np.array([(yy - 1) % N_s ,xx])
                        # root_to_change_to = root_to_change_to - np.array([yy ,xx])
                        roots[((yy - 1) % N_s + root_to_be_changed[0]) % N_s][
                            ((xx - 1) % N_s + root_to_be_changed[1]) % N_s] = root_to_change_to + np.array(
                            [1, 1]) - root_to_be_changed

                    else:
                        sizes[
                            all_label_ids.index(label[(yy - 1) % N_s][(xx - 1) % N_s])] = sizes[all_label_ids.index(
                            label[yy][xx])] + sizes[
                                                                                              all_label_ids.index(
                                                                                                  label[(yy - 1) % N_s][
                                                                                                      (xx - 1) % N_s])]
                        del sizes[all_label_ids.index(label[yy][xx])]

                        merge_label = label[yy][xx]
                        all_label_ids.remove(merge_label)
                        for xxx in x:
                            for yyy in y:
                                if label[yyy][xxx] == merge_label:
                                    label[yyy][xxx] = label[(yy - 1) % N_s][(xx - 1) % N_s]
                        root_to_be_changed, roots = find_root(roots, yy, xx)
                        root_to_change_to, roots = find_root(roots, (yy - 1) % N_s, (xx - 1) % N_s)
                        # root_to_be_changed = root_to_be_changed - np.array([yy, xx])
                        # root_to_change_to = root_to_change_to - np.array([(yy - 1) % N_s, xx])
                        # print(root_to_change_to, "HAHAHA ROOT TO CHANGE TO")
                        roots[(yy + root_to_be_changed[0]) % N_s][
                            (xx + root_to_be_changed[1]) % N_s] = root_to_change_to + np.array(
                            [-1, -1]) - root_to_be_changed
                else:
                    # print(roots,"e before root change")

                    root_to_be_changed, roots = find_root(roots, (yy - 1) % N_s, (xx - 1) % N_s)
                    root_to_change_to, roots = find_root(roots, yy, xx)
                    # print(root_to_be_changed)
                    # print(root_to_change_to)
                    # print(roots)
                    # print(root_to_change_to)
                    # print(root_to_be_changed)
                    # print(roots,"e")
                    # print(yy,xx)
                    # print(np.abs(roots[yy][ (xx - 1) % N_s] - roots[yy][xx]))
                    # print(roots[yy][ (xx - 1) % N_s] ,roots[yy][xx] )
                    # print(roots[(yy - 1) % N_s][xx], roots[yy][xx])
                    # print(np.array([yy,xx]) + roots[yy][xx], "ga" , np.array([(yy-1)%N_s,xx]) + roots[(yy-1)%N_s][xx])
                    # print(yy,xx)
                    # print(label)
                    # print(sizes)
                    diff = np.abs(roots[(yy - 1) % N_s][(xx - 1) % N_s] - roots[yy][xx])
                    # print(diff, diff[0] , diff[1] , percolated)
                    if diff[0] == 1. and diff[1] == 1.:
                        pass


                    else:
                        # print("da hanb ich dich")
                        return True

            label_id = label_id + 1

    # print("asd",sizes,"brerr")
    # print(all_label_ids)
    # print(roots)
    return False


@jit(nopython=True, cache=True, )
def find_root(roots, yyy, xxx):
    # print(type(roots))
    # print(roots,yyy,xxx)
    N = len(roots)
    new_pointer = roots[yyy][xxx]
    # print(new_pointer,"AGAGAGAGAGAGAGAGA")
    if new_pointer[0] == 0 and new_pointer[1] == 0:

        return np.array([0, 0], dtype=np.int64), roots
    else:
        temp, roots = find_root(roots, (yyy + new_pointer[0]) % N, (xxx + new_pointer[1]) % N)
        roots[yyy][xxx] = np.array([int(temp[0] + roots[yyy][xxx][0]), int(temp[1] + roots[yyy][xxx][1])],
                                   dtype=np.int64)
        vec = np.array([roots[yyy][xxx][0], roots[yyy][xxx][1]], dtype=np.int64)
        return vec, roots


def debug():
    grid = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1],
                     [1, 0, 1, 1, 1]], dtype=int)
    label, all_label_ids, percolated = fill_bonds_identify_clusters_check_periodic_perco(1, 1, 0, grid)
    print(percolated)

    grid = (np.array([[
        [0., 1],
        [0, 1.],
        [0, 0.],
        [0, 0.]],

        [[0., 1],
         [0., 1],
         [0, 1],
         [1., 0.]],

        [[0., 1],
         [0., 1],
         [0, 1],
         [1., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [1, 0.]]], dtype=int))
    print(find_root(grid, 1, 0))
    print(find_root(grid, 2, 0))


def critical(J_x, J_y, J_d):
    def f(B):
        return sinh(2 * J_x * (B)) * sinh(2 * J_y * (B)) + sinh(2 * J_d * (B)) * (
                sinh(2 * J_x * (B)) + sinh(2 * J_y * (B))) - 1

    B = fsolve(f, 2, maxfev=400000000)
    return B

def criticalq3(J_x, J_y, J_d):
    def f(B):
        return np.sqrt(3)*((np.exp(2*J_x*B)-1)/np.sqrt(3))*((np.exp(2*J_y*B)-1)/np.sqrt(3))*((np.exp(2*J_d*B)-1)/np.sqrt(3)) + ((np.exp(2*J_x*B)-1)/np.sqrt(3))*((np.exp(2*J_y*B)-1)/np.sqrt(3)) + ((np.exp(2*J_x*B)-1)/np.sqrt(3))*((np.exp(2*J_d*B)-1)/np.sqrt(3))+((np.exp(2*J_d*B)-1)/np.sqrt(3))*((np.exp(2*J_y*B)-1)/np.sqrt(3)) -1

    B = fsolve(f, 10)
    return 1/B
