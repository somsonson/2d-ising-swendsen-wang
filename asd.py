import matplotlib.pyplot as plt
import numpy as np




grid = np.array([[0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0,],
 [1, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0],
 [1, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0]])

print(len(set(np.where(grid == 1)[0])))
print(np.where(grid == 1))