from numba_plyfit import fit_poly
import numpy as np

x = np.asarray([1,2])
y = np.asarray([2,4])

print(         fit_poly(x,y,deg=1 ))