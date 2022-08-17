
'''

grid = np.array([
    [1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, ],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1]])

bondx = np.array([[1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, ],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

bondy = np.array([[1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, ],
                  [1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0]])

# grid = initgrid(7,0.4)


label, bondx, bondy = fill_bonds_identify_clusters(1, 1, 0, grid)

print(check_percolation(label, bondx, bondy))

fig, ax2 = plt.subplots(4, 1)
ax2[0].imshow(label)
ax2[1].imshow(grid)
ax2[2].imshow(bondx)
ax2[3].imshow(bondy)
for i in range(0, label.shape[0]):
    for j in range(0, label.shape[1]):
        c = label[j, i]
        g = grid[j, i]
        ax2[0].text(i, j, str(c), va='center', ha='center')
        ax2[1].text(i, j, str(g), va='center', ha='center')

plt.show()

'''
''' 
@jit(nopython=True, parallel=True)
def squarelatticepercolation():
    proba = []
    prob2 = []
    for i in range(0,400):
        print(i)
        p = 0.5 #i*0.002
        proba.append(10 + i)
        proby1 = 0
        proby2 = 0
        for j in range(0,10):
            grid = initgrid(50+i,1)
            label , bondx , bondy = fill_bonds_identify_clusters(p, p, 0, grid, False)
            #total_filled_bonds.append(n_bonds)
            #if check_percolation(label , bondx , bondy) == True:
            #    proby1 = proby1 +1
            if check_percolation_nonperiodic(label) == True:
                proby2 = proby2 +1
            else:
                pass
        #prob1.append(proby1 / 40)
        prob2.append(proby2 / 10)




    return 0 ,prob2, proba

prob1,prob2, n_bonds = squarelatticepercolation()

#plt.plot( n_bonds , prob1, color='r')
plt.plot( n_bonds , prob2)
plt.hlines(0.5, 0 , 300, color='r')
#plt.vlines(0.59, 0 , 1)
plt.show()
'''

'''



def swendsen_wang_chain_magnetisation(N_s,p):

    grid = initgrid(150, 0.5)

    for iteration in range(0,1000):

        label , bondx , bondy , label_ids = fill_bonds_identify_clusters(p, p, 0, grid, True)

        grid_flipped = 1 - grid
        label_flipped , bondx_flipped , bondy_flipped , label_ids_flipped = fill_bonds_identify_clusters(p, p, 0, grid_flipped, True)

        grid_updated = np.zeros(N_s,N_s)
        for id in label_ids:
            if decision(0.5):
                for xx in range(0,len(grid)):
                   for yy in range(0,len(grid)):
                       if label[yy][xx] == id:
                           grid[yy][xx] = 1 - grid[yy][xx]

        for id in label_ids_flipped:
            if decision(0.5):
                for xx in range(0,len(grid)):
                   for yy in range(0,len(grid)):
                       if label[yy][xx] == id:
                           grid[yy][xx] = 1 - grid[yy][xx]
    magnetisation = 0.
    for xx in range(0, len(grid)):
        for yy in range(0, len(grid)):
            magnetisation = magnetisation + grid[yy][xx]
    return magnetisation

'''