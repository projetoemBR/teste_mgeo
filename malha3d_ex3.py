#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:09:00 2019

@author: akel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:04:53 2019

@author: akel
"""

from SimPEG import Mesh, Utils
from discretize.utils import mkvc, refine_tree_xyz
import numpy as np
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt
import time
plt.close('all')
t = time.time()

style_list = ['default', 'classic'] + sorted(
        style for style in plt.style.available if style != 'classic')

"""
dh x nbc define as dimensoões.
eixo-x 40x128=5120
eixo-y 40x64= 2560
eixo-z 40x64= 2560

"""

#Dimensoes na horizontal
dh = 40   # minimum cell width (base mesh cell width)

#Dimensao eixo vertical( base 2)
nbcx =128# number of base mesh cells in x
nbcy =128
nbcz =128
# Define base mesh (domain and finest discretization)
hx = dh*np.ones(nbcx)
hy = dh*np.ones(nbcy)
hz = dh*np.ones(nbcz)

M =  Mesh.TreeMesh([hx,hy,hz])


hz=[1000] # definir fronteira das camadas

n_camadas=len(hz)
#=========================================
#loop de criação e camadas
#=========================================

for i in range(0,n_camadas,1):
         xp, yp,zp = np.meshgrid( [0., 5120.],[0., 5120.], [hz[i]-1, hz[i]+1])
         print(hz[i])
         xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors
         M = refine_tree_xyz(
         M, xyz, octree_levels=[1, 1, 1], method='box', finalize=False)




#=========================================
#Target
#=========================================
xp, yp, zp = np.meshgrid( [2400., 2600.],[2400.,2600], [2000 , 1999]) #goal

xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors

# Discretize to finest cell size within rectangular box
M = refine_tree_xyz(
    M, xyz, octree_levels=[4,4,4], method='radial', finalize=False
    )



M.finalize()  # Must finalize tree mesh before use
ax=M
M.plotGrid(showIt=True)
ax = plt.gca()
ax.invert_zaxis()
plt.show()

print("\n the mesh has {} cells".format(M.nC))
print("\n Elapsed Time = {:1.2f} s".format(time.time() - t))