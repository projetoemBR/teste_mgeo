#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:59:20 2019

@author: akel
"""

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

t = time.time()

# sphinx_gallery_thumbnail_number = 4

###############################################
# Basic Example
# -------------


style_list = ['default', 'classic'] + sorted(
        style for style in plt.style.available if style != 'classic')
#
# Here we demonstrate the basic two step process for creating a 2D tree mesh
# (QuadTree mesh). The region of highest discretization if defined within a
# rectangular box. We use the keyword argument *octree_levels* to define the
# rate of cell width increase outside the box.
#
plt.close('all')
"""
dh x nbc define as dimensoões.
eixo-x 40x128=5120
eixo-y 40x64= 2560
eixo-z 40x64= 2560

"""

#Dimensoes na horizontal
dh = 10   # minimum cell width (base mesh cell width)

dhx=dh
dhy=dh
dhz=1.5

#Dimensao eixo vertical( base 2)
nbcx =2**12# number of base mesh cells in x
nbcy =2**12
nbcz =2**15
# Define base mesh (domain and finest discretization)
hx = dhx*np.ones(nbcx)
hy = dhy*np.ones(nbcy)
hz = dhz*np.ones(nbcz)

M =  Mesh.TreeMesh([hx,hy,hz])
#M.x0 = np.r_[-(nbcx*dhx)/2,-(nbcy*dhy)/2,-(nbcz*dhz)/2]
M.x0 = np.r_[-(nbcx*dhx)/2,-(nbcy*dhy)/2,-nbcz*dhz+15000]



hz=[0000] # definir fronteira das camadas

n_camadas=len(hz)
#=========================================
#Criando mais uma área de dricretização
#=========================================

for i in range(0,n_camadas,1):
         xp, yp,zp = np.meshgrid( [-(nbcx*dhx)/32, (nbcx*dhx)/32],[-(nbcy*dhy)/32, (nbcy*dhy)/32], [hz[i]-1, hz[i]+1])
#         print(hz[i])
         xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors
         M = refine_tree_xyz(
         M, xyz, octree_levels=[2, 2, 4], method='radial', finalize=False)

 #Define corner points for disc 
xp, yp, zp = np.meshgrid( [-(nbcx*dhx)/32, (nbcx*dhx)/64],[-(nbcy*dhy)/64, (nbcy*dhy)/32], [-150, -350])
#
xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors
#
 #Discretize to finest cell size within rectangular box
M = refine_tree_xyz(
    M, xyz, octree_levels=[3, 3,3], method='radial', finalize=False
    )
#=========================================
#Criando mais uma área de dricretização
#=========================================
#xp, yp,zp = np.meshgrid( [0., 5120.],[0., 5120.], [1000., 999.])
#xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors

# Discretize to finest cell size within rectangular box



M.finalize()  # Must finalize tree mesh before use
ax=M
M.plotGrid(showIt=True)
ax = plt.gca()
##ax.invert_zaxis()
plt.show()


       # collect_obj, line_obj = M.plotSlice(np.log(sig), grid=True, normal='y',ax=axes)



print("\n the mesh has {} cells".format(M))
print("\n Elapsed Time = {:1.2f} s".format(time.time() - t))