#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:43:22 2019

@author: ellen
"""

from SimPEG import Mesh, Utils
from discretize.utils import mkvc, refine_tree_xyz
import numpy as np
import matplotlib.pyplot as plt


style_list = ['default', 'classic'] + sorted(
        style for style in plt.style.available if style != 'classic')

plt.close('all')

# sphinx_gallery_thumbnail_number = 4

###############################################
# Basic Example
# -------------
#
# Here we demonstrate the basic two step process for creating a 2D tree mesh
# (QuadTree mesh). The region of highest discretization if defined within a
# rectangular box. We use the keyword argument *octree_levels* to define the
# rate of cell width increase outside the box.
#

#size dh x nbc(base2)
#Dimensoes na horizontal
dh = 10   # minimum cell width (base mesh cell width)

dhx=10
dhy=1.5
dhz=dh

#Dimensao eixo vertical( base 2)
nbcx =2**12# number of base mesh cells in x
nbcy =2**15
#nbcz =2**15
# Define base mesh (domain and finest discretization)
hx = dhx*np.ones(nbcx)
hy = dhy*np.ones(nbcy)
#hz = dhz*np.ones(nbcz)


M =  Mesh.TreeMesh([hx,hy])

M.x0 = np.r_[-(nbcx*dhx)/2,-nbcy*dhy+15000]


#definir a camada 

xp, yp = np.meshgrid( [-(nbcx*dhx)/2, (nbcx*dhx)/2], [1., -1.]) #layer
xy = np.c_[mkvc(xp), mkvc(yp)]  # mkvc creates vectors

# Discretize to finest cell size within rectangular box
M = refine_tree_xyz(
    M, xy, octree_levels=[1,1], method='box', finalize=False
    )



# Define objeto
xp, yp = np.meshgrid( [-(nbcx*dhx)/32, (nbcx*dhx)/32], [-10150, -10650.]) #goal
xy = np.c_[mkvc(xp), mkvc(yp)]  # mkvc creates vectors

# Discretize to finest cell size within rectangular box
M = refine_tree_xyz(
    M, xy, octree_levels=[4, 4], finalize=False,method='radial'
    )


#=========================================
#Criando mais uma área de dricretização
#=========================================




M.finalize()  # Must finalize tree mesh before use
x=M
M.plotGrid(showIt=True)
ax = plt.gca()
#ax.invert_yaxis()
plt.show()

nC = M.nC
print(M)
#print("Aqui!")
#mesh.plotGrid(showIt=True)