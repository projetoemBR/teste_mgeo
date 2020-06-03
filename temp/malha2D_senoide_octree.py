#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:28:32 2019

@author: akel
"""
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
dx = 50    # minimum cell width (base mesh cell width) in x
dy = 50    # minimum cell width (base mesh cell width) in y

x_length = 30000.    # domain width in x
y_length = 30000.    # domain width in y

# Compute number of base mesh cells required in x and y
nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))

# Define the base mesh
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
M = TreeMesh([hx, hy], x0='CC')


# Refine mesh near points
xx = np.linspace(-10000,10000,3000)
yy = 400*np.sin((2*xx*np.pi)/1000)
pts = np.c_[mkvc(xx), mkvc(yy)]
M = refine_tree_xyz(M, pts, octree_levels=[2, 2], method='radial', finalize=False
    )

M.finalize()
print("\n the mesh has {} cells".format(M))
ccMesh=M.gridCC
print('indices:',np.size(ccMesh))


# We can apply the plotGrid method and output to a specified axes object
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
M.plotGrid(ax=ax)
ax.set_xbound(M.x0[0], M.x0[0]+np.sum(M.hx))
ax.set_ybound(M.x0[1], M.x0[1]+np.sum(M.hy))
ax.set_title('QuadTree Mesh')