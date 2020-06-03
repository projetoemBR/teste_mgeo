#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Nov 20 12:27:13 2019

@author: akel
"""

from discretize import TreeMesh
from discretize.utils import matutils, meshutils
import SimPEG as simpeg
import matplotlib.pyplot as plt
import numpy as np
plt.close('all') 

# sphinx_gallery_thumbnail_number = 4
dx = 20    # minimum cell width (base mesh cell width) in x
dy = 10   # minimum cell width (base mesh cell width) in y

x_length = 30000.    # domain width in x
y_length = 40000.    # domain width in y

# Compute number of base mesh cells required in x and y
nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))

# Define the base mesh
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
#deslocar eixo z em -15000. "C" centraliza o eixo na direção escolhida
M = TreeMesh([hx, hy], x0=['C',-15000])

## definir camada
xx = M.vectorNx
yy = np.zeros(nbcx+1) #vetor linha(poderia ser uma função ou pontos)
pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy)]
M = meshutils.refine_tree_xyz(M, pts,octree_levels=[1, 1], method='surface',finalize=False)

xx = M.vectorNx
yy = np.zeros(nbcx+1)-150 #vetor linha(poderia ser uma função ou pontos)
pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy)]
M = meshutils.refine_tree_xyz(M, pts,octree_levels=[1, 1], method='surface',finalize=False)

xx = M.vectorNx
yy = np.zeros(nbcx+1)-350 #vetor linha(poderia ser uma função ou pontos)
pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy)]
M = meshutils.refine_tree_xyz(M, pts,octree_levels=[1, 1], method='surface',finalize=False)

M.finalize()  # Must finalize tree mesh before use

#informações do meshgrid

print("\n the mesh has {} cells".format(M))
ccMesh=M.gridCC
print('indices:',np.size(ccMesh))


     
conds = [1e-2,1]
sig = simpeg.Utils.ModelBuilder.defineBlock(M.gridCC, [-5040 , -150], [5040, -350], conds)
#       
sig[M.gridCC[:, 1] > 0] = 1e-8
sigBG = np.zeros(M.nC) + conds[1]
sigBG[M.gridCC[:, 1] > 0] = 1e-8

M.plotImage(np.log(sig), grid=True )




