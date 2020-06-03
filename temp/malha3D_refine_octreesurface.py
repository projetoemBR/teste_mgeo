#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:22:20 2019

@author: akel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:35:53 2019

@author: akel
"""

import time
import inspect
from SimPEG import Mesh
from discretize.utils import matutils, meshutils

from discretize.utils import mkvc, refine_tree_xyz
from SimPEG.EM import NSEM
import SimPEG as simpeg
import numpy as np
import matplotlib.pyplot as plt
import math 
import cmath

try:
    from pymatsolver import Pardiso as Solver
except:
    from SimPEG import Solver
  
    
t= time.time() 
style_list = ['default', 'classic'] + sorted(
        style for style in plt.style.available if style != 'classic')  
plt.close('all') 
   
def run(plotIt=True):        
      
    nFreq = 13
    freqs = np.logspace(2, -2, nFreq)
    
    # x and y grid parameters
#    dh = 10   # minimum cell width (base mesh cell width)
    
    dx = 15    # minimum cell width (base mesh cell width) in x
    dy = 15  
    dz =10
    
    x_length = 10000.    # domain width in x
    y_length = 10000.    
    z_length = 40000. 

# Compute number of base mesh cells required in x and y
    nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
    nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))
    nbcz = 2**int(np.round(np.log(z_length/dz)/np.log(2.)))

# Define the base mesh
    hx = [(dx, nbcx)]
    hy = [(dy, nbcy)]
    hz = [(dz, nbcz)]

    M =  Mesh.TreeMesh([hx,hy,hz],x0=['C','C',-15000])
    
    
# camadas
    
    xx = M.vectorNx
    yy = M.vectorNy
    zz = np.zeros(nbcx +1) #vetor linha(poderia ser uma função ou pontos)
    pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy), matutils.mkvc(zz)]
    M = meshutils.refine_tree_xyz(
    M, pts, octree_levels=[1, 1, 1], method='surface', finalize=False)
    
    
    xx = M.vectorNx
    yy = M.vectorNy
    zz = np.zeros(nbcx +1) -150#vetor linha(poderia ser uma função ou pontos)
    pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy), matutils.mkvc(zz)]
    M = meshutils.refine_tree_xyz(
    M, pts, octree_levels=[1, 1, 1], method='surface', finalize=False)
    
    
    xx = M.vectorNx
    yy = M.vectorNy
    zz = np.zeros(nbcx +1)-350 #vetor linha(poderia ser uma função ou pontos)
    pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy), matutils.mkvc(zz)]
    M = meshutils.refine_tree_xyz(
    M, pts, octree_levels=[1, 1, 1], method='surface', finalize=False)

#        
    M.finalize()   
    print("\n the mesh has {} cells".format(M))

    ccMesh=M.gridCC
    print('indices:',np.size(ccMesh))
###   
    
    conds = [1e-2,1]
    sig = simpeg.Utils.ModelBuilder.defineBlock(M.gridCC, [-15000,15000, -350], [15000,1500, -150], conds)
       
    sig[M.gridCC[:, 2] > 0] = 1e-8
    sigBG = np.zeros(M.nC) + conds[1]
    sigBG[M.gridCC[:, 2] > 0] = 1e-8
    

    fig = plt.figure('slice: malha + modelo')
    ax = fig.add_subplot()
    collect_obj,=M.plotSlice(np.log(sig), normal='x', ax=ax, grid=True)
   
if __name__ == '__main__':
    run()
    plt.show()
    print('Program Finished!')
    tf = time.time()
    print('tempo: ',round(tf-t,2),'s')