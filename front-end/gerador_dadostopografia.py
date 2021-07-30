#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:27:55 2021

@author: akel
"""

import SimPEG as simpeg
#from SimPEG import Mesh, Utils
import discretize
from SimPEG import utils

from discretize.utils import mkvc, refine_tree_xyz
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit

from scipy.interpolate import griddata as gd
import pandas as pd
plt.close('all')
plt.style.use('ggplot')
t = time.time()


dh=100

x = np.arange(-3200,3200,dh)
q=x
print(len(x))
y = np.arange(-3200,3200,dh)
print(len(x))

x, y = np.meshgrid(x,y)
print(len(x))


#funções para teste
#zz = 400*np.sin(2*np.pi*xx/8000) +400*np.sin(2*np.pi*yy/4000) -2000 #superficie definida
#z = 0.0*np.sin(2*np.pi*x/8000) +000.0*np.sin(2*np.pi*y/4000) -1000

z =-1000+5*(x+3200)/32+0*(y+3200)/128
print(z.max()-z.min())
#z =(x+3000)**2/(90000*3.93361)+0*np.random.rand(np.size(x,0),np.size(y,1))-300

#z=25*np.sin(2*np.pi*x/8000) +30*np.sin(2*np.pi*y/4000)-300
#z/=x/60+50-300  #0*np.random.rand(54,80)




fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z)
fig.colorbar(surf, shrink=0.5, aspect=5)



filename='teste_topografia_dado_temp.dat'
f=open(filename,'w')
for i in range(np.size(z,0)):
    for j in range(np.size(z,1)):
        f.write('{0:9.5F} {1:9.5F} {2:5.5F} \n'.format(x[i,j],y[i,j],z[i,j]))            
f.close()


def loadtopo(filename,delta):
    tp=np.loadtxt(filename);

    X=tp[:,0];
    Y=tp[:,1];
    Z=tp[:,2];
    
    LX=max(X)-(min(X))
    LY=max(Y)-(min(Y))
    
    dxi=delta
    dyi=delta
    
    
    nxi=LX/dxi #número de celulas em x
    nyi=LY/dyi #numero de celulas em y
    
    
    xi = np.linspace(min(X), max(X), int(nxi))
    yi = np.linspace(min(Y), max(Y), int(nyi))
    xi, yi = np.meshgrid(xi, yi)
    zi = gd((X, Y), Z,(xi, yi) ,method='cubic')
    zi=zi

    
    dh=dxi/2.0
    X=8*1024 #50x1024
    Y=8*1024
    Z=8*1024
    
    zz=zi
    
    xx = xi
    yy = yi
    
    #Dimensao eixo vertical( base 2)
    nbcx = 2**int(np.round(np.log(X/dh)/np.log(2.)))
    nbcy = 2**int(np.round(np.log(Y/dh)/np.log(2.)))
    nbcz = 2**int(np.round(np.log(Z/dh)/np.log(2.)))
    # Define base mesh (domain and finest discretization)
    hx = [(dh, nbcx)]
    hy = [(dh, nbcy)]
    hz = [(dh, nbcz)]
    
    
    M=discretize.TreeMesh([hx, hy, hz], x0='CCC')
    xyz = np.c_[mkvc(xx), mkvc(yy),mkvc(zz)]
    M = refine_tree_xyz(M, xyz, octree_levels=[3,3,3], 
                        method='surface', finalize=False) #refinamento topografia
    
    #refinamento nivel do mar
    zz0= zz*0
    xyz0 = np.c_[mkvc(xx), mkvc(yy),mkvc(zz0)]
    M = refine_tree_xyz(M, xyz0, octree_levels=[1,1,1], method='surface', finalize=False)
    
    M.finalize()
    s=np.zeros(M.nC) +1 #geology
    
    print('k-->',M.nC)
    
    X=M.gridCC[:,0];
    Y=M.gridCC[:,1];
    
    
    actv = utils.surface2ind_topo(M, xyz)
    nC = int(actv.sum())
    actv_ocean=np.invert(actv)
    index_ocean=np.where(actv_ocean)
    s[index_ocean]=0.01 #ocean
    
    s[(M.gridCC[:,2]  > 0) ] = 1.0e-18
    
    return M,s






Me,sig=loadtopo(filename,100)

fig, a1 = plt.subplots()
fig.canvas.set_window_title('Slice Y')
Me.plotSlice(np.log10(sig),grid=True, normal='y',ax=a1)
plt.xlim((-3200,3200))
plt.ylim((-3000,500))
plt.show()
plt.show()

fig, a2 = plt.subplots()
fig.canvas.set_window_title('Slice X')
Me.plotSlice(np.log10(sig),grid=True, normal='x',ax=a2)
plt.xlim((-3200,3200))
plt.ylim((-3000,500))
plt.show()

print("\n the mesh has {} cells".format(Me))
print('terminado em: ', time.ctime())
print("\n Elapsed Time = {:1.2f} s".format(time.time() - t))
