
import scipy.optimize as optimize
import pprint
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import leastsq
from scipy import optimize
from scipy.optimize import minimize
from scipy.constants import mu_0, epsilon_0 as eps_0
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import discretize
from SimPEG import utils
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
import pandas as pd
import pyvista as pv
from pyvista import examples
#
def ip():

    xyz=pd.read_csv('../models/survey_data.csv')  
#    xyz = []
#    with open("../models/survey_data.csv", 'r') as fobj:
#            lines =  fobj.readlines()
#    for line in lines:
#        line = line.rstrip()
#        line = line.split('\t')
#        print(line)
#        xyz.extend([float(dummy) for dummy in line][:])
    print(np.shape(xyz))
    xyz = np.asarray(xyz, dtype = float)
    xyz = np.reshape(xyz, (-1,3))
    print(f"xyz {np.shape(xyz)}")
    print(xyz[:10])
#
    return xyz
#
# -------------- Define seed surface   ----------------------------
def surf():

    ent = input(f'\n Enter 1 (read topo) or 2 (generate it): ')
    ent = int(ent)
    if ent == 1:
        xyz = ip()
        xyz = xyz[xyz[:, 0].argsort()]
        xyz = xyz[xyz[:, 1].argsort()]
        x = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
#-- Normalize to [0,1]
        x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))
        z = (z - np.amin(z)) / ( np.amax(z) -  np.amin(z))
    elif ent == 2:
        #f  = lambda x, y: 1 * np.cos(1 * x) * np.sin(1/2 * y) + 1
        f  = lambda x, y: (1 - x - y) / 2 
#-- Generate seed surface   
        xmin, xmax, xstep = -1., 1., .05
        #xmin, xmax, xstep = -np.pi/4., np.pi/6., np.pi/300.
        x = np.arange(xmin, xmax + xstep, xstep)
        y = np.arange(xmin, xmax + xstep, xstep)
        z = f(x, y)       
    else:
        raise SystemExit
#
#-- Normalize to [0,1]
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))
    z = (z - np.amin(z)) / ( np.amax(z) -  np.amin(z))
    a=np.array([np.amin(x), np.amin(y), np.amin(z)])
    b=np.array([np.amax(x), np.amax(y), np.amax(z)])
#-- Assemble
    xyz = np.vstack((x, y))
    xyz = np.vstack((xyz, z))
    xyz = np.transpose(xyz)
#-- Mesh it
    x, y = np.meshgrid(x, y)            #, sparse=True
#                |--(n,D)--| |--(n,)--| |(m,D)|    
    zz = griddata(xyz[:,:-1], xyz[:,-1], (x, y), method='nearest')
    print(np.shape(zz))
#-- Plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    surf = ax.plot_surface(x, y, zz, rstride=1, cstride=1, cmap='jet', linewidth=1, antialiased=True)
    #ax.scatter(xy[:,0],xy[:,1],zz,color="k",s=20)
    ax.set_zlim(np.amin(z.flatten(), axis=0), np.amax(z.flatten(), axis=0))
    fig.colorbar(surf, aspect=1)
    plt.show()
#
    return  a, b, xyz
#
# -------------- write Files   ----------------------------
def wfile(X, Name=None):
    """ Write files for O/P
    """ 
    #
    #------------  Open and write data
    text = 'File for O/P'
    print(text.center(60, "-"))
    path = input('Enter subdirectory for I/P and O/P (rtn=../models):') or '../models'
    path = path + "/"
    print(f"\n Directory {path}")
    #-- Choose data file name
    file = path + Name + ".btm"
    print(f"Open: {file}")
    #                  file object
    with open(file, 'w') as fobj:
        np.savetxt(fobj, X, delimiter='\t')
    text = 'Close OP File'
    print(text.center(60, "-"))
    fobj.close()
    return
#
# -------------- Main  ----------------------------
#-- Map   [a,b]|->[c,d]
f_t=lambda a,b,    c,d, t: c + (d-c) / (b-a) * (t - a)

#def make_example_data():
#    surface = examples.download_saddle_surface()
#    pprint.pprint(dir(surface))
#    points = examples.download_sparse_points()
#    poly = surface.interpolate(points, radius=12.0)
#    pprint.pprint(dir(poly))
#    print(f"points {np.shape(poly)}")
#
#    
#    return poly
#
#poly = make_example_data()
#
#poly.plot()

a, b, xyz = surf()
#-- Model size
# dznew = Desired z variation 200m/40km= 0.5%
# dznew = 200m/12km= 10%
dznew = 500.
z = xyz[:,2] * dznew
# Half-space
c = np.zeros(3)
d = np.asarray([20000,14000,10000], dtype = float)
#-- Model topography
x = f_t(a[0],b[0],c[0],d[0], xyz[:,0])
y = f_t(a[1],b[1],c[1],d[1], xyz[:,1])
print(f"Mapped surface: x={np.shape(x)} y={np.shape(y)} z={np.shape(z)}")
print(f"Mapped Xmin={np.amin(x.flatten(), axis=0)}, Xmax={np.amax(x.flatten(), axis=0)}")
print(f"Mapped Ymin={np.amin(y.flatten(), axis=0)}, Ymax={np.amax(y.flatten(), axis=0)}")
print(f"Mapped Zmin={np.amin(z.flatten(), axis=0)}, Zmax={np.amax(z.flatten(), axis=0)}")
#
#-- xyz file for output
#-- Assemble
xyz = np.vstack((x, y))
xyz = np.vstack((xyz, z))
xyz = np.transpose(xyz)
print(f"Write xyz file of shape {np.shape(xyz)}")
#
ent = input(f'\n Write o/p file (dflt=False): ') or False
if ent: wfile(xyz, Name="ramp")
#wfile(xyz, Name="test")
#-- Mesh it
x, y = np.meshgrid(x, y)            #, sparse=True
#           |--(n,D)--| |--(n,)--| |(m,D)|    
z = griddata(xyz[:,:-1], z[:], (x, y), method='nearest')
#
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='jet', linewidth=1, antialiased=True)
ax.set_zlim(np.amin(z.flatten(), axis=0), np.amax(z.flatten(), axis=0))
fig.colorbar(surf, aspect=1)
plt.show()