#!/usr/bin/python3
#--------  Code Dependencies   ----------
#\__________General Utilities____________/
import pprint
import time
import argparse
import sys
import os
import subprocess
import platform
import psutil
import gc
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize
from scipy.interpolate import griddata
#\_____________________________________/
#
#\__________Local functions__________/
import ipop_fun as ipop             #I/P and O/P functions
import plt_fun as p                 #Plot functions
#\_____________________________________/
#\__________Specialized stuff__________/
import discretize
from SimPEG import utils
from discretize.utils import mkvc, refine_tree_xyz
#\_____________________________________/
#
# -------------- estlev  ----------------------------
"""
      Estimate an optimzed number of octree levels
 d0 -> one dimension of initial discretization volume
 d1 -> one dimension of final discretization volume
"""
def estlev(d0, d1):
#
#------------ Objective inner function
    def objective(x,coeff):
        a, b = x
        return np.abs(a * 2**b - coeff)
#
#------------ Optimization
    x0 = np.array([1., 1.])
    mycoeffs = d1 / d0
    myoptions={'disp':True}
    results = minimize(objective,x0,args=mycoeffs,
                method="Nelder-Mead",                #option: "Powell"
                options = myoptions)
#
    return np.int(np.around(results.x[-1]))
#
# -------------- End of function   ---------------------
#
# -------------- otaid  ----------------------------
"""
                 Aid to octree_levels
(I/P)
   data       -> I/P data
   args[:,-1] -> Arguments
   args[-1]   -> A string specifying refinement

A) args[-1] = 'box' -> Work with a cuboid defining a given space. Fill space with
                        3D points using two strategies (see below).
   data    -> (x y z)=[6,] vertices coordinates of the given space in BOX FORMAT
  Two strategies, depending on args[0]
  1) args[0] = numpy.ndarray  -> sizes of the discretization element. This may result in a 
                                  dense discretization.
  2) args[0] = N (float, int) -> N is the number of points along each direction. This 
                                  results in a less dense discretization (preferred). If
                                  N is absent, then N=3. N**3 points in N surfaces of a 3D box (see flat)
B) args[-1] = 'srp' -> Work around a point.
   data    -> (x y z)=[3,] point coordinates
   args[0] -> (X Y Z)=[3,]  sizes of the smallest discretization element.

C) args[-1] = 'topo' -> Work around topography
   data     -> data dictionary

D) args[-1] = 'flat_topo' or 'strc_topo' -> Densify topography for plotting purposes
   data    -> data['hspace'] or data['topo']
   args[-2] = receivers and transmitters

"""
def otaid(data, *args):
# -------------- Check on last args' list argument 
  if not isinstance(args[-1], str):
    raise ipop.CmdInputError('@otaid=> Illegal specification: ', args[-1])
# -------------- Construct a mesh in a convex hull defined by 3D points.
  if args[-1] == 'box':
    if len(args) == 1:
#-- Generate 27 points in 3 surfaces of a 3D box (see flat)
      _, xyz = flat(data)
    elif isinstance(args[0], (float, int)):
#-- Generate N**3 points in N surfaces of a 3D box (see flat)
      _, xyz = flat(data, num=args[0])
    else:
#-- Use the discretization element sizes to constrict a point cloud. This may result
#    in a large number of points, depending on the discretization element size.
#-- Total number of points along each direction
      nxyz  = round2( (data[1::2]-data[0::2]) / args[0], ty ='int' )
#-- Construct the coordinates 
      x     = np.linspace(data[0],  data[1], nxyz[0] )
      y     = np.linspace(data[2],  data[3], nxyz[1] )
      z     = np.linspace(data[4],  data[5], nxyz[2] )
#-- Construct the cloud
      x,y,z = np.meshgrid(x,y,z)
#-- Ravel the O/P from meshgrid to a 3-column (x,y,z) matrix
      xyz   = np.c_[x.ravel(), y.ravel(), z.ravel()]
#      nxyz  = np.shape(xyz)[0]
#
#-- Plot generated points
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], marker='o')
    #plt.show()
#
    return xyz
#
#------------ Construct a cloud around a point
  elif  args[-1] == 'srp':
    raise ipop.CmdInputError('@otaid=> Deprecated specification: ', args[-1])
#-- Surround point data with 8 discretization elements. Make a box.
    xyz   = np.ravel(np.stack((data - args[0], 
                             data + args[0]), axis=0), order='F')
#-- Generate 8 points in 2 surfaces, top and bottom of the box (see flat)
    _, xyz = flat(xyz, num=2)
#
    return xyz
#
#------------ Construct a cloud around structured topography
  elif  args[-1] == 'topo':
#------------ Structured topography.
#-- Reads topography file.
    xyz = ipop.iofile(data['topo'], mode='topo')
    pline(['- Topography file with shape ', *np.shape(xyz)], width=18, align='<')
    pline('    and has size of '+human_size(xyz.nbytes))
#-- Sanity check on topo file.
#-- Assumes it has the horizontal sizes half-space. Origin at lower SW front corner.
    dummy = np.amax(xyz, axis=0) - np.amin(xyz, axis=0)
    if not np.all(dummy[:2] >= data['hspace'][1:-1:2]):
        pline([' Topo size(km):     ', *dummy[:2] / 1000.], width=20, align='<')
        raise ipop.CmdInputError("@otaid=> Topo xy limits fall short of half-space's.\n")
    pline('- Make min(z) of topography touches top of half-space.')
    xyz = xyz  + np.asarray([0.,0.,data['hspace'][-1]], dtype = float)
#-- xyz_u is a (x y z) topography file NOT a meshgrid!
    xyz_u = xyz.copy()
#-- Mesh topography
    x, y = np.meshgrid(xyz[:,0], xyz[:,1])        #, sparse=True
#               |--(n,D)--| |--(n,)--|  |(m,D)|    
    z = griddata(xyz[:,:-1], xyz[:,-1], (x, y), method='nearest')
#-- Mesh topography to array (x y z)
    xyz = np.c_[x.ravel(), y.ravel(), z.ravel()]
    #xyz = np.c_[utils.mkvc(x), utils.mkvc(y), utils.mkvc(z)]
    nxyz  = np.shape(xyz)[0]
    pline('- Gridded topography has size of '+human_size(xyz.nbytes))
#-- Glue receivers to topography. Get number of receivers
    dummy = data['rec'].ndim
    if dummy == 1:                              # 1 receiver
      pline('- Glue 1 receiver to topography.')
      xx, yy = data['rec'][0:-1]
#
      data['rec'][-1] = griddata(xyz[:,:-1], xyz[:,-1], (xx, yy), method='nearest')
    else:
      pline('- Glue '+ str(np.shape(data['rec'])[0])+' receivers to topography.')
      for dummy in data['rec']:
        xx, yy = dummy[0:-1]
        dummy[-1] = griddata(xyz[:,:-1], xyz[:,-1], (xx, yy), method='nearest')
#-- Glue transmitter clearance to topography
    if 'src' in data:
      pline('- Glue source to topography.')
      xx, yy = data['src'][0:-1]
#-- Source clearance from sea bottom
      dummy = data['src'][-1] - data['hspace'][-1]
#-- Glue clearance
      data['src'][-1] = griddata(xyz[:,:-1], xyz[:,-1], 
                        (xx, yy), method='nearest')
      data['src'][-1] = data['src'][-1] + dummy
#
    return xyz_u, xyz
#
#------------ Construct a cloud around flat topography for plotting purposes
  elif args[-1] == 'flat_topo' or 'strc_topo':
#-- Flat topo    
    if args[-1] == 'flat_topo':
      x = np.linspace(data[0], data[1], num=100)
      y = np.linspace(data[2], data[3], num=100)
#-- Mesh it
      x,y = np.meshgrid(x,y)
      z   = np.ones(np.shape(x)) * data[-1]
      p.pltv(args[-2], x, y, z, 'surf')
    else:
#-- Srtructured topo. Mesh it.
      x,y = np.meshgrid(data[:,0], data[:,1])
      z = griddata(data[:,:-1], data[:,-1], (x, y), method='nearest')
      p.pltv(args[-2], x, y, z, 'surf')
#
#-- Error message
  else:
    raise ipop.CmdInputError('@otaid=> Illegal entry\n')
#
# -------------- End of function   ---------------------
#
# -------------- coordtrf  ----------------------------
def coordtrf(ipdata):
  """ User origin is on the front southwest (bottom) corner of half-space.
                              _________ 
                             /.       /|
                            / .      / |              
                         z1/________/  |              
                  z        |  .     |  |              
                  | /y     |y1. .  .| .|              
                  |/       | .      | /              
                  +---x    |._______|/              
                  0        0        x1              
                       x0,y0,z0
 Notes:
  1) Half-space left, lower, front corner defines the user origin. 
  2) Transform elements other than box to box format refereed to user origin.
  3) Estimates amount of extrusion needed BUT DO NOT EXTRUDE ANY LONGER.
  """
#------------  Original ipdata['hspace'] in box format
  ipdata['hspace'] = np.array([
                                0,ipdata['hspace'][0],                  #x0,x1
                                0,ipdata['hspace'][1],                  #y0,y1
                                0,ipdata['hspace'][2]], dtype=float)    #z0,z1
# 
#------------  Original ipdata['sea'] in box format 
  nwhspace = ipdata['hspace'][-1]
  if 'sea' in ipdata:
      ipdata['sea']  = np.hstack([ ipdata['hspace'][:-2],
                                    nwhspace,
                                    nwhspace + ipdata['sea'] ])
      nwhspace = ipdata['sea'][-1]
#
#------------  Air comes on the top of the half-space or of the sea
  ipdata['air']     = np.hstack([ ipdata['hspace'][:-2],
                                  nwhspace,
                                  nwhspace + ipdata['air'] ])
#
  text = 'Model elements in box format and user coordinates'
  print(text.center(60, "-"))
#------------  Estimate amount of extrusion needed. IT DOES NOT EXTRUDE!
  """
                    +~~+---------------+ <- upward air extension
                    :~~:      Air      :
                    :~~|      Sea      |
           z        +~~+---------------+
           | /y     :~~:               :
           |/       :~~|  half space   |
           +---x    +~~+---------------+
                    :~~0               :
                    +~~^~~~~~~~~~~~~~~~+ <- downward hspace extension; 
      New origin -> 0' |-- Old origin        sea does not change.
  """
#-- Print original model info 
  pline('\n 1) Original domain in box format and user coordinates, from top to bottom:')
#
#-- Print model info so far
  pline(['   ', 'x(0)', 'x(1)', 'y(0)', 'y(1)', 'z(0)', 'z(1)'], width=8, align='^')
  pline(['air', *ipdata['air']], width=10, align='>')
  if 'sea' in ipdata:
    pline(['sea', *ipdata['sea']], width=10, align='>')
    pline(['hsp', *ipdata['hspace']], width=10, align='>') 

  for dummy in ipdata['rec']:
      pline(['rec', *np.stack((dummy, np.zeros(3)), axis=0).flatten('F')], width=10, align='>')
  if 'src' in ipdata:
          pline(['src', *np.stack((ipdata['src'], np.zeros(3)), axis=0).flatten('F')], width=10, align='>')
  if 'box' in ipdata:
    if ipdata['box'].ndim == 1:
      pline(['box', *ipdata['box']], width=10, align='>')
    else:
      for dummy in ipdata['box']:
        pline(['box', *dummy], width=10, align='>')
#
#  pline('\n 2) Core limits as from the anomalies and source.')
##-- Find a box that contains all anomalies
#  ws = np.array([[1, 1]], dtype = float).T * ipdata['box'] if ipdata['box'].ndim == 1 else ipdata['box']
#  ws = np.array([ [np.amin(ws[:,:2]), np.amin(ws[:,2:4]), np.amin(ws[:,4:])],
#                  [np.amax(ws[:,:2]), np.amax(ws[:,2:4]), np.amax(ws[:,4:])] ])
##-- Enlarge box to contain source and receivers
#  if ipdata['survey'] == 'mCSEM':
#    ws = np.array([ [np.minimum(ipdata['src'][0],ws[0,0]),
#                     np.minimum(ipdata['src'][1],ws[0,1]),
#                     np.minimum(ipdata['src'][2],ws[0,2])],
#                    [np.maximum(ipdata['src'][0],ws[1,0]),
#                     np.maximum(ipdata['src'][1],ws[1,1]),
#                     ws[1,2]] ])
##
#  ipdata['core'] = ws.flatten('F')
#  pline(['core= ', *ipdata['core']], width=10, align='>')
#------------  Estimate amount to extrude from core (Plessix, 2007, Geophys., 72:5, SM179, column 2)
#  pline('\n 2) Minimum to extrude in x, y and z:')
#  ipdata['ext'] = np.array([
#                  round10(ipdata['mad'][0] * ipdata['dxyz2'][0]),
#                  round10(ipdata['mad'][0] * ipdata['dxyz2'][1]),
#                  round10(ipdata['mad'][0] * ipdata['dxyz2'][2])
#                  ],dtype=float)
#-- half-space extends only downwards
#  pline(['\u0394(x,y,z)=', *ipdata['ext']*np.array([2.,2.,1.])], width=10, align='>')
#
#------------  Extrusion will be performed in main
#
  return ipdata
#
# -------------- End of function   ---------------------
#
# -------------- mBuilder  ----------------------------
"""
  Build a base mesh using TreeMesh or mesh_builder_xyz.
  x0 => origin or ‘anchor point’ of the mesh for TreeMesh
  x0 = 'CCC' -> center
  x0 = '000' -> first node location
"""
def mBuilder(data):                        #def mBuilder(data, N=3):
#------------ Base mesh constructor
#     Adjust domain dimensions to a 2**n range. hspace z dimension is extended downwards and
#       air upward. x and y are extended in both directions.
# ------------ Power of 2 estimator
  nbc_lf = lambda a, b, c: 2**int( np.round( np.log((b - a) / c)/np.log( 2.)))
#
# ------------ Decide on base mesh constructor
#  pline('\n 1) Base Mesh constructor bmc:\n')
#  pline(' 1.1) abs(bmc) = 1   -> Base mesh is constructed on half space dimensions.\n')
#  pline(' 1.2) abs(bmc) > 1   -> Base mesh is constructed by a n-point cloud xyz [n x dim].')
#  pline('                         the cloud has N points along each dimension of half-space,')
#  pline('                         defining a total of 3**N 3-D points. Fend expands domain.')
#  pline('                         so it cannot be used with topography.\n')
#  pline(' 1.3) sign(bmc) = 1  -> Base mesh is anchored at its bottom lower vertex.')
#  pline('                = -1 -> Base mesh is anchored its middle point.\n')
  pline('\n 1) Base Mesh constructor on half space dimensions:')
  pline(' 1.1) bmc = 0   -> Base mesh is anchored at its bottom lower vertex.')
  pline(' 1.2) bmc = 1   -> Base mesh is anchored at its middle point.\n')
#-- Enter N
  N = input(f'\n Enter bmc (rtn = 0):\n') or int(0)
  if isinstance(N, str):
      N = int(N)
  N = 0 if np.logical_and(N !=1 , N != 0) else N
#-- anchor point
  x0 = '000' if N == 0 else 'CCC'
# ------------ Vector h, a list of tuples, is used in discretize. TreeMesh to 
#               define base mesh for the whole domain; it includes air layer.
#              
  nbc = [nbc_lf(data['hspace'][0],data['hspace'][1],data['dxyz0'][0]),
         nbc_lf(data['hspace'][2],data['hspace'][3],data['dxyz0'][1]),
         nbc_lf(data['hspace'][4],data['air'][5],   data['dxyz0'][2])]
#
  h = [dummy for dummy in np.ravel((data['dxyz0'],nbc), order='F')]
  hx = [tuple([h[0], int(h[1])])]
  hy = [tuple([h[2], int(h[3])])]
  hz = [tuple([h[4], int(h[5])])]
# ------------ Base mesh
  mesh = discretize.TreeMesh([ hx, hy, hz ], x0=x0)
#
# ------------ Branch to a point cloud to expand domain (DEPRECATED)
  if N > 1:
#------------ Core defining points xyz = [n x dim]. Generate N**3 points in
#              N surfaces inside a 3-D box (see flat).
    pline('\n 2) Expand domain using padding.')
    pline('             [W,E]              [N,S]              [Down,Up]')
#-- Core padding_distance
#                                        |-----rb-----|
    pdist = data['dxyz2'] * np.stack(([data['mad'][0]]*3,
                                      [data['mad'][0]]*3), axis=0)
    pdist = np.ravel(pdist, order='F')
#-- Padding up = air layer    
    pdist[-1] = data['air'][-1] - data['hspace'][-1]
    pline(['Padding', *pdist], width=10, align='^')
    pdist = [pdist[dummy:dummy+2].tolist() for dummy in np.arange(0,6,2)]
#------------ Generate a Tree mesh given a cloud of xyz points
    xyz = otaid(data['hspace'], N, 'box')
#-- .
#------------ Expand Tree mesh with a cloud of xyz points. Expansion_factor not used in tree meshes.
#   Mesh with a smallest cell size width of h, within a convex hull defined by xyz.
#-- xyz [n x dim]     -> Point cloud.
#-- h = [1 x ndim]    -> Cell size for the core mesh: data['dxyz0']
#                        NB: h will be inherited by future calls to refine_tree_xyz
#-- padding_distance  -> Padding distances [[W,E], [N,S], [Down,Up]]
#                         based on data['dxyz2']
#-- base_mesh         -> discretize.BaseMesh
#                        For a mesh defined in Cartesian coordinates
#                         origin is the bottom southwest corner.
#-- depth_core        -> Depth of core mesh below xyz  -> depth_core=None
#-- expansion_factor  -> Expansion factor for padding  -> expansion_factor=not used
    mesh = discretize.utils.mesh_builder_xyz(
                            xyz, data['dxyz0'],
                            base_mesh = mesh,
                            padding_distance = pdist,
                            mesh_type='tree')
#-- Base mesh info
#  msize = np.array(np.shape(mesh.h)[1]) * data['dxyz0']
#  pline(['\n Base mesh tot. cells:   ', str(np.array(mesh.h).size)], width=22, align='^')
#  pline([  ' Base mesh # cells/xyz:  ', str(np.shape(mesh.h)[1])],   width=22, align='^')
#  pline([' Mesh grid size(km):', *msize / 1000.], width=20, align='<')  
  pline(['\n Base mesh origin  ', *mesh.x0], width=20, align='^')
#
  return mesh
#
# -------------- End of function   ---------------------
#
# -------------- IndBlock  ----------------------------
"""
    Builds a vector with block indices in the cell centers mesh. Vector is
     returned as a tuple.
    box    -> A box in box format.
    ccMesh -> A cell-centered mesh restricted to the acive indexes below topography.
"""
def IndBlock(box,ccMesh):
#-- Find box indexes in mesh
    indX = (box[0] <= ccMesh[:, 0]) & (ccMesh[:, 0] <= box[1])
    indY = (box[2] <= ccMesh[:, 1]) & (ccMesh[:, 1] <= box[3])
    indZ = (box[4] <= ccMesh[:, 2]) & (ccMesh[:, 2] <= box[5])
#-- Return box active cells as a tuple
    return indX & indY & indZ
#
# -------------- End of function   ---------------------
#
# -------------- flat  ----------------------------
"""
  Generate num points to define a flat surface or a box.
  If num=3 generate 9 points to define a flat surface or 27 for a box.

     *-----*------*           *---*---*
     |            |          /       /--*
     *     *      *         *   *   *  /--*
     |            |        /       /  *  /
     *-----*------*       *---*---*  /  *
                            *---*---*  /
                              *---*---*
  1) box -> Two extreme vertex of a 3-D box in box format. If initial and
             final z are the same -> it is a flat surface.
  2) num -> Number of points along x, y and z. num=3 -> 9 (surface) or 27 (box).

"""
def flat(box, num=3):
#.................. x_min   x_max
    x = np.linspace(box[0], box[1], num)
#.................. y_min   y_max
    y = np.linspace(box[2], box[3], num)
#-- Surface or box
    if np.isclose(box[-2],box[-1]):
#-- Flat
      z = np.full(np.shape(x), box[-1])
    else:
#..................... z_min    z_max
      z = np.linspace(box[-2], box[-1], num)
#-- Mesh it
    x,y,z = np.meshgrid(x,y,z)
#-- Ravel the O/P from meshgrid to a 3-column (x,y,z) matrix
    xyz = np.c_[x.ravel(), y.ravel(), z.ravel()]
#-- Same result can be obtained with: xyz = np.c_[mkvc(x.T), mkvc(y.T), mkvc(z.T)]
#-- Supress repeated lines from meshgrid
    xyz_u  = np.unique(xyz, axis=0)
#-- Plot generated points
#    fig = plt.figure()
#    ax = fig.add_subplot(projection='3d')
#    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], marker='o')
#    plt.show()
    return xyz_u,xyz
#
# -------------- End of function   ---------------------
#
# -------------- pline  ----------------------------
"""
Print stuff along a line
line -> A string or a list of strings or one string followed by real numbers.
        The string can be a blank
"""
def pline(line, width=None, align='^'):
  if isinstance(line, str):
    width = len(line) if width == None else width
    print ('{:{align}{width}}'.format(line, align=align, width=width))
  elif all(isinstance(dummy, str) for dummy in line):
    n = len(line)
    f = ['{:{align}{width}} '] * n
    print(''.join(f).format(*line, align=align, width=width))
  else:
    n = len(line)
    f = ['{:^'+str(len(line[0]))+'}']
    dummy =  line[1:]
    if dummy == np.float:
      f.extend(['{:{align}{width}.1f} '] * (n - 1))
    else:
      f.extend(['{:{align}{width}}'] * (n - 1))
    print(''.join(f).format(line[0], *dummy, align=align, width=width))
# -------------- End of function   ---------------------
#
# -------------- line  ----------------------------
"""
A line linking two points P to Q.
"""
def line(x, P, Q):
    chi = (Q[1] - P[1]) / (Q[0] - P[0])
    return chi * x + P[1] - chi * P[0]
#
# -------------- End of function   -------------------
#
# -------------- Round10  ----------------------------
"""
      Rounds up to nearest multiple of a suitable power of 10.
 value -> ditto.
"""
def round10(value):
    a = np.round(value/10.**np.int(np.log10(value)))
    a = a * 10.**np.int(np.log10(value))
    return float(a)
#
# -------------- End of function   ---------------------
#
# -------------- Round2  ----------------------------
"""
      Rounds up to nearest multiple of a suitable power of 2.
 value -> dim=1 numpy array.
 ty    -> float or int.
"""
def round2(value, ty = 'float'):
    a = np.round(value/2.**np.floor(np.log10(value)/np.log10(2)))
    a = a * 2.**np.floor(np.log10(value)/np.log10(2))
    return np.array(a, dtype=ty)
#
# -------------- End of function   ---------------------
#
# -------------- ramon  ----------------------------
"""Monitor RAM usage and release unreferenced memory
Args:
    size (int): file size in bytes.
    a_kilobyte_is_1024_bytes (boolean) - true for multiples of 1024, false for multiples of 1000.
Returns:
    Human-readable (string).
"""
def ramon(dict=False):
  if dict == False:
#-- Release unreferenced memory 
    gc.collect(generation=2)
#-- Monitor RAM
    # retrieves the current RAM usage from the total installed
#    pline('- RAM in use '+util.human_size(int(
#            psutil.virtual_memory().total - psutil.virtual_memory().available) ))
    # reports the RAM usage as a percentage.
    dummy = psutil.virtual_memory().percent
    pline('** RAM usage: '+ str(dummy) + ' %')
    if(dummy > 90): pline('\n\t\t ** WARNING **\n\t Running out of RAM.')
#-- Obtain a dictionary size
  else:
    size = int(0)
    size += sum([sys.getsizeof(dict) for v in dict.values()])
    size += sum([sys.getsizeof(dict) for k in dict.keys()])
    return size
#
# -------------- End of function   ---------------------
#
# -------------- human_size  ----------------------------
"""Convert a file size to human-readable form.
Args:
    size (int): file size in bytes.
    a_kilobyte_is_1024_bytes (boolean) - true for multiples of 1024, false for multiples of 1000.
Returns:
    Human-readable (string).
"""
def human_size(size, a_kilobyte_is_1024_bytes=False):

    suffixes = {1000: ['KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'], 1024: ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']}
    if size < 0:
        raise ValueError('Number must be non-negative.\n')
    multiple = 1024 if a_kilobyte_is_1024_bytes else 1000
    for suffix in suffixes[multiple]:
        size /= multiple
        if size < multiple:
            return '{:.3g}{}'.format(size, suffix)
    raise ValueError('Number is too large.\n')
#
# -------------- End of function   ---------------------
#
# -------------- get_host_info  ----------------------------
"""
Get information about the machine, CPU, RAM, and OS. Assumes sys.platform == Linux
Returns:
    hostinfo (dict): Manufacturer and model of machine; description of CPU
            type, speed, cores; RAM; name and version of operating system.
"""
def get_host_info():
#-- Default to 'unknown' if any of the detection fails
    manufacturer = model = cpuID = sockets = threadspercore = 'unknown'

#-- Manufacturer/model
    try:
        manufacturer = subprocess.check_output("cat /sys/class/dmi/id/sys_vendor", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
        model = subprocess.check_output("cat /sys/class/dmi/id/product_name", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        pass
    machineID = manufacturer + ' ' + model

#-- CPU information
    try:
        my_env = os.environ.copy()
        my_env["LC_ALL"] = "C"
        cpuIDinfo = subprocess.check_output("cat /proc/cpuinfo", shell=True, env=my_env, stderr=subprocess.STDOUT).decode('utf-8').strip()
        for line in cpuIDinfo.split('\n'):
            if re.search('model name', line):
                cpuID = re.sub('.*model name.*:', '', line, 1).strip()
        allcpuinfo = subprocess.check_output("lscpu", shell=True, env=my_env, stderr=subprocess.STDOUT).decode('utf-8').strip()
        for line in allcpuinfo.split('\n'):
            if 'Socket(s)' in line:
                sockets = int(re.sub("\D", "", line.strip()))
            if 'Thread(s) per core' in line:
                threadspercore = int(re.sub("\D", "", line.strip()))
            if 'Core(s) per socket' in line:
                corespersocket = int(re.sub("\D", "", line.strip()))
    except subprocess.CalledProcessError:
        pass

    physicalcores = sockets * corespersocket
    logicalcores = sockets * corespersocket * threadspercore

#-- OS version
    osversion = platform.platform()

#-- Dictionary of host information
    hostinfo = {}
    hostinfo['hostname'] = platform.node()
    hostinfo['machineID'] = machineID.strip()
    hostinfo['sockets'] = sockets
    hostinfo['cpuID'] = cpuID
    hostinfo['osversion'] = osversion

#-- Hyperthreading
    if logicalcores != physicalcores:
        hostinfo['hyperthreading'] = True
    else:
        hostinfo['hyperthreading'] = False

    hostinfo['logicalcores'] = logicalcores
#-- Number of physical CPU cores, i.e. avoid hyperthreading with OpenMP
    hostinfo['physicalcores'] = physicalcores

#-- Handle case where cpu_count returns None on some machines
    if not hostinfo['physicalcores']:
        hostinfo['physicalcores'] = hostinfo['logicalcores']
    hostinfo['ram'] = psutil.virtual_memory().total
    return hostinfo
