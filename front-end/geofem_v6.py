#!/usr/bin/env python3
# -*- coding: utf-8 -*--------------------------------------------------------
"""
Created on Wed Mar 31 17:14:28 2021

@author: akel
"""

import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from SimPEG import utils
from scipy.constants import mu_0

from SimPEG.electromagnetics import natural_source as NSEM
from discretize.utils import sdiag, mkvc

import NMT3D_v6 as model
import loadgeofemv2 as load


try:
    from pymatsolver import Pardiso as Solver
except:
    from SimPEG import Solver

from scipy.constants import mu_0, epsilon_0 as eps_0
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
#============================================================================

plt.close('all')
plt.style.use('ggplot')
t = time.time()
print('start : ', time.ctime())

#out=load.inputfiles('MT3D_input_ex0.in')   #Leitura dos dados
out=load.inputfiles('MT3D_input_ex0_topo.in') 
#out=model.modelmesh(out,level=1)
  #Leitura dos dados
#M,sig,sigBG=model.modelmesh(out,level=1)
#M,sig,sigBG,sigBG1d=model.modelmesh(out,level=1) #Criação do Mesh e modelo





M,sig,sigBG,sigBG1d=model.modelmesh(out,level=1) #Criação do Mesh e modelo
t = time.time()
print('start : ', time.ctime())




#ver malha modelo
fig, a1 = plt.subplots()
fig.canvas.set_window_title('topo file Slice Y')
M.plotSlice(np.log10(sig),grid=True, normal='y',ax=a1)
plt.xlim((M.nodes_x[0],M.nodes_x[-1]))
plt.ylim((M.nodes_z[0],M.nodes_z[-1]))
plt.show()


fig, a2 = plt.subplots()
fig.canvas.set_window_title('topo file Slice X')
M.plotSlice(np.log10(sig),grid=True, normal='x',ax=a2)
plt.xlim((M.nodes_x[0],M.nodes_x[-1]))
plt.ylim((M.nodes_z[0],M.nodes_z[-1]))
plt.show()

print("\n the mesh has {} cells".format(M))
print('terminado em: ', time.ctime())
print("\n Elapsed Time = {:1.2f} s".format(time.time() - t))
#simulação  começa abaixo

# f, rho_app,phs=model.runMT(M,sig,sigBG,sigBG1d,out['freq']) #Run simulacao

# print("\n the mesh has {} cells".format(M))
# print('terminado em: ', time.ctime())
# print("\n Elapsed Time = {:1.2f} s".format(time.time() - t))


# plt.figure()
# plt.plot(f,rho_app)
# plt.yscale('log')
# plt.xscale('log')
# plt.grid(which = 'minor',axis = 'both')
# #plt.ylim((1, 500))
# # plt.xlim((-3000, 3000))
# plt.xlabel('frequency (Hz)')
# plt.ylabel('Resistividade Aparente (Ohm.m)')
# plt.title('octree')
# #plt.plot(x,np.real(hx_py),x,np.imag(hx_py))
# plt.legend(('geofem'))
# plt.gca().invert_xaxis()









