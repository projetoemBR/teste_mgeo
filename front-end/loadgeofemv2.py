#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:56:12 2020

@author: akel
Leitura dos arquivos de input 
input --> arquivo com as instruções do modelo geologico e caracteristicas
          da simulação. Todas as instruções são identificadas com um label
          seguidos ":" com a informação. Por exemplo, para o método geofísico,
          temos duas opções MT3D e MCSEM. Assim a instrução no arquivo fica.
          metodo: MT3D
          
          
v2----inclusão da leitura dos dados de topografia.          
"""

import time

#import discretize
#from SimPEG import utils

#from discretize.utils import mkvc, refine_tree_xyz
import numpy as np
#from scipy.interpolate import griddata as gd

import sys 

#def loadtopo(filename,delta):
    # tp=np.loadtxt(filename);

    # X=tp[:,0];
    # Y=tp[:,1];
    # Z=tp[:,2];
    
    # LX=max(X)-(min(X))
    # LY=max(Y)-(min(Y))
    
    # dxi=delta
    # dyi=delta
    
    
    # nxi=LX/dxi #número de celulas em x
    # nyi=LY/dyi #numero de celulas em y
    
    
    # xi = np.linspace(min(X), max(X), int(nxi))
    # yi = np.linspace(min(Y), max(Y), int(nyi))
    # xi, yi = np.meshgrid(xi, yi)
    # zi = gd((X, Y), Z,(xi, yi) ,method='cubic')
    # zi=zi

    
    # dh=dxi/2.0
    # X=8*1024 #50x1024
    # Y=8*1024
    # Z=8*1024
    
    # zz=zi
    
    # xx = xi
    # yy = yi
    
    # #Dimensao eixo vertical( base 2)
    # nbcx = 2**int(np.round(np.log(X/dh)/np.log(2.)))
    # nbcy = 2**int(np.round(np.log(Y/dh)/np.log(2.)))
    # nbcz = 2**int(np.round(np.log(Z/dh)/np.log(2.)))
    # # Define base mesh (domain and finest discretization)
    # hx = [(dh, nbcx)]
    # hy = [(dh, nbcy)]
    # hz = [(dh, nbcz)]
    
    
    # M=discretize.TreeMesh([hx, hy, hz], x0='CCC')
    # xyz = np.c_[mkvc(xx), mkvc(yy),mkvc(zz)]
    # M = refine_tree_xyz(M, xyz, octree_levels=[1,1,1], 
    #                     method='surface', finalize=False)
    
    # #refinamento nivel do mar
    # zz0= zz*0
    # xyz0 = np.c_[mkvc(xx), mkvc(yy),mkvc(zz0)]
    # M = refine_tree_xyz(M, xyz0, octree_levels=[1,1,1], method='surface', finalize=False)
    
    # M.finalize()
    # s=np.zeros(M.nC) + 10000 #ocean
    
    # print('k-->',M.nC)
    
    # X=M.gridCC[:,0];
    # Y=M.gridCC[:,1];
    
    
    # actv = utils.surface2ind_topo(M, xyz)
    # actv_geology=np.invert(actv)
    # index_geology=np.where(actv_geology)
    # s[index_geology]=100
    
    # s[(M.gridCC[:,2]  > 0) ] = 1.0
    
#    return M,s

def inputfiles(filename):
    """ Leitura arquivo input"""
    out={'info input migeo': time.ctime()} #informação da hora da leitura
    arq=open(filename,"rt")
    count=0
    while True:
        linha=arq.readline() 
        if not linha:
            break
        linha=linha[:-1]
        try:
            st,temp=linha.split(":",1)
        except ValueError:
            break
        if st[0]==str('#'):
            pass
        elif st.lower()==str('metodo'):
            M=temp.split(",")
            out['metodo']=str(M).lower()
        elif st.lower()==str('topo'):
            tp=temp.split(",")
            tp2=str(tp)[2:-2].lstrip()
            out['topofile']=tp2
         #info_domain  
        elif st.lower()==str('freq'):
            f=np.array(temp.split(","),dtype=float)
            freq=np.logspace(np.log10(f[0]),np.log10(f[1]),int(f[2]))
            out['freq']= freq
        elif st.lower()==str('x'):
            X=np.array(temp.split(","),dtype=float)
            out['x']=X
        elif st.lower()==str('y'):
            Y=np.array(temp.split(","),dtype=float)
            out['y']=Y
        elif st.lower()==str('z'):   
            Z=np.array(temp.split(","),dtype=float)
            out['z']=Z
        #cellsize
        elif st.lower()==str('dxdydz'):
            dx,dy,dz=np.array(temp.split(","),dtype=float)
            out['dxdydz']=dx,dy,dz
        elif st.lower()==str('layer'):
            layer=np.array(temp.split(","),dtype=float)
            out['layer']=layer
        elif st.lower()==str('cond'):
            cond=np.array(temp.split(","),dtype=float)
            out['cond']=cond
        elif st.lower()==str('box'):
            count+=1
            if count==1:
                box=np.array(temp.split(","),dtype=float)
            else:
                tmp_box=np.array(temp.split(","),dtype=float)
                box = np.concatenate((box,tmp_box))
            out['box']=box
           
        #infomeshgrid

        #infomodelMT1D    
        elif st.lower()==str('thi'):
            thi=np.array(temp.split(","),dtype=float)
            out['thi']=thi
        elif st.lower()==str('res'):
            res=np.array(temp.split(","),dtype=float)
            out['res']=res
        else:
            nome = None
            print("Label",st, "invalido")
            sys.exit()
    arq.close()
    return out




# if __name__ == "__main__":
#     import sys
#     print(input_1(str(sys.argv[1])))