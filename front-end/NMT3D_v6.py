#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:14:28 2021


*V5- 
@author: akel
controi modelo e malha MT3D(input,opt='string')
input --> dicionário contendo informações da simulação,modelo geologico e meshgrid.
          veja a função loadmigeo para maiores informações.
          
opt ---> Variavel string que define o tipo de malha usada 'tensor' ou 'octree'
         se não declarado opt o código roda com a malha tensor.
         
         
         

def utilizadas 
         
         easymodelbox    Cria uma estrutura 3D com uma condutividade definida
         easymodellayer  Cria camadas planas 
         layermesh       Realiza mesh nas camadas
         boxmesh         Realiza mesh nas estruturas do tipo box
         runMT           Realiza mesh nas estruturas do tipo box



*V2- versão 2 adaptada para rodar com simpeg2020.Sem mesh do tipo octree !
*V3- em processo. adatação para rodar octree
*V4- Adaptações/correções/aperfeicoamentos do mesh octree em camadas
*V5- Inclusão Simulador MT(freq)

**V5.5- Incluir Opt offset MT
**V6- Inclusão topografia               
"""

#import SimPEG as simpeg
import numpy as np
import time
import discretize
from SimPEG import utils
from SimPEG.electromagnetics import natural_source as NSEM
from discretize.utils import mkvc, refine_tree_xyz
from discretize.utils import sdiag
from scipy.constants import mu_0, epsilon_0 as eps_0

from scipy.interpolate import griddata as gd
t = time.time()
print('start NMT3D : ', time.ctime())


def modelmesh(input_var,**kwargs):
    
    op=kwargs.get('opt') #tensor
    lv=kwargs.get('level') #grau de refinemanto
    
    if lv==None:
        lv=1
        pass

    dx=input_var['dxdydz'][0]
    dy=input_var['dxdydz'][1]
    dz=input_var['dxdydz'][2]
    x_length = input_var['x']    # tamanho do dominio em x
    y_length = input_var['y']    # tamanho do dominio em y
    z_length = input_var['z']    # tamanho do dominio em z
    
#Definções de mesh   
#    # Compute number of base mesh cells required in x and y
    nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
    nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))
    nbcz = 2**int(np.round(np.log(z_length/dz)/np.log(2.)))

    hx = [(dx, nbcx)]
    hy = [(dy, nbcy)]
    hz = [(dz, nbcz)]
    
    
    if op ==  None :
        M = discretize.TreeMesh([hx, hy, hz], x0='CCC')
        layermesh(M,input_var['layer'],lv,opt='S')
        if 'box' in input_var:
            boxmesh(M,input_var['box'],lv)
            pass
        M.finalize()
        
        
    if op == 'tensor':
        #M=discretize.TensorMesh([hx, hy,hz], x0=['C', 'C','C'])
        hx = [(200, 8, -1.5), (500.0, 40), (200, 8, 1.5)]
        hy = [(100, 10, -1.5), (500.0, 80), (100, 10, 1.5)]
        hz = [(200, 10, -1.6), (2500.0, 22), (500, 10,1.4)]
        
        M = discretize.TensorMesh([
            hx,
            hy,
            hz,],x0=["C", "C",-120000])
        pass   
##"Contrução" do modelo (add a condutividade )  
    sig=np.zeros(M.nC) + 1.0e-18 # define 
    

        
    
#    inclusão de camadas, se for o caso    
    if 'layer' in input_var:
        print('Add layers')
        M,sigBG,sigBG1d=easymodellayer(M,sig,input_var['layer'],input_var['cond'],opt='S')
        
        pass
    
    if 'topofile' in input_var:
        print('Building  topography')
        M=discretize.TreeMesh([hx, hy, hz], x0='CCC')
        M,xyz,Zo=loadtopo(M,input_var['topofile'],dx)
        if 'box' in input_var:
            boxmesh(M,input_var['box'],2)
            pass
        M.finalize()
        sigBG=np.zeros(M.nC) + input_var['cond'][1] #geology
    
#        print('k-->',M.nC)
        actv = utils.surface2ind_topo(M, xyz)
        print('surface2ind_topo',time.ctime())
        actv_ocean=np.invert(actv)
        print('np.invert',time.ctime())
        index_ocean=np.where(actv_ocean)
        print('np.where',time.ctime())
        sigBG[index_ocean]=input_var['cond'][0] #ocean
        
        sigBG[(M.gridCC[:,2]  > 0) ] =1.0e-18 #atm
        
     #   MESH 1D (para modelo de background) #teste
        mesh1d = discretize.TensorMesh([M.hz], np.array([M.x0[2]]))
        sigBG1d = np.zeros(mesh1d.nC) + 0
        sigBG1d[mesh1d.gridCC > 0] = 1e-18  #atm
        sigBG1d[mesh1d.gridCC < 0] = input_var['cond'][0] #ocean
        sigBG1d[mesh1d.gridCC < Zo] = 1.0e-1 #final layer       
        print('endtopfile',time.ctime())

        pass
    
#=====    
#incluir aqui  sigBG1d e add ao return abaixo
#==================
          
#   inclusão de estruturas , se for o caso    
    if 'box' in input_var:
        print('Add geologic body')
        M,sig=easymodelbox(M,sigBG,input_var['box'])
        pass    
    return M,sig,sigBG,sigBG1d 

#add sigBG1d ??

def easymodelbox(M,S,B):
    print('Build Blocks')
    modelBG=S
    n_box=len(B)/7
    for i in range(0,int(n_box),1):
        x=B[0+int(i*7)]
        Lx=B[1+int(i*7)]
        y=B[2+int(i*7)]
        Ly=B[3+int(i*7)]
        z=B[4+int(i*7)]
        Lz=B[5+int(i*7)]
        aim_cond=B[6+int(i*7)]
        modelBG = utils.model_builder.addBlock(
        M.gridCC, modelBG, [x, y, z], [x+Lx, y+Ly, z+Lz],aim_cond)
        S= utils.mkvc(modelBG)
    return M,S

#função para criar as camadas
def easymodellayer(M,S,camada,cond,**kwargs):
    op=kwargs.get('opt')
    if op=='B':
        print('Model do tipo box')
        S[M.gridCC[:, 2] >= 0] = 1.0e-18  # cond. ar
        c=0
        for i in range(0,len(camada),1):
            c=0+c
            S[(M.gridCC[:,2]  < -c) & (M.gridCC[:,2]  >= -camada[i])]=cond[i]
            c=camada[i]
            
        S[(M.gridCC[:,2]  < -c) ]=cond[i+1]
        pass
    if op=='S':
        print('Building Layers')
        print('Model do tipo surface')
        
        M1d = discretize.TensorMesh([M.hz], np.array([M.x0[2]]))
        sigBG1d = np.zeros(M1d.nC)
        sigBG1d[M1d.gridCC > 0] = 1.0e-18 #cond.ar
        
        X=M.gridCC[:,0];
        Z=np.zeros(len(X))
        S[(M.gridCC[:,2]  < Z) ] = 1.0e-18  #cond. ar
        c=0
        for i in range(0,len(camada),1):
            c=0+c
            S[(M.gridCC[:,2]  < -c) & (M.gridCC[:,2]  >= -camada[i])]=cond[i]
            sigBG1d[(M1d.gridCC  < -c) & (M1d.gridCC  >= -camada[i])]=cond[i]
                    
            c=camada[i]
            
        S[(M.gridCC[:,2]  < -c) ]=cond[i+1]
        sigBG1d[(M1d.gridCC < -c) ]=cond[i+1]

        print('E. conductivity',c,cond[i+1])
        pass        
    
    return M,S,sigBG1d

def layermesh(M,camada,lv,**kwargs):
    lv=2 # camadasfixasdas com level->2 comentar p/ level igual dos boxs
    op=kwargs.get('opt')
    if op=='B':
        print('Mesh do tipo box')

        #z=0
        xp, yp, zp = np.meshgrid( [-np.sum(M.h[0])/2, np.sum(M.h[0])/2],
                                   [-np.sum(M.h[1])/2,np.sum(M.h[1])/2],
                                   [-0-1*M.h[2][0],-0+1*M.h[2][0]])
    
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)] 
        M = refine_tree_xyz(
                M, xyz, octree_levels=[lv,lv,lv], method='box',
                finalize=False)
        #add camadas
        for i in range(0,len(camada),1):
            xp, yp, zp = np.meshgrid( [-np.sum(M.h[0])/2, np.sum(M.h[0])/2],
                                       [-np.sum(M.h[1])/2,np.sum(M.h[1])/2], 
                                       [-camada[i]-1*M.h[2][0],
                                        -camada[i]+1*M.h[2][0]])
            xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]
            M = refine_tree_xyz(M, xyz, octree_levels=[lv,lv,lv], method='box',
                                finalize=False)
        pass
    
    if op=='S':
        print('Mesh do tipo surface!!')
        dc=3 # reduz o tamanho da linha de refinamento da camada
        xx = np.arange(-np.sum(M.h[0])/dc,np.sum(M.h[0])/dc,M.h[0][0]) #np.sum(Msurf.hxh2) ou dg --> dá as bordas
        yy = np.arange(-np.sum(M.h[0])/dc,np.sum(M.h[0])/dc,M.h[0][0])
        xx, yy = np.meshgrid(xx, yy)
        zz=np.zeros([len(xx),len(xx)])-0
        #função de superficie
        xyz = np.c_[mkvc(xx), mkvc(yy),mkvc(zz)]
        M = refine_tree_xyz(M, xyz, octree_levels=[lv,lv,lv], method='surface', finalize=False)
        
#        add camadas
        for i in range(0,len(camada),1):
            zz=np.zeros([len(xx),len(xx)])-camada[i]
            xyz = np.c_[mkvc(xx), mkvc(yy),mkvc(zz)]
            M = refine_tree_xyz(M, xyz, octree_levels=[lv,lv,lv], method='surface', finalize=False)
        pass
            
    return

def boxmesh(M,box,lv):
#    lv=2
    n_box=len(box)/7
    fb=4 #fator de discretização ao redor
    for i in range(0,int(n_box),1):
        x1=box[0+int(i*7)]
        x2=x1+box[1+int(i*7)]
        y1=box[2+int(i*7)]
        y2=y1+box[3+int(i*7)]
        z1=box[4+int(i*7)]
        z2=z1+box[5+int(i*7)]
    #plano1 XY-zbotton
        xp, yp, zp = np.meshgrid( [x1, x2],[y1,y2], [z1-fb*M.h[2][0],z1+fb*M.h[2][0]])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  
        M = refine_tree_xyz(
                M, xyz, octree_levels=[lv,lv,lv], method='box', finalize=False
                )
    #plano2 XY-ztop
        xp, yp, zp = np.meshgrid( [x1,x2],[y1,y2], [z2-fb*M.h[2][0],z2+fb*M.h[2][0]])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)] 
        M = refine_tree_xyz(
                M, xyz, octree_levels=[lv,lv,lv], method='box', finalize=False
                )
    #plano3 XZ-yleft
        xp, yp, zp = np.meshgrid( [x1-fb*M.h[0][0],x1+fb*M.h[0][0]],[y1-fb*np.sqrt(3)*M.h[1][0],y2+fb*np.sqrt(3)*M.h[1][0]], [z2+fb*np.sqrt(3)*M.h[2][0],z1-fb*np.sqrt(3)*M.h[2][0]])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)] 
        M = refine_tree_xyz(
                M, xyz, octree_levels=[lv,lv,lv], method='box', finalize=False
                )
    #plano4 XZ-yrigth
        xp, yp, zp = np.meshgrid( [x2-fb*M.h[0][0],x2+fb*M.h[0][0]],[y1-fb*np.sqrt(3)*M.h[1][0],y2+fb*np.sqrt(3)*M.h[1][0]], [z2+fb*np.sqrt(3)*M.h[2][0],z1-fb*np.sqrt(3)*M.h[2][0]])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors
        M = refine_tree_xyz(
                M, xyz, octree_levels=[lv,lv,lv], method='box', finalize=False
                )
    #plano5 YZ-Xleft
        xp, yp, zp = np.meshgrid( [x1,x2],[y1-fb*M.h[1][0],y1+fb*M.h[1][0]], [z2+fb*np.sqrt(3)*M.h[2][0],z1-fb*np.sqrt(3)*M.h[2][0]])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors
        M = refine_tree_xyz(
                M, xyz, octree_levels=[lv,lv,lv], method='box', finalize=False
                )
    #plano6 YZ-Xrigth
        xp, yp, zp = np.meshgrid( [x1,x2],[y2-fb*M.h[1][0],y2+fb*M.h[1][0]], [z2+fb*np.sqrt(3)*M.h[2][0],z1-fb*np.sqrt(3)*M.h[2][0]])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors
        M = refine_tree_xyz(
                M, xyz, octree_levels=[lv,lv,lv], method='box', finalize=False
                )
    return

def pyvista_view(input_var,ftr):
    import pyvista as pv


    dx=input_var['dxdydz'][0]
    dy=input_var['dxdydz'][1]
    dz=input_var['dxdydz'][2]
    x_length = input_var['x']    # tamanho do dominio em x
    y_length = input_var['y']    # tamanho do dominio em y
    z_length = input_var['z']    # tamanho do dominio em z
    
    #Definções de mesh   
   # Compute number of base mesh cells required in x and y
    nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
    nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))
    nbcz = 2**int(np.round(np.log(z_length/dz)/np.log(2.)))

    hx = [(dx, nbcx)]
    hy = [(dy, nbcy)]
    hz = [(dz, nbcz/2)]
    
    M=discretize.TensorMesh([hx, hy,hz], x0=['C', 'C', -dz*nbcz/2])
    
    sig=np.zeros(M.nC) + 1e-18 # define 
    
    #inclusão de camadas, se for o caso    
    if 'layer' in input_var:
        easymodellayer(M,sig,input_var['layer'],input_var['cond'],opt='S')
        
        pass
        
    sigBG = sig
  
   #inclusão de estruturas , se for o caso    
    if 'box' in input_var:
        easymodelbox(M,sigBG,input_var['box'])
        pass
    
    models = {'res':np.log10(sig)}
    dataset = M.toVTK(models)
    p = pv.Plotter(notebook=0)
    p.show_grid(location='outer')
#    
#    
    p.add_mesh(dataset.slice('x'), opacity=0.75, name='x-slice')
    p.add_mesh(dataset.slice('y'), opacity=0.75, name='y-slice')
    p.add_mesh(dataset.slice('z'), opacity=0.75, name='z-slice')
#    p.add_mesh(threshed, name='vol')
    p.add_mesh(dataset.threshold([np.log10(ftr)-0.1,np.log10(ftr)]), name='vol')
    p.show() 
    
    # 
    return


def runMT(M,S,Sbg,Sbg1d,fq):
    try:
        from pymatsolver import Pardiso as Solver
    except:
        from SimPEG import Solver
        
        
        
        
    nFreq = len(fq)

    rx_x = np.array([0.])
    rx_y = np.array([0.])
    rx_loc = np.hstack((utils.mkvc(rx_x, 2), utils.mkvc(rx_y, 2),  np.zeros((np.prod(rx_x.shape), 1))))

    # Receivers
    rxList = []
    for rx_orientation in ['xx', 'xy', 'yx', 'yy']:
        rxList.append(NSEM.Rx.Point_impedance3D(rx_loc, rx_orientation, 'real'))
        rxList.append(NSEM.Rx.Point_impedance3D(rx_loc, rx_orientation, 'imag'))
    for rx_orientation in ['zx', 'zy']:
        rxList.append(NSEM.Rx.Point_tipper3D(rx_loc, rx_orientation, 'real'))
        rxList.append(NSEM.Rx.Point_tipper3D(rx_loc, rx_orientation, 'imag'))
    
    # Source list,
    srcList = []
    for freq in fq:
        srcList.append(NSEM.Src.Planewave_xy_1Dprimary(rxList, freq, Sbg1d, Sbg))
        
# Make the survey
    survey = NSEM.Survey(srcList)
    
# Set the problem
    problem = NSEM.Problem3D_ePrimSec(M, sigma=S, sigmaPrimary=Sbg)
    problem.pair(survey)
    problem.Solver = Solver

# Calculate the data
    fields = problem.fields()
    
    # Calculate the data
    fields = problem.fields()    # returns secondary field
    
    
    rec_x = np.array([-200]) #M.getTensor('Ex')[0]
    rec_y = np.zeros(np.prod(rec_x.shape))
    rec_z = np.zeros(np.prod(rec_x.shape))
    pos_rx = np.hstack((utils.mkvc(rec_x,2),utils.mkvc(rec_y,2),utils.mkvc(rec_z,2)))
    
    
    grid_field_px = np.empty((M.nE,nFreq),dtype=complex)
    grid_field_py = np.empty((M.nE,nFreq),dtype=complex)
    for i in range(nFreq):
        grid_field_px[:,i] = np.transpose(fields._getField('e_pxSolution', i))
        grid_field_py[:,i] = np.transpose(fields._getField('e_pySolution', i))


    # campos E e H calculado em todas as arestas d malha
    e_px_full  = fields._e_px(grid_field_px, srcList)
    e_py_full  = fields._e_py(grid_field_py, srcList)
    h_px_full  = fields._b_px(grid_field_px, srcList)/mu_0
    h_py_full  = fields._b_py(grid_field_py, srcList)/mu_0
    
    
    # Interpolando campos nos locais do receptor
    Pex = M.getInterpolationMat(pos_rx,'Ex')
    ex_px = Pex*e_px_full
    ex_py = Pex*e_py_full
    
    Pey = M.getInterpolationMat(pos_rx,'Ex')
    ey_px = Pey*e_px_full
    ey_py = Pey*e_py_full
    
    Pbx = M.getInterpolationMat(pos_rx,'Fx')
    hx_px = Pbx*h_px_full
    hx_py = Pbx*h_py_full
    
    Pby = M.getInterpolationMat(pos_rx,'Fy')
    hy_px = Pby*h_px_full
    hy_py = Pby*h_py_full



    hd = sdiag(mkvc(1.0/ (sdiag(mkvc(hx_px,2)) * mkvc(hy_py,2) - sdiag(mkvc(hx_py,2)) * mkvc(hy_px,2)),2))
    
    orientation = 'xy'
    if "xx" in orientation:
        Zij = ex_px * hy_py - ex_py * hy_px
    elif "xy" in orientation:
        Zij = -ex_px * hx_py + ex_py * hx_px
    elif "yx" in orientation:
        Zij = ey_px * hy_py - ey_py * hy_px
    elif "yy" in orientation:
        Zij = -ey_px * hx_py + ey_py * hx_px
    # Calculate the complex value
    Imped = hd * mkvc(Zij,2)
    
    rho_app = np.empty((nFreq),dtype=float)
    phs = np.empty((nFreq),dtype=float)
    for i in range(nFreq):
        print("freq:", fq[i])
        rho_app[i] = 1/(2*np.pi*fq[i]*mu_0) * abs(Imped[i])**2
        phs[i]     = np.arctan2(Imped[i].imag, Imped[i].real)*(180./np.pi)
        
 
    return fq,rho_app,phs
  

    
def loadtopo(M,filename,delta):
    
    
    tp=np.loadtxt(filename);
    
    lv=2

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
#    zi=zi

    Zo=zi[int(nxi/2-1)][int(nyi/2-1)]
    
    dh=dxi/2.0
    X=8*1024 #50x1024
    Y=8*1024
    Z=8*1024
    
    zz=zi
    
    dc=1
    xx = xi/dc
    yy = yi/dc
    
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
    M = refine_tree_xyz(M, xyz, octree_levels=[lv,lv,lv], 
                        method='surface', finalize=False) # topografia
    
    #refinamento nivel do mar
    dc=1
    xx0 = np.arange(-np.sum(M.h[0])/dc,np.sum(M.h[0])/dc,M.h[0][0]) #np.sum(Msurf.hxh2) ou dg --> dá as bordas
    yy0 = np.arange(-np.sum(M.h[0])/dc,np.sum(M.h[0])/dc,M.h[0][0])
    xx0, yy0 = np.meshgrid(xx0, yy0)
    zz0=np.zeros([len(xx0),len(xx0)])-0 #mesh sea level
        #função de superficie
    xyz0 = np.c_[mkvc(xx0), mkvc(yy0),mkvc(zz0)]
    M = refine_tree_xyz(M, xyz0, octree_levels=[lv,lv,lv], method='surface', finalize=False)
        
    
    
    
    

    
    return M,xyz,Zo
   
    
