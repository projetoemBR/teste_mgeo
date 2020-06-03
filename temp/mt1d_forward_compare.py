#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 27 23:23:20 2019

MT3D forward modeling with 1D geoelectric model

@author: diego
"""
import time
import inspect
#from SimPEG import (
#    Mesh, Utils, Maps, DataMisfit, Regularization, Optimization,
#    InvProblem, Directives, Inversion, Versions
#)
#from SimPEG.Utils import plotLayer
import SimPEG as simpeg
from SimPEG.EM import NSEM
import numpy as np
import matplotlib.pyplot as plt
import math 
import cmath

try:
    from pymatsolver import Pardiso as Solver
except:
    from SimPEG import Solver
  
    
t= time.time()    
   
def run(plotIt=True):        
    
#    nFreq = 13
#    freqs = np.logspace(2, -2, nFreq)
#    
#    # x and y grid parameters
#    ctx   = 10 #100
#    npadx = 22
#    incx  = 1.3
#    nx    = [(ctx,npadx,-incx),(ctx,6),(ctx,npadx,incx)]            
#    
#    # z grid parameters
#    ct1   = 10
#    npad1 = 21
#    inc1  = 1.3
#    
#    ct    = 10
#    npad  = 20
#    
#    ct2   = 10
#    npad2 = 20
#    inc2  = 1.3
    
    
        
    nFreq = 13
    freqs = np.logspace(2, -2, nFreq)
    
    # x and y grid parameters
    ctx   = 10 #100
    npadx = 16
    incx  = 1.5
    nx    = [(ctx,npadx,-incx),(ctx,6),(ctx,npadx,incx)]            
    
    # z grid parameters
    ct1   = 30
    npad1 = 15
    inc1  = 1.3
    
    ct    = 25
    npad  = 20
    
    ct2   = 1
    npad2 = 27
    inc2  = 1.3     
      
    
    
    
    
    nz = [(ct1,npad1,-inc1),(ct,npad),(ct2,npad2,inc2)]   
    
    # Origem do sistema de coordenadas da malha
    ztop  = 0
    for i in range(1,npad1+1):
        ztop += ct1*inc1**(i)
    ztop += 1.*ct*npad #9*ct
    # Make a mesh
    M = simpeg.Mesh.TensorMesh([nx,nx,nz], x0=['C', 'C', -ztop])
    #M = simpeg.Mesh.TensorMesh([nx,nx,nz], x0=['C', 'C', 'C']) 
    ccMesh=M.gridCC
    print(np.size(ccMesh))
    #M.setCellGradBC('dirichlet')
    #print(inspect.signature(M.setCellGradBC))  # visualize input parameters of functions
    
    # Setup the model
    # Profundidade do topo de cada camada (precisa coincidir com a malha!)
    ztop  = M.vectorNz[-1]
    
#    zh    = np.array([ztop,0.,-200.,-350])    
#    zh    = np.array([ztop,0.,-200.,-300])
#    conds = np.array([1e-8,1.,0.01,1.]) 
    
    
    #sig = simpeg.Utils.ModelBuilder.layeredModel(ccMesh, zh, conds)

     
    conds = [1e-2,1]
    sig = simpeg.Utils.ModelBuilder.defineBlock(
        M.gridCC, [-15000, -15000, -350], [15000, 15000, -150], conds)
        
    sig[M.gridCC[:, 2] > 0] = 1e-8
    #sig[M.gridCC[:, 2] < -1000] = 1 # 1e-1
    
    #sigBG = sig
    
    sigBG = np.zeros(M.nC) + conds[1]
    sigBG[M.gridCC[:, 2] > 0] = 1e-8
    
  
    
   
    
    if plotIt:
        fig,axes = plt.subplots(num=1,clear=True)
        collect_obj, line_obj = M.plotSlice(np.log(sig), grid=True, normal='y',ax=axes)
        plt.colorbar(collect_obj) 
        plt.show()
     
    #%% Setup the layout of the survey, set the sources and the connected receivers
    # Receiver locations
#    rx_x, rx_y = np.meshgrid(np.arange(-600, 601, 100), np.arange(-600, 601, 100))
    rx_x = np.array([0.])
    rx_y = np.array([0.])
    #rx_z = np.array([0.])
    rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x, 2), simpeg.Utils.mkvc(rx_y, 2), np.zeros((np.prod(rx_x.shape), 1))))
    #rx_loc = np.reshape([rx_x,rx_y,rx_z],(1,3))
    
    # Receivers    
    rxList = []
    for rx_orientation in ['xx', 'xy', 'yx', 'yy']:
        rxList.append(NSEM.Rx.Point_impedance3D(rx_loc, rx_orientation, 'real'))
        rxList.append(NSEM.Rx.Point_impedance3D(rx_loc, rx_orientation, 'imag'))
    for rx_orientation in ['zx', 'zy']:
        rxList.append(NSEM.Rx.Point_tipper3D(rx_loc, rx_orientation, 'real'))
        rxList.append(NSEM.Rx.Point_tipper3D(rx_loc, rx_orientation, 'imag'))
        
    
    # Source list
    srcList = []
    for freq in freqs:
            srcList.append(NSEM.Src.Planewave_xy_1Dprimary(rxList, freq))
    # Make the survey
    survey = NSEM.Survey(srcList)
    #survey.mtrue = m_true          
    
    # Set the problem
    problem = NSEM.Problem3D_ePrimSec(M, sigma=sig, sigmaPrimary=sigBG)
    problem.pair(survey)
    problem.Solver = Solver   
    
    
    # Calculate the data
    fields = problem.fields()
    dataVec = survey.eval(fields)
   
    
    
    
#    # Add uncertainty to the data - 10% standard
#    # devation and 0 floor
#    dataVec.standard_deviation.fromvec(
#        np.ones_like(simpeg.mkvc(dataVec)) * 0.0
#    )
#    dataVec.floor.fromvec(
#        np.zeros_like(simpeg.mkvc(dataVec))
#    )

    
    if plotIt:
        #plotting
        #    fig,ax1 = plt.subplots(clear=True)
        out1 = dataVec.plot_app_res(np.array([0.,0.]),components=['xy','yx'],
                                    comp_plot_dict = {'xy':{'marker':'None','ls':'-'},
                                                      'yx':{'marker':'None','ls':'--'}  } )
#        out1.autoscale(tight=True)
#        out1.set_yscale('linear')
#        out1.grid(True)
        #out1.set_ylim([0.,300.])    
        
        
        
        
    mu = 4*math.pi*1E-7; #Magnetic Permeability (H/m)
    resistivities = np.array([1.,100.,1.]) #[100,10,200] #[300, 2500, 0.8, 3000, 2500];
    
    thicknesses = [150.,200] #[300,300] #[200, 400, 40, 500];
    n = len(resistivities);
    
    nFreq = 13
    frequencies = np.logspace(2, -2, nFreq) #[1,5,10,25,50,100] #data[:,0] ##[0.0001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,10000];
    
    rhoapp      = np.zeros(len(frequencies))
    #print('freq\tares\t\t\tphase');
    for i,frequency in enumerate(frequencies):   
        w =  2*math.pi*frequency;       
        impedances = list(range(n));
        #compute basement impedance
        impedances[n-1] = cmath.sqrt(w*mu*resistivities[n-1]*1j);
       
        for j in range(n-2,-1,-1):
            resistivity = resistivities[j];
            thickness = thicknesses[j];
      
            # 3. Compute apparent resistivity from top layer impedance
            #Step 2. Iterate from bottom layer to top(not the basement) 
            # Step 2.1 Calculate the intrinsic impedance of current layer
            dj = cmath.sqrt((w * mu * (1.0/resistivity))*1j);
            wj = dj * resistivity;
            # Step 2.2 Calculate Exponential factor from intrinsic impedance
            ej = cmath.exp(-2*thickness*dj);                     
        
            # Step 2.3 Calculate reflection coeficient using current layer
            #          intrinsic impedance and the below layer impedance
            belowImpedance = impedances[j + 1];
            rj = (wj - belowImpedance)/(wj + belowImpedance);
            re = rj*ej; 
            Zj = wj * ((1 - re)/(1 + re));
            impedances[j] = Zj;    
    
        # Step 3. Compute apparent resistivity from top layer impedance
        Z = impedances[0];
        absZ = abs(Z);
        apparentResistivity = (absZ * absZ)/(mu * w);
        rhoapp[i] = apparentResistivity
        phase = math.atan2(Z.imag, Z.real)
    #    print(frequency, '\t', apparentResistivity, '\t', phase);

    #Plot result
        
    #fig,ax = plt.subplots(num=1,clear=True) 
    ax = out1   
    #ax.plot(frequencies,rhoapp,frequencies,data[:,1],'--')
    ax.plot(frequencies,rhoapp)
    ax.autoscale(tight=True)
    ax.legend(('Analytic','Numeric'))
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('Apparent Resistivity (Rho.m)')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.invert_xaxis()
    ax.grid()         
        
    out1.autoscale(tight=True)
    out1.set_yscale('linear')
    out1.grid(True)    
        
        
        
             
    
    
if __name__ == '__main__':
    run()
    plt.show()
    print('Program Finished!')
    tf = time.time()
    print('tempo: ',round(tf-t,2),'s')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    