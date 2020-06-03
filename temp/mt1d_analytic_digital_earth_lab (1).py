import math
import cmath
#import time
import matplotlib.pyplot as plt
import numpy as np


print('====================================');
print('1D MAGNETOTELLURIC MODELLING PROGRAM');
print('====================================');
print('    LAST UPDATED 17TH DECEMBER 2013 ');
print('    DEVELOPED BY ANDREW PETHICK     ');
print('      WWW.DIGITIALEARTHLAB.COM      ');
print('====================================');
print('');
print('licensed under WTFPL')
print('');
#start = time.clock();


## Open and read file 
#with open("data_isadora.txt") as f:
#    content = f.readlines()
#content = [x.strip() for x in content]
#mat = []
#for line in content:
#    s = line.split()
#    if len(s) == 3:
#        mat.append(s)
#data = np.asarray(mat,dtype=float)


mu = 4*math.pi*1E-7; #Magnetic Permeability (H/m)
resistivities = np.array([1.,100.,1.]) #[100,10,200] #[300, 2500, 0.8, 3000, 2500];

thicknesses = [500,100] #[300,300] #[200, 400, 40, 500];
n = len(resistivities);

nFreq = 26
frequencies = np.logspace(2, -3, nFreq) #[1,5,10,25,50,100] #data[:,0] ##[0.0001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,10000];

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
fig,ax = plt.subplots(num=1,clear=True) 
#ax.plot(frequencies,rhoapp,frequencies,data[:,1],'--')
ax.plot(frequencies,rhoapp)
ax.legend(('Digital Earth Lab','Isadora data'))
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('Apparent Resistivity (Rho.m)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.invert_xaxis()
ax.grid()  


  
    
#print('');
#print('time taken = ', time.clock() - start, 's');