"""
Created on Fri Sep 25 15:17:36 2020

@author: akel
"""
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata as gd
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import functions as df


plt.close('all')

#def dat_topo(filename):


#leitura do arquivo

file = r'../data/Miocene-mr3d.xyz'
zmax=6000
tp = pd.read_table(file, header=None)
tp  = tp.to_numpy()
tpt = tp.copy()	
tpt[:,2] = 0			
tp[:,2]  = zmax - tp[:,2]					
tpt[:,2] = zmax - tpt[:,2]				

x=tp[:,0]
y=tp[:,1]
z=tp[:,2]

dx=x[0]-x[1] # 

LX=max(x)-min(x) # Tamanho em x
LY=max(y)-min(y) # Tamanho em y


dxi=25 # valor discretização
dyi=25 #

rstx=LX%dxi #calculo do resto
rsty=LY%dxi


indmax_x=np.where(x == x.max())
indmin_x=np.where(x == x.min())
indmax_y=np.where(y == y.max())
indmin_y=np.where(y == y.min())



while rstx != 0:
    print('resto x não é zero')
    indmax_x=np.where(x == x.max())
    x[indmax_x]=x[indmax_x]-rstx
    LX=max(x)-min(x)
    rstx=LX%dxi

while rsty != 0:
    print('resto não é zero')
    indmax_y=np.where(y == y.max())
    y[indmax_y]=y[indmax_y]-rsty
    LY=max(y)-min(y)
    rsty=LY%dyi
    
        
nxi=LX/dxi
nyi=LY/dxi
#print(LX,dxi)
#print(LY,dyi)
xi = np.linspace(min(x), max(x), nxi)
yi = np.linspace(min(y), max(y), nyi)
xi, yi = np.meshgrid(xi, yi)
zi = gd((x, y), z,(xi, yi) ,method='cubic')


print('LX   ',LX)
print('dxi  ',dxi)
print('nxi  ',nxi)
print('resto',rstx)

print('LY   ',LY)
print('dyi  ',dyi)
print('nyi  ',nyi)
print('resto',rsty)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xi, yi, zi,cmap=cm.jet,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
