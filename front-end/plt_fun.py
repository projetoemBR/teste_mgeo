#!/usr/bin/python3
#--------  Code Dependencies   ----------
#\__________General Utilities____________/
import os
import re
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
#\__________Local functions__________/
import ipop_fun as ipop             #I/P and O/P functions
import util_fun as util             #Utility functions
#\_____________________________________/
#
# -------------- plot_pyvista  ----------------------------
"""
Plot mesh with PyVista.
https://github.com/OpenGeoVis/GeothermalDesignChallenge/blob/master/docs/project/gravity-inversion/01-gravity-mesh-refine.rst
https://docs.pyvista.org/examples/00-load/create-structured-surface.html?highlight=surface
Plot 
A) Parameters
    1. mesh  = a given mesh
    2. data  = data dictionary
    3. model = Quantity to plot

B) Plotting options
B.1) 'domain' -> Plots user's model. Does not include air
  Parameters:
    1. data     = model dictionary to plot
    2. args[-1] = 'domain'

B.2) 'grid' -> Plots the grid
    1. mesh     = a given mesh
    2. data  = not used
    3. model = cell volumes to plot
    4. args[-1] = 'grid'

B.3) 'surf' -> Plots a 2D structured grid to 3D
    1. mesh     = Topography
    2. data     = Top of half-space
    3. model    = receivers and source 
    4. args[-1] = 'surf'

"""
def plot_pyvista(mesh, data, model, *args):
#
#------------  Plot domain. Does not include air or topography.
    if args[-1] == 'domain':
        p = pv.Plotter()
#-- Source and receivers radius based on hspace or anomaly size
        if 'box' in data.keys():
            rd = (data['box'].flatten()[1]-data['box'].flatten()[0]) * 0.03
        else:
            rd = (data['hspace'].flatten()[1]-data['hspace'].flatten()[0]) * 0.006
#-- Source
        if 'src' in data:
            dummy = tuple(data['src'])
            box = pv.Sphere(radius=rd, center=dummy)
            _ = p.add_mesh(box, color="red")
            #union['src'] = p.add_mesh(box, color="red")
#-- Receivers
        if data['rec'].ndim == 1:
            dummy = tuple(data['rec'])
            box = pv.Sphere(radius=rd, center=dummy)
            _ = p.add_mesh(box, color="black")
            #union['rec'] = p.add_mesh(box, color="black")
        else:
            for dummy in data['rec']:
                index = 0
                dummy = tuple(dummy)
                box = pv.Sphere(radius=rd, center=dummy)
                _ = p.add_mesh(box, color="black")
                #union['rec'+str(index)] = p.add_mesh(box, color="black")
                index = index + 1
#-- Sea
        if 'sea' in data:
            box = pv.Box(tuple(data['sea']))
            res = np.log10(data['sea_r'])
            r = np.ones(len(box.points)) * res
            _ = p.add_mesh(box, color="white", opacity=0.2, show_edges=True,)
#            _ = p.add_mesh(box, scalars=r, cmap='jet', opacity=0.3, show_edges=True,)
#-- Deals with Boxes
        if 'box' in data.keys():
            res  = np.log10(data['box_r'])
            index = 0
            if data['box'].ndim == 1:
                box = pv.Box(tuple(data['box']))
                r = res if len(res) ==1 else res[0]
                r =  np.ones(len(box.points)) * r
                _ = p.add_mesh(box, opacity=0.7, show_edges=True,
                    show_scalar_bar=False)      #scalars=r, cmap='jet', 
            else:
               for dummy in data['box']:
                box = pv.Box(tuple(dummy))
                r = res[index] if len(res[index]) ==1 else res[index][0]
                _ = p.add_mesh(box, opacity=0.7, show_edges=True,
                    show_scalar_bar=False)      #scalars=r, cmap='jet', 
                index = index + 1
#-- Half space
        box = pv.Box(tuple(data['hspace'])) 
        res  = np.log10(data['hspace_r'])
        #r =  np.ones(len(box.points)) * res
        _ = p.add_mesh(box, opacity=0.5, show_edges=True,
            show_scalar_bar=False)            #scalars=r, cmap='jet', scalar_bar_args={'title': 'log(rho)'}
#        _ = p.add_scalar_bar('log(rho)', interactive=True, vertical=True,
#                                   width=.3, height=.5,
#                                   title_font_size=14,
#                                   label_font_size=14,
#                                   outline=False, fmt='%3.2f')
#        p.add_scalar_bar('log(rho)', vertical=True)
        p.show_grid()
        return p.show()   
#
#------------  Plots grid
    elif args[-1] == 'grid':
#-- Convert TreeMesh to VTK
        dataset = mesh.toVTK()
        dataset.cell_arrays['log2(vol)'] = model            #Magnitude
#        dataset.cell_arrays['Active'] = args[1]            # Topo active cells
        dataset.active_scalars_name = 'log2(vol)'           #'Magnitude'
#-- Instantiate plotting window
        p = pv.Plotter()
        #p = pv.BackgroundPlotter()
#-- Show axes labels
        p.show_grid(all_edges=False,)
#-- Add a bounding box of original mesh to see total extent
        p.add_mesh(dataset.outline(), color='k')
#-- Plotting params
        d_params = dict(
                show_edges=False,
                cmap='jet',                     #'jet', 'hot', 'rainbow'
                scalars='log2(vol)',            #'Magnitude'
                scalar_bar_args=dict(label_font_size=20, title_font_size=25,
                                     vertical=True),)
#-- Clip volume in half
#        p.add_mesh(threshed.clip('-y'), **d_params)
#-- Add slices for opacity
#        slices = threshed.slice_along_axis(n=5, axis='x')
#        p.add_mesh(slices, name='slices', **d_params)
#-- The mesh
        p.add_mesh(dataset.extract_all_edges(), **d_params)             #, color='r'
        return p.show(window_size=[1024, 768])    
#
#------------  Plots 2D structured grid to 3D
    elif args[-1] == 'surf':
#-- Plot topography in km.
        mesh  = mesh  / np.array([1000.,1000., 1.])    # - np.array([0.,0., data])
        model = model / np.array([1000.,1000., 1.])    # - np.array([0.,0., data])
        p = pv.Plotter()
        grid = pv.PolyData(mesh)
        dargs = dict(scalars=mesh[:,-1], cmap='rainbow',show_edges=True)       #scalars='Elevation', 
        p.add_mesh(grid, interpolate_before_map=True,
            scalar_bar_args={'title': 'Height'},                         #'title': 'Height'
            **dargs)
        p.add_mesh(model[:-1,:], color="b", point_size=20.0, render_points_as_spheres=True,)
        p.add_mesh(model[-1,:],  color="r", point_size=20.0, render_points_as_spheres=True,)
        p.enable_eye_dome_lighting()
        p.show()
        return
        #
# -------------- pltv  ----------------------------
"""
Plot some mesh views depending on plot parameter args[-1].
A) plot='slice' -> Plots two slices at middle of grid
    1. mesh     = a given mesh
    2. args[0]  = a quantity to plot.
    3. args[-2] = (y-axis_min, y-axis_max)
    4. args[-1] = 'slice'

B) plot='grid' -> Plots the grid
    1. mesh     = a given mesh
    2. args[-2] = active cells to plot
    3. args[-1] = 'grid'

C) plot='surf'
    1. mesh     = not used
    2. args[-1] = 'surf'
    3. args[0] = structured x
    4. args[1] = structured y
    5. args[2] = structured z

"""
def pltv(mesh, *args):
# -------------- Check on last args' list argument 
    if not isinstance(args[-1], str):
        raise ipop.CmdInputError('@plt=> Illegal args[-1]', args[-1])
# -------------- Task pick
    if args[-1] == 'slice':
        range_y= args[-2]
        qty    = args[0]
        clim   = (np.floor(np.min(qty)), np.ceil(np.max(qty)))
#
        fig = plt.figure(figsize=(14, 6))
#        ax1 = fig.add_axes([0.13, 0.1, 0.6, 0.85])
        ax = fig.add_subplot(121)
        ind_slice = int(mesh.hy.size / 2)
        mesh.plotSlice(qty, normal='Y', ax=ax,
                        range_y=range_y, ind=int(mesh.hy.size / 2),
                        grid=True, clim=clim,
                        )
        ax = fig.add_subplot(122)
        mesh.plotSlice(args[0], normal='X',ax=ax, 
                        range_y=range_y, ind=int(mesh.hx.size / 2),
                        grid=True, clim=clim,
                        )
#                         [left, bottom, width, height] 
        ax = fig.add_axes([0.91, 0.1, 0.02, 0.75])      #[0.75, 0.1, 0.05, 0.85]
        norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
        dummy = "${%.1f}$"              #if linlog == 'lin' else "$10^{%.1f}$"
        cbar = mpl.colorbar.ColorbarBase(ax, norm=norm, orientation="vertical",
                                        format=dummy)
        cbar.set_label("log(Resistivity)", rotation=270, labelpad=15, size=12)
        return plt.show()
#
# -------------- Plot grid
    elif args[-1] == 'grid':
        ax = plt.subplot(111)
        mesh.plotImage(args[-2], ax=ax)
        mesh.plotGrid(centers=True, ax=ax)
        ax.set_title("CC")
#
        #mesh.plotImage(mesh, ax=ax)
        #mesh.plotGrid(centers=True, ax=ax)
        return plt.show()
#
# -------------- Plot topography
    elif args[-1] == 'surf':
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('x(km)')
        ax.set_ylabel('y(km)')
        ax.set_zlabel('z(km)')
        x = args[0]/1000.; y = args[1]/1000.; z = args[2]/1000.
        surf = ax.plot_surface(x, y, z, 
                rstride=1, cstride=1, cmap='jet', linewidth=1, antialiased=True)
        ax.scatter(mesh[:,0]/1000.,mesh[:,1]/1000.,mesh[:,2]/1000.,color="k",s=10)
        a = np.round(np.amin(z.flatten(), axis=0),1)
        b = np.round(np.amax(z.flatten(), axis=0),1)
        if a==b: a = a * .9; b = b * 1.1; 
        ax.set_zlim(a, b)
        cbar = fig.colorbar(surf, aspect=20)
        cbar.set_label('km')
        plt.show()