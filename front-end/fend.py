#!/usr/bin/python3
#--------  Code Dependencies   ----------
#\__________General Utilities__________/
import pprint
import time
import sys
import os
import re
import numpy as np
#\__________Local functions__________/
import ipop_fun as ipop				#I/P and O/P functions
import util_fun as util             #Utility functions
import plt_fun as p                 #Plot functions
#\__________Specialized stuff__________/
import discretize
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from discretize.utils import active_from_xyz
from SimPEG import utils, maps
from SimPEG.utils import mkvc, model_builder, surface2ind_topo
try:
    from pymatsolver import Pardiso as Solver
except:
    from SimPEG import Solver
from scipy.constants import mu_0, epsilon_0 as eps_0
from scipy.interpolate import griddata
from scipy.stats.mstats import gmean
#\_____________________________________/
#
# -------------- Main Program --------------
def main():
    print( )
    text = "Process Information"
    print(text.center(60, "-"))
    dummy = util.get_host_info()
    pprint.pprint(dummy)
    util.pline('** Total RAM: '+util.human_size(dummy['ram']))
    dummy.clear()
#-- RAM monitoring and estimate of O/P file.
    util.ramon()
#------------  Read I/P file
    args = ipop.read_param()
    data = ipop.iofile(args)
# -------------- Construct model --------------
#-- Suply needed parameters
    data = ipop.sup_param(data)
#-- Assemble for numpy
    data = ipop.flt_param(data)
    print()
    text = 'I/P dictionary:'
    print(text.center(60, "-"))
    pprint.pprint(data)
    print()
# -------------- Extrude domain --------------
    data = util.coordtrf(data)
#
#------------  Plot model with PyVista
    _ = p.plot_pyvista(None, data, None, 'domain')
#
# -------------- Base Mesh --------------
    print( )
    text = "Base mesh constructor"
    print(text.center(60, "-"))
#
# -------------- Build the core with mesh_builder_xyz
    """
                    +~~+--------------+~~+ <- upward padding
                    :~~|              |~~:
           z        +~~+--------------+~~+
           | /y     :~~:              :~~:
           |/       :~~|     core     |~~:
           +---x    +~~+--------------+~~+
                    :~~|              :~~:
                    +~~+~~~~~~~~~~~~~~+~~+ <- max depth_core
    Lateral padding--^-----------------^
    - Padding should be 2-3 times the LARGE diffusion distance -> data['dxyz2']
    """
# -------------- Build  base mesh
    mesh = util.mBuilder(data)
#-- Boolean to store meshgrided or xyz-unique surface descriptors for finding
#    the cells which lie below a set of xyz points defining a surface:
#    - meshgrid is required by surface2ind_topo;
#    - xyz-unique is required by SimPEG.utils' discretize.utils.active_from_xyz
    uniq = False        # Stores meshgrided descriptors
#
#-- RAM monitoring
    util.ramon()
# -------------- Refine Mesh --------------
    print( )
    text = "Build mesh"
    print(text.center(60, "-"))
#
#------------ Refine Tree mesh around topography
    util.pline('\n 2) Refine flat or structured topography.')
#-- Works with both flat and tructured topography.
#    Structured topography:
#     1) data['topo'] now holds the xyz topography data.
#     2) Inherits half-space resistivity, data['topo_r'] = data['hspace_r'].
#     3) Glues receivers and transmitters to topography
#     4) Size of topgraphy depends of user's model! Grid does not enlarge it!
#
#-- Structured topography
    if data['topo']:
        data['topo'] = args.folder + data['topo']
#-- Get griddata xyz and glues receivers and sources to topography
        xyz_u, xyz = util.otaid(data, 'topo')
#-- Plot surface, source and receivers.
        dummy = np.vstack((data['rec'],data['src'])) if 'src' in data else data['rec']       
        util.otaid(xyz_u,   dummy, 'strc_topo')
    else:
#-- Flat topography
        xyz_u, xyz = util.flat(np.append([data['hspace'][:-2]] , [data['hspace'][-1]] * 2),
                                num=3)
#-- Plot it
        dummy = np.vstack((data['rec'],data['src'])) if 'src' in data else data['rec']
        util.otaid(data['hspace'],   dummy, 'flat_topo')
#-- store meshgrided or xyz-unique topography surface descriptor
    data['topo'] = xyz_u if uniq else xyz
#-- Refine topography now.
    mesh = refine_tree_xyz(mesh, xyz,
            octree_levels=data['mad'][-1],
            method="surface", finalize=False,)
#-- RAM monitoring
    del xyz_u, xyz
    util.ramon()
#
#------------ Refine around sea.
    if 'sea' in data:
        util.pline('\n 3) Refining sea surface.')
#-- A flat, calm, sea.
        xyz_u, xyz = util.flat(np.append([data['sea'][:-2]] , [data['sea'][-1]] * 2),
                                num=3)
#-- store meshgrided or xyz-unique sea surface descriptor
        data['sea'] = xyz_u if uniq else xyz
#-- Refine sea surface now.
        mesh = refine_tree_xyz(mesh, xyz,
                octree_levels=data['mad'][-1],
                method="surface", finalize=False,)
#-- RAM monitoring
        del xyz_u, xyz
        util.ramon()
#
#------------ Refine Tree mesh around receivers
    util.pline('\n 4) Refining around '+
        str(len(np.shape(data['rec'])) if data['rec'].ndim == 1 else np.shape(data['rec'])[0])+
        ' receivers.')
#
    xyz = np.vstack([data['rec'],data['rec']]) if data['rec'].ndim == 1 else data['rec']
#-- A point cloud around each receiver
    mesh = refine_tree_xyz(mesh, xyz,
            octree_levels=data['mad'][-1],
            method="radial", finalize=False)
    del xyz
#-- RAM monitoring
    util.ramon()
#
#------------ Refine Tree mesh around source.
    if 'src' in data:
        util.pline('\n 5) Refining around source.')
        xyz = np.vstack([data['src'],data['src']])
        mesh = refine_tree_xyz(mesh, xyz,
                octree_levels=data['mad'][-1],
                method="radial", finalize=False)
    del xyz
#-- RAM monitoring
    util.ramon()
#
#------------ Refine Tree mesh around boxes
    if 'box' in data.keys():
        util.pline('\n 6) Refining around '+str(data['box'].ndim)+' anomalies.')
#
        if data['box'].ndim == 1:
            util.pline(['box', *data['box']], width=15, align='>')
#-- Defining box points xyz = [n x dim]
            xyz = util.otaid(data['box'], 'box')
            mesh = refine_tree_xyz(mesh, xyz,
                    octree_levels=data['mad'][-1],
                    method="box", finalize=False)
        else:
            for dummy in data['box']:
                util.pline(['box', *dummy], width=15, align='>')
#-- Defining box points xyz = [n x dim]
                xyz = util.otaid(dummy, 'box')
                mesh = refine_tree_xyz(mesh, xyz,
                        octree_levels=data['mad'][-1],
                        method="box", finalize=False)
        del xyz
#-- RAM monitoring
        util.ramon()
#
# -------------- Finalize mesh
    mesh.finalize()
# -------------- Print mesh info
#-- hx, hy, hz           -> tuples with finest discretization and # of cells of base mesh
#-- mesh.x0              -> bottom west corner
#-- mesh.vectorNx        -> Nodal grid vector (1D), x direction.
#-- mesh.gridCC          -> an (nC, 3) array containing the cell-center locations
#-- mesh.cellBoundaryInd -> a boolean array specifying which cells lie on the boundary
#-- mesh.vol             -> Cell volumes
#-- mesh.max_level       -> ditto
    util.pline('\n 7) Mesh information:')
    util.pline(['Bottom SW x0:', *mesh.x0], width=18, align='^')
    util.pline(['Mesh size:', np.shape(mesh)[0]], width=18, align='^')
    util.pline(['Cell-centers:', *np.shape(mesh.gridCC)], width=18, align='^')
# A boolean array specifying which cells lie on the boundary
    #bInd = mesh.cellBoundaryInd
    #pprint.pprint(mesh.cellBoundaryInd)
    util.pline(['Mesh grid size ', *np.shape(mesh.h_gridded)], width=18, align='^')
    util.pline(['Mesh max_level:', mesh.maxLevel], width=18, align='^')
    util.pline(['Nodal grid:',*[np.shape(mesh.vectorNx)[0],
                                np.shape(mesh.vectorNy)[0],
                                np.shape(mesh.vectorNz)[0]]], width=18, align='^')
    pprint.pprint(mesh)
#-- RAM monitoring
    util.ramon()
#
#-------------- Add model to mesh
    util.pline('\n 8) Anchor model onto Mesh.')
#------------ Model building
    """
    Active cells below surfaces A and B
              ...F.....F... )----> F
              ...F.....F... )----> F
    z      A->+--|-----F--+ )----> .
    | /y      |  T     F  | )----> T
    |/        |  T     F  | )----> T
    +---x  B->+--T-----|--+ )----> .
                 T     T    )----> F
                 T     T    )----> F
#------------ Active cells below the topography. Set model0
        +---------------+                        
        |   air or sea  | <- data['topo'] = False
        |     +-----+   |    
        +-----+     +---+ <- topography          
        |  half space   | <- data['topo'] = True 
        +---------------+                       
    """
#-- Active cells below sea surface. data['topo'] -> a boolean with True below topography
    data['topo']  = active_from_xyz(mesh, data['topo'], grid_reference='N',method='nearest')
#    data['topo']  = surface2ind_topo(mesh, data['topo'],method='cubic')    <- not to be used!
#-- Alocate and include topography in model. model has length of data['topo'].sum()
#    1) All cells below topography inherit hspace resistivity
#    2) All cells above topography inherit either air or sea resistivity
    model  = np.zeros(np.shape(data['topo']))
    model[data['topo']] = data['hspace_r']
    topres = data['sea_r'] if 'sea' in data else data['air_r']
    model[np.logical_not(data['topo'])] = topres 
    util.pline('\n '+str(np.sum(data['topo']))+' topography cells added to model')
#    util.pline(' '+str(np.shape(model_map))+' elements in model map')
#
#------------ Sea active cells; below the air layer, above topography
    """
        +---------------+
        |       Air     |
        +---------------+
        |       Sea     | <- data['sea']  = True
        |     +-----+   |    
        +-----+     +---+ <- topography
        |   half space  | <- data['topo'] = True
        +---------------+
    """
    if 'sea' in data:
#-- Active cells below sea surface and above topography. data['sea'] holds a
#    meshgrided sea surface descriptor. NB: Model assumes sea above topography.
#        data['sea']  = surface2ind_topo(mesh, data['sea'])
#        data['sea']  = data['sea'] & np.logical_not(data['topo'])
#-- Add air layer onto sea top, i.e., on inactive data['topo']. It meshgrids
#    air bottom surface.
        xyz_u, xyz = util.flat(np.append([data['air'][:-2]] , [data['air'][-2]] * 2),
                                num=3)
#-- Set an active map to grid cells below air bottom surface.
        xyz = xyz_u if uniq else xyz
        xyz  = active_from_xyz(mesh, xyz, grid_reference='N',method='nearest')
#        xyz  = surface2ind_topo(mesh, xyz)
        model[np.logical_not(xyz)] = data['air_r']
        del xyz_u, xyz
#------------ Deals with anomalies
    if 'box' in data.keys():
#-- If any box is anisotropic assigns just the x value to mesh. New data['box'] holds mesh cells 
#    defining a box: a boolean with True within box, restricted to the acive indexes below topography.
#
        if data['box'].ndim == 1:
#-- One block only.                 
#                                                      \cell-centered mesh/
            data['box']        = util.IndBlock(data['box'],mesh.gridCC)
            model[data['box']] = data['box_r'][0]
            util.pline(' '+str(np.sum(data['box']))+' box cells added to model')
#-- Several blocks         
        else:
            for ind, dummy in enumerate(data['box']):
                ind_blk = util.IndBlock(dummy,mesh.gridCC)
                box = ind_blk if ind == 0 else np.vstack((box, ind_blk))
                model[ind_blk] = data['box_r'][ind,0]
                util.pline(' '+str(np.sum(box))+' box cells added to model')
                ddummy  = np.round(np.sum(dummy) / np.shape(mesh)[0] * 100., 1)
            data['box'] = np.copy(box)
            del box, ind_blk
#-- Plot slices
#    exponential_map = maps.ExpMap()
#    reciprocal_map = maps.ReciprocalMap()
#    model_map = model_map * model_a_map
    ent = input(f'\n Plot air in slice?(rtn = yes)\n') or 'yes'
    if ent=='yes':
        _ = p.pltv(mesh, np.log10(model),
                (data['hspace'][-2], data['air'][-1]), 'slice')
    else:
        _ = p.pltv(mesh, np.log10(model),
                (data['hspace'][-2], data['air'][-2]), 'slice')
#
#-------------- 3-D mesh plot
    util.pline('\n 9) Plot a 3-D grid of mesh volumes? That is prone to Segmentation Fault.')
    dummy = input(f' Plot? (rtn=no): \n') or False
    if dummy != False: p.plot_pyvista(mesh,  None, np.log2(mesh.vol), 'grid')
#-------------- Tide up and prepare to dump
    _=data.pop('air'); _=data.pop('air_r'); _=data.pop('dxyz0'); _=data.pop('dxyz2'); _=data.pop('hspace')
    _=data.pop('hspace_r'); _=data.pop('mad'); _=data.pop('sea'); _=data.pop('sea_r')
#
    data['model'] = model
    data['mesh'] = mesh
    util.pline('\n The following data will be dumped onto a file:')
    print(list(data.keys()))
    #-- RAM monitoring and estimate of O/P file.
    util.ramon()
#    dummy = util.human_size(util.ramon(dict=data))
#    util.pline('\n    Size of data to be dumped onto O/P file: '+str(dummy))
#------------  Write relevant data to an O/P file with pickle
#             -> Pickle files can be hacked!
    print()
    text = 'Dump to an O/P file'
    print(text.center(70, "-"))
#-- Construct a default
    dummy = args.folder + args.ipfile
    dummy = dummy[:len(dummy)-dummy[::-1].find(".")]
    dummy = dummy + 'out'
#-- Enter file name
    util.pline('\n Options for O/P file:')
    util.pline('  1) Need to provide folder and filename.')
    util.pline('  2) Defaults to the same folder and I/P filename.')
    util.pline('  3) Any single character aborts O/P.')
    file = input(f'\n Name of O/P file (rtn = '+dummy+'):\n') or dummy
#
    if len(file) > 1:
        import pickle
#-- File atribute
        file = open(file, 'wb')
#-- dump data to file
        pickle.dump(data, file)
#-- Closes file
        file.close()
        text = 'Data dumped to binary file '
        print(text.center(70, "-"))
#
#------------  Program terminates
    text = 'Program terminates '
    print(text.center(70, "-"))
#
# -------------- Controls Execution --------------
if __name__ == "__main__":
    main()
#    plt.show()
#
raise SystemExit
#