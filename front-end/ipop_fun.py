#!/usr/bin/python3
#
#--------  Code Dependencies   ----------
#\__________General Utilities____________/
import pprint
import time
import argparse
import sys
import os
import re
import numpy as np
from scipy.stats.mstats import gmean
from collections import defaultdict
#\__________Local functions__________/
import util_fun as util				#I/P and O/P functions
#\_____________________________________/
#
# -------------- Read/Write Files   ----------------------------
def read_param():
#------------  Create the parser
    parser = argparse.ArgumentParser(prog='fend', 
        description='Frontend to GeoFem', usage='%(prog)s [options] ipfile',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ipfile', metavar='ipfile', type=str,
        help='Path and name of inputfile')
    parser.add_argument('--log_op', action='store_true',
        default='>&1 | tee op_fend.txt', 
        help='File to hold screen output')
    args = parser.parse_args()
    args.folder = '/'.join(args.ipfile.split('/')[0:-1]) + '/'
    args.ipfile = args.ipfile.split('/')[-1]
    return args
#
# -------------- End of function   ----------------------------
#
# -------------- Read/Write Files   ----------------------------
def iofile(args=None, mode='input'):
    """ Open files for I/P and O/P
    Action is controlled by parameter "mode":
     1. mode = 'input' 
        Open I/P parameter file. Return data.
     2. mode = 'topo'
        Open topgraphy file. Return xyz.
     3. mode = 'close'
        Close file. Returns "None"
    """ 
#
#------------  Input File
    if mode == 'input':
        data = defaultdict(list)
        text = 'Opens File for I/P'
        print(text.center(70, "-"))
        print('I/P  => {}'.format(args.folder + args.ipfile))
# --                file object-->|
        with open(args.folder + args.ipfile, 'r') as fobj:
            lines =  fobj.readlines()
        for line in lines:
            dummy = line.rstrip()               #get rid of eol
            try:
                if not (dummy[0] == '#'  or 
                        dummy[0] == '\t' or 
                        dummy[0] == ' '  ):
                    key, value = dummy.split(':')
                    data[key].append([value])
            except:
              print("Empty line read; ignored.")
              continue
        dummy = {}
        for lines in data.keys():
            line = np.asarray(data[lines]).shape[0]
            if line == 1:
            #dummy[lines] = np.asarray(data[lines][0][0].split(','),dtype=np.float32)
                dummy[lines] = data[lines][0][0].split(',')
            else:
                for k in range(line):
                    dummy[lines + str(k + 1)] = data[lines][k][0].split(',')
        data.clear()
        data.update(dummy)
        dummy.clear()
#
        util.pline('Closes I/P File.')
        fobj.close()
        return data
#
#------------  Topography File
    elif mode == 'topo':
        xyz = []
        util.pline(['- Reading topography file: ', args], width=20, align='^')
        with open(args, 'r') as fobj:
            lines =  fobj.readlines()
        for line in lines:
            line = line.rstrip()
            line = line.split('\t')
            xyz.extend([float(dummy) for dummy in line][:])
#
        xyz = np.asarray(xyz, dtype = float)
        xyz = np.reshape(xyz, (-1,3))
        util.pline('- Closes topo file.')
        fobj.close()
        return xyz
#
#------------  Closes File
    elif mode == 'close':
        text = 'Close IP/OP File'
        print(text.center(60, "-"))
        fobj.close()
#
    else:
        raise NameError("Wrong entry in iofile.\n")
#
# -------------- End of function   ----------------------------
#
# -------------- Read in parameters for user's model   ----------
def sup_param(data):
#------------  Check minimum data was supplied
#-- Mandatory parameters
    man_par = ['hspace','freq','rec']            #, 'box'
#-- Optional parameters
#                0      1      2       3      4
    opt_par = ['air', 'sea', 'src', 'dipo', 'box']
#-- Get keys
    ipkeys = set(data.keys())   
#-- Check all mandatory parameters are suplied (check 3 1st characters only)
    if(not all(dummy in [dummy[0:3] for dummy in ipkeys]
           for dummy in [dummy[0:3] for dummy in man_par])):
        raise CmdInputError('foo.in needs parameters: ',man_par)
#-- Deals with half-space. Check with data['topo'] is None
    data['hspace_r'] = [data['hspace'][3]]
    data['topo'] = None if len(data['hspace']) < 5 else data['hspace'][-1]
    data['hspace']   = data['hspace'][:3]
#-- Supply defaults whether needed
    mask = np.isin(opt_par,list(ipkeys))
    if not mask[0]: data[opt_par[0]] = [str(pow(10, 4) * 5)]            #air thickness (m)
    data[opt_par[0]+'_r'] = [str(pow(10, 8))]                           #air resistivity (ohm.m)
    #
    if mask[1]:
        data[opt_par[1]+'_r'] = [str(0.3)]                              #sea resistivity ohm.m
        if mask[2]:
            data['survey'] = 'mCSEM'                                    #sea+src => mCSEM
        else:
            data['survey'] = 'mMT'                                      #sea+~src => mMT
    else:
        data['survey'] = 'landMT'                                       #~sea+~src => MT
#-- O/P     
    return data
#
# -------------- End of function   --------------------------
#
# -------------- Supply parameters for user's model   ----------
def flt_param(data):
#-- Deals with frequency. Sort in increasing order
    lst = [dummy for dummy in list(data.keys()) if dummy[0:4] == 'freq']
    num = len(lst)                                          # Number of entries
    ent = len(data.get(lst[0]))                             # Number of elemnts per entry
#-- 1 frequency line
    if  num == 1:
        dummy = np.shape(data['freq'])[0]
#-- A sequence of frequencies
        if dummy == 3:
            data['freq'] = np.asarray(data['freq'], dtype = float)
            if data['freq'][-1] < 0:
                data['freq'] = np.linspace(data['freq'][0], data['freq'][1], 
                        num=int(np.absolute(data['freq'][-1])))
            else:
                data['freq'] = np.logspace(data['freq'][0], data['freq'][1], 
                        num=int((data['freq'][1] - data['freq'][0]) * data['freq'][-1] + 1.))
#-- Just one frequency
        elif dummy == 1:
            data['freq'] = np.asarray(data['freq'], dtype = float)
        else:
            raise CmdInputError('@flt_param=> Supy 1 or 3 frequency parameters: ',data['freq'])
#-- Several frequency lines
    elif num > 1:
        data['freq'] = []
        for temp, dummy in  np.ndenumerate(lst):
            data['freq'].append(data.pop(dummy)[0])
        data['freq'] = np.asarray(data['freq'], dtype = float)
        data['freq'] = np.sort(data['freq'], kind='mergesort')
#
#-- Deals with several parameters
    man_par = ['hspace', 'hspace_r', 'air', 'air_r', 'sea', 'sea_r']
    dummy = [0, 1, 2, 3, 4, 5] if 'sea' in data else [0, 1, 2, 3]
    man_par = [man_par[dummy] for dummy in dummy]
    for dummy in man_par:
        data[dummy] = np.asarray(data[dummy], dtype = float)
#-- Deals with receivers
    lst = [dummy for dummy in list(data.keys()) if dummy[0:3] == 'rec']
    num = len(lst)                                          # Number of entries
    ent = len(data.get(lst[0]))                             # Number of elemnts per entry
    print(f"Total of {num} receiver entries with {ent} parameters each.")
#-- 1 receiver
    if   (num == 1 and ent == 2):
        data['rec'] = np.asarray(data['rec'], dtype = float)
        data['rec'] = np.hstack((data['rec'],data['hspace'][-1]))
#-- Receiver file
    elif (num == 1 and ent == 1):                           # 2-column receiver file
        raise CmdInputError('@flt_param: not implemented: ',data['rec'])
#-- Receiver line:          Rx0,Ry0,Rx1,Ry1,N
#                           [0,  1,  2,  3, 4]
    elif (num == 1 and ent == 5):
        dummy = np.linspace(np.asarray(data['rec'][0], dtype = float),
                            np.asarray(data['rec'][2], dtype = float),
                            np.asarray(data['rec'][-1], dtype = int))
        temp  = util.line(dummy,
                            np.asarray(data['rec'][0:2], dtype = float),
                            np.asarray(data['rec'][2:4], dtype = float))
        data['rec'] = np.vstack((dummy, temp, [data['hspace'][-1]] * len(dummy))).T
#-- Several receivers
    elif (num > 1 and ent == 2):
        for temp, dummy in  np.ndenumerate(lst):
            ddummy = np.hstack((np.asarray(data.pop(dummy), dtype = float),data['hspace'][-1]))
            data['rec'] = np.vstack((data['rec'], ddummy)) if temp[0] != 0 else ddummy
    data['rec'] = np.asarray(data['rec'], dtype = float)
#
#-- Deals with Transmitters
    if 'src' in data:
        lst = [dummy for dummy in list(data.keys()) if dummy[0:3] == 'src']
        num = len(lst)                                          # Number of entries (src lines)
        ent = len(data.get(lst[0]))                             # Number of elements per entry
        print(f"Total of {num} transmitter(s) with {ent} parameters.")   
    #-- 1 transmitter
        if num == 1:
            if ent == 2:
                data['src_d'] = 'x'
                data['src'].append('0')
            elif ent == 3:
                if (data['src'][-1] == 'x' or data['src'][-1] == 'y'):
                    data['src_d'] = data['src'][-1]
                    data.update({'src':data['src'][:-1]})
                    data['src'].append('0')
                else:
                    data['src_d'] = 'x'
            elif ent == 4:
                data['src_d'] = data['src'][-1]
                data.update({'src':data['src'][:-1]})
        else:
            raise CmdInputError('@flt_param: >1 transmitter ',num)
        data['src']     = np.asarray(data['src'], dtype = float)
        data['src'][-1] = data['src'][-1] + data['hspace'][-1]
    #
# several transmitters
#   elif (num > 1 and ent == 2):
#       for temp, dummy in  np.ndenumerate(lst):
#           #data['src'] = [data.pop(dummy) if temp[0] == 0 else
#           #               np.vstack((data['src'],data.pop(dummy)))]
#           if temp[0] == 0:
#               data['src'] = data.pop(dummy)
#           else:
#               data['src'] = np.vstack((data['src'],data.pop(dummy)))
#   data['src'] = np.asarray(data['rec'], dtype = float)
        if 'dipo' in data:
            raise CmdInputError('@flt_param: entry dipo not yet implemented.\n')
#-- Deals with anomalies
    lst = [dummy for dummy in list(data.keys()) if dummy[0:3] == 'box']
    num = len(lst)                                          # Number of entries
    if num !=0:
        print(f"Total of {num} boxes read.")
        for temp, dummy in np.ndenumerate(lst):
            blen = len(data[dummy])
            if np.logical_and(blen != 7, blen != 9):
                raise CmdInputError('@flt_param=> Box needs 7 or 9 parameters.\n')
            if temp[0] == 0:           
                data['box_r'] = data[dummy][-1] if blen == 7 else data[dummy][-3:]
                data['box']   = data.pop(dummy)[:6]
            else:
                ddummy        = data[dummy][-1] if blen == 7 else data[dummy][-3:]
                data['box_r'] = np.vstack((data['box_r'],ddummy))
                data['box']   = np.vstack((data['box'],  data.pop(dummy)[:6]))
#

        data['box_r'] = np.asarray(data['box_r'], dtype = float)
        data['box']   = np.asarray(data['box'],   dtype = float)
#
#------------  Estimates a skin depth
    inpt(data, mode='sdepth')
#-- O/P     
    return data
#
# -------------- End of function   --------------------------
#
# -------------- General utility for I/P   ----------------------------
def inpt(data, mode='None'):
    """ Asks for I/P
    Action is controlled by parameter "mode":
     1. mode = 'sdepth' -> Estimates a skin depth
     2. mode = 'intP'   -> Asks for a given point P = (x y z) coordinates => Deprecated
    """ 
#
# -------------- Deals with skin depth. gmean()=geometric average; sd()=skin depth;
#                   (f,res)=freq, resistivity.
    sd = lambda f,r: 503. * np.sqrt(r / f)                  # skin depth
#
#------------  Estimates a skin depth
    if mode == 'sdepth':
#-- stack all resistivities but air
        res  = np.hstack((np.ravel(data['box_r']),data['hspace_r'])) if 'box' in data.keys() else data['hspace_r']
        res  = np.hstack((res,data['sea_r'])) if 'sea' in data.keys() else res
        res  = np.sort(res, kind='mergesort') if np.shape(res)[0] !=1 else res
        res  = np.unique(res)                                   # Assure unique elements
        res  = [res[0],gmean(res),res[-1]] if np.shape(res)[0] !=1 else [res]*3
#-- Flip to max to min
        freq = np.flip(data['freq']) if data['freq'].size !=1 else data['freq']        
        freq = np.array([freq[0], np.median(freq), freq[-1]]) if np.size(freq) !=1 else np.repeat(freq, 3)
#-- Metrics
        text = 'Metrics - Skin Depth'
        print(text.center(60, "-"))
        util.pline('1)   The skin depth (sd) is the metrics used for the discretization of the')
        util.pline('      computational domain by OcTree, implying that sd(f,rho) = 2**N. sd is used')
        util.pline('      as a discretization unit. The choice of sd dictates the results accuracy,')
        util.pline('      especially critical at source and receiver location in the total field formulation')
        util.pline('      used for mCSEM, as well as the overall mesh size.')
#
        util.pline('\n2)   As from the minimum, geometric average and maximum values fo resistivities,')
        util.pline(['min(res)', 'gmean(res)', 'max(res)'], width=14, align='^')
        util.pline([' ', *np.array(res)], width=12, align='^')
        util.pline('  the following table gives some possible choices for sd(f,rho), rounded to 2**N:')
        util.pline(['Freq', 'min(res)', 'gmean(res)', 'max(res)'], width=12, align='>')
        for temp, dummy in np.ndenumerate(freq):
            ddummy = ['', dummy, util.round2(sd(dummy,res[0])), 
                        util.round2(sd(dummy,res[1])),util.round2(sd(dummy,res[-1]))]
            util.pline(ddummy, width=12, align='>')
#--  Minimum default skin depth/discretization of the computational domain
#                                      max(freq) -> |-----|  |----| <- min(res)
        data['dxyz0'] = np.asarray([ util.round2(sd(freq[0], res[0])) ] * 3, dtype = float)
#--  Large skin depth/discretization for overall model discretization
#                                   median(freq) -> |-----|  |----| <- gmean(res)
        data['dxyz1'] = np.asarray([ util.round2(sd(freq[1], res[1])) ] * 3, dtype = float)
#--  Largest skin depth/discretization for overall model discretization
#                                      min(freq) -> |-----|  |---| <- gmean/max(res)
        data['dxyz2'] = np.asarray([ util.round2(sd(freq[2], res[-1])) ] * 3, dtype = float)
#
        res = data['dxyz0'][0]
        util.pline('\n3)   Provide the minimum discretization volume for the computational domain,')
        util.pline('      which can be either a cube or a cuboid: dxyz.')
        util.pline('     Suply dxyz with 1 or 3 entries. If 1 entry is given the volume is a cube,')
        util.pline('      otherwise it is a cuboid. Values are transformed to power of 2, if needed.')
        util.pline('      Entries are space separated, e.g.,')
        util.pline('      (i)  a cube  -> '+str(res)) 
        util.pline('      (ii) a cuboid-> '+str(res)+' '+str(res*2)+' '+str(res*2))
#
        ent = input(f'\n Enter discretization volume dxyz(rtn = '+str(data['dxyz0'][0])+'):\n') or True
        if isinstance(ent, str):
            dummy = ent.rstrip()
            data['dxyz0'] = dummy.split(' ')
            data['dxyz0'] = data['dxyz0'] if len(data['dxyz0']) == 3 else data['dxyz0'] * 3
            data['dxyz0'] = util.round2(np.asarray(data['dxyz0'], dtype = float))
#-- Remove from dictionary
        data.pop('dxyz1')
#
#------------  Print sd info so far
        util.pline('\n   Adopted metrics for the discretization of the computational domain:')
        util.pline(['  dxyz->', *data['dxyz0']], width=12, align='^')
#-- Sanity check
        if 'box' in data.keys():
            dummy = [ data['box'][1::2] - data['box'][0:-1:2] if data['box'].ndim == 1 else
                      data['box'][:,1::2] - data['box'][:,0:-1:2] ]
            dummy = np.array(dummy)
            if not np.all(dummy >= data['dxyz0']):
                raise CmdInputError("@inpt=> dxyz0 >= anomaly(ies).\n")
#
#------------ Mesh adaptation parameters
        text = 'Mesh adaptation parameters'
        print(text.center(60, "-"))
        util.pline(' The mesh adaptation parameters, mad, control mesh building.')
#-- Deals with Octree levels now
        util.pline('\n1) The octree levels is a list of expanding mesh volumes of length k.')
        util.pline('          i =       |~~~~~0~~~~~|  |~~~~~1~~~~~| ... |~~~~k-1~~~~|')
        util.pline('   [Ni*2**i*dxyz] = [N0*2**0*dxyz, N1*2**1*dxyz,..., Nk*2**k*dxyz]\n')
        util.pline('1.1)  dxyz= '+str(data['dxyz0'][0])+' may be bridged to maximum sd= '+str(data['dxyz2'][0]))        
        util.pline('      with k octree levels. Estimate k through optimization now.\n')
#-- Optimzed k
        k = util.estlev(np.amin(data['dxyz0']),np.amax(data['dxyz2']))
        util.pline('\tEstimated k= '+str(k))
#
        k = input(f'\n Enter your choice of octree length k (rtn k={k}): \n') or k
        if isinstance(k, str):
            k = int(k)
#-- Specify the octree levels
        data['mad'] = [int(4)]
        util.pline('\n2) Provide the values for each of the '+str(k)+' octree levels. Each value represent ')
        util.pline(' the number of cells of a given size around a interface. The')
        util.pline(' suggested minimum for the first level is rb= '+str(data['mad'][0])+'. Typical values are')
        util.pline(' 4 <= rb <= 8, representing an energy decay between 98% (4*sd) and 99.97% (8*sd).')
#-- Deals with octree levels.
        otl = [data['mad'][0]] * k
        util.pline('\nThe '+str(k)+' octree levels default to N0...Nk = '+str(otl)+'.')
#
        util.pline('Provide '+str(k)+' octree levels. If one value is given, repeat it '+str(k)+' times.')
        ent = input(f'\n Enter space separated octree levels (dflt='+str(otl)+'): \n') or otl
        if isinstance(ent, str):
            ent = ent.split(' ')
            l = len(ent)
            otl = [int(ent) for ent in ent]
            otl = otl if l == k else otl * k
#-- Appends octree levels to data['mad']
        data['mad'].append(otl)
#
#------------  Print mad info. NB: data['mad'] is a list!
        util.pline('\n3) Adopted mesh adaptation parameters:')
        util.pline(['rb      -> ',  *data['mad'][:-1]], width=12, align='<')
        util.pline(['N0...Nk -> ', *data['mad'][-1]],  width=12, align='<')
        util.pline('\n OcTree levels mesh growth:')
        l=[]
        l.append([str(data['mad'][-1][i])+'*2**'+str(i)+'*dxyz0' for i in range(k)])
        print(f"{l[:]}")
        l = np.asarray([data['mad'][-1][i]*2.**i*data['dxyz0'][0] for i in np.arange(k)]
                        , dtype = float)
        util.pline(['Levels span =', *l[:]],  width=12, align='<')
        util.pline('Total span  = ' + str(np.sum(l))+'\n')
#
#------------  Asks for a given point P = (x y z)
    elif mode == 'intP':
        raise CmdInputError('@inpt=> Deprecated. ,\n') 
        util.pline('\n  Ploting 3 perpendicular slices meeting at a given point P = (x y z).')
        util.pline('\n  1) Enter x y z (space separated), or')
        util.pline('\n  2) Paste a line from the table below, index = 0 is half-space,')
        util.pline('      symbols give position slice touchs a box face (<=min, ^=center, >=max).')    
        util.pline(['loc/box', ' ', 'x', 'y', 'z'], width=8, align='<')
        ent   = np.array([])
        hsp_box = np.concatenate((np.reshape(data['hspace'], (1,-1)), data['box']), axis=0)
#-- Prints a table
        for dummy in np.arange(np.shape(hsp_box)[0]):
            util.pline([ ' <  ', *np.stack(([dummy]*3, 
                        hsp_box[dummy, ::2][:]), axis=0).astype(int).flatten()[-4:]], 
                        width=10, align='<')
            util.pline([ ' ^  ', *np.stack(([dummy]*3, 
                        (hsp_box[dummy,1::2][:] + hsp_box[dummy, ::2][:]) / 2), axis=0).astype(int).flatten()[-4:]], 
                        width=10, align='<')
            ent = np.append(ent,(hsp_box[dummy,1::2][:] + hsp_box[dummy, ::2][:]) / 2)
            util.pline([ ' >  ', *np.stack(([dummy]*3, 
                        hsp_box[dummy,1::2][:]), axis=0).astype(int).flatten()[-4:]], 
                        width=10, align='<')
#-- Provides a default
        ent = np.average(np.reshape(ent, (-1,3)), axis=0)
        util.pline('\n   Provide intercession point P=(x y z). rtn defaults to')
        util.pline(['rtn = ', *ent], width=10, align='<')    
        ent = input(f'\n Enter x y z (space separated):') or ent
        if isinstance(ent, str):
            ent = [temp for temp in ent.split(' ') if temp]
            if len(ent) == 3:
                ent = np.asarray(ent, dtype = float)
            else:
                raise CmdInputError('@inpt=> You need to provide x y and z.\n')
#
        return ent
#
#------------  Error message
    else:
        raise CmdInputError('@inpt=> I/P error.\n')  
#
# -------------- Fatal error messages   ----------------------------
    """ Send error message and stops program.
    Ex.: raise f.CmdInputError('receiver positions required.')
         raise f.CmdInputError(st.lower() + ' requires 5 parameters')
    """ 
class CmdInputError(ValueError):
    """Handles errors in user specified commands. Subclasses the ValueError class."""
    def __init__(self, message, *args):
        self.message = message
        super(CmdInputError, self).__init__(message, *args)
        print(message)
        #raise SystemExit
# -------------- End of class ----------------------------
