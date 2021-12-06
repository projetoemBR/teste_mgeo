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
from SimPEG import utils
try:
    from pymatsolver import Pardiso as Solver
except:
    from SimPEG import Solver
import pickle
from scipy.constants import mu_0, epsilon_0 as eps_0
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
#-- RAM monitoring
    util.ramon()
#------------  Read I/P file
    text = 'Open File for I/P'
    print(text.center(70, "-"))
    args = ipop.read_param()
    dummy = args.folder + args.ipfile
    util.pline([' Reading I/P file: ', dummy], width=30, align='^')
#                  file object-->|
    with open(dummy, 'rb') as fobj:
        data = pickle.load(fobj)
#-- RAM monitoring
    util.ramon()
#------------  List data read
    util.pline('\n Dictionary data[...] read:')
    print(list(data.keys()))







#
# -------------- Controls Execution --------------
if __name__ == "__main__":
    main()
#
raise SystemExit    