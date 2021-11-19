import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')

import numpy as np
import os.path
#from cqtoolbox import CQModel
from newtonStepper import NewtonIntegrator
from linearcq import Conv_Operator
from rkmethods import RKMethod
from data_generators import compute_densities

T = 5
m = 2
am_space = 7
am_time  = 9
for space_index in range(am_space):
    for time_index in range(am_time):
        h   = 2**(-space_index*1.0/2)
        N = 4*2**time_index
        tau = T*1.0/N
        gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
        
        filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
        if os.path.isfile(filename):
            print("File "+filename+" already computed, jumped.")
            continue
        rk = RKMethod("RadauIIA-"+str(m),tau)
        sol = compute_densities(N,gridfilename,T,rk)
        resDict = dict()
        resDict["sol"] = sol
        resDict["T"] = T
        resDict["m"] = rk.m
        resDict["N"] = N
        np.save(filename,resDict)


