import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')
import warnings
warnings.filterwarnings('error')
import numpy as np
import time 
import os.path
#from cqtoolbox import CQModel
from newtonStepper import NewtonIntegrator
from linearcq import Conv_Operator
from rkmethods import RKMethod
from data_generators import compute_densities
start = time.time()
T = 6
m = 2
am_space = 1
am_time  = 1
alpha = 1
for space_index in range(am_space):
    for time_index in range(am_time):
        h   = 2**(-space_index*1.0/2)
        N   = 1025*2**time_index
        #STILL WORKS until at least 85% : N   = 600*2**time_index
        tau = T*1.0/N
        gridfilename='data/grids/sphereh'+str(np.round(h,3))+'.npy'
        #gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
        filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
        if os.path.isfile(filename):
            print("File "+filename+" already computed, jumped.")
        #    continue
        rk = RKMethod("RadauIIA-"+str(m),tau)
        sol = compute_densities(alpha,N,gridfilename,T,rk)
        print(np.linalg.norm(sol[:,::m],axis = 0))
        print(sol[0,::m])
        resDict = dict()
        resDict["sol"] = sol
        resDict["T"] = T
        resDict["m"] = rk.m
        resDict["N"] = N
        #np.save(filename,resDict)


end = time.time()
print("COMPLETE DURATION: "+str(end-start))