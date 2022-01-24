import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')
#import warnings
#warnings.filterwarnings('error')
import numpy as np
import time 
import os.path
#from cqtoolbox import CQModel
from newtonStepper import NewtonIntegrator
from bemppStepp import compute_linear_densities
from linearcq import Conv_Operator
from rkmethods import RKMethod
from id_bc_rk import scattering_solution
from data_generators import compute_densities
start = time.time()
T = 6
m = 3
diffs = np.load('data/diffs.npy')
diffsAbstract = np.load('data/diffsAbstract.npy')
print(diffs)
print("Abstract = ",diffsAbstract)
am_space = 1
am_time  = 4
alpha = 1
diffs = np.zeros(am_time)
diffsAbstract = np.zeros(am_time)
for space_index in range(am_space):
    for time_index in range(am_time):
        h   = 2**(-(space_index+2)*1.0/2)
        #h = 0.7
        N   = int(np.round(1000*2**time_index))
        #N   = 320*2**time_index
        #### MAX DIFFERENCE IS 0.012 for N:
        #N   = 255*2**time_index
        #STILL WORKS until at least 85% : N   = 600*2**time_index
        tau = T*1.0/N
        gridfilename='data/grids/sphereh'+str(np.round(h,3))+'.npy'
        #gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
        filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
        if os.path.isfile(filename):
            print("File "+filename+" already computed, jumped.")
        #    continue
        rk = RKMethod("RadauIIA-"+str(m),tau)
        #print("TIME STEPPING : ")
        solLin = scattering_solution(gridfilename,h,N,T,m)
        solAbstract = compute_linear_densities(N,gridfilename,T,rk)
        #sol = compute_densities(alpha,N,gridfilename,T,rk)
        #print("sol = ",np.linalg.norm(sol, axis = 0))
        #print("solLin = ",np.linalg.norm(solLin,axis = 0))
        #print(np.linalg.norm(sol[:,::m]-solLin,axis = 0))
        #diffs[time_index] = max(np.linalg.norm(solAbstract[:,::m]-solLin,axis = 0))
        #diffs[time_index] = max(np.linalg.norm(solLin,axis = 0))
        diffs[time_index] = max(np.linalg.norm(solLin-solAbstract[:,::m],axis = 0))
        #diffsAbstract[time_index] = max(np.linalg.norm(solAbstract[:,::m]-sol[:,::m],axis = 0))
        print("Max N = ",N," MAX DIFFERENCE : ",diffs)
        #print("Max N = ",N," MAX DIFFERENCE ABSTRACT: ",diffsAbstract)
       # resDict = dict()
       # resDict["sol"] = sol
       # resDict["T"] = T
       # resDict["m"] = rk.m
       # resDict["N"] = N
       # np.save(filename,resDict)
        np.save("data/diffs",diffs)
        #np.save("data/diffsAbstract",diffsAbstract)
#np.save("data/diffs",diffs)
end = time.time()
print("COMPLETE DURATION: "+str(end-start))
