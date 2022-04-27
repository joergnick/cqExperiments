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
from bemppSteppDirect import compute_linear_densities_direct
from linearcq import Conv_Operator
from rkmethods import RKMethod
from id_bc_rk import scattering_solution
from data_generators import compute_densities
T = 1
#T = 6
m = 7
diffs = np.load('data/diffs.npy')
#diffsAbstract = np.load('data/diffsAbstract.npy')
print(diffs)
#print("Abstract = ",diffsAbstract)
am_space = 1
am_time  = 1
alpha = 0.5
diffs = np.zeros(am_time)
norms_direct = np.zeros(am_time)
diffs_direct = np.zeros(am_time)
for space_index in range(am_space):
    for time_index in range(am_time):
        h   = 2**(-(space_index)*1.0/2)
        N   = int(np.round(80*2**time_index))
        #### MAX DIFFERENCE IS 0.012 for N:
        #N   = 255*2**time_index
        #STILL WORKS until at least 85% : N   = 600*2**time_index
        tau = T*1.0/N
        gridfilename='data/grids/sphereh'+str(np.round(h,3))+'.npy'
        #gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
        filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
        if os.path.isfile(filename):
            print("File "+filename+" already computed, jumped.")
            #continue
        rk = RKMethod("RadauIIA-"+str(m),tau)
        #print("TIME STEPPING : ")

        #start = time.time()
        #sol_direct = compute_linear_densities_direct(N,gridfilename,T,rk,debug_mode=False)
        #end = time.time()
        #print("Direct computation finished, time: ",end-start)
        #start = time.time()
        #sol = compute_linear_densities(N,gridfilename,T,rk,debug_mode = False)
        #end = time.time()
        #print("Recursive computation finished, time: ",end-start)
        #start = time.time()
        #sol_lin = scattering_solution(gridfilename,h,N,T,m)
        #end = time.time()
        #print("Forward computation finished, time: ",end-start)
        sol_newt = compute_densities(alpha,N,gridfilename,T,rk)
        #diffs[time_index] = max(np.linalg.norm(solAbstract[:,::m]-solLin,axis = 0))
        #diffs[time_index] = max(np.linalg.norm(solLin,axis = 0))
        #diffs_direct[time_index] = max(np.linalg.norm(sol_lin-sol_direct[:,::m],axis = 0))
        #norms_direct[time_index] = max(np.linalg.norm(sol_direct[:,::],axis = 0))
        #norms_direct[time_index] = max(np.linalg.norm(sol[:,::],axis = 0))
        #norm_lin =sum(np.linalg.norm(sol_lin[:,::],axis = 0))
        #norm_dir =sum(np.linalg.norm(sol_direct[:,::m],axis = 0))
        norm_newt =sum(np.linalg.norm(sol_newt[:,::m],axis = 0))


        #diffs[time_index] = sum(np.linalg.norm(sol_newt[:,::m]-sol_lin[:,::],axis = 0))
        #print("Max N = ",N," MAX DIFFERENCE : ",diffs)
        #print("DIFFS = ",diffs)
        # print("Max N = ",N," MAX DIFFERENCE DIRECT: ",diffs_direct)
        #print("Max N = ",N," NORM FORWARD SOLUTION: ",norm_lin)
        #print("Max N = ",N," NORM DIRECT SOLUTION: ",norm_dir)
        print("Max N = ",N," NORM NEWTON SOLUTION: ",norm_newt)
        #print("Max N = ",N," MAX DIFFERENCE RECURSIVE: ",diffs)
        resDict = dict()
        resDict["sol"] = sol_newt
        resDict["T"] = T
        resDict["m"] = rk.m
        resDict["N"] = N
        np.save(filename,resDict)
        #np.save("data/diffs",diffs)
        #np.save("data/diffsAbstract",diffsAbstract)
        import matplotlib 
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.semilogy(np.linalg.norm(sol_newt[:,::m],axis = 0),color='r')
        #plt.semilogy(np.linalg.norm(sol_lin,axis = 0),color='b')
        #plt.semilogy(np.linalg.norm(sol_lin-sol_newt[:,::m],axis = 0),linestyle='dashed')
        plt.savefig('temp.png')
        #np.save("data/diffs",diffs)
        #print("COMPLETE DURATION: "+str(end-start))
