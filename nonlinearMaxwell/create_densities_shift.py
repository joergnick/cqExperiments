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
from data_generators_II_shift import compute_densities
#T = 1
T = 2
m = 3
#diffsAbstract = np.load('data/diffsAbstract.npy')
#print("Abstract = ",diffsAbstract)
alpha = 0.5
import time


am_space = 1
am_time  = 8
m = 5
for space_index in range(am_space):
    for time_index in range(am_time):
        h   = 2**(-(space_index+2)*1.0/2)
        N   = int(np.round(16*2**(time_index*0.5)))
        tau = T*1.0/N
        gridfilename = 'data/grids/two_cubes_h_'+str(np.round(h,3))+'.npy'
        rk = RKMethod("RadauIIA-"+str(m),tau)
        start = time.time()
        
        alpha = 0.5
        filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+ '_T_'+str(T)+'_II_shift.npy'
        #filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+ '_T_2_II_shift.npy'
        if os.path.isfile(filename):
            print("File "+filename+" already computed, jumped.")
            continue
        sol_newt = compute_densities(alpha,N,gridfilename,T,rk)
      #  try:
      #      sol_newt = compute_densities(alpha,N,gridfilename,T,rk)
      #  except:
      #      print("Computation of "+filename+" failed, continue.")
      #      continue
        norm_newt =sum(np.linalg.norm(sol_newt[:,::m],axis = 0))
        end = time.time()
        print("Max N = ",N," NORM NEWTON SOLUTION: ",norm_newt, " Length of computation: ", (end-start)/60.0)
        resDict = dict()
        resDict["sol"] = sol_newt
        resDict["T"] = T
        resDict["m"] = rk.m
        resDict["N"] = N
        np.save(filename,resDict)
        #continue 
        #alpha = 0.5
        #filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+ '_II.npy'
        #if os.path.isfile(filename):
        #    print("File "+filename+" already computed, jumped.")
        #    continue
        #try:
        #    sol_newt = compute_densities(alpha,N,gridfilename,T,rk)
        #except:
        #    print("Computation of "+filename+" failed, continue.")
        #    continue
        #norm_newt =sum(np.linalg.norm(sol_newt[:,::m],axis = 0))
        #end = time.time()
        #print("Max N = ",N," NORM NEWTON SOLUTION: ",norm_newt, " Length of computation: ", (end-start)/60.0)
        #resDict = dict()
        #resDict["sol"] = sol_newt
        #resDict["T"] = T
        #resDict["m"] = rk.m
        #resDict["N"] = N
        #np.save(filename,resDict)
        #
        #alpha = 0.75
        #filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+ '_II.npy'
        #if os.path.isfile(filename):
        #    print("File "+filename+" already computed, jumped.")
        #    continue
        #try:
        #    sol_newt = compute_densities(alpha,N,gridfilename,T,rk)
        #except:
        #    print("Computation of "+filename+" failed, continue.")
        #    continue
        #norm_newt =sum(np.linalg.norm(sol_newt[:,::m],axis = 0))
        #end = time.time()
        #print("Max N = ",N," NORM NEWTON SOLUTION: ",norm_newt, " Length of computation: ", (end-start)/60.0)
        #resDict = dict()
        #resDict["sol"] = sol_newt
        #resDict["T"] = T
        #resDict["m"] = rk.m
        #resDict["N"] = N
        #np.save(filename,resDict)
#m=3
#
#h   = 2**(-(7)*1.0/2)
#N   = 256
#tau = T*1.0/N
#gridfilename = 'data/grids/two_cubes_h_'+str(np.round(h,3))+'.npy'
#filename = 'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+ '.npy'
#rk = RKMethod("RadauIIA-"+str(m),tau)
#sol_newt = compute_densities(alpha,N,gridfilename,T,rk)
#resDict = dict()
#resDict["sol"] = sol_newt
#resDict["T"] = T
#resDict["m"] = rk.m
#resDict["N"] = N
#np.save(filename,resDict)
#
#h   = 2**(-(5)*1.0/2)
#N   = 256
#tau = T*1.0/N
#gridfilename = 'data/grids/two_cubes_h_'+str(np.round(h,3))+'.npy'
#filename = 'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+ '.npy'
#rk = RKMethod("RadauIIA-"+str(m),tau)
#sol_newt = compute_densities(alpha,N,gridfilename,T,rk)
#resDict = dict()
#resDict["sol"] = sol_newt
#resDict["T"] = T
#resDict["m"] = rk.m
#resDict["N"] = N
#np.save(filename,resDict)
#import matplotlib 
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#plt.semilogy(np.linalg.norm(sol_newt[:,::m],axis = 0),color='r')
##plt.semilogy(np.linalg.norm(sol_lin,axis = 0),color='b')
##plt.semilogy(np.linalg.norm(sol_lin-sol_newt[:,::m],axis = 0),linestyle='dashed')
#plt.savefig('temp.png')
#np.save("data/diffs",diffs)
#print("COMPLETE DURATION: "+str(end-start))



#h   = 2**(-(5)*1.0/2) 
#gridfilename = 'data/grids/two_cubes_h_'+str(np.round(h,3))+'.npy'
#N = 256
#tau = T*1.0/N
#rk = RKMethod("RadauIIA-"+str(m),tau)
#filename = 'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+ '.npy'
#sol_newt = compute_densities(alpha,N,gridfilename,T,rk)
#h   = 2**(-(7)*1.0/2) 
#gridfilename = 'data/grids/two_cubes_h_'+str(np.round(h,3))+'.npy'
#N = 256
#tau = T*1.0/N
#rk = RKMethod("RadauIIA-"+str(m),tau)
#filename = 'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+ '.npy'
#sol_newt = compute_densities(alpha,N,gridfilename,T,rk)
#alpha = 0.75
#filename = 'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+ '.npy'
#sol_newt = compute_densities(alpha,N,gridfilename,T,rk)
