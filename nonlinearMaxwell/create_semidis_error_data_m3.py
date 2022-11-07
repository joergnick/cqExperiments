import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')
import numpy as np

from data_generators import evaluate_densities

import scipy.io
########### Space discretization #########
## Loading reference solution 
#h_ref   = 2**(-(8)*1.0/2)
#N_ref   = 32
#m_ref = 3
#gridfilename='data/grids/two_cubes_h_'+str(np.round(h_ref,3))+'.npy'
#alpha = 0.5
##filename = 'data/density_two_cubes_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+ '_a_'+str(alpha)+'.npy'
#filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+ '_a_'+str(alpha)+'_II.npy'
#sol_ref,T,dof = evaluate_densities(filename,gridfilename)
#print("Ref DOF = "+str(dof))
#Am_space = 7
#Am_time  = 1
#tau_s=np.zeros(Am_time)
#h_s=np.zeros(Am_space)
#errors=np.zeros((Am_space,Am_time))
#taus=np.zeros(Am_time)
#hs=np.zeros(Am_space)
#m = 3
#for space_index in range(Am_space):
#    for time_index in range(Am_time):
#        h   = 2**(-(space_index)*1.0/2)
#        hs[space_index] = h
#        N   = int(np.round(32*2**time_index))
#        gridfilename='data/grids/two_cubes_h_'+str(np.round(h,3))+'.npy'
#        filename = 'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '_a_'+str(alpha)+'.npy'
#        #filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '_a_'+str(alpha)+'_II.npy'
#        num_sol,T,dof = evaluate_densities(filename,gridfilename)
#        tau = T*1.0/N
#        taus[time_index] = tau
#        speed=int(N_ref/N)
#        print("speed = ",speed)
#        resc_ref=np.zeros((3,N+1))
#    #   resc_ref=sol_ref
#        for j in range(N+1):
#            resc_ref[:,j]      = sol_ref[:,j*speed]
#        errors[space_index,time_index]=np.max(np.abs(resc_ref-num_sol))
#        print(errors)
#res = dict()
#res["errors"]=errors
#res["T"] = T
#res["hs"]= hs
#res["taus"]= taus
#scipy.io.savemat("data/error_m_2_space_I.mat",res)
#print("Space discretization completed.")
#

########### Time discretization #########
## Loading reference solution 
h_ref   = 2**(-(4)*1.0/2)
N_ref   = 512
m_ref = 3
gridfilename='data/grids/two_cubes_h_'+str(np.round(h_ref,3))+'.npy'
#filename = 'data/density_two_cubes_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+ '_a_'+str(alpha)+'.npy'
filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+ '_a_'+str(alpha)+'_II.npy'
sol_ref,T,dof = evaluate_densities(filename,gridfilename)

Am_space = 1
Am_time  = 5
tau_s=np.zeros(Am_time)
h_s=np.zeros(Am_space)
errors=np.zeros((Am_space,Am_time))
taus=np.zeros(Am_time)
hs=np.zeros(Am_space)
m = 2
for space_index in range(Am_space):
    for time_index in range(Am_time):
        h   = 2**(-(space_index+4)*1.0/2)
        hs[space_index] = h
        N   = int(np.round(16*2**time_index))
        gridfilename='data/grids/two_cubes_h_'+str(np.round(h,3))+'.npy'
        filename = 'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '_a_'+str(alpha)+'.npy'
        #filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '_a_'+str(alpha)+'_II.npy'
        num_sol,T,dof = evaluate_densities(filename,gridfilename)
        tau = T*1.0/N
        taus[time_index] = tau
        speed=int(N_ref/N)
        print("speed = ",speed)
        resc_ref=np.zeros((3,N+1))
    #   resc_ref=sol_ref
        for j in range(N+1):
            resc_ref[:,j]      = sol_ref[:,j*speed]
        errors[space_index,time_index]=np.max(np.abs(resc_ref-num_sol))
        print(errors)
res = dict()
res["errors"]=errors
res["T"] = T
res["hs"]= hs
res["taus"]= taus
scipy.io.savemat("data/error_m_2_time_I.mat",res)
