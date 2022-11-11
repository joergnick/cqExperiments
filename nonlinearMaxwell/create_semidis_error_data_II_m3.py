import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')
import numpy as np
import bempp.api
from data_generators import evaluate_densities,load_grid,extract_densities

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
#        #filename = 'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '_a_'+str(alpha)+'.npy'
#        filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '_a_'+str(alpha)+'_II.npy'
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
#scipy.io.savemat("data/error_m_3_space_II.mat",res)
#print("Space discretization completed.")
#
import math
def X_Gamma_norm(grid,phi):
    NC_space=bempp.api.function_space(grid, "NC",0)
    #NC_space=bempp.api.function_space(grid, nrspace_string,0)
    RT_space=bempp.api.function_space(grid, "RT",0)
    #RT_space=bempp.api.function_space(grid, space_string,0)
    s = 1
    elec = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*s)
    magn = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*s)
    identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
    identity2=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
    blocks=np.array([[None,None], [None,None]])
    blocks[0,0] = -elec.weak_form()
    blocks[0,1] =  magn.weak_form()
    blocks[1,0] = -magn.weak_form()
    blocks[1,1] = -elec.weak_form()
    Cald_weak = bempp.api.BlockedDiscreteOperator(blocks)
    return math.sqrt(np.real(np.conj(phi).dot(Cald_weak*phi)))
########### Time discretization #########
## Loading reference solution 
h_ref   = 2**(-(2)*1.0/2)
N_ref   = 362
m_ref = 3
alpha = 0.5
gridfilename='data/grids/two_cubes_h_'+str(np.round(h_ref,3))+'.npy'
T = 2
#filename = 'data/density_two_cubes_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+ '_a_'+str(alpha)+'.npy'
filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+ '_a_'+str(alpha)+'_T_'+str(T)+'_II.npy'
sol_ref,T,dof = extract_densities(filename)
grid = load_grid(gridfilename)

#sol_ref,T,dof = evaluate_densities(filename,gridfilename)
from linearcq import Conv_Operator
print(T)
#print("||phi_ref[T]|| = ",X_Gamma_norm(grid,sol_ref[:,-1]))

Am_space = 1
Am_time  = 7
errors=np.zeros((4,Am_time))
for m_index in range(1,5):
    m = m_index
    if m_index==4:
        m = 5
 
    tau_s=np.zeros(Am_time)
    h_s=np.zeros(Am_space)
    taus=np.zeros(Am_time)
    hs=np.zeros(Am_space)
    for space_index in range(Am_space):
        for time_index in range(Am_time):
            h   = 2**(-(space_index+2)*1.0/2)
            hs[space_index] = h
            N   = int(np.round(16*2**(time_index*0.5)))
            #filename = 'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '_a_'+str(alpha)+'.npy'
            filename = 'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '_a_'+str(alpha)+'_T_'+str(T)+'_II.npy'
            #num_sol,T,dof = evaluate_densities(filename,gridfilename)
            num_sol,T,dof = extract_densities(filename)
            tau = T*1.0/N
            taus[time_index] = tau
            errors[m_index-1,time_index] = X_Gamma_norm(grid,num_sol[:,-1]-sol_ref[:,-1])
            #errors[m-1,time_index] = np.max(np.abs(num_sol[:,-1]-sol_ref[:,-1]))
           # print("num_sol = ",num_sol)
           # print("num_sol[T] = ",num_sol[:,-1])
    print(errors)
    res = dict()
    res["errors"]=errors
    res["T"] = T
    res["hs"]= hs
    res["taus"]= taus
    #scipy.io.savemat("data/error_multiple_m_Gamma_time_II.mat",res)
