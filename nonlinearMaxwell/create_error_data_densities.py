import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')



import bempp.api
import numpy as np

from data_generators import extract_densities,load_grid
from linearcq import Conv_Operator
def sinv(s,b):
    return s**(-1)*b
IntegralOperator = Conv_Operator(sinv)


h_ref   = 2**(-(0)*1.0/2)
N_ref   = 256
#### MAX DIFFERENCE IS 0.012 for N:


m = 2
gridfilename='data/grids/sphereh'+str(np.round(h_ref,3))+'.npy'
grid = load_grid(gridfilename)
space_string = "RT"
rspace_string = "NC"
NC_space=bempp.api.function_space(grid, rspace_string,0)
RT_space=bempp.api.function_space(grid, space_string,0)
elec = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*1)
magn = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*1)
identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
identity2=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
blocks=np.array([[None,None], [None,None]])
blocks[0,0] = -elec.weak_form()
blocks[0,1] =  magn.weak_form()-1.0/2*identity.weak_form()
blocks[1,0] = -magn.weak_form()-1.0/2*identity.weak_form()
blocks[1,1] = -elec.weak_form()
Cald_1 = bempp.api.BlockedDiscreteOperator(blocks)
def xg_norm(x):
    return np.sqrt(np.real(np.dot(x,Cald_1*x)))
RT_space=bempp.api.function_space(grid, space_string,0)

#gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
filename = 'data/density_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m)+ '.npy'
sol_ref,T,m = extract_densities(filename)
sol_ref = IntegralOperator.apply_RKconvol(sol_ref,T,method="RadauIIA-"+str(m),show_progress=False)

#sol_ref,T = evaluate_densities(filename,gridfilename)
#sol_abs = np.linalg.norm(sol,axis = '0')
tt_ref=np.linspace(0,T,N_ref+1)
Am_space=1
Am_time=5
tau_s=np.zeros(Am_time)
h_s=np.zeros(Am_space)
errors=np.zeros((Am_space,Am_time))
for space_index in range(Am_space):
    for time_index in range(Am_time):
        h   = 2**(-(space_index+0)*1.0/2)
        N   = int(np.round(16*2**time_index))
        gridfilename='data/grids/sphereh'+str(np.round(h,3))+'.npy'
        #gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
        filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
        num_sol,T,m_sol = extract_densities(filename)
        num_sol = IntegralOperator.apply_RKconvol(num_sol,T,method="RadauIIA-"+str(m_sol),show_progress=False)
        num_sol = num_sol[:,1::m_sol]
        tau = T*1.0/N
        speed=N_ref/N
        print("speed = ",speed)
        resc_ref=np.zeros((len(num_sol[:,0]),N))
    #   resc_ref=sol_ref
        for j in range(N):
            resc_ref[:,j]      = sol_ref[:,(j*m+1)*speed]
        err = resc_ref-num_sol
        errors[space_index,time_index]=np.sqrt(tau*np.sum([xg_norm(err[:,j])**2 for j in range(N) ]))
        #errors[space_index,time_index]=np.sqrt(tau*np.sum(np.linalg.norm(resc_ref-num_sol,axis = 0)**2))
        print(errors)

        tt=np.linspace(0,T,len(resc_ref[0,:]))
        #import scipy.io

        import matplotlib 
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        #plt.plot(tt,[xg_norm(resc_ref[:,j]) for j in range(N)])
        plt.semilogy(tt,[xg_norm(err[:,j]) for j in range(N)],'r')
        #plt.plot(tt,[xg_norm(num_sol[:,j]) for j in range(N)])
#        plt.semilogy(np.linalg.norm(resc_ref-num_sol,axis = 0),'b')
#        plt.semilogy(np.linalg.norm(resc_ref,axis = 0),'r')
#        plt.semilogy(np.linalg.norm(num_sol,axis = 0),'g')
#        plt.plot(tt,np.linalg.norm(num_sol[:,::],axis = 0),color='b')
#
#plt.plot(tt,np.linalg.norm(resc_ref[:,::],axis = 0),color='r')
plt.savefig('temp_sol')
