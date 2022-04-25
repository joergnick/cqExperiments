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
    return b
IntegralOperator = Conv_Operator(sinv)

h_ref   = 2**(-(4)*1.0/2)
N_ref   = 256
#### MAX DIFFERENCE IS 0.012 for N:
m = 2
gridfilename='data/grids/sphereh'+str(np.round(h_ref,3))+'.npy'
grid = load_grid(gridfilename)
space_string = "RT"
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
        h   = 2**(-(space_index+4)*1.0/2)
        N   = int(np.round(16*2**time_index))
        gridfilename='data/grids/sphereh'+str(np.round(h,3))+'.npy'
        #gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
        filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
        num_sol,T,m = extract_densities(filename)
        num_sol = IntegralOperator.apply_RKconvol(num_sol,T,method="RadauIIA-"+str(m),show_progress=False)
        num_sol = num_sol[:,1::m]
        tau = T*1.0/N
        speed=N_ref/N
        print("speed = ",speed)
        resc_ref=np.zeros((len(num_sol[:,0]),N))
    #   resc_ref=sol_ref
        for j in range(N):
            resc_ref[:,j]      = sol_ref[:,(j*m+1)*speed]
        errors[space_index,time_index]=np.sqrt(tau*np.sum(np.linalg.norm(resc_ref-num_sol,axis = 1)**2))
        print(errors)

        tt=np.linspace(0,T,len(resc_ref[0,:]))
        #import scipy.io

        import matplotlib 
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot(tt,np.linalg.norm(num_sol[:,::],axis = 0),color='b')

plt.plot(tt,np.linalg.norm(resc_ref[:,::],axis = 0),color='r')
plt.savefig('temp_sol')
