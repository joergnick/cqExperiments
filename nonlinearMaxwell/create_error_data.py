
import numpy as np

from data_generators import evaluate_densities

h_ref   = 2**(-(7)*1.0/2)
N_ref   = 256
#### MAX DIFFERENCE IS 0.012 for N:
m_ref = 3
gridfilename='data/grids/two_cubes_h_'+str(np.round(h_ref,3))+'.npy'
#gridfilename='data/grids/sphereh'+str(np.round(h_ref,3))+'.npy'
#gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
filename = 'data/density_two_cubes_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+ '_a_0.5.npy'
#filename = 'data/density_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m)+ '.npy'
sol_ref,T,dof = evaluate_densities(filename,gridfilename)
print("DOF = ",dof)
#sol_abs = np.linalg.norm(sol,axis = '0')
tt_ref=np.linspace(0,T,N_ref+1)
#Am_space = 3
#Am_time=1
Am_space = 7
Am_time=6
tau_s=np.zeros(Am_time)
h_s=np.zeros(Am_space)
errors=np.zeros((Am_space,Am_time))
taus=np.zeros(Am_time)
hs=np.zeros(Am_space)
m = 2
for space_index in range(Am_space):
    for time_index in range(Am_time):
        h   = 2**(-(space_index)*1.0/2)
        hs[space_index] = h
        N   = int(np.round(8*2**time_index))
        gridfilename='data/grids/two_cubes_h_'+str(np.round(h,3))+'.npy'
        #gridfilename='data/grids/sphereh'+str(np.round(h,3))+'.npy'
        #gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
        #filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
        filename = 'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '_a_0.5.npy'
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

        tt=np.linspace(0,T,N+1)

        #import matplotlib 
        #matplotlib.use('Agg')
        #import matplotlib.pyplot as plt
        #plt.plot(tt_ref,np.linalg.norm(sol_ref[:,::],axis = 0),color='r')
        #plt.plot(tt,np.linalg.norm(num_sol[:,::],axis = 0),color='b')
        #plt.savefig('temp_sol')

import scipy.io
res = dict()
res["errors"]=errors
res["T"] = T
res["hs"]= hs
res["taus"]= taus
scipy.io.savemat("error_m_2.mat",res)
