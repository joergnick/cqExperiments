
import numpy as np

from data_generators import evaluate_densities

h_ref   = 2**(-(0)*1.0/2)
N_ref   = 512
#### MAX DIFFERENCE IS 0.012 for N:
m = 2
gridfilename='data/grids/sphereh'+str(np.round(h_ref,3))+'.npy'
#gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
filename = 'data/density_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m)+ '.npy'
sol_ref,T = evaluate_densities(filename,gridfilename)
#sol_abs = np.linalg.norm(sol,axis = '0')
tt_ref=np.linspace(0,T,N_ref+1)
Am_space=1
Am_time=6
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
        num_sol,T = evaluate_densities(filename,gridfilename)

        tau = T*1.0/N
        speed=N_ref/N
        print("speed = ",speed)
        resc_ref=np.zeros((3,N+1))
    #   resc_ref=sol_ref
        for j in range(N+1):
            resc_ref[:,j]      = sol_ref[:,j*speed]
    
        errors[space_index,time_index]=np.max(np.abs(resc_ref-num_sol))
        print(errors)

        tt=np.linspace(0,T,N+1)
        import scipy.io

        import matplotlib 
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot(tt_ref,np.linalg.norm(sol_ref[:,::],axis = 0),color='r')
        plt.plot(tt,np.linalg.norm(num_sol[:,::],axis = 0),color='b')
        plt.savefig('temp_sol')
        
