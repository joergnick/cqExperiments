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
from rkmethods import RKMethod

h_ref   = 2**(-(0)*1.0/2)
#gridfilename='data/grids/sphereh'+str(np.round(h_ref,3))+'.npy'
gridfilename='data/grids/sphere_python3_h'+str(np.round(h_ref,3))+'.npy'

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
blocks[0,1] =  magn.weak_form()
blocks[1,0] = -magn.weak_form()
blocks[1,1] = -elec.weak_form()
Cald_1 = bempp.api.BlockedDiscreteOperator(blocks)
def xg_norm(x):
    #return np.linalg.norm(x)
    #return np.max(np.abs(x))
    return np.sqrt(np.real(np.dot(x,Cald_1*x)))
RT_space=bempp.api.function_space(grid, space_string,0)

#gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
#filename = 'data/density_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m)+ '.npy'
sigma = 0
alpha = 0.5

m_ref = 3
shift = 1

N_ref   = 256
shifts = np.linspace(0,1,4)
sigma = shifts[1]
filename = 'data/density_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+'_a_'+str(alpha)+'_shift_'+str(sigma)+'.npy'
#filename = 'data/density_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+'_a_'+str(alpha)+'_shift_0.0.npy'
#filename = 'data/density_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+'_a_'+str(alpha)+'_shift_'+str(sigma)+'.npy'
sol_ref,T,m_ref = extract_densities(filename)
#sol_ref = IntegralOperator.apply_RKconvol(sol_ref,T,method="RadauIIA-"+str(m_ref),show_progress=False)

tau_ref = T*1.0/N_ref
rk = RKMethod("RadauIIA-"+str(m_ref),tau_ref)
tt_ref = rk.get_time_points(T)

from scipy.interpolate import CubicSpline
cs_ref = CubicSpline(tt_ref,sol_ref,axis=1)
#print(sol_ref[0,:])
#print(sol_ref.shape)
sol_ref_plot = sol_ref
sol_ref = sol_ref[:,m_ref::m_ref]
#print(sol_ref[0,:])
#print(sol_ref.shape)
#sol_ref,T,dof = evaluate_densities(filename,gridfilename)
#sol_abs = np.linalg.norm(sol,axis = '0')
#shifts = [1.0]
Am_time = 6
errors=np.zeros((len(shifts),Am_time))
taus = np.zeros(Am_time)
m=3
for time_index in range(Am_time):
    for shift_index in range(len(shifts)):
        h   = 2**(-(0)*1.0/2)
        N   = 4*2**(time_index)
        #N   = int(np.round(32))
        #gridfilename='data/grids/sphereh'+str(np.round(h,3))+'.npy'
        gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
        sigma = shifts[shift_index]
        def sinv(s,b):
            #return b
            return (s+sigma)**(-1)*b
        IntegralOperator = Conv_Operator(sinv)


        filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+'_shift_'+str(sigma)+'.npy'
        #filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
        num_sol,T,m_sol = extract_densities(filename)
        #num_sol = IntegralOperator.apply_RKconvol(num_sol,T,method="RadauIIA-"+str(m_sol),show_progress=False)
        num_sol_plot = num_sol
       # num_sol = np.real(num_sol[:,m_sol::m_sol])
         
        tau = T*1.0/N
        taus[time_index] = tau
        rk = RKMethod("RadauIIA-"+str(m_sol),tau)
        resc_ref2 = cs_ref(rk.get_time_points(T))
        speed=int(N_ref/N)
        #print("speed = ",speed,)
#        resc_ref=np.zeros((len(num_sol[:,0]),N))
    #   resc_ref=sol_ref
#        for j in range(N-1):
#            resc_ref[:,j+1]      = sol_ref[:,(j+1)*speed+1]
#
        err = resc_ref2-num_sol
        err = IntegralOperator.apply_RKconvol(err,T,method="RadauIIA-"+str(m_sol),show_progress=False)
        errors[shift_index,time_index]=np.sqrt(tau*np.sum([xg_norm(err[:,j])**2 for j in range(len(err[0,:])) ]))
        #print("||num_sol|| = ",np.sqrt(tau*np.sum([xg_norm(num_sol[:,j])**2 for j in range(N) ])))
        #errors[space_index,time_index]=np.sqrt(tau*np.sum(np.linalg.norm(resc_ref-num_sol,axis = 0)**2))
print(errors)
    
        #import scipy.io
import scipy.io
res = dict()
res["errors"]=errors
res["T"] = T
res["shifts"]= shifts
res["taus"]= taus
scipy.io.savemat("data/error_m_3_shifts.mat",res)
#
#import matplotlib 
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
##plt.plot(tt,[xg_norm(resc_ref[:,j]) for j in range(N)])
#tt = rk.get_time_points(T)
#plt.plot(tt,cs_ref(tt)[:2,:].T,color='r')
##plt.plot(tt_ref,sol_ref_plot[:2,:].T,linestyle='dashed',color='b')
##plt.plot(tt,num_sol_plot[:2,:].T,color='g',marker='o')
##plt.semilogy(tt[::m_sol],np.abs(num_sol_plot[:5,::m_sol].T-cs_ref(tt)[:5,::m_sol].T),color='g')
##plt.semilogy(tt[::m_sol],np.abs(num_sol_plot[:5,::m_sol].T-cs_ref(tt)[:5,::m_sol].T),color='g')
##plt.semilogy([xg_norm(resc_ref[:,j]) for j in range(len(resc_ref[0,:]))],'r')
##plt.semilogy([xg_norm(num_sol[:,j]-resc_ref[:,j]) for j in range(len(num_sol[0,:]))],'b')
##plt.semilogy([xg_norm(resc_ref2[:,j]) for j in range(len(num_sol[0,:]))],'g')
##plt.ylim((10**(-10),10**(0)))
#plt.savefig('test.png')
#    #plt.plot(tt,[xg_norm(num_sol[:,j]) for j in range(N)])
##        plt.semilogy(np.linalg.norm(resc_ref-num_sol,axis = 0),'b')
##        plt.semilogy(np.linalg.norm(resc_ref,axis = 0),'r')
##        plt.semilogy(np.linalg.norm(num_sol,axis = 0),'g')
##        plt.plot(tt,np.linalg.norm(num_sol[:,::],axis = 0),color='b')
##
##plt.plot(tt,np.linalg.norm(resc_ref[:,::],axis = 0),color='r')
##plt.savefig('temp_sol')
