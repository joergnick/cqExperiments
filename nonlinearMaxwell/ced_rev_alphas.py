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


from data_generators_rev import evaluate_densities
#### MAX DIFFERENCE IS 0.012 for N:



#filename = 'data/density_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+'_a_'+str(alpha)+'_shift_0.0.npy'
#filename = 'data/density_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+'_a_'+str(alpha)+'_shift_'+str(sigma)+'.npy'
#sol_ref = IntegralOperator.apply_RKconvol(sol_ref,T,method="RadauIIA-"+str(m_ref),show_progress=False)


alphas = [0.33,0.66,1]
#alphas = [0.33,0.66,1.0]
Am_space = 6
errors=np.zeros((len(alphas),Am_space))
hs = np.zeros(Am_space)
m=3

h_ref   = 2**(-(6)*1.0/2)
sigma = 0
N   = 16

N_ref   = 16
m_ref = 3
for alpha_index in range(len(alphas)):
    alpha = alphas[alpha_index]
    gridfilename='data/grids/sphere_python3_h'+str(np.round(h_ref,3))+'.npy'
    filename = 'data/density_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m_ref)+'_a_'+str(alpha)+'_shift_'+str(sigma)+'.npy'
    sol_ref,T,dof = evaluate_densities(filename,gridfilename)
    for space_index in range(Am_space):
        h   = 2**(-(space_index)*1.0/2)
        hs[space_index] = h
        #N   = int(np.round(32))
        #gridfilename='data/grids/sphereh'+str(np.round(h,3))+'.npy'
        gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
        filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+'_shift_'+str(sigma)+'.npy'
        #filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
        num_sol,T,dof = evaluate_densities(filename,gridfilename)
        tau = T*1.0/N         
        err = sol_ref-num_sol
        errors[alpha_index,space_index]=np.sqrt(tau*np.sum([np.abs(err[:,j])**2 for j in range(len(err[0,:])) ]))
        #print("||num_sol|| = ",np.sqrt(tau*np.sum([xg_norm(num_sol[:,j])**2 for j in range(N) ])))
        #errors[space_index,time_index]=np.sqrt(tau*np.sum(np.linalg.norm(resc_ref-num_sol,axis = 0)**2))
        print(errors)
    
        #import scipy.io
import scipy.io
res = dict()
res["errors"]=errors
res["T"] = T
res["alphas"]= alphas
res["hs"]= hs
scipy.io.savemat("data/error_m_3_alphas.mat",res)
#
#import matplotlib 
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
##plt.plot(tt,[xg_norm(resc_ref[:,j]) for j in range(N)])
#plt.plot(sol_ref[:,:].T,linestyle='dashed',color='b')
#plt.plot(num_sol[:,:].T,color='g',marker='o')
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
