#import numpy as np
#import bempp.api
#import math
#from RKconv_op import *
import bempp.api
import numpy as np
import math
#from RKconv_op import *
import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')
from linearcq import Conv_Operator
from rml_main import density2evals
import math

import time

## Initialize points in domain
am_p = 1
#points = np.array([np.zeros(am_p),0.0*np.ones(am_p),np.linspace(-0.5,0.5,am_p)])
points = np.array([[0],[0],[0]])

## Load reference solution
#N_ref=256
N_ref=2048
T=8

h_ref = 2**(-6/2)
#h_ref = 2**(-4)
m = 2
filename = 'data/rml_densities_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m)+ '.npy'
#filename = 'data/rml_densities_sphere_h_'+str(np.round(h_ref,3)) +'_N_'+str(N_ref)+'_m_'+str(m)+ '.npy'
sol_ref = density2evals(h_ref,N_ref,T,m,points,filename,use_sphere=True) ## (Stages are dropped)

print(sol_ref.shape)
#######################################
###############
am_space = 8
am_time  = 9
h_s = np.zeros(am_space)
tau_s = np.zeros(am_time)
error_s = np.zeros((am_space,am_time))
m = 2
T = 8
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
plt.plot(np.linalg.norm(sol_ref,axis = 0))
plt.savefig('test.png')
for space_index in range(am_space):
    for time_index in range(am_time):
        h                 = 2**(-(space_index)*1.0/2)
        h_s[space_index]  = h
        N                 = int(np.round(8*2**time_index))
        tau               = T*1.0/N
        tau_s[time_index] = tau
        filename = 'data/rml_densities_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
        num_sol  = density2evals(h,N,T,m,points,filename,use_sphere=True)
        #import matplotlib.pyplot as plt
        #plt.plot(np.linalg.norm(num_sol,axis=1))
        #plt.show()
 ########## Rescaling reference solution:        
        speed=N_ref/N
        resc_ref=np.zeros((3*am_p,N+1))
    #   resc_ref=sol_ref
        for j in range(N+1):
            resc_ref[:,j]      = sol_ref[:,int(j*speed)]
        print(error_s)
        error_s[space_index,time_index] = 1.0/np.sqrt(am_p*N)*np.linalg.norm(resc_ref - num_sol)
import scipy.io
scipy.io.savemat('data/Err_data_rml.mat', dict( ERR=error_s,h_s=h_s,tau_s=tau_s))
 
