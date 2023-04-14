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
from rml_main import scattering_solution
import math

import time

N=256
T=10

#Generate points
x_a=-2
x_b=2
y_a=-2
y_b=2
#x_a=-1.5
#x_b=1.5
#y_a=-1.5
#y_b=1.5
n_grid_points=200
nx=n_grid_points
nz=n_grid_points
#######################################
#Initialize empty file, which will be overwritten continously with condition numbers#and the frequencies
###############
plot_grid = np.mgrid[y_a:y_b:1j*n_grid_points, x_a:x_b:1j*n_grid_points]
#plot_grid = np.mgrid[-0.5:1:1j*n_grid_points, -1.5:1.5:1j*n_grid_points]
#print(plot_grid)
#points = np.vstack( ( plot_grid[0].ravel()  , 0.5*np.ones(plot_grid[0].size) , plot_grid[1].ravel()) )

points = np.vstack( ( plot_grid[0].ravel()  , 0*np.ones(plot_grid[0].size) , plot_grid[1].ravel()) )
        
evals=scattering_solution(0.2,N,T,2,points)
u_ges=np.zeros((n_grid_points**2,N+1))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
for j in range(N+1):    
        #matplotlib inline
        
        # Adjust the figure size in IPython
        t=j*T*1.0/N
        def incident_field(x):
                inc_vals =  np.array([np.exp(-50*(x[2]-t+4)**2), 0. * x[2], 0. * x[2]])
                for j in range(len(x[0,:])):
                    if np.linalg.norm(x[:,j])<1:
                        inc_vals[:,j] = 0*inc_vals[:,j]
                return inc_vals
                  

        incident_field_data = incident_field(points)
        #scat_eval=np.zeros(nx*nz*3)
        #incident_field_data[radius<1]=0
        scat_eval=evals[:,j].reshape(3,nx*nz)
#       print(scat_eval)
        field_data = scat_eval + incident_field_data
#       field_data = scat_eval 
#       field_data = incident_field_data 
#       print("Points: ")
#       print(points)
#       print("Data: ")
        #print(field_data)
        squared_field_density = np.sum(field_data * field_data,axis = 0)
        u_ges[:,j]=squared_field_density.T
        #squared_field_density=field_data[2,:]
        
        #squared_field_density[radius<1]=np.nan
        #print("MAX FIELD DATA: " , max(squared_field_density)) 
        print(max(np.abs(squared_field_density)))
        
        plt.imshow(squared_field_density.reshape((nx, nz)).T,
                   cmap='coolwarm', origin='lower',
                   extent=[x_a, x_b, y_a, y_b])
        if j==10:
                plt.colorbar()
        plt.clim((-1,1))
        plt.title("Squared Electric Field Density")

        plt.savefig("data/wave_images/Screen_n{}.png".format(j))
import scipy.io
scipy.io.savemat('data/sim_data.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,points=points))







#
#       tp1=time.time()
#       print("Dense Matrix assembled, time : ", tp1-tp0)
#
#       normA=np.linalg.norm(blocks_mat,ord=2)
#       tp2=time.time()
#       print("Norm A calculated, value ",normA,  " time : ",tp2-tp1)
#       cond=np.linalg.cond(blocks_mat)
#       tp3=time.time() 
#       print("Cond A calculated, value ", cond," time : ",tp3-tp2)
#       norminv=np.linalg.norm(np.linalg.inv(blocks_mat),ord=2)
#       print("Inv A calculated, direct : ", norminv, " Previous estimate : ", cond/normA)

