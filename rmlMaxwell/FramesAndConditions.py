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

N=2096
T=8

#Generate points
x_a=-1
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
## Two cubes:
#points = np.vstack( ( plot_grid[0].ravel()  , 0.5*np.ones(plot_grid[0].size) , plot_grid[1].ravel()) )
## Sphere:
points = np.vstack( ( plot_grid[0].ravel()  , 0*np.ones(plot_grid[0].size) , plot_grid[1].ravel()) )
import time
start = time.time() 
evals=scattering_solution(np.sqrt(2)**(-8),N,T,2,points)
end = time.time()
u_ges=np.zeros((n_grid_points**2,N+1))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
for j in range(N+1):    
        # Adjust the figure size in IPython
        t=j*T*1.0/N
        print("t = "+str(t))
        rot = np.array([[-1.0/np.sqrt(2),0 , - 1.0/np.sqrt(2)]
                       ,[0, 1,0]
                       ,[ -1.0/np.sqrt(2) ,0,1.0/np.sqrt(2)]]) 
        def incident_field(x):
               # inc_vals =  np.array([np.exp(-50*(x[2]-t+4)**2), 0. * x[2], 0. * x[2]])
                x_rot = rot.dot(x)
                #for j in range(len(x[0,:])):
                #    x_rot[:,j] = rot.dot(x[:,j])
                inc_vals = rot.dot(np.array([  np.exp(-100*(x_rot[2]+t-5)**2),   0. * x_rot[2], 0. * x_rot[2]]))
                #inc_vals = np.array([  np.sin(20*(x[2]+t-5))*np.exp(-2*(x[2]+t-5)**2),   0. * x[2], 0. * x[2]])    
                for j in range(len(x[0,:])):
                    if (0.25<x[0,j]) and (x[0,j]<1.25) and( 0<x[2,j]) and (x[2,j]<1)  :
                        inc_vals[:,j] = 0*inc_vals[:,j]
                    if (-1.25<x[0,j]) and (x[0,j]<-0.25)and( 0<x[2,j]) and (x[2,j]<1) :
                        inc_vals[:,j] = 0*inc_vals[:,j]
                return inc_vals
                  

        incident_field_data = incident_field(points)
        #scat_eval=np.zeros(nx*nz*3)
        #incident_field_data[radius<1]=0
        scat_eval=evals[:,j].reshape(3,nx*nz)
##       print(scat_eval)
        field_data = scat_eval + incident_field_data
        #field_data = scat_eval 
        #field_data = incident_field_data 
##       print("Points: ")
##       print(points)
##       print("Data: ")
        #print(field_data)
        squared_field_density = np.sum(np.real(field_data * field_data),axis = 0)
        u_ges[:,j]=squared_field_density.T
        #squared_field_density=field_data[2,:]
        
        #squared_field_density[radius<1]=np.nan
        #print("MAX FIELD DATA: " , max(squared_field_density)) 
        #print(np.linalg.norm(scat_eval))
        
        plt.imshow(squared_field_density.reshape((nx, nz)).T,
                   cmap='coolwarm', origin='lower',
                   extent=[x_a, x_b, y_a, y_b])
        if j==10:
                plt.colorbar()
        plt.clim((0,1))
        plt.title("Squared Electric Field Density")

        plt.savefig("data/wave_images/ScreenSphere_n{}.png".format(j))
#print("Computation runtime is: ",end-start)
#import scipy.io
#scipy.io.savemat('data/sim_data_rml.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,points=points))







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

