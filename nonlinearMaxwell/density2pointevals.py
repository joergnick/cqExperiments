import numpy as np
import bempp.api
import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
from data_generators import load_grid,extract_densities

#gridfilename='data/grids/.npy'
#h = (2)**(-6*1.0/2)
h = (2)**(-5*1.0/2)
#gridfilename='data/grids/sphereh'+str(np.round(h,3))+'.npy'
#gridfilename='data/grids/cubeh'+str(np.round(h,3))+'.npy'
gridfilename='data/grids/two_cubes_h_'+str(np.round(h,3))+'.npy'
#gridfilename='data/grids/angle_oriented.npy'
grid = load_grid(gridfilename)
N = 256
#N = 128
m = 2
#filename =  'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
#filename =  'data/density_cube_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
#filename =  'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_0.25_II.npy'
#filename =  'data/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_0.5.npy'
#filename = 'data/density_angle_oriented_refined_N_'+str(N)+'_m_'+str(m)+ '.npy'''
#filename = 'data/density_angle_transformed_N_'+str(N)+'_m_'+str(m)+ '.npy'''
#### Create Points
x_a=-0.5
x_b=1.5
y_a=-1.5
y_b=1.5
n_grid_points= 150
nx = n_grid_points
nz = n_grid_points
plot_grid = np.mgrid[y_a:y_b:1j*n_grid_points, x_a:x_b:1j*n_grid_points]
points = np.vstack( ( plot_grid[0].ravel() ,0.5*np.ones(plot_grid[0].size) ,  plot_grid[1].ravel() ) )

RT_space = bempp.api.function_space(grid, "RT",0)
def kirchhoff_repr(s,lambda_data):
    if (np.linalg.norm(lambda_data)<10**(-5)) or (np.real(s)>200):
        print("Jumped, ||phi|| = ",np.linalg.norm(lambda_data), " s = ",s)
        return np.zeros(n_grid_points**2*3)
    if np.isnan(lambda_data).any():
        print("NAN VALUE IN LAMBDA DATA! ")
        print(lambda_data)
    phigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[0:dof],dual_space=RT_space)
    psigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[dof:2*dof],dual_space=RT_space)
    slp_pot = bempp.api.operators.potential.maxwell.electric_field(RT_space, points, s*1j)
    dlp_pot = bempp.api.operators.potential.maxwell.magnetic_field(RT_space, points, s*1j)
    scattered_field_data = -slp_pot * phigrid+dlp_pot*psigrid
    if np.isnan(scattered_field_data).any():
        print("nan-value detected, has been replaced by zero, s = "+str( s))
        print("norm(density)=",np.linalg.norm(lambda_data))
        scattered_field_data[np.isnan(scattered_field_data)] = 0 
    return scattered_field_data.reshape(n_grid_points**2*3,1)[:,0]
#alpha=0.25
#filename =  'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+'_II.npy'
#sol,T,mcheck = extract_densities(filename)
### Sanity checks: 
#if np.isnan(sol).any():
#    raise ValueError(" Solution contains NaN values, terminating. ")
#if (len(sol[0,:])-1)/m != N:
#    raise ValueError("Difference in N, N = ",N," Ncheck = ", (len(sol[0,:])-1)/m, " Terminating.")
#dof = int(len(sol[:,0])/2)
#print("DOF = ",dof)
#
#from linearcq import Conv_Operator
#mSpt_Dpt = Conv_Operator(kirchhoff_repr)
#uscatStages = mSpt_Dpt.apply_RKconvol(sol,T,method = "RadauIIA-"+str(m),cutoff=10**(-5),prolonge_by=0,factor_laplace_evaluations=3)
#uscat = uscatStages[:,::m]
#uscat = np.concatenate((np.zeros((len(uscat[:,0]),1)),uscat),axis = 1)
#u_ges=np.zeros((n_grid_points**2,N+1))
#for j in range(N+1):    
#    # Adjust the figure size in IPython
#    t=j*T*1.0/N
#    def incident_field(x):
#        return np.array([np.exp(-100*(x[2]+t-2)**2), 0. * x[2], 0. * x[2]])
#    incident_field_data = incident_field(points)
#    scat_eval=uscat[:,j].reshape(3,nx*nz)
#    field_data = scat_eval + incident_field_data
#    squared_field_density = np.real(np.sum(field_data * field_data,axis = 0))
#    u_ges[:,j]=squared_field_density.T
#import scipy.io
##scipy.io.savemat('data/DonutFieldDataDOF340N200.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,points=points))
#scipy.io.savemat('data/thesis_nonlinear/AngleTransformedFields_a_'+str(alpha)+'N'+str(N)+'.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,points=points))
#

alpha=0.75
filename =  'data/thesis_nonlinear/density_two_cubes_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+'_a_'+str(alpha)+'_II.npy'
sol,T,mcheck = extract_densities(filename)
## Sanity checks: 
if np.isnan(sol).any():
    raise ValueError(" Solution contains NaN values, terminating. ")
if (len(sol[0,:])-1)/m != N:
    raise ValueError("Difference in N, N = ",N," Ncheck = ", (len(sol[0,:])-1)/m, " Terminating.")
dof = int(len(sol[:,0])/2)
print("DOF = ",dof)

from linearcq import Conv_Operator
mSpt_Dpt = Conv_Operator(kirchhoff_repr)
uscatStages = mSpt_Dpt.apply_RKconvol(sol,T,method = "RadauIIA-"+str(m),cutoff=10**(-5),prolonge_by=0,factor_laplace_evaluations=3)
uscat = uscatStages[:,::m]
uscat = np.concatenate((np.zeros((len(uscat[:,0]),1)),uscat),axis = 1)
u_ges=np.zeros((n_grid_points**2,N+1))
for j in range(N+1):    
    # Adjust the figure size in IPython
    t=j*T*1.0/N
    def incident_field(x):
        return np.array([np.exp(-100*(x[2]+t-2)**2), 0. * x[2], 0. * x[2]])
    incident_field_data = incident_field(points)
    scat_eval=uscat[:,j].reshape(3,nx*nz)
    field_data = scat_eval + incident_field_data
    squared_field_density = np.real(np.sum(field_data * field_data,axis = 0))
    u_ges[:,j]=squared_field_density.T
import scipy.io
#scipy.io.savemat('data/DonutFieldDataDOF340N200.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,points=points))
scipy.io.savemat('data/thesis_nonlinear/AngleTransformedFields_a_'+str(alpha)+'N'+str(N)+'.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,points=points))
