
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
import math
import bempp.api
from helpers import load_grid,save_densities,extract_densities
def create_timepoints(c,N,T):
    m=len(c)
    time_points=np.zeros((1,m*N))
    for j in range(m):
        time_points[0,j:m*N:m]=c[j]*1.0/N*np.ones((1,N))+np.linspace(0,1-1.0/N,N)
    return T*time_points

## Dispersive model taken from "Electromagnetic Scattering From Dispersive Dielectric Scatterers Using the Finite Difference Delay Modeling Method"

def mu_p(s):
    return 1.0
def eps_p(s):
    return 1.0

def mu_m(s):
    return 2.0
    #return 2.0
def eps_m(s):
    #return 2.0
    return 2.0+1.0/(1.0 + np.sqrt(s))
    #return 1.0+s**(-1)*(1.0/(1.0+np.exp(-s)))

def create_rhs(grid,dx,N,T,m):
#   grid=bempp.api.shapes.cube(h=1)
#
    OrderQF = 9
    #tol= np.finfo(float).eps
    bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
    bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
    bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
    bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
    bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
    bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
    bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
    bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
    bempp.api.global_parameters.quadrature.double_singular = OrderQF
    bempp.api.global_parameters.hmat.eps=10**-6
    bempp.api.global_parameters.hmat.admissibility='strong'
    
    if (m==1):
        c_RK=np.array([1])
    if (m==2):
        c_RK=np.array([1.0/3,1])
    if (m==3):
        c_RK=np.array([2.0/5-math.sqrt(6)/10,2.0/5+math.sqrt(6)/10,1])

    from bempp.api.operators.boundary import maxwell
    from bempp.api.operators.boundary import sparse
    multitrace = maxwell.multitrace_operator(grid, 1)
    NC_space = bempp.api.function_space(grid,"NC",0)
    RT_space = bempp.api.function_space(grid,"RT",0)
      
    dof=multitrace.range_spaces[0].global_dof_count
    print("Total degree of freedom is: ",dof)
    rhs=np.zeros((dof+dof,N*m))
    curls=np.zeros((dof,N*m))
    time_points=create_timepoints(c_RK,N,T)
    #rot = np.array([[1.0/np.sqrt(2),1.0/np.sqrt(2),0],[1.0/np.sqrt(2),-1.0/np.sqrt(2),0],[0,0,1]]) 
    for j in range(m*N):
        t=time_points[0,j]
        rot = np.array([[-1.0/np.sqrt(2),0 , - 1.0/np.sqrt(2)]
                       ,[0, 1,0]
                       ,[ -1.0/np.sqrt(2) ,0,1.0/np.sqrt(2)]])
        def func_Einc(x,n,domain_index,result):
            x    = rot.dot(x)
            Einc =rot.dot(np.array([  np.exp(-100*(x[2]+t-4)**2),   0. * x[2], 0. * x[2]]))
            #Einc = np.array([  np.sin(20*(x[2]+t-4))*np.exp(-2*(x[2]+t-4)**2),   0. * x[2], 0. * x[2]])    
            result[:] = np.cross(n,np.cross(Einc,n))

        def func_Hinc(x,n,domain_index,result):
            x    = rot.dot(x)
            Hinc =  -rot.dot(np.array([0. * x[2], np.exp(-100*(x[2]+t-4)**2), 0. * x[2]])    )
            #Hinc =  -np.array([0. * x[2], np.sin(20*(x[2]+t-4))*np.exp(-2*(x[2]+t-4)**2), 0. * x[2]])    
            result[:] = np.cross(n,np.cross(Hinc,n))
        #trace_fun= -bempp.api.GridFunction(RT_space, fun=tangential_trace,dual_space=NC_space)
        #E_fun= bempp.api.GridFunction(RT_space, fun=func_Einc,dual_space=NC_space)
        E_fun= bempp.api.GridFunction(RT_space, fun=func_Einc,dual_space=RT_space)
        H_fun= bempp.api.GridFunction(RT_space, fun=func_Hinc,dual_space=RT_space)
        rhs[0:dof,j]=E_fun.coefficients 
        rhs[dof:,j]=H_fun.coefficients 
#        curlCoeffs=curl_fun.coefficients
##        if np.linalg.norm(curlCoeffs)>10**-9:
#        curls[0:dof,j]=curlCoeffs
#
#        #print("RHS NORM :", np.linalg.norm(trace_fun.coefficients))
#
#    def sinv(s,b):
#        return mu_p(s)**(-1)*s**(-1)*b
#    IntegralOperator=Conv_Operator(sinv)
#    Hinc=IntegralOperator.apply_RKconvol(curls,T,method="RadauIIA-"+str(m),show_progress=False,first_value_is_t0=False)
#    rhs[dof:,:]= np.real(Hinc)
#   
    return rhs

def harmonic_calderon(s,b,grid):
    #points=np.array([[0],[0],[2]])
    #normb=np.linalg.norm(b[0])+np.linalg.norm(b[1])+np.linalg.norm(b[2])
    normb=np.max(np.abs(b))
    #bound=np.abs(s)**3*1.0/(s.real)*normb
    bound=np.abs(s)**3*np.exp(-s.real)*1.0/(s.real)*normb
    #print("s: ",s, " maxb: ", normb, " bound : ", bound)

    b = np.exp(-s)*b
    OrderQF = 9
    
    #tol= np.finfo(float).eps
    bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
    bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
    bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
    bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
    bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
    bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
    bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
    bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
    bempp.api.global_parameters.quadrature.double_singular = OrderQF
    bempp.api.global_parameters.hmat.eps=10**-7
    bempp.api.global_parameters.hmat.admissibility='strong'
###    Define Spaces
    NC_space=bempp.api.function_space(grid, "NC",0)
    RT_space=bempp.api.function_space(grid, "RT",0)
    #print("GLOBAL REFERENCE DOF: ", RT_space.global_dof_count) 
    alpha_s_p = np.emath.sqrt(mu_p(s)*eps_p(s))*s
    elec_p = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*alpha_s_p)
    magn_p = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*alpha_s_p)

    alpha_s_m = np.emath.sqrt(mu_m(s)*eps_m(s))*s
    elec_m = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*alpha_s_m)
    magn_m = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*alpha_s_m)



    dof=NC_space.global_dof_count
    if (bound <10**(-8)) or (normb < 10**(-6)):
        return np.zeros(dof*4)
    #trace_E= bempp.api.GridFunction(RT_space, coefficients=b[0:dof],dual_space=RT_space)
    #trace_H= bempp.api.GridFunction(RT_space, coefficients=b[dof:],dual_space=RT_space)
######## End condition, by theoretical bound:
    #normb=trace_E.l2_norm()
#    bound=np.abs(s)**3*np.exp(-2*s.real)*normb
#    if bound <10**(-5):
#    #    print("JUMPED")
#        return np.zeros(3)

    
    #rhs=[trace_fun,zero_fun]
    #b[0:dof]=id_discrete*b[0:dof]
    rhs = 1j*np.zeros((4*dof))
    b   = 0.5*b

    identity= bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
    #identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
    id_weak = identity.weak_form()
    rhs[0:dof]       = id_weak*(- b[0:dof])
    rhs[dof:2*dof]   = id_weak*(- b[dof:2*dof] )
    rhs[2*dof:3*dof] = id_weak*(  b[0:dof])
    rhs[3*dof:]      = id_weak*(  b[dof:2*dof] )
#    rhs[0:dof]       = id_weak*(-b[0:dof])
#    rhs[dof:2*dof]   = id_weak*(-b[dof:2*dof]            )
#    rhs[2*dof:3*dof] = id_weak*( b[0:dof])
#    rhs[3*dof:]      = id_weak*( b[dof:2*dof]            )
    blocks=np.array([[None,None,None,None], [None,None,None,None],[None,None,None,None],[None,None,None,None]])



    blocks[0,0] = -np.emath.sqrt(mu_p(s)/eps_p(s))*elec_p.weak_form()
    blocks[0,1] =  magn_p.weak_form()
    blocks[1,0] = -magn_p.weak_form()
    blocks[1,1] = -np.emath.sqrt(eps_p(s)/mu_p(s))*elec_p.weak_form()

    blocks[2,2] = -np.emath.sqrt(mu_m(s)/eps_m(s))*elec_m.weak_form()
    blocks[2,3] =  magn_m.weak_form()
    blocks[3,2] = -magn_m.weak_form()
    blocks[3,3] = -np.emath.sqrt(eps_m(s)/mu_m(s))*elec_m.weak_form()

    identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
    id_weak = identity.weak_form()
    blocks[0,3] = -0.5*id_weak
    blocks[1,2] =  0.5*id_weak
    blocks[2,1] =  0.5*id_weak
    blocks[3,0] = -0.5*id_weak


    blocks=bempp.api.BlockedDiscreteOperator(blocks)
 
    from scipy.sparse.linalg import gmres

    lambda_data,info = gmres(blocks, rhs,tol=10**-9)
    return lambda_data

def repr_formula(s,lambda_data,points,grid):
    RT_space=bempp.api.function_space(grid, "RT",0)
    dof=RT_space.global_dof_count
    #print("GLOBAL REFERENCE DOF: ", RT_space.global_dof_count) 
    alpha_s_p = np.emath.sqrt(mu_p(s)*eps_p(s))*s                           
    alpha_s_m = np.emath.sqrt(mu_m(s)*eps_m(s))*s
##### Outer evals:
    phigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[0:dof],dual_space=RT_space)
    psigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[dof:2*dof],dual_space=RT_space)

    slp_pot = bempp.api.operators.potential.maxwell.electric_field(RT_space, points, alpha_s_p*1j)
    dlp_pot = bempp.api.operators.potential.maxwell.magnetic_field(RT_space, points, alpha_s_p*1j)
    scattered_field_data_p = -np.emath.sqrt( mu_p(s)/eps_p(s))*(slp_pot*phigrid)+dlp_pot*psigrid
##### Inner evals:
    phigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[2*dof:3*dof],dual_space=RT_space)
    psigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[3*dof:4*dof],dual_space=RT_space)

    slp_pot = bempp.api.operators.potential.maxwell.electric_field(RT_space, points, alpha_s_m*1j)
    dlp_pot = bempp.api.operators.potential.maxwell.magnetic_field(RT_space, points, alpha_s_m*1j)
    scattered_field_data_m = -np.emath.sqrt( mu_m(s)/eps_m(s))*(slp_pot*phigrid)+dlp_pot*psigrid
    scattered_field_data =  scattered_field_data_p+scattered_field_data_m
    if np.isnan(scattered_field_data).any():
        print("NAN Warning, s = ", s)
        scattered_field_data=np.nan_to_num(scattered_field_data)
    return scattered_field_data.reshape(3*len(points[0,:]),1)[:,0]

def scattering_solution(dx,N,T,m,points):
    #grid=bempp.api.shapes.cube(h=dx)
    #grid=bempp.api.shapes.sphere(h=dx)
    gridfilename = "data/grids/two_cubes_h_"+str(np.round(dx,3))+".npy"
    grid = load_grid(gridfilename)
    #grid.plot()
    rhs=create_rhs(grid,dx,N,T,m)
    print("RHS completed.")
    def CaldSystem(s,b):
        return harmonic_calderon(s,b,grid)
    def repr_form(s,b):
        return repr_formula(s,b,points,grid)
    AOperator=Conv_Operator(CaldSystem)
    WOperator=Conv_Operator(repr_form) 
    filename= 'data/rml_densities_h_'+str(np.round(dx,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
    if (m==1):
        dens_sol_stages=AOperator.apply_RKconvol(rhs,T,show_progress=False,cutoff=10**(-7),method="RadauIIA-1",first_value_is_t0=False)
        num_solStages=WOperator.apply_RKconvol(dens_sol_stages,T,show_progress=False,cutoff=10**(-7),method="RadauIIA-1",first_value_is_t0=False)
    #num_sol=ScatOperator.apply_convol(rhs,T)
    if (m==2):
        dens_sol_stages=AOperator.apply_RKconvol(rhs,T,show_progress=False,cutoff=10**(-7),method="RadauIIA-2",first_value_is_t0=False)
        num_solStages=WOperator.apply_RKconvol(dens_sol_stages,T,show_progress=False,cutoff=10**(-7),method="RadauIIA-2",first_value_is_t0=False)
    if (m==3):
        dens_sol_stages=AOperator.apply_RKconvol(rhs,T,show_progress=True,cutoff=10**(-7),method="RadauIIA-3",first_value_is_t0=False)
        num_solStages=WOperator.apply_RKconvol(dens_sol_stages,T,show_progress=True,cutoff=10**(-7),method="RadauIIA-3",first_value_is_t0=False)
    save_densities(filename,dens_sol_stages,T,m,N)
    num_sol=np.zeros((len(num_solStages[:,0]),N+1)) 
    num_sol[:,1:N+1]=np.real(num_solStages[:,m-1:N*m:m])
    return num_sol

def compute_densities(dx,N,T,m,use_sphere=False):
    gridfilename = "data/grids/two_cubes_h_"+str(np.round(dx,3))+".npy"
    if use_sphere:
        gridfilename = "data/grids/sphere_python3_h"+str(np.round(dx,3))+".npy"
    grid = load_grid(gridfilename)
    rhs=create_rhs(grid,dx,N,T,m)
    def CaldSystem(s,b):
        return harmonic_calderon(s,b,grid)
    AOperator=Conv_Operator(CaldSystem)
    filename= 'data/rml_densities_h_'+str(np.round(dx,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
    if use_sphere:
        filename= 'data/rml_densities_sphere_h_'+str(np.round(dx,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
    dens_sol_stages=AOperator.apply_RKconvol(rhs,T,show_progress=False,cutoff=10**(-7),method="RadauIIA-"+str(m),first_value_is_t0=False)
    save_densities(filename,dens_sol_stages,T,m,N)
    print("Computed "+filename)
    #num_sol=np.zeros((len(num_solStages[:,0]),N+1)) 
    #num_sol[:,1:N+1]=np.real(num_solStages[:,m-1:N*m:m])
    return 0

def density2evals(dx,N,T,m,points,filename,use_sphere=False):
    gridfilename = "data/grids/two_cubes_h_"+str(np.round(dx,3))+".npy"
    if use_sphere:
        gridfilename = "data/grids/sphere_python3_h"+str(np.round(dx,3))+".npy"
    grid = load_grid(gridfilename)
    dens_sol_stages,T,m = extract_densities(filename)
    def repr_form(s,b):
        return repr_formula(s,b,points,grid)
    WOperator=Conv_Operator(repr_form) 
    num_solStages=WOperator.apply_RKconvol(dens_sol_stages,T,show_progress=False,cutoff=10**(-7),method="RadauIIA-"+str(m),first_value_is_t0=False)
    num_sol=np.zeros((len(num_solStages[:,0]),N+1)) 
    num_sol[:,1:N+1]=np.real(num_solStages[:,m-1:N*m:m])
    return num_sol
####
####
####T=6
####N_ref=2**11
#####N_ref=2**10
####tt_ref=np.linspace(0,T,N_ref+1)
#####dx_ref=np.sqrt(2)**(-0)
####dx_ref=np.sqrt(2)**(-10)
####m=3
####
#####import matplotlib.pyplot as plt
####start=time.time()
#####sol_ref=scattering_solution(dx_ref,N_ref,T)
#####sol_ref2=scattering_solution(dx_ref,N_ref,T)
#####
#####tt=np.linspace(0,T,N_ref+1)
#####plt.plot(tt,np.abs(sol_ref[0,:]))
#####plt.plot(tt,np.abs(sol_ref[0,:]-sol_ref2[0,:]))
######plt.plot(tt,resc_ref[0,:],linestyle='dashed') 
######plt.plot(tt,num_sol[0,:])
#####plt.show()
####
#####sol_ref = np.load("data/sol_ref_rml_N2h"+str(dx_ref)+"_dxsqrt2m7RK3.npy")
####print("Computing reference solution.")
####sol_ref=scattering_solution(dx_ref,N_ref,T,m)
####print("||sol_ref||_Linfty = ",np.max(np.abs(sol_ref)))
####
####np.save("data/sol_ref_rml_N2h11_dxsqrt2m9RK3.npy",sol_ref)
####
#####Current Reference solutions:
#####np.save("data/sol_ref_absorbing_delta0p1_N212_dxsqrt2m9RK5.npy",sol_ref)
####
####
#####np.save("data/sol_ref_absorbing_delta001_N212_dxsqrt2m9RK3.npy",sol_ref)
#####sol_ref=np.load("data/sol_ref_absorbing_delta1_N212_dxsqrt2m9RK3.npy")
#####sol_ref=np.load("data/sol_ref_absorbing_delta001_N212_dxsqrt2m9RK3.npy")
#####sol_ref=np.load("data/sol_ref_absorbing_N212_dxsqrt2m7RK5.npy")
#####import scipy.io
#####scipy.io.loadmat('data/Err_data_delta1.mat')
#####tt=np.linspace(0,T,N_ref+1)
#####plt.plot(tt,sol_ref[0,:])
#####plt.show()
####
#####plt.plot(sol_ref[0,:]**2+sol_ref[1,:]**2+sol_ref[2,:]**2)
#####plt.show()
####
####Am_space=9
####Am_time=9
#####Am_space=1
#####Am_time=8
####tau_s=np.zeros(Am_time)
####h_s=np.zeros(Am_space)
####errors=np.zeros((Am_space,Am_time))
####
####m=2
####for ixSpace in range(Am_space):
####    for ixTime in range(Am_time):
####        N=8*2**(2+ixTime)
####        tau_s[ixTime]=T*1.0/N
####        tt=np.linspace(0,T,N+1)
####        dx=np.sqrt(2)**(-(ixSpace))
####        h_s[ixSpace]=dx
####
############## Rescaling reference solution:        
####        speed=N_ref/N
####        resc_ref=np.zeros((3,N+1))
####    #   resc_ref=sol_ref
####        for j in range(N+1):
####            resc_ref[:,j]      = sol_ref[:,int(j*speed)]
####        #num_sol = calc_ref_sol(N,dx,F_transfer)    
####        num_sol  = scattering_solution(dx,N,T,m)
####    #   plt.plot(tt,num_sol[0,:]**2+num_sol[1,:]**2+num_sol[2,:]**2)
####    #   plt.plot(tt_ref,sol_ref[0,:]**2+sol_ref[1,:]**2+sol_ref[2,:]**2,linestyle='dashed')
#####       plt.show()
####    
####        errors[ixSpace,ixTime]=np.max(np.abs(resc_ref-num_sol))
####        print(errors)
####        import scipy.io
####        scipy.io.savemat('data/Err_data_h0.mat', dict( ERR=errors,h_s=h_s,tau_s=tau_s))
####        #scipy.io.savemat('data/Err_data_delta0p1_long.mat', dict( ERR=errors,h_s=h_s,tau_s=tau_s))
####end=time.time()
####print("Script Runtime: "+str((end-start)/60) +" Min")
####
