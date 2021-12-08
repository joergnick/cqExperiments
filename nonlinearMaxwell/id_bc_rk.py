import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')

import bempp.api
import numpy as np
import math
from linearcq import Conv_Operator
print("Bempp version used : " + bempp.api.__version__)
def create_timepoints(c,N,T):
    m=len(c)
    time_points=np.zeros((1,m*N))
    for j in range(m):
        time_points[0,j:m*N:m]=c[j]*1.0/N*np.ones((1,N))+np.linspace(0,1-1.0/N,N)
    return T*time_points

def create_rhs(grid,dx,N,T,m):
    OrderQF = 8
    bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
    bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
    bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
    bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
    bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
    bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
    bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
    bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
    bempp.api.global_parameters.quadrature.double_singular = OrderQF
    bempp.api.global_parameters.hmat.eps=10**-4
    bempp.api.global_parameters.hmat.admissibility='strong'

    if (m==2):
        c_RK=np.array([1.0/3,1])
    if (m==3):
        c_RK=np.array([2.0/5-math.sqrt(6)/10,2.0/5+math.sqrt(6)/10,1])

    from bempp.api.operators.boundary import maxwell
    from bempp.api.operators.boundary import sparse
    
    NC_space = bempp.api.function_space(grid,"NC",0)
    RT_space = bempp.api.function_space(grid,"RT",0)

    dof=RT_space.global_dof_count
    print(" DOF: ", dof)
    rhs=np.zeros((dof+dof,N*m))
    curls=np.zeros((dof,N*m))
    time_points=create_timepoints(c_RK,N,T)
    for j in range(m*N):
        t=time_points[0,j]
        def incident_field(x):
            return np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])
        def tangential_trace(x, n, domain_index, result):
            result[:] = np.cross(n,np.cross(incident_field(x), n))
        def curl_trace(x,n,domain_index,result):
            curlU=np.array([ 0. * x[2],-100*(x[2]-t+2)*np.exp(-50*(x[2]-t+2)**2), 0. * x[2]])
            result[:] = np.cross(curlU , n)

        curl_fun = bempp.api.GridFunction(RT_space, fun=curl_trace,dual_space=RT_space)
        trace_fun= bempp.api.GridFunction(RT_space, fun=tangential_trace,dual_space=RT_space)
        rhs[0:dof,j]=trace_fun.coefficients 
        curlCoeffs=curl_fun.coefficients
        if np.linalg.norm(curlCoeffs)>10**-9:
            curls[0:dof,j]=curlCoeffs

    def sinv(s,b):
        return s**(-1)*b
    IntegralOperator=Conv_Operator(sinv)
    def HarmonicImpedance(s,b):
        return 0.1*s**(0.5)*b
    TimeImpedance=Conv_Operator(HarmonicImpedance)  
    if (m==2):
        curls=IntegralOperator.apply_RKconvol(curls,T,method="RadauIIA-2",show_progress=False)
        ZptNeuTrace=TimeImpedance.apply_RKconvol(curls,T,method="RadauIIA-2",show_progress=False)
    if (m==3):
        curls=IntegralOperator.apply_RKconvol(curls,T,method="RadauIIA-3",show_progress=False)
        ZptNeuTrace=TimeImpedance.apply_RKconvol(curls,T,method="RadauIIA-3",show_progress=False)
    rhs[0:dof,:]=np.real(ZptNeuTrace)-rhs[0:dof,:]
    return rhs

def harmonic_calderon(s,b,grid):
    points=np.array([[0],[0],[2]])
    #normb=np.linalg.norm(b[0])+np.linalg.norm(b[1])+np.linalg.norm(b[2])
    normb=np.max(np.abs(b))
    bound=np.abs(s)**4*np.exp(-s.real)*normb
    OrderQF = 8
    
    bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
    bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
    bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
    bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
    bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
    bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
    bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
    bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
    bempp.api.global_parameters.quadrature.double_singular = OrderQF
    bempp.api.global_parameters.hmat.eps=10**-4
    bempp.api.global_parameters.hmat.admissibility='strong'
###    Define Spaces
    NC_space=bempp.api.function_space(grid, "NC",0)
    RT_space=bempp.api.function_space(grid, "RT",0)
        
    elec = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*s)
    magn = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*s)
    identity2=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
    identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
    dof=NC_space.global_dof_count
    trace_fun= bempp.api.GridFunction(RT_space, coefficients=b[0:dof],dual_space=RT_space)
######## End condition, by theoretical bound:
    normb=trace_fun.l2_norm()
    bound=np.abs(s)**3*np.exp(-s.real)*normb
    #if bound <10**(-8):
    #    print("JUMPED")
    #    return np.zeros(2*dof)
    id_discrete=identity2.weak_form()
    b[0:dof]=id_discrete*b[0:dof]

    blocks = np.array([[None,None], [None,None]])
    blocks[0,0] = -elec.weak_form()+identity2.weak_form()
    blocks[0,1] =  magn.weak_form()-1.0/2*identity.weak_form()
    blocks[1,0] = -magn.weak_form()-1.0/2*identity.weak_form()
    blocks[1,1] = -elec.weak_form()
    blocks = bempp.api.BlockedDiscreteOperator(blocks)
    from scipy.sparse.linalg import gmres
    print("Start GMRES : ")
    lambda_data,info = gmres(blocks, b,tol=10**-5,maxiter=300)
    print("INFO :", info)
    phigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[0:dof],dual_space=RT_space)
    psigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[dof:2*dof],dual_space=RT_space)

    slp_pot = bempp.api.operators.potential.maxwell.electric_field(RT_space, points, s*1j)
    dlp_pot = bempp.api.operators.potential.maxwell.magnetic_field(RT_space, points, s*1j)
    print("Evaluate field : ")  
    scattered_field_data = -slp_pot * phigrid+dlp_pot*psigrid
    if np.isnan(scattered_field_data).any():
        print("NAN Warning, s = ", s)
        scattered_field_data=np.zeros(np.shape(scattered_field_data))
    return lambda_data

def scattering_solution(gridfilename,dx,N,T,m):
    print('GRIDFILENAME: ',gridfilename)
    mat_contents = np.load(gridfilename).item()
    Nodes        = mat_contents['Nodes']
    Elements     = mat_contents['Elements']

    grid=bempp.api.grid_from_element_data(Nodes,Elements)
    rhs=create_rhs(grid,dx,N,T,m)
    def ellipticSystem(s,b):
        return harmonic_calderon(s,b,grid)
    ScatOperator=Conv_Operator(ellipticSystem)
    if (m==2):
        num_solStages=ScatOperator.apply_RKconvol(rhs,T,cutoff=10**(-5),method="RadauIIA-2")
    if (m==3):
        num_solStages=ScatOperator.apply_RKconvol(rhs,T,cutoff=10**(-5),method="RadauIIA-3")
    num_sol=np.zeros((len(num_solStages[:,0]),N+1)) 
    num_sol[:,1:N+1]=np.real(num_solStages[:,m-1:N*m:m])
    return num_sol
import time

#sol = scattering_solution(1,30,4,2)

T=6
#N_ref=2**4
N_ref=2**3
tt_ref=np.linspace(0,T,N_ref+1)
dx_ref=np.sqrt(2)**(0)
#dx_ref=np.sqrt(2)**(-9)
m=2

import matplotlib.pyplot as plt
start=time.time()
gridfilename = 'data/grids/sphereh1.0.npy'
sol_ref=scattering_solution(gridfilename,dx_ref,N_ref,T,m)
print(np.linalg.norm(sol_ref,axis = 0))
#np.save("data/sol_ref_absorbing_delta0p1_N2h11_dxsqrt2m9RK5.npy",sol_ref)
#Am_space=8
#Am_time=7
#tau_s=np.zeros(Am_time)
#h_s=np.zeros(Am_space)
#errors=np.zeros((Am_space,Am_time))
#m=2
#for ixSpace in range(Am_space):
#    for ixTime in range(Am_time):
#        N=8*2**(ixTime)
#        tau_s[ixTime]=T*1.0/N
#        tt=np.linspace(0,T,N+1)
#        dx=np.sqrt(2)**(-ixSpace)
#        h_s[ixSpace]=dx
########### Rescaling reference solution:        
#        speed=N_ref/N
#        resc_ref=np.zeros((3,N+1))
#    #   resc_ref=sol_ref
#        for j in range(N+1):
#            resc_ref[:,j]      = sol_ref[:,j*speed]
#        num_sol  = scattering_solution(dx,N,T,m)
#        errors[ixSpace,ixTime]=np.max(np.abs(resc_ref-num_sol))
#        print(errors)
#        import scipy.io
#        scipy.io.savemat('data/Err_data_delta01.mat', dict( ERR=errors,h_s=h_s,tau_s=tau_s))
#        #scipy.io.savemat('data/Err_data_delta0p1_long.mat', dict( ERR=errors,h_s=h_s,tau_s=tau_s))
#end=time.time()
#print("Script Runtime: "+str((end-start)/60) +" Min")