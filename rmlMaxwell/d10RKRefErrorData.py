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
import math
def create_timepoints(c,N,T):
    m=len(c)
    time_points=np.zeros((1,m*N))
    for j in range(m):
        time_points[0,j:m*N:m]=c[j]*1.0/N*np.ones((1,N))+np.linspace(0,1-1.0/N,N)
    return T*time_points
def mu_p(s):
    return 1
def eps_p(s):
    return 1
def mu_m(s):
    return 1.0/math.sqrt(2)
def eps_m(s):
    return 1.0/math.sqrt(2)
def create_rhs(grid,dx,N,T,m):
#   grid=bempp.api.shapes.cube(h=1)
#
    OrderQF = 8

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
    bempp.api.global_parameters.hmat.eps=10**-4

    bempp.api.global_parameters.hmat.admissibility='strong'

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
    rhs=np.zeros((dof+dof,N*m))
    curls=np.zeros((dof,N*m))
    time_points=create_timepoints(c_RK,N,T)
    for j in range(m*N):
        t=time_points[0,j]
        def incident_field(x):
            return np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])
            #return np.array([np.exp(-200*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])

        def tangential_trace(x, n, domain_index, result):
            result[:] = np.cross(n,np.cross(incident_field(x), n))

        def curl_trace(x,n,domain_index,result):
            curlU=np.array([ 0. * x[2],-100*(x[2]-t+2)*np.exp(-50*(x[2]-t+2)**2), 0. * x[2]])
            result[:] = np.cross(curlU , n)

        curl_fun = bempp.api.GridFunction(RT_space, fun=curl_trace,dual_space=RT_space)
        trace_fun= bempp.api.GridFunction(RT_space, fun=tangential_trace,dual_space=RT_space)
        rhs[0:dof,j]=trace_fun.coefficients 
        curlCoeffs=curl_fun.coefficients
#        if np.linalg.norm(curlCoeffs)>10**-9:
        curls[0:dof,j]=curlCoeffs

        #print("RHS NORM :", np.linalg.norm(trace_fun.coefficients))

    def sinv(s,b):
        return mu_p(s)**(-1)*s**(-1)*b
    IntegralOperator=Conv_Operator(sinv)
    Hinc=IntegralOperator.apply_RKconvol(curls,T,method="RadauIIA-"+str(m),show_progress=False,first_value_is_t0=False)
    rhs[dof:,:]= np.real(Hinc)
   
    return rhs

def harmonic_calderon(s,b,grid):
    points=np.array([[0],[0],[2]])
    #normb=np.linalg.norm(b[0])+np.linalg.norm(b[1])+np.linalg.norm(b[2])
    normb=np.max(np.abs(b))
    bound=np.abs(s)**3*np.exp(-s.real*2)*normb
    #print("s: ",s, " maxb: ", normb, " bound : ", bound)
    if bound <10**(-7):
        #print("JUMPED")
        return np.zeros(3)
    if normb <10**(-7):
        #print("JUMPED")
        return np.zeros(3)

    OrderQF = 7
    
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
    alpha_s_p = math.sqrt(mu_p(s)*eps_p(s))*s
    elec_p = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*alpha_s_p)
    magn_p = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*alpha_s_p)

    alpha_s_m = math.sqrt(mu_m(s)*eps_m(s))*s
    elec_m = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*alpha_s_m)
    magn_m = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*alpha_s_m)



    identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
    dof=NC_space.global_dof_count
    
    trace_E= bempp.api.GridFunction(RT_space, coefficients=b[0:dof],dual_space=RT_space)
    trace_H= bempp.api.GridFunction(RT_space, coefficients=b[dof:],dual_space=RT_space)
######## End condition, by theoretical bound:
    normb=trace_E.l2_norm()
    bound=np.abs(s)**3*np.exp(-2*s.real)*normb
    if bound <10**(-5):
    #    print("JUMPED")
        return np.zeros(3)

    
    #rhs=[trace_fun,zero_fun]
    #b[0:dof]=id_discrete*b[0:dof]
    rhs = 1j*np.zeros((4*dof))
    b   = 0.5*b

    identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
    id_weak = identity.weak_form()
    rhs[0:dof]       = id_weak*(-mu_p(s)*b[dof:2*dof])
    rhs[dof:2*dof]   = id_weak*(-b[0:dof]            )
    rhs[2*dof:3*dof] = id_weak*( mu_p(s)*b[dof:2*dof])
    rhs[3*dof:]      = id_weak*( b[0:dof]            )
    blocks=np.array([[None,None,None,None], [None,None,None,None],[None,None,None,None],[None,None,None,None]])



    blocks[0,0] = -1.0/math.sqrt(mu_p(s)*eps_p(s))*elec_p.weak_form()
    blocks[0,1] =  magn_p.weak_form()
    blocks[1,0] = -magn_p.weak_form()
    blocks[1,1] = -math.sqrt(mu_p(s)*eps_p(s))*elec_p.weak_form()

    blocks[2,2] = -1.0/math.sqrt(mu_m(s)*eps_m(s))*elec_m.weak_form()
    blocks[2,3] =  magn_m.weak_form()
    blocks[3,2] = -magn_m.weak_form()
    blocks[3,3] = -math.sqrt(mu_m(s)*eps_m(s))*elec_m.weak_form()

    blocks[0,3] = -0.5*identity.weak_form()
    blocks[1,2] =  0.5*identity.weak_form()
    blocks[2,1] =  0.5*identity.weak_form()
    blocks[3,0] = -0.5*identity.weak_form()


#
#    blocks[0,0] = -1.0/math.sqrt(mu_p(s)*eps_p(s))*elec_p.strong_form()
#    blocks[0,1] =  magn_p.strong_form()
#    blocks[1,0] = -magn_p.strong_form()
#    blocks[1,1] = -math.sqrt(mu_p(s)*eps_p(s))*elec_p.strong_form()
#
#    blocks[2,2] = -1.0/math.sqrt(mu_m(s)*eps_m(s))*elec_m.strong_form()
#    blocks[2,3] =  magn_m.strong_form()
#    blocks[3,2] = -magn_m.strong_form()
#    blocks[3,3] = -math.sqrt(mu_m(s)*eps_m(s))*elec_m.strong_form()
#
#    blocks[0,3] = -0.5*identity.strong_form()
#    blocks[1,2] =  0.5*identity.strong_form()
#    blocks[2,1] =  0.5*identity.strong_form()
#    blocks[3,0] = -0.5*identity.strong_form()
#
#    blocks[0,0] = -elec_p.weak_form()
#    blocks[0,1] =  magn_p.weak_form()
#    blocks[1,0] = -magn_p.weak_form()
#    blocks[1,1] = -elec_p.weak_form()
#
#    blocks[2,2] = -elec_m.weak_form()
#    blocks[2,3] =  magn_m.weak_form()
#    blocks[3,2] = -magn_m.weak_form()
#    blocks[3,3] = -elec_m.weak_form()
#
#    blocks[0,3] = -0.5*identity.weak_form()
#    blocks[1,2] =  0.5*identity.weak_form()
#    blocks[2,1] =  0.5*identity.weak_form()
#    blocks[3,0] = -0.5*identity.weak_form()
#


    blocks=bempp.api.BlockedDiscreteOperator(blocks)
#    A_mat=bempp.api.as_matrix(blocks)
#    print("A_mat : ",A_mat)
#    e,D=np.linalg.eig(A_mat)
#    print("Eigs : ", e)
#    print("Cond : ", np.linalg.cond(A_mat))
##
##  trace_fun= bempp.api.GridFunction(multitrace.range_spaces[0], coefficients=b[0:dof],dual_space=multitrace.dual_to_range_spaces[0])
##
##  zero_fun= bempp.api.GridFunction(multitrace.range_spaces[1],coefficients = b[dof:],dual_space=multitrace.dual_to_range_spaces[1])
##
##  rhs=[trace_fun,zero_fun]
##
##  #print("Still living")
##  
    #from bempp.api.linalg import gmres 
    from scipy.sparse.linalg import gmres
    #print("Start GMRES : ")
#   def print_res(rk):
#       print("Norm of residual: "+ str(np.linalg.norm(rk)))
    #print(np.linalg.norm(lambda_data))
    #lambda_data,info = gmres(blocks, b,tol=10**-4,restart=50,maxiter=100,callback=print_res)
    #lambda_data,info = gmres(blocks, rhs,tol=10**-9,maxiter=100)
    lambda_data,info = gmres(blocks, rhs,tol=10**-9)
    #print("INFO :", info, " ||phi+|| = ",np.linalg.norm(lambda_data[0:2*dof]))

    phigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[0:dof],dual_space=RT_space)
    psigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[dof:2*dof],dual_space=RT_space)

    slp_pot = bempp.api.operators.potential.maxwell.electric_field(RT_space, points, alpha_s_p*1j)
    dlp_pot = bempp.api.operators.potential.maxwell.magnetic_field(RT_space, points, alpha_s_p*1j)
    scattered_field_data = -1.0/math.sqrt(eps_p(s)*mu_p(s))*(slp_pot*phigrid)+dlp_pot*psigrid

    #print("Evaluated field, || E+|| = ",np.linalg.norm(scattered_field_data))  
    if np.isnan(scattered_field_data).any():
        print("NAN Warning, s = ", s)
        scattered_field_data=np.zeros(np.shape(scattered_field_data))
    return scattered_field_data.reshape(3,1)[:,0]

def scattering_solution(dx,N,T,m):
    grid=bempp.api.shapes.sphere(h=dx)
    rhs=create_rhs(grid,dx,N,T,m)
    def ellipticSystem(s,b):
        return harmonic_calderon(s,b,grid)
    ScatOperator=Conv_Operator(ellipticSystem)
    #num_sol=ScatOperator.apply_convol(rhs,T)
    if (m==2):
        num_solStages=ScatOperator.apply_RKconvol(rhs,T,show_progress=False,cutoff=10**(-7),method="RadauIIA-2",first_value_is_t0=False)
    if (m==3):
        num_solStages=ScatOperator.apply_RKconvol(rhs,T,show_progress=True,cutoff=10**(-7),method="RadauIIA-3",first_value_is_t0=False)
    num_sol=np.zeros((len(num_solStages[:,0]),N+1)) 
    num_sol[:,1:N+1]=np.real(num_solStages[:,m-1:N*m:m])
    return num_sol
import time


T=6
N_ref=2**11
#N_ref=2**10
tt_ref=np.linspace(0,T,N_ref+1)
#dx_ref=np.sqrt(2)**(-0)
dx_ref=np.sqrt(2)**(-10)
m=3

#import matplotlib.pyplot as plt
start=time.time()
#sol_ref=scattering_solution(dx_ref,N_ref,T)
#sol_ref2=scattering_solution(dx_ref,N_ref,T)
#
#tt=np.linspace(0,T,N_ref+1)
#plt.plot(tt,np.abs(sol_ref[0,:]))
#plt.plot(tt,np.abs(sol_ref[0,:]-sol_ref2[0,:]))
##plt.plot(tt,resc_ref[0,:],linestyle='dashed') 
##plt.plot(tt,num_sol[0,:])
#plt.show()

#sol_ref = np.load("data/sol_ref_rml_N2h"+str(dx_ref)+"_dxsqrt2m7RK3.npy")
print("Computing reference solution.")
sol_ref=scattering_solution(dx_ref,N_ref,T,m)
print("||sol_ref||_Linfty = ",np.max(np.abs(sol_ref)))

np.save("data/sol_ref_rml_N2h11_dxsqrt2m9RK3.npy",sol_ref)

#Current Reference solutions:
#np.save("data/sol_ref_absorbing_delta0p1_N212_dxsqrt2m9RK5.npy",sol_ref)


#np.save("data/sol_ref_absorbing_delta001_N212_dxsqrt2m9RK3.npy",sol_ref)
#sol_ref=np.load("data/sol_ref_absorbing_delta1_N212_dxsqrt2m9RK3.npy")
#sol_ref=np.load("data/sol_ref_absorbing_delta001_N212_dxsqrt2m9RK3.npy")
#sol_ref=np.load("data/sol_ref_absorbing_N212_dxsqrt2m7RK5.npy")
#import scipy.io
#scipy.io.loadmat('data/Err_data_delta1.mat')
#tt=np.linspace(0,T,N_ref+1)
#plt.plot(tt,sol_ref[0,:])
#plt.show()

#plt.plot(sol_ref[0,:]**2+sol_ref[1,:]**2+sol_ref[2,:]**2)
#plt.show()

Am_space=9
Am_time=9
#Am_space=1
#Am_time=8
tau_s=np.zeros(Am_time)
h_s=np.zeros(Am_space)
errors=np.zeros((Am_space,Am_time))

m=2
for ixSpace in range(Am_space):
    for ixTime in range(Am_time):
        N=8*2**(2+ixTime)
        tau_s[ixTime]=T*1.0/N
        tt=np.linspace(0,T,N+1)
        dx=np.sqrt(2)**(-(ixSpace))
        h_s[ixSpace]=dx

########## Rescaling reference solution:        
        speed=N_ref/N
        resc_ref=np.zeros((3,N+1))
    #   resc_ref=sol_ref
        for j in range(N+1):
            resc_ref[:,j]      = sol_ref[:,int(j*speed)]
        #num_sol = calc_ref_sol(N,dx,F_transfer)    
        num_sol  = scattering_solution(dx,N,T,m)
    #   plt.plot(tt,num_sol[0,:]**2+num_sol[1,:]**2+num_sol[2,:]**2)
    #   plt.plot(tt_ref,sol_ref[0,:]**2+sol_ref[1,:]**2+sol_ref[2,:]**2,linestyle='dashed')
#       plt.show()
    
        errors[ixSpace,ixTime]=np.max(np.abs(resc_ref-num_sol))
        print(errors)
        import scipy.io
        scipy.io.savemat('data/Err_data_h0.mat', dict( ERR=errors,h_s=h_s,tau_s=tau_s))
        #scipy.io.savemat('data/Err_data_delta0p1_long.mat', dict( ERR=errors,h_s=h_s,tau_s=tau_s))
end=time.time()
print("Script Runtime: "+str((end-start)/60) +" Min")

