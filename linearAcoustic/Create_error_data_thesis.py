import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')
from linearcq import Conv_Operator
from rkmethods import RKMethod


import numpy as np
import scipy.io
import bempp.api
from acoustic_models import Incident_wave,spherical_Incident_wave

def create_rhs(N,T,dx,m):
    grid = bempp.api.shapes.sphere(h=dx)
    dp0_space = bempp.api.function_space(grid,"DP",0)
    p1_space  = bempp.api.function_space(grid,"P",1)
    rk = RKMethod("RadauIIA-"+str(m) ,T*1.0/N)    
    time_points = rk.get_time_points(T)
    dof0=dp0_space.global_dof_count
    dof1=p1_space.global_dof_count
    dof=dof0+dof1
    rhs=np.zeros((dof,m*N+1))

    u_inc=spherical_Incident_wave()
    ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
    u_incs=np.zeros((dof1,m*N+1))   
    pnu_incs=np.zeros((dof0,m*N+1))

    for j in range(0,m*N+1):
        #tj=j*T*1.0/N
        tj = time_points[j]
        
        def u_inc_fun(x,normal,domain_index,result):
            result[0]=u_inc.eval(tj,x)
        def u_neu_fun(x,normal,domain_index,result):
            result[0]=u_inc.eval_dnormal(tj,x,normal)

        gridfun_inc=bempp.api.GridFunction(p1_space,fun=u_inc_fun)
        gridfun_neu=bempp.api.GridFunction(dp0_space,fun=u_neu_fun)

        u_incs[:,j]=gridfun_inc.coefficients
        pnu_incs[:,j]=gridfun_neu.coefficients

        rhs[0:dof1,j]=gridfun_inc.coefficients
        rhs[dof1:dof0+dof1,j]=gridfun_neu.coefficients
    return rhs


def apply_elliptic_scat(s,b,F_transfer,dx):
    #print("In apply_elliptic")
    Points = np.array([[2],[0],[0]])
    grid = bempp.api.shapes.sphere(h=dx)
    OrderQF = 6
    #tol= np.finfo(float).eps
    bempp.api.global_parameters.hmat.eps=10**-4
    bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
    bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
    bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
    bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
    bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
    bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
    bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
    bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
    bempp.api.global_parameters.quadrature.double_singular = OrderQF

    #Define space
    dp0_space = bempp.api.function_space(grid,"DP",0)
    p1_space = bempp.api.function_space(grid, "P" ,1)

    dof0 = dp0_space.global_dof_count
    dof1 = p1_space.global_dof_count

    dof = dof0+dof1

    blocks = np.array([[None,None],[None,None]])
    ## Definition of Operators

#   slp = bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
#   dlp = bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,dp0_space,s)
#   adlp = bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,p1_space,s)
#   hslp = bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s,use_slp=True)
#
#   ident_0 = bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,p1_space)
#   ident_1 = bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,dp0_space)
#
#   ident_10 = bempp.api.operators.boundary.sparse.identity(p1_space,dp0_space,p1_space)
#   ident_00 = bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,p1_space)
    slp = bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,dp0_space,dp0_space,s)
    dlp = bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,dp0_space,dp0_space,s)
    adlp = bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,p1_space,p1_space,s)
    hslp = bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,p1_space,p1_space,s,use_slp=True)
    ident_0 = bempp.api.operators.boundary.sparse.identity(dp0_space,p1_space,p1_space)
    ident_1 = bempp.api.operators.boundary.sparse.identity(p1_space,dp0_space,dp0_space)
    ident_10 = bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
    ident_00 = bempp.api.operators.boundary.sparse.identity(dp0_space,p1_space,p1_space)


    #grid_ginc = grid_pnu - grid_Fsu
    ginc =ident_00.weak_form()*b[dof1:dof0+dof1] -ident_10.weak_form()*(s*F_transfer(s)*b[:dof1])
    rhs = 1j*np.zeros(dof0+dof1)
    rhs[dof0:dof0+dof1] = ginc
    ## Building Blocked System

    #blocks=bempp.api.BlockedOperator(2,2)
    blocks=np.array([[None,None], [None,None]])
    blocks[0,0] =(s*slp).weak_form()
    blocks[0,1] = (dlp.weak_form())-1.0/2*ident_1.weak_form()
    blocks[1,0] = -(adlp.weak_form())+1.0/2*ident_0.weak_form()
    blocks[1,1] = (1.0/s*hslp.weak_form()+F_transfer(s)*ident_10.weak_form())
    #blocks[1,1] = (1.0/s*hslp+s**(-1.0/2)*ident_1)
    #blocks[1,1] = 1.0/s*hslp   
    B_weak_form=bempp.api.BlockedDiscreteOperator(blocks)
    #B_weak_form=blocks
    #B_weak_form=bempp.api.BlockedDiscreteOperator(np.array(blocks))
    #print("Weak_form_assembled")
    #from bempp.api.linalg import gmres
    from scipy.sparse.linalg import gmres
    sol,info=gmres(B_weak_form,rhs,maxiter=300,tol=10**(-5))
    #sol=B_weak_form*[grid_rhs1,grid_rhs2]
    if info>0:
        #res=np.linalg.norm(B_weak_form*sol-b)
        print("Linear Algebra warning")

    #phi[:dof0]=sol[0].coefficients
    #phi[dof0:dof0+dof1]=sol[1].coefficients

    return sol

def pot_vals(s,phi,grid):
    OrderQF = 9
    #tol= np.finfo(float).eps
    bempp.api.global_parameters.hmat.eps=10**-4
    bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
    bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
    bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
    bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
    bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
    bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
    bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
    bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
    bempp.api.global_parameters.quadrature.double_singular = OrderQF

    Points = np.array([[2],[0],[0]])

    dp0_space = bempp.api.function_space(grid,"DP",0)
    p1_space = bempp.api.function_space(grid, "P" ,1)

    dof0=dp0_space.global_dof_count
    dof1=p1_space.global_dof_count
    dof=dof0+dof1
    slp_pot = bempp.api.operators.potential.modified_helmholtz.single_layer(dp0_space,Points,s)
    dlp_pot = bempp.api.operators.potential.modified_helmholtz.double_layer(p1_space, Points,s)
    
    varph=bempp.api.GridFunction(dp0_space,coefficients=phi[:dof0])
    psi=bempp.api.GridFunction(p1_space,coefficients=phi[dof0:dof])
    
    eval_slp=slp_pot*varph
    eval_dlp=dlp_pot*psi
    del slp_pot
    del dlp_pot
    evaluated_elli_sol=(eval_slp+s**(-1)*eval_dlp)
    return evaluated_elli_sol[0]

def calc_ref_sol_fast(N,F_transfer,m):
    import scipy.io
    #workspace=scipy.io.loadmat('data/Ref_spherical.mat')
    T=4
    tau=T*1.0/N
    rk = RKMethod("RadauIIA-"+str(m),tau)
    def combined_inverse(s,b):
        rhs=-s*F_transfer(s)*b[0]+b[1]
        phi=1j*np.zeros(2)
        #result[1]=(1+1.0/s)**(-1)*b[0]
        psi=(1+1.0/s+F_transfer(s))**(-1)*rhs
        uhat=s**(-1)*psi
        return [uhat]
    ons=np.ones(m*N+1)
    tt = rk.get_time_points(T)
    rhss=np.zeros((2,m*N+1))
    rhss[0,:]=np.exp(-5*(ons-(3*ons-tt))**2)
    rhss[1,:]=(19*ons-10*tt-10*ons)*rhss[0,:]

    RHS_to_DIR=Conv_Operator(combined_inverse)
    u_trace=RHS_to_DIR.apply_RKconvol(rhss,T,method="RadauIIA-"+str(m),show_progress=False,first_value_is_t0=True,cutoff=10**(-9))
    u_trace = u_trace[0][::m]
    u_P=np.zeros(N+1)
#Works only if 1/tau is natural 
    for j in range(0,N+1):
        tj=j*T*1.0/N
        t_transf=tj-1
        if t_transf>0:  
            u_P[j]=1.0/2*u_trace[int(t_transf*N/T)]
    return u_P

def scattering_solution(dx,N,F_transfer,m):
    T = 4
    grid=bempp.api.shapes.sphere(h=dx)
    def elli_pot_vals(s,b):
        return pot_vals(s,b,grid)
    Pot_time=Conv_Operator(elli_pot_vals)
    rhs = create_rhs(N,T,dx,m)
    def gibc_elli(s,b):
        return apply_elliptic_scat(s,b,F_transfer,dx)
    Scat_op = Conv_Operator(gibc_elli)
    psi_num = Scat_op.apply_RKconvol(rhs[:,1:],T,method="RadauIIA-"+str(m),show_progress=False,cutoff=10**(-6))
    num_sol = Pot_time.apply_RKconvol(psi_num,T,method="RadauIIA-"+str(m),show_progress=False,cutoff=10**(-6))
    return np.real(num_sol[0,:])

Am_time=7
Am_space=6

errors=np.zeros((Am_space+1,Am_time))
tau_s = np.zeros(Am_time)
h_s = np.zeros(Am_space)

### Calculating reference solution:

###########Define b.c. ################
def F_transfer(s):
###########Absorbing b.c. #############
    return 100*s**(-0.5)-s**(-1)
###########Homog. Neumann b.c. ########
#   return 0
###########Acoustic b.c. ##############
#   return (s+1+s**(-1))**(-1)
##
##N_ref=2**15
##dx_ref=0.005


N_ref=2**14
T=4
m=3
#dx_ref=2**(-6)
sol_ref = calc_ref_sol_fast(N_ref,F_transfer,m)
m=2
#sol_ref_bempp = calc_ref_sol(N_ref,dx_ref,F_transfer)
##def F2_transfer(s):
##  return 0
##
##sol_ref_neumann =calc_ref_sol(N_ref,dx_ref,F2_transfer)
##
#plt.plot(sol_ref,linestyle='dashed')
#plt.plot(sol_ref_bempp)
##plt.plot(sol_ref_neumann,linestyle='dashed')
#np.save("data/sol_ref_absorbinge2_N216.npy",sol_ref)
#sol_ref=np.load("data/sol_ref_absorbinge2_N216.npy")
import scipy.io
#mat_contents= scipy.io.loadmat('data/Err_200527_plot_data')
T=4
##errors=mat_contents['ERR']
##print(errors)
##h_s=mat_contents['h_s'][0]
##tau_s=mat_contents['tau_s'][0]
start_time=0
start_space=0
ttref=np.linspace(0,T,N_ref+1)
for ixTime in range(Am_time):
    N=4*2**(ixTime)
## Rescaling reference solution:        
    speed=N_ref/N
    resc_ref=np.zeros(N+1)
    for j in range(N+1):
        resc_ref[j]      = sol_ref[int(j*speed)]
    num_sol = calc_ref_sol_fast(N,F_transfer,m) 
    #num_sol  = scattering_solution(dx,N,F_transfer)
    num_sol = num_sol[::]
    errors[Am_space,ixTime]=max(np.abs(resc_ref-num_sol))
    print(errors)
    #plt.plot(ttref,sol_ref,linestyle='dashed')
    #plt.plot(np.linspace(0,T,N+1),num_sol)
#   plt.semilogy(np.linspace(0,4,N+1),np.abs(resc_ref-num_sol))

#   plt.show()
m=2
for ixSpace in range(start_space,Am_space):
    for ixTime in range(start_time,Am_time):
        N=8*2**(ixTime)
        tau_s[ixTime]=T*1.0/N
        tt=np.linspace(0,T,N+1)
        dx=2**(-ixSpace)
        h_s[ixSpace]=dx
########## Rescaling reference solution:        
        speed=N_ref/N
        resc_ref=np.zeros(N+1)
        for j in range(N+1):
            resc_ref[j]      = sol_ref[int(j*speed)]
        #num_sol = calc_ref_sol(N,dx,F_transfer)    
        num_sol  = scattering_solution(dx,N,F_transfer,m)
        num_sol = num_sol[m-1::m]
        errors[ixSpace,ixTime]=max(np.abs(resc_ref[1:]-num_sol))
        print(errors)
        #import matplotlib.pyplot
        #plt.plot(tt[1:],resc_ref[1:],linestyle='dashed')   
        #plt.plot(tt[1:],num_sol)
        #plt.show()
        #plt.semilogy(np.linspace(0,5,N+1),np.abs(resc_ref,num_sol))
        scipy.io.savemat('data/ERR_DATA_ACOUSTIC_dp0.mat', dict( ERR=errors,h_s=h_s,tau_s=tau_s))

