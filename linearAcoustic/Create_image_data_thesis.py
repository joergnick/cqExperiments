import sys,os
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

#from conv_op import *
from acoustic_models import Incident_wave,spherical_Incident_wave

def create_rhs(N,T,m):
    grid=bempp.api.import_grid('data/grids/magnet_h05_h01.msh')
    dp0_space = bempp.api.function_space(grid,"DP",0)
    p1_space  = bempp.api.function_space(grid,"P",1)
    rk = RKMethod("RadauIIA-"+str(m) ,T*1.0/N)    
    time_points = rk.get_time_points(T)
    dof0=dp0_space.global_dof_count
    dof1=p1_space.global_dof_count
    dof=dof0+dof1
    rhs=np.zeros((dof,m*N+1))
    u_inc=Incident_wave(-100,np.array([0,-1.0,0]),-3)
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

def apply_elliptic_scat(s,b,F_transfer,grid):
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

    slp = bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
    dlp = bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,dp0_space,s)
    adlp = bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,p1_space,s)
    hslp = bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s,use_slp=True)

    ident_0 = bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,p1_space)
    ident_1 = bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,dp0_space)

    ident_10 = bempp.api.operators.boundary.sparse.identity(p1_space,dp0_space,p1_space)
    ident_00 = bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,p1_space)
    
    ## Bringing RHS into GridFuncion - type ;
    #Fsu = s*F_transfer(s)*b[:dof0]
    #grid_Fsu = bempp.api.GridFunction(dp0_space,coefficients=Fsu)
    #grid_pnu = bempp.api.GridFunction(dp0_space,coefficients=b[dof0:dof])
    

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

def apply_elliptic_scat_thin_layer(s,b,delta,grid):
    #print("In apply_elliptic")
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
    blocks = np.array([[None,None],[None,None]])
    ## Definition of Operators
    slp = bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
    dlp = bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,dp0_space,s)
    adlp = bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,p1_space,s)
    hslp = bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s,use_slp=True)
    ident_0 = bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,p1_space)
    ident_1 = bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,dp0_space)
    ident_10 = bempp.api.operators.boundary.sparse.identity(p1_space,dp0_space,p1_space)
    ident_00 = bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,p1_space)
    lb=bempp.api.operators.boundary.sparse.laplace_beltrami(p1_space,dp0_space,p1_space)
    ## Bringing RHS into GridFuncion - type ;
    #Fsu = s*F_transfer(s)*b[:dof0]
    #grid_Fsu = bempp.api.GridFunction(dp0_space,coefficients=Fsu)
    #grid_pnu = bempp.api.GridFunction(dp0_space,coefficients=b[dof0:dof])
    #grid_ginc = grid_pnu - grid_Fsu
    ginc =ident_00.weak_form()*b[dof1:dof0+dof1] -delta*(s*ident_10.weak_form()+1.0/s*lb.weak_form())*b[:dof1]
    #ginc =ident_00.weak_form()*b[dof1:dof0+dof1] -ident_10.weak_form()*(s*F_transfer(s)*b[:dof1])
    rhs = 1j*np.zeros(dof0+dof1)
    rhs[dof0:dof0+dof1] = ginc
    ## Building Blocked System

    #blocks=bempp.api.BlockedOperator(2,2)
    blocks=np.array([[None,None], [None,None]])
    blocks[0,0] =(s*slp).weak_form()
    blocks[0,1] = (dlp.weak_form())-1.0/2*ident_1.weak_form()
    blocks[1,0] = -(adlp.weak_form())+1.0/2*ident_0.weak_form()
    blocks[1,1] = (1.0/s*hslp.weak_form()+delta*1.0/s*lb.weak_form()+delta*s*ident_10.weak_form())
    B_weak_form=bempp.api.BlockedDiscreteOperator(blocks)
    from scipy.sparse.linalg import gmres
    sol,info=gmres(B_weak_form,rhs,maxiter=300,tol=10**(-5))
    #sol=B_weak_form*[grid_rhs1,grid_rhs2]
    if info>0:
        #res=np.linalg.norm(B_weak_form*sol-b)
        print("Linear Algebra warning")
    return sol
def pot_vals(s,phi,grid,Points):
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

def scattering_solution(N,T,F_transfer,m,delta=False):
    n_grid_points=200
    ########DRAFT MAGNET PICTURE DATA:
    x_a=-0.75
    x_b=0.75
    y_a=-0.25
    y_b=1.25
###############################################
    plot_grid = np.mgrid[y_a:y_b:1j*n_grid_points, x_a:x_b:1j*n_grid_points]
    Points = np.vstack( ( plot_grid[0].ravel() , plot_grid[1].ravel() , 0.25*np.ones(plot_grid[0].size) ) )
    
    #grid=bempp.api.import_grid('data/grids/magnet_h05_h01.msh')
    grid=bempp.api.import_grid('data/grids/magnet_h05_h01.msh')
    def elli_pot_vals(s,b):
        return pot_vals(s,b,grid,Points)
    Pot_time=Conv_Operator(elli_pot_vals)
    rhs = create_rhs(N,T,m)
    if delta:
        def gibc_elli(s,b):
            return apply_elliptic_scat(s,b,F_transfer,grid)
    else:
        def gibc_elli(s,b):
            return apply_elliptic_scat_thin_layer(s,b,delta,grid)
    Scat_op = Conv_Operator(gibc_elli)
    psi_num = Scat_op.apply_RKconvol(rhs[:,1:],T,method="RadauIIA-"+str(m),show_progress=False,cutoff=10**(-6))
    num_sol = Pot_time.apply_RKconvol(psi_num,T,method="RadauIIA-"+str(m),show_progress=False,cutoff=10**(-6))
    u_eval= np.real(num_sol[:,::m])

    u_ges=np.zeros((n_grid_points**2,N))
    for indt in range(0,int(N)):
        u_tp=u_eval[:,indt]
        uinc=np.zeros(n_grid_points**2)
        uinc_wave=Incident_wave(-100,np.array([0,-1.0,0]),-3)
        for k in range(n_grid_points**2):
            uinc[k]=uinc_wave.eval(T*indt*1.0/N,Points[:,k])
        u_ges[:,indt]=u_tp+uinc
    return u_ges,plot_grid,Points

N=200
m=3
T=5

boundary_cond="GIBC"
F_transfer=None
delta=0.1
u_ges,plot_grid,Points = scattering_solution(N,T,F_transfer,m,delta=False)
scipy.io.savemat('data/'+boundary_cond+'.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,Points=Points))

boundary_cond="Absorbing"
def F_transfer(s):
###########Absorbing b.c. #############
    return s**(1.0/2)/0.1
u_ges,plot_grid,Points = scattering_solution(N,T,F_transfer,m,delta=False)
scipy.io.savemat('data/'+boundary_cond+'.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,Points=Points))

boundary_cond="Acoustic"
def F_transfer(s):
###########Acoustic b.c. #############
    return (s+1+s**(-1))**(-1)*s
u_ges,plot_grid,Points = scattering_solution(N,T,F_transfer,m,delta=False)
scipy.io.savemat('data/'+boundary_cond+'.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,Points=Points))

