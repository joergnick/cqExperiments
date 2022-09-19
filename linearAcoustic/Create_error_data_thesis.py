import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')
from linearcq import Conv_Operator
from rkmethods import RKMethod


import numpy as np
import bempp.api

#from conv_op import *
from acoustic_models import Incident_wave,spherical_Incident_wave
import matplotlib.pyplot as plt

def create_timepoints(c,N,T):
    m=len(c)
    time_points=np.zeros((1,m*N))
    for j in range(m):
        time_points[0,j:m*N:m]=c[j]*1.0/N*np.ones((1,N))+np.linspace(0,1-1.0/N,N)
    return T*time_points
def create_rhs(N,T,dx,m):
	grid = bempp.api.shapes.sphere(h=dx)
	dp0_space = bempp.api.function_space(grid,"P",1)
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
	pnu_incs=np.zeros((dof1,m*N+1))

	for j in range(0,m*N+1):
		#tj=j*T*1.0/N
		tj = time_points[j]
		
		def u_inc_fun(x,normal,domain_index,result):
			result[0]=u_inc.eval(tj,x)
		def u_neu_fun(x,normal,domain_index,result):
			result[0]=u_inc.eval_dnormal(tj,x,normal)

		gridfun_inc=bempp.api.GridFunction(p1_space,fun=u_inc_fun)
		gridfun_neu=bempp.api.GridFunction(p1_space,fun=u_neu_fun)

		u_incs[:,j]=gridfun_inc.coefficients
		pnu_incs[:,j]=gridfun_neu.coefficients

		rhs[0:dof0,j]=gridfun_inc.coefficients
		rhs[dof0:dof0+dof1,j]=gridfun_neu.coefficients
	return rhs


def apply_elliptic_scat(s,b,F_transfer,dx):
	#print("In apply_elliptic")
	Points = np.array([[2],[0],[0]])
	order = 2

	grid = bempp.api.shapes.sphere(h=dx)

	OrderQF = 5
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
	dp0_space = bempp.api.function_space(grid,"P",1)
	p1_space = bempp.api.function_space(grid, "P" ,1)

	dof0 = dp0_space.global_dof_count
	dof1 = p1_space.global_dof_count

	dof = dof0+dof1

	blocks = np.array([[None,None],[None,None]])
	## Definition of Operators

	slp = bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
	dlp = bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,p1_space,s)
	adlp = bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,dp0_space,s)
	hslp = bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s,use_slp=True)

	ident_0 = bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,dp0_space)
	ident_1 = bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
	ident_10 = bempp.api.operators.boundary.sparse.identity(p1_space,dp0_space,p1_space)
	
	## Bringing RHS into GridFuncion - type ;
	Fsu = s*F_transfer(s)*b[:dof0]
	grid_Fsu = bempp.api.GridFunction(dp0_space,coefficients=Fsu)
	grid_pnu = bempp.api.GridFunction(p1_space,coefficients=b[dof0:dof])
	

	grid_ginc = grid_pnu - grid_Fsu

	## Building Blocked System

	blocks=bempp.api.BlockedOperator(2,2)

	blocks[0,0] =(s*slp)
	blocks[0,1] = (dlp)-1.0/2*ident_1
	blocks[1,0] = -(adlp)+1.0/2*ident_0
	blocks[1,1] = (1.0/s*hslp+F_transfer(s)*ident_1)
	#blocks[1,1] = (1.0/s*hslp+s**(-1.0/2)*ident_1)
	#blocks[1,1] = 1.0/s*hslp	

	B_weak_form=blocks
	#B_weak_form=bempp.api.BlockedDiscreteOperator(np.array(blocks))
	#print("Weak_form_assembled")
	from bempp.api.linalg import gmres
	#from scipy.sparse.linalg import gmres
	sol,info=gmres(B_weak_form,[0*grid_ginc,grid_ginc],maxiter=300,tol=10**(-5))
	#sol=B_weak_form*[grid_rhs1,grid_rhs2]
	if info>0:
		#res=np.linalg.norm(B_weak_form*sol-b)
		print("Linear Algebra warning")

	phi=1j*np.zeros(dof0+dof1)

	phi[:dof0]=sol[0].coefficients
	phi[dof0:dof0+dof1]=sol[1].coefficients

	return phi

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

	dp0_space = bempp.api.function_space(grid,"P",1)
	p1_space = bempp.api.function_space(grid, "P" ,1)

	dof0=dp0_space.global_dof_count
	dof=2*dof0
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


def calc_ref_sol(N,dx,F_transfer,m):
	import scipy.io
	#workspace=scipy.io.loadmat('data/Ref_spherical.mat')
	T=4
	grid = bempp.api.shapes.sphere(h=dx)

	dp0_space = bempp.api.function_space(grid,"P",1)
	p1_space  = bempp.api.function_space(grid, "P" ,1)


	def combined_inverse(s,b):
		
		rhs=-s*F_transfer(s)*b[0]+b[1]
		phi=1j*np.zeros(2)
		#result[1]=(1+1.0/s)**(-1)*b[0]
		phi[1]=(1+1.0/s+F_transfer(s))**(-1)*rhs
		phi[0]=(1+s)**(1)*s**(-1)*phi[1]

		dof      = p1_space.global_dof_count
		phi_space  =1j*np.zeros(2*dof)+ np.ones(2*dof)
	
		phi_space[:dof] = phi[0]*phi_space[:dof]
		phi_space[dof:2*dof] = phi[1]*phi_space[dof:2*dof]

		eval_Point=pot_vals(s,phi_space,grid)
		return eval_Point
	
	ons=np.ones(N+1)
	tt=np.linspace(0,T,N+1)
	rhss=np.zeros((2,N+1))
	rhss[0,:]=np.exp(-5*(ons-(3*ons-tt))**2)
	rhss[1,:]=(19*ons-10*tt-10*ons)*rhss[0,:]

#	def Fs_transfer(s,b):
#		return s*F_transfer(s)*b

##	ptFpt=Conv_Operator(Fs_transfer)
##	
##	rhs=-ptFpt.apply_convol(u,T,show_progress=False)+pnu
	
	PotmAptm1=Conv_Operator(combined_inverse)
	sol_ref=PotmAptm1.apply_RKconvol(rhss,T,method="RadauIIA-"+str(m),show_progress=True,cutoff=10**(-8))
	#ref_sol=workspace['ref_sol'][0]                           #
	#N_ref=len(ref_sol)-1                                      #

####
####	grid = bempp.api.shapes.sphere(h=dx)
####
####	dp0_space = bempp.api.function_space(grid,"P",1)
####	p1_space  = bempp.api.function_space(grid, "P" ,1)
####
####	dof      = p1_space.global_dof_count
####	psi_ref  = np.ones((2*dof,N_ref+1))
####
####	psi_ref[:dof,:] = ref_dens[0,:]*psi_ref[:dof,:]
####	psi_ref[dof:2*dof,:] = ref_dens[1,:]*psi_ref[dof:2*dof,:]
######	for j in range(N+1):
######		psi_ref[0:dof,j]     = ref_dens[0,j]*psi_ref[0:dof,j]
######		psi_ref[dof:2*dof,j] = ref_dens[1,j]*psi_ref[dof:2*dof,j]
####
####
####	def elli_pot_vals(s,b):
####		return pot_vals(s,b,dx)
####	Pot_time=Conv_Operator(elli_pot_vals)
####	
#####	sol_ref = Pot_time.apply_convol(psi_ref,T,show_progress=True)
####
######		psi[0:dof,j]     = ref_sol[0,j*speed]*psi[0:dof,j]
######		psi[dof:2*dof,j] = ref_sol[1,j*speed]*psi[dof:2*dof,j]
######		pnu_resc[j]      = pnu[j*speed]

	return sol_ref


def calc_ref_sol_fast(N,F_transfer,m):
	import scipy.io
	#workspace=scipy.io.loadmat('data/Ref_spherical.mat')

	Point = np.array([[2],[0],[0]])
	r=2
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
	tt=np.linspace(0,T,m*N+1)
	rhss=np.zeros((2,m*N+1))
	rhss[0,:]=np.exp(-5*(ons-(3*ons-tt))**2)
	rhss[1,:]=(19*ons-10*tt-10*ons)*rhss[0,:]
#	def Fs_transfer(s,b):
#		return s*F_transfer(s)*b
##	ptFpt=Conv_Operator(Fs_transfer)
##	
##	rhs=-ptFpt.apply_convol(u,T,show_progress=False)+pnu
	RHS_to_DIR=Conv_Operator(combined_inverse)
	u_trace=RHS_to_DIR.apply_RKconvol(rhss,T,method="RadauIIA-"+str(m),show_progress=True,cutoff=10**(-7))
	u_P=np.zeros(m*N+1)
#Works only if 1/tau is natural	
	tpts = rk.get_time_points(T)
	for j in range(0,m*N+1):
		#tj=j*T*1.0/N
		tj = tpts[j]
		t_transf=tj-1
		if t_transf>0:	
			#u(x,t)= 1/R*u(y,t-(R-1))
			u_P[j]=1.0/2*u_trace[0][t_transf*N/T]
	#ref_sol=workspace['ref_sol'][0]                           #
	#N_ref=len(ref_sol)-1                                      #

####
####	grid = bempp.api.shapes.sphere(h=dx)
####
####	dp0_space = bempp.api.function_space(grid,"P",1)
####	p1_space  = bempp.api.function_space(grid, "P" ,1)
####
####	dof      = p1_space.global_dof_count
####	psi_ref  = np.ones((2*dof,N_ref+1))
####
####	psi_ref[:dof,:] = ref_dens[0,:]*psi_ref[:dof,:]
####	psi_ref[dof:2*dof,:] = ref_dens[1,:]*psi_ref[dof:2*dof,:]
######	for j in range(N+1):
######		psi_ref[0:dof,j]     = ref_dens[0,j]*psi_ref[0:dof,j]
######		psi_ref[dof:2*dof,j] = ref_dens[1,j]*psi_ref[dof:2*dof,j]
####
####
####	def elli_pot_vals(s,b):
####		return pot_vals(s,b,dx)
####	Pot_time=Conv_Operator(elli_pot_vals)
####	
#####	sol_ref = Pot_time.apply_convol(psi_ref,T,show_progress=True)
####
######		psi[0:dof,j]     = ref_sol[0,j*speed]*psi[0:dof,j]
######		psi[dof:2*dof,j] = ref_sol[1,j*speed]*psi[dof:2*dof,j]
######		pnu_resc[j]      = pnu[j*speed]
	return u_P


def scattering_solution(dx,N,F_transfer,m):
###########Calculate reference Solution###########
	T = 4
##################Calculate numerical solution##########
	grid=bempp.api.shapes.sphere(h=dx)
	def elli_pot_vals(s,b):
		return pot_vals(s,b,grid)
	
	Pot_time=Conv_Operator(elli_pot_vals)
	rhs = create_rhs(N,T,dx,m)
	
	def gibc_elli(s,b):
		return apply_elliptic_scat(s,b,F_transfer,dx)
	
	Scat_op = Conv_Operator(gibc_elli)
	psi_num = Scat_op.apply_RKconvol(rhs,T,method="RadauIIA-"+str(m),cutoff=10**(-6))
	num_sol = Pot_time.apply_RKconvol(psi_num,T,method="RadauIIA-"+str(m),cutoff=10**(-6))
	####################################################
	
	#Calc Reference for Neumann- b.c.
	#tt=np.linspace(0,T,N+1)
	#Time_ref=np.zeros(N+1)
	#u_inc=spherical_Incident_wave()
	#for j in range(N+1):
	#	Time_ref[j]=u_inc.eval((j*T*1.0/N)-2,Point)
#	#print(time_evalsp[0])

#	plt.plot(tt,sol_num)
#	plt.plot(tt,sol_ref,linestyle='dashed')
	return num_sol
	
#	plt.ylim((-0.5,0.5))
		
		
#Am_time=6
#Am_space=2


Am_time=8
Am_time=3
Am_space=5

errors=np.zeros((Am_space+1,Am_time))
tau_s = np.zeros(Am_time)
h_s = np.zeros(Am_space)

### Calculating reference solution:

###########Define b.c. ################
def F_transfer(s):
###########Absorbing b.c. #############
	return 100*s**(-0.5)-s**(-1)
###########Homog. Neumann b.c. ########
#	return 0
###########Acoustic b.c. ##############
#	return (s+1+s**(-1))**(-1)
##
##N_ref=2**15
##dx_ref=0.005


N_ref=2**10
T=4
m=2
#dx_ref=2**(-6)
sol_ref = calc_ref_sol_fast(N_ref,F_transfer,m)
sol_ref = sol_ref[::m]
#sol_ref_bempp = calc_ref_sol(N_ref,dx_ref,F_transfer)
##def F2_transfer(s):
##	return 0
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
ttref=np.linspace(0,4,N_ref+1)
for ixTime in range(Am_time):
	N=32*2**(ixTime)
	tt=np.linspace(0,5,N+1)

## Rescaling reference solution:		
	speed=N_ref/N
	resc_ref=np.zeros(N+1)
	for j in range(N+1):
		resc_ref[j]      = sol_ref[j*speed]
	num_sol = calc_ref_sol_fast(N,F_transfer,m)	
	#num_sol  = scattering_solution(dx,N,F_transfer)

	num_sol = num_sol[::m]
	errors[Am_space,ixTime]=max(np.abs(resc_ref-num_sol))
	print(errors)
	#plt.plot(ttref,sol_ref,linestyle='dashed')
	#plt.plot(np.linspace(0,T,N+1),num_sol)
#	plt.semilogy(np.linspace(0,4,N+1),np.abs(resc_ref-num_sol))

#	plt.show()
m = 2
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
			resc_ref[j]      = sol_ref[j*speed]
		#num_sol = calc_ref_sol(N,dx,F_transfer)	
		num_sol  = scattering_solution(dx,N,F_transfer,m)
		num_sol = num_sol[:,::m]
		print(np.abs(resc_ref[1:]-num_sol))
		print(max(np.abs(resc_ref[1:]-num_sol)))
		errors[ixSpace,ixTime]=max(np.abs(resc_ref[1:]-num_sol)[0])
		print(errors)
#		plt.plot(tt,resc_ref,linestyle='dashed')	
#		plt.plot(tt,num_sol)
	#	plt.semilogy(np.linspace(0,5,N+1),np.abs(resc_ref,num_sol))

		
	

		import scipy.io
		scipy.io.savemat('data/ERR_DATA.mat', dict( ERR=errors,h_s=h_s,tau_s=tau_s))

	






		





