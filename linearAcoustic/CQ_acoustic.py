
import numpy as np
import bempp.api



def delta(zeta,order):
	if order==1:
	
		return 1-zeta
	else:
		if order==2:
		
			return 1.5-2.0*zeta+0.5*zeta**2
		else:
			if order ==3:
				#return 1-zeta**3
				return (1-zeta)+0.5*(1-zeta)**2+1.0/3.0*(1-zeta)**3
			else:
				Print("Order not availible")

def dirichlet_data(t,x,n, domain_index,result):
	#result[0]=t**2-x[2]**2
	result[0]=t**8
def neumann_data(t,x,n, domain_index,result):
	#result[0]=t**2-x[2]**2
	result[0]=t**8

def CQ_RHS(t):
	return (32.0/35.0)*t**(7.0/2.0)/np.sqrt(np.pi)

def Lap_f(s):
	import numpy as np
	return 1.0/np.sqrt(s)

def Evaluate_phi(phi_sol,delta_func,grid,space,n_plot):
	import bempp.api
	import numpy as np
	dof=phi_sol[:,0].size
	N=phi_sol[0,:].size-1
	L=2*N


	dt=(1.0*T)/N
	L=2*N
	tol=10**(-10)
	rho=tol**(1.0/(2*L))
	# STEP 1
	phi_scale=np.zeros((dof,L+1))
	phi_fft=1j*np.ones((dof,L+1))
	for j in range(0,dof):
		phi_scale[j,:]=np.concatenate((rho**(np.linspace(0,N,N+1))*phi_sol[j,:],np.zeros(N)),axis=0)		
		phi_fft[j,:]=np.fft.fft(phi_scale[j,:])
	

	#CREATING GRID TO PLOT
	plot_grid = np.mgrid[-1:1:n_plot*1j, -1:1:n_plot*1j]


	points = np.vstack((plot_grid[0].ravel(),
				plot_grid[1].ravel(),
				np.zeros(plot_grid[0].size)))

	
	u_hat=1j*np.ones((n_plot**2,L+1))

	#Calculating the Unit Roots
	Unit_Roots=np.exp(-1j*2*np.pi*np.linspace(0,L,L+1)/(L+1))
	#Calculating zetavect
	Zeta_vect=1j*np.ones(L+1)
	Zeta_vect=map(lambda y: delta_func( rho* y)/dt , Unit_Roots)

	# STEP 2

	for j in range(0,L+1):
		slp_pot = bempp.api.operators.potential.modified_helmholtz.single_layer(space, points,Zeta_vect[j])

		GridFun=bempp.api.GridFunction(space,coefficients=phi_fft[:,j])
	
		u_hat[:,j]=slp_pot*GridFun
	
	# STEP 3
	u_scaled=1j*np.ones((n_plot**2,L+1))
	u_sol=np.zeros((n_plot**2,N+1))
	for j in range(0,n_plot**2):
	
		u_scaled[j,:]=np.fft.ifft(u_hat[j,:])
		u_sol[j,:]=np.real(rho**(-np.linspace(0,N,N+1))*u_scaled[j,:N+1])


	return u_sol,points ,plot_grid







def Inverse_SCALAR(Dirichlet_data,delta_func,dx,Lap_func,T,N):
	import numpy as np
	
	#Set up right hand sides
	RHS_scal=np.zeros((1,N+1))
	RHS_scal=[CQ_RHS(x*1.0/N) for x in range(0,N+1)]
	#print('RHS_scal:')
	#print(RHS_scal)
	# Setting parameters
	dt=(1.0*T)/N
	L=2*N
	tol=10**(-9)
	#print(L)

	rho=tol**(1.0/(2.0*L))
	#print('rho:')
	
	#print(tol)
	#print(rho)
	
	## SCALING AND FFT (STEP 1)
	
	RHS_scal=np.concatenate((rho**(np.linspace(0,N,N+1))*RHS_scal,np.zeros(N)),axis=0)
	#print('RHS_scal:')
	#print(RHS_scal)
	RHS_scal_fft=np.fft.fft(RHS_scal)
	
	#Calculating the Unit Roots
	Unit_Roots=np.exp(-1j*2*np.pi*np.linspace(0,L,L+1)/(L+1))
	#print(Unit_Roots)
	#Calculating zetavect
	#Zeta_vect=1j*np.ones(L+1)
	Zeta_vect=map(lambda y: delta_func( rho* y)/dt , Unit_Roots)
	#print('Zeta_vect:')
	#print(Zeta_vect)
	#print(map(lambda y: delta_func( rho* y), Unit_Roots))
	#print(Zeta_vect)
	#SOLVING THE BLOCK DIAGONAL
	Laps=Lap_func(Zeta_vect)
 	
	#print('Laps')
	#print(Laps)
	phi_hat_scal=np.zeros(L+1)*1j
	for j in range(0,L+1):
		phi_hat_scal[j]=RHS_scal_fft[j]/Laps[j] 				

	##RESCALING AND IFFT (STEP 3)
#######################################################################################################

	
	ift_phi_scal=1j*np.ones(L+1)
	phi_sol_scal=np.zeros(N+1)
	
	ift_phi_scal=np.fft.ifft(phi_hat_scal)
	#print('iftu')
	#print(ift_phi_scal)
	
	phi_sol_scal=np.real(rho**(-np.linspace(0,N,N+1))*ift_phi_scal[0:N+1])
	#print('u')
	#print(phi_sol_scal)
	return phi_sol_scal




	



def GET_INTEGRATION_PARAMETERS(T,N):
	L=N
	

	dt=(T*1.0)/N

	tol=10**(-10)

	rho=tol**(1.0/(2*L))
	#rho=max(dt**(3.0/N),tol**(1.0/(2*N)))
	return L,dt,tol,rho

def u_inc(C,a,t_0,t,x):
	import numpy as np
	y=np.exp(C*(np.dot(a,x)-t-t_0))
	return y
def u_inc_ddot(C,a,t_0,t,x):
	import numpy as np
	y=np.exp(C*(np.dot(a,x)-t-t_0))*(4*C**2*(np.dot(a,x)-t-t_0)**2+2*C)
	return y

def pn_u_inc(C,a,t_0,t,x):
	import numpy as np
	ax=np.dot(a,x)	
	y1=(ax-t-t_0)
	y=ax*np.exp(C*y1**2)*2*C*(ax-t-t_0)
	return y

def create_u_inc_RHS(grid,p1_space,N,T,C1,C2,a1,a2,t_1,t_2,epsilon):

	dof=p1_space.global_dof_count
	RHS=np.zeros((dof,N+1))	
	lb=bempp.api.boundary.sparse.laplace_beltrami(p1_space,p1_space,p1_space)
	for j in range(0,N+1):
		t=j*1.0/N
		func_sum=lambda (x) :-epsilon*(u_inc(C1,a1,t_1,t,x)+u_inc(C2,a2,t_2,t,x))
		func_rhs=lambda (x): pn_u_inc(C1,a1,t_1,t,x)-pn_u_inc(C1,a1,t_1,t,x)-epsilon*(u_inc_ddot(C1,a1,t_1,t,x)+u_inc_ddot(C2,a2,t_2,t,x))
		gridfun_lb=lb*bempp.api.GridFunction(p1_space,fun=func_sum)
		gridfun_rhs=bempp.api.GridFunction(p1_space,fun=func_rhs)
		RHS[:,j]=(gridfun_lb+gridfun_rhs).coefficients
		
	return RHS


def SCALE_FFT(A,dof,N,L,rho):
	import numpy as np
	
	A_hat=1j*np.ones((dof,L+1))
	A_fft=1j*np.ones((dof,L+1))
	for j in range(0,dof):
		A_hat[j,:]=np.concatenate((rho**(np.linspace(0,N,N+1))*A[j,:],np.zeros(L-N)),axis=0)
		A_fft[j,:]=np.fft.fft(A_hat[j,:])
	return(A_fft)

def RESCALE_IFFT(A,dof,N,L,rho):
	import numpy as np
	ift_A=1j*np.ones((dof,L+1))
	A_sol=np.zeros((dof,N+1))
	for j in range(0,dof):
		ift_A[j,:]=np.fft.ifft(A[j,:])
		A_sol[j,:]=np.real(rho**(-np.linspace(0,N,N+1))*ift_A[j,0:N+1])
	return(A_sol)

def GET_ZETA_VECT(L,delta_func,rho,dt):
	import numpy as np
	#Calculating the Unit Roots
	Unit_Roots=np.exp(-1j*2*np.pi*(np.linspace(0,L,L+1)/(L+1)))
	#Calculating zetavect
	Zeta_vect=map(lambda y: delta_func( rho* y)/dt , Unit_Roots)
	return Zeta_vect



	
def Inverse_Acoustic_GIBC(grid,p1_space,dp0_space, Neumann_data,delta_func,T,N,OrderQF,eps):
	import bempp.api
	import numpy as np
	
	bempp.api.global_parameters.quadrature.double_singular = OrderQF
	#bempp.api.global_parameters.quadrature.medium.max_rel_dist = 4
	#bempp.api.global_parameters.quadrature.medium.single_order = 3
	#bempp.api.global_parameters.quadrature.medium.double_order = 3
	bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
	bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
	bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
	
	bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
	bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
	bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2


	bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
	bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3

######################################################################################################

########Precalculating Operators
	ident_0=bempp.api.operators.boundary.sparse.identity			(dp0_space,dp0_space,dp0_space)
	ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
	lb=bempp.api.boundary.sparse.laplace_beltrami(p1_space,p1_space,p1_space)

#######################################################################################################
	dof_const=dp0_space.global_dof_count
	# Setting parameters
	dof=p1_space.global_dof_count
	dof_const=dp0_space.global_dof_count
	# CONVOLUTION PARAMETERS
	L,dt,tol,rho=GET_INTEGRATION_PARAMETERS(T,N)	

	print("DOF:",dof+dof_const)
	RHS=np.zeros((dof+dof_const,N+1))
	#RHS[dof:2*dof,:N+1]=np.ones((dof,N+1))*np.linspace(0,1,N+1)**4
	#RHS[dof:dof+dof_const,:N+1]=np.ones((dof_const,N+1))*np.exp(-100*(np.linspace(0,T,N+1)-0.5*np.ones(N+1))**2)
	a=np.sqrt(3)*np.array([1,1,1])
	C=-100
	t_0=-2.5
	
	RHS[dof:dof+dof_const,:N+1]=create_u_inc_RHS(grid,p1_space,N,T,C,a,t_0)
	#for j in range(0,N+1):
	#	tj=(T*1.0)*j*1.0/N
		#RHS[:,j]=bempp.api.GridFunction(p1_space,fun=(lambda z : Dirichlet_data(z,tj)))
	#	Dirichl=lambda x,n, domain_index,result :dirichlet_data(tj,x,n, domain_index,result)
	#	RHS[:,j]=bempp.api.GridFunction(p1_space,fun=Dirichl).coefficients

#####################################################################################################
	## STEP 1
	RHS_fft=SCALE_FFT(RHS,dof+dof_const,N,L,rho)



	Zeta_vect=GET_ZETA_VECT(L,delta_func,rho,dt)


######################################################################################################
	## STEP 2
	normsRHS=np.ones(L+1)
	from scipy.sparse.linalg import gmres 
	phi_hat=1j*np.zeros((dof_const+dof,L+1))
	Half=int(np.ceil(float(L)/2.0))
	for j in range(0,Half+1):
		#from scipy.sparse.linalg import aslinearoperator
		#from bempp.api.fenics_interface import FenicsOperator
		normsRHS[j]=np.max(np.abs(RHS_fft[:,j]))
		print("normRHS:",normsRHS[j])
		if normsRHS[j]>10**-9:
			print("j:",j,"L:",Half+1)
			s=Zeta_vect[j]
			Bimp = bempp.api.BlockedOperator(2,2)
			blocks=[[None,None],[None,None]]

			slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
			dlp=bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,p1_space,s)
			adlp=bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,dp0_space,s)
			hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s)
		
			#trace_op = aslinearoperator(trace_matrix)

			#A = FenicsOperator((dolfin.inner(dolfin.nabla_grad(u),
	        	#                        dolfin.nabla_grad(v)) -k**2 * n**2 * u * v) * dolfin.dx)
			#ident_str=ident.strong_form()
#			from multiprocessing import Pool
#			pool=Pool()
#			b00=pool.apply_async((s*slp).strong_form)
#			b01=pool.apply_async((dlp-1.0/2.0*ident_1).strong_form)
#			b10=pool.apply_async((-adlp+1.0/2.0*ident_0).strong_form)
#			b11=pool.apply_async((1.0/s*hslp).strong_form)
#			
#			blocks[0][0]=b00.get(timeout=10)
#			blocks[0][1]=b01.get(timeout=10)
#			blocks[1][0]=b10.get(timeout=10)
#			blocks[1][1]=b11.get(timeout=10)	
			
			
			blocks[0][0] =(s*slp).strong_form()
			blocks[0][1] = (dlp-1.0/2.0*ident_1).strong_form()
			blocks[1][0] = (-adlp+1.0/2.0*ident_0).strong_form()
			blocks[1][1] = (1.0/s*hslp+eps*(s*ident_0-1.0/s*lb)).strong_form()
			B_strong_form=bempp.api.BlockedDiscreteOperator(np.array(blocks))
			print("Strong_form calculated")
#			Bimp[0, 0] = s*slp
#			Bimp[0, 1] = dlp-1.0/2.0*ident
#			Bimp[1, 0] = -adlp+1.0/2.0*ident
#			Bimp[1, 1] = 1.0/s*hslp

#			B_strong_form=Bimp.strong_form()
			if j>0:
				phi_hat[:,j], info=gmres(B_strong_form,RHS_fft[:,j],x0=phi_hat[:,j-1],maxiter=20)
			else:
				phi_hat[:,j], info=gmres(B_strong_form,RHS_fft[:,j],maxiter=50)
			
	for j in range(Half+1,L+1):
		phi_hat[:,j]=np.conj(phi_hat[:,L+1-j])	

#	##RESCALING AND IFFT (STEP 3)
#######################################################################################################

	phi_sol=RESCALE_IFFT(phi_hat,dof_const+dof,N,L,rho)


	
	return phi_sol,grid,p1_space,dof


def Inverse_Acoustic_Neumann(grid,p1_space,dp0_space, Neumann_data,delta_func,T,N,OrderQF):
	import bempp.api
	import numpy as np
	
	bempp.api.global_parameters.quadrature.double_singular = OrderQF
	#bempp.api.global_parameters.quadrature.medium.max_rel_dist = 4
	#bempp.api.global_parameters.quadrature.medium.single_order = 3
	#bempp.api.global_parameters.quadrature.medium.double_order = 3
	bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
	bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
	bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
	
	bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
	bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
	bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2


	bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
	bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3

	dof_const=dp0_space.global_dof_count
	# Setting parameters
	dof=p1_space.global_dof_count
	dof_const=dp0_space.global_dof_count
	# CONVOLUTION PARAMETERS
	L,dt,tol,rho=GET_INTEGRATION_PARAMETERS(T,N)	

	print("DOF:",dof+dof_const)
	RHS=np.zeros((dof+dof_const,N+1))
	#RHS[dof:2*dof,:N+1]=np.ones((dof,N+1))*np.linspace(0,1,N+1)**4
	RHS[dof:dof+dof_const,:N+1]=np.ones((dof_const,N+1))*np.exp(-100*(np.linspace(0,T,N+1)-0.5*np.ones(N+1))**2)
		
	#for j in range(0,N+1):
	#	tj=(T*1.0)*j*1.0/N
		#RHS[:,j]=bempp.api.GridFunction(p1_space,fun=(lambda z : Dirichlet_data(z,tj)))
	#	Dirichl=lambda x,n, domain_index,result :dirichlet_data(tj,x,n, domain_index,result)
	#	RHS[:,j]=bempp.api.GridFunction(p1_space,fun=Dirichl).coefficients

#####################################################################################################
	## STEP 1
	RHS_fft=SCALE_FFT(RHS,dof+dof_const,N,L,rho)



	Zeta_vect=GET_ZETA_VECT(L,delta_func,rho,dt)


######################################################################################################
	## STEP 2
	normsRHS=np.ones(L+1)
	from scipy.sparse.linalg import gmres 
	phi_hat=1j*np.zeros((dof_const+dof,L+1))
	Half=int(np.ceil(float(L)/2.0))
	for j in range(0,Half+1):
		#from scipy.sparse.linalg import aslinearoperator
		#from bempp.api.fenics_interface import FenicsOperator
		normsRHS[j]=np.max(np.abs(RHS_fft[:,j]))
		print("normRHS:",normsRHS[j])
		if normsRHS[j]>10**-9:
			print("j:",j,"L:",Half+1)
			s=Zeta_vect[j]
			Bimp = bempp.api.BlockedOperator(2,2)
			blocks=[[None,None],[None,None]]
			ident_0=bempp.api.operators.boundary.sparse.identity			(dp0_space,dp0_space,dp0_space)
			ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
			slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
			dlp=bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,p1_space,s)
			adlp=bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,dp0_space,s)
			hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s)
			#trace_op = aslinearoperator(trace_matrix)

			#A = FenicsOperator((dolfin.inner(dolfin.nabla_grad(u),
	        	#                        dolfin.nabla_grad(v)) -k**2 * n**2 * u * v) * dolfin.dx)
			#ident_str=ident.strong_form()
#			from multiprocessing import Pool
#			pool=Pool()
#			b00=pool.apply_async((s*slp).strong_form)
#			b01=pool.apply_async((dlp-1.0/2.0*ident_1).strong_form)
#			b10=pool.apply_async((-adlp+1.0/2.0*ident_0).strong_form)
#			b11=pool.apply_async((1.0/s*hslp).strong_form)
#			
#			blocks[0][0]=b00.get(timeout=10)
#			blocks[0][1]=b01.get(timeout=10)
#			blocks[1][0]=b10.get(timeout=10)
#			blocks[1][1]=b11.get(timeout=10)	
			
			
			blocks[0][0] =(s*slp).strong_form()
			blocks[0][1] = (dlp-1.0/2.0*ident_1).strong_form()
			blocks[1][0] = (-adlp+1.0/2.0*ident_0).strong_form()
			blocks[1][1] = (1.0/s*hslp).strong_form()
			B_strong_form=bempp.api.BlockedDiscreteOperator(np.array(blocks))
			print("Strong_form calculated")
#			Bimp[0, 0] = s*slp
#			Bimp[0, 1] = dlp-1.0/2.0*ident
#			Bimp[1, 0] = -adlp+1.0/2.0*ident
#			Bimp[1, 1] = 1.0/s*hslp

#			B_strong_form=Bimp.strong_form()
			if j>0:
				phi_hat[:,j], info=gmres(B_strong_form,RHS_fft[:,j],x0=phi_hat[:,j-1],maxiter=20)
			else:
				phi_hat[:,j], info=gmres(B_strong_form,RHS_fft[:,j],maxiter=50)
			
	for j in range(Half+1,L+1):
		phi_hat[:,j]=np.conj(phi_hat[:,L+1-j])	

#	##RESCALING AND IFFT (STEP 3)
#######################################################################################################

	phi_sol=RESCALE_IFFT(phi_hat,dof_const+dof,N,L,rho)


	
	return phi_sol,grid,p1_space,dof



	
def Inverse_Acoustic_Dirichlet(grid, p1_space, Dirichlet_data, delta_func,T,N,OrderQF):
	import bempp.api
	import numpy as np
	

	bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
	bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
	bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
	
	bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
	bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
	bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2


	bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
	bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3

	bempp.api.global_parameters.quadrature.double_singular = OrderQF
	print("ORDERQF:",OrderQF)
	#from CQ_acoustic import Dirichlet
	dp0_space=bempp.api.function_space(grid,"P",1)
	dof_const=dp0_space.global_dof_count
	# Setting parameters
	dof=p1_space.global_dof_count
	# CONVOLUTION PARAMETERS
	L,dt,tol,rho=GET_INTEGRATION_PARAMETERS(T,N)	

	
	RHS=np.zeros((dof,N+1))
	#RHS=np.ones((dof,N+1))*np.linspace(0,1,N+1)**8
	RHS=np.ones((dof,N+1))*np.exp(-100*(np.linspace(0,T,N+1)-0.5*np.ones(N+1))**2)
	#RHS=np.ones((dof,N+1))*(np.sin(2*np.pi*(np.linspace(0,1,N+1))))**3
	#for j in range(0,N+1):
	#	tj=(T*1.0)*j*1.0/N
		#RHS[:,j]=bempp.api.GridFunction(p1_space,fun=(lambda z : Dirichlet_data(z,tj)))
	#	Dirichl=lambda x,n, domain_index,result :dirichlet_data(tj,x,n, domain_index,result)
	#	RHS[:,j]=bempp.api.GridFunction(p1_space,fun=Dirichl).coefficients

#####################################################################################################
	## STEP 1
	RHS_fft=SCALE_FFT(RHS,dof,N,L,rho)

	Zeta_vect=GET_ZETA_VECT(L,delta_func,rho,dt)



#	Half=int(np.ceil(float(L)/2.0))
#	import matplotlib.pyplot as plt
#	x_s=np.real(Zeta_vect[0:Half+1])
#	y_s=np.imag(Zeta_vect[0:Half+1])
#	plt.scatter(x_s,y_s,color='red')
#	print(Zeta_vect[0])
#	print(Zeta_vect[Half])
#	plt.show()

####################################################################################################	
	## SOLVING THE BLOCK DIAGONAL SYSTEM (STEP 2)
	#p0_space=bempp.api.function_space(grid,"P",1)
	#dof_const=p0_space.global_dof_count
	phi_hat=1j*np.zeros((dof_const,L+1))
	phi_hat2=1j*np.zeros((dof_const,L+1))
	norms=np.ones(L+1)
	normsRHS=np.ones(L+1)
	normsRHS=np.ones(L+1)
	#p0_space=bempp.api.function_space(grid,"DP",0)
	#from bempp.api.linalg.iterative_solvers import gmres
	from scipy.sparse.linalg import gmres 	
	#from bempp.api.linalg.iterative_solvers import gmres
	
	Half=int(np.ceil(float(L)/2.0))
	
	for j in range(0,Half+1):
		normsRHS[j]=np.max(np.abs(RHS_fft[:,j]))
		print("normRHS:",normsRHS[j])
		if normsRHS[j]>10**-9:
			print("j:",j,"L:",Half+1)
			#Initializing the Operator
			slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,Zeta_vect[j])
		
			#Initializing the Grid function on the right hand side
		
			#b=bempp.api.GridFunction(p1_space,coefficients=RHS_fft[:,j])
			slp_discrete=slp.strong_form()
			#phi_grid,info = gmres(slp,b,tol=10**(-10))

		
			#print("NORM:",np.linalg.norm(phi_hat[:,j]))
			##################################################
			#slp_discrete=slp.strong_form()
		
			phi_grid,info = gmres(slp_discrete,RHS_fft[:,j],tol=10**-9,maxiter=300)
			phi_hat[:,j]=phi_grid
			#phi_grid=np.linalg.solve(bempp.api.as_matrix(slp_discrete),RHS_fft[:,j])
			#####################################################
			#RES=slp_discrete*phi_grid-RHS_fft[:,j]
			#normRes=np.linalg.norm(RES)

			##CATCH IF GMRES DOESNT CONVERGE	
			if info>0:
				#slp_discrete=slp.strong_form()
	
				RES=slp_discrete*phi_hat[:,j]-RHS_fft[:,j]
				normRes=np.linalg.norm(RES)
				print("WARNING, INFO:",info,"NORMRES:",normRes)	

			
				phi_vect=np.linalg.solve(bempp.api.as_matrix(slp_discrete),RHS_fft[:,j])	
				RES=slp_discrete*phi_vect-RHS_fft[:,j]
				normRes=np.linalg.norm(RES)
				print("NEW_norm_RES:",normRes)
				#print(info)		
				phi_hat[:,j]=phi_vect


	for j in range(Half+1,L+1):
		phi_hat[:,j]=np.conj(phi_hat[:,L+1-j])
		
	for j in range(0,L+1):
		norms[j]=np.max(np.abs(phi_hat[:,j]))
		normsRHS[j]=np.max(np.abs(RHS_fft[:,j]))

	#import matplotlib.pyplot as plt
	#plt.semilogy(norms,'r')
	#plt.semilogy(normsRHS,'b')
	#plt.show()



#	for j in range(0,L+1):
#		#print("j:",j,"L:",L)
#		#Initializing the Operator
#		slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,Zeta_vect[j])
#		
#		#Initializing the Grid function on the right hand side
#		b=bempp.api.GridFunction(p1_space,coefficients=RHS_fft[:,j])
#
#		phi_grid,info = gmres(slp,b,tol=10**(-10),maxiter=100)
#
#		phi_hat2[:,j]=phi_grid.coefficients
#		##################################################
#		#slp_discrete=slp.strong_form()
#		
#		#phi_grid,info = cg(slp_discrete,RHS_fft[:,j],tol=10**-10,maxiter=50)
#		
#		#phi_grid=np.linalg.solve(bempp.api.as_matrix(slp_discrete),RHS_fft[:,j])
#		#####################################################
#		#RES=slp_discrete*phi_grid-RHS_fft[:,j]
#		#normRes=np.linalg.norm(RES)
#
##CATCH IF GMRES DOESNT CONVERGE	
#		if info>0:
#			slp_discrete=slp.strong_form()
#
#			RES=slp_discrete*phi_hat2[:,j]-RHS_fft[:,j]
#			normRes=np.linalg.norm(RES)
#			print("WARNING, INFO:",info,"NORMRES:",normRes)	
#
#			
#			phi_vect=np.linalg.solve(bempp.api.as_matrix(slp_discrete),RHS_fft[:,j])	
#			RES=slp_discrete*phi_vect-RHS_fft[:,j]
#			normRes=np.linalg.norm(RES)
#			print("NEW_norm_RES:",normRes)
#			#print(info)		
#			phi_hat2[:,j]=phi_vect
#
#		#print(np.linalg.norm(RES))
#		#phi_grid,info = cg(slp,b,tol=10**(-16))
#
#	print(np.linalg.norm(phi_hat-phi_hat2))
#	##RESCALING AND IFFT (STEP 3)
#######################################################################################################

	phi_sol=RESCALE_IFFT(phi_hat,dof_const,N,L,rho)




	return phi_sol,grid,p1_space,dof


def Acoustic_error_plot(Dirichlet_data,delta_func,T,Amount_time,Amount_space):
	import bempp.api
	import numpy as np

	bempp.api.global_parameters.hmat.eps=10**-8

	h_s=np.zeros(Amount_space)
	dof_s=np.zeros(Amount_space)
	err_Linf=np.zeros((Amount_space,Amount_time))
	#err_eucl=np.zeros(Amount)
	tau_s=np.zeros(Amount_time)
	#Define grid
	OrderQF=8



	for i in range(0,Amount_space):

		print('Space_index:')			
		print(i)
		dx=2**(-i)
		print(dx)
		grid=bempp.api.shapes.sphere(h=dx)
		#Define space
		p1_space=bempp.api.function_space(grid, "P" , 1)
		
		dof=p1_space.global_dof_count


		dof_s[i]=dof
		print('DOF:')
		print(dof)

		for j in range(0,Amount_time):

			print('Time_index:')			
			print(j)
			N=2*2**j
			


			
			phi_sol,grid,p1_space,dof_const=Inverse_Acoustic_Dirichlet(grid,p1_space,Dirichlet_data,delta_func,T,N,OrderQF)
			print(phi_sol)
			#print(phi_sol_scal)
			#print(phi_sol[:,N])
			#start=time.time()
			#phi_sol_scal=Inverse_SCALAR(Dirichlet_data,delta_func,dx,Lap_f,T,N)
			#end=time.time()
			#print(end-start)
			
	
			
			## EVALUATIONS FOR PLOTTING 		######################################################################################################################################################
			#phi_ex=5.0/8.0*np.ones((dof,N+1))
			#phi_ex=8.0*np.ones((dof_const,N+1))
			#phi_ex=np.ones((dof_const,N+1))*16.0*(np.linspace(0,1,N+1))**7
			phi_ex=np.ones((dof,N+1))*np.exp(-100*(np.linspace(0,T,N+1)-0.5*np.ones(N+1))**2)*-400*(np.linspace(0,T,N+1)-0.5*np.ones(N+1))
			#phi_ex=np.ones((dof,N+1))*(np.sin(2*np.pi*np.linspace(0,1,N+1))**2)*2*2*np.pi*3*np.cos(2*np.pi*np.linspace(0,1,N+1))
			#print(phi_ex)
			#print(phi_ex)
			tau_s[j]=T*1.0/N
			h_s[i]=dx
			#err_eucl[j]=np.abs(phi_ex[N]-	phi_sol_scal[N])
			#err_Linf[j]=np.max(np.abs(phi_ex-phi_sol_scal))

			#err_eucl[j]=np.linalg.norm(phi_sol[:,N]-phi_ex[:,N]) /dof
			err_mat=phi_sol[:,:]-phi_ex[:,:]
			#print("err_mat:",err_mat[:4,:])
			err_mat=tau_s[j]*err_mat**2
			#print("err_mat_squared:",err_mat[:4,:])
			err_vec=np.sum(err_mat,axis=1)
			#print("err_vec:",err_vec)
			#err_Linf[i,j]=np.max(np.abs(phi_sol[:,N]-phi_ex[:,N]))
			err_Linf[i,j]=np.sqrt(np.max(err_vec))			
			k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
			#print(err_eucl)
			print(err_Linf)
			#import matplotlib.pyplot as pyplot
			#pyplot.semilogy(np.abs(phi_sol[k,:N/2]-phi_ex[k,:N/2]),'b')
			#pyplot.plot(phi_ex[k,:N/2],'r')
			#pyplot.show()


	#k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
	#EX=6*(np.linspace(0,N,N+1)/(N))**2
	#import matplotlib.pyplot as pyplot
	#pyplot.plot(phi_sol[k,:],'b')
	#pyplot.plot(phi_ex[k,:],'r')
	#pyplot.show()
	#print(phi_ex[:,N]-phi_sol[:,N])
			

	return tau_s,h_s,dof_s,err_Linf

def Error_plots_difQF(Dirichlet_data,delta_func,T,Amount_time,Amount_QF):
	import bempp.api
	import numpy as np
	
	bempp.api.global_parameters.hmat.eps=10**-8

	QF_s=np.zeros(Amount_QF)

	err_Linf=np.zeros((Amount_QF,Amount_time))
	#err_eucl=np.zeros(Amount)
	tau_s=np.zeros(Amount_time)
	#Define grid
	dx=0.5
	grid=bempp.api.shapes.sphere(h=0.5)
	#Define space
	p1_space=bempp.api.function_space(grid, "P" , 1)
	dof=p1_space.global_dof_count


	for i in range(0,Amount_QF):

		print('QF_index:')			
		print(i)
		OrderQF=4+2*i
		print(OrderQF)

		
		


		QF_s[i]=OrderQF

		for j in range(0,Amount_time):

			print('Time_index:')			
			print(j)
			N=20*2**j
			


			
			phi_sol,grid,p1_space,dof_const=Inverse_Acoustic_Neumann(grid,p1_space,Dirichlet_data,delta_func,T,N,OrderQF)
	
			#print(phi_sol_scal)
			#print(phi_sol[:,N])
			#start=time.time()
			#phi_sol_scal=Inverse_SCALAR(Dirichlet_data,delta_func,dx,Lap_f,T,N)
			#end=time.time()
			#print(end-start)
			
	
			
			## EVALUATIONS FOR PLOTTING 		######################################################################################################################################################
			#phi_ex=5.0/8.0*np.ones((dof,N+1))
			#phi_ex=8.0*np.ones((dof_const,N+1))
			#phi_ex=np.ones((dof_const,N+1))*16.0*(np.linspace(0,1,N+1))**7
			phi_ex=np.ones((dof,N+1))*np.exp(-100*(np.linspace(0,T,N+1)-0.5*np.ones(N+1))**2)*-400*(np.linspace(0,T,N+1)-0.5*np.ones(N+1))
			#phi_ex=np.ones((dof,N+1))*(np.sin(2*np.pi*np.linspace(0,1,N+1))**2)*2*2*np.pi*3*np.cos(2*np.pi*np.linspace(0,1,N+1))
			#print(phi_ex)
			#print(phi_ex)
			tau_s[j]=T*1.0/N
			
			#err_eucl[j]=np.abs(phi_ex[N]-	phi_sol_scal[N])
			#err_Linf[j]=np.max(np.abs(phi_ex-phi_sol_scal))

			#err_eucl[j]=np.linalg.norm(phi_sol[:,N]-phi_ex[:,N]) /dof
			err_mat=phi_sol[:,:N/2]-phi_ex[:,:N/2]
			#print("err_mat:",err_mat[:4,:])
			err_mat=tau_s[j]*err_mat**2
			#print("err_mat_squared:",err_mat[:4,:])
			err_vec=np.sum(err_mat,axis=1)
			#print("err_vec:",err_vec)
			#err_Linf[i,j]=np.max(np.abs(phi_sol[:,N]-phi_ex[:,N]))
			err_Linf[i,j]=np.sqrt(np.max(err_vec))			
			k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
			#print(err_eucl)
			print(err_Linf)
			#import matplotlib.pyplot as pyplot
			#pyplot.semilogy(np.abs(phi_sol[k,:N/2]-phi_ex[k,:N/2]),'b')
			#pyplot.plot(phi_ex[k,:N/2],'r')
			#pyplot.show()


	#k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
	#EX=6*(np.linspace(0,N,N+1)/(N))**2
	#import matplotlib.pyplot as pyplot
	#pyplot.plot(phi_sol[k,:],'b')
	#pyplot.plot(phi_ex[k,:],'r')
	#pyplot.show()
	#print(phi_ex[:,N]-phi_sol[:,N])
			

	return tau_s,QF_s,err_Linf

def Error_plots_difQF_n(Dirichlet_data,delta_func,T,Amount_time,Amount_QF):
	import bempp.api
	import numpy as np
	
	bempp.api.global_parameters.hmat.eps=10**-8

	QF_s=np.zeros(Amount_QF)

	err_Linf=np.zeros((Amount_QF,Amount_time))
	#err_eucl=np.zeros(Amount)
	tau_s=np.zeros(Amount_time)
	#Define grid
	dx=0.5
	grid=bempp.api.shapes.sphere(h=0.5)
	#Define space
	p1_space=bempp.api.function_space(grid, "P" , 1)
	dof=p1_space.global_dof_count


	for i in range(0,Amount_QF):

		print('QF_index:')			
		print(i)
		OrderQF=4+2*i
		print(OrderQF)

		
		


		QF_s[i]=OrderQF

		for j in range(0,Amount_time):

			print('Time_index:')			
			print(j)
			N=20*2**j
			


			
			phi_sol,grid,p1_space,dof_const=Inverse_Acoustic_Neumann(grid,p1_space,Dirichlet_data,delta_func,T,N,OrderQF)
	
			#print(phi_sol_scal)
			#print(phi_sol[:,N])
			#start=time.time()
			#phi_sol_scal=Inverse_SCALAR(Dirichlet_data,delta_func,dx,Lap_f,T,N)
			#end=time.time()
			#print(end-start)
			
	
			
			## EVALUATIONS FOR PLOTTING 		######################################################################################################################################################
			#phi_ex=5.0/8.0*np.ones((dof,N+1))
			#phi_ex=8.0*np.ones((dof_const,N+1))
			#phi_ex=np.ones((dof_const,N+1))*16.0*(np.linspace(0,1,N+1))**7
			#phi_ex=np.ones((dof,N+1))*np.exp(-100*(np.linspace(0,1,N+1)-0.5*np.ones(N+1))**2)*-400*(np.linspace(0,1,N+1)-0.5*np.ones(N+1))
			phi_ex=np.ones((dof,N+1))*np.exp(-100*(np.linspace(0,1,N+1)-0.5*np.ones(N+1))**2)
			#phi_ex=np.ones((dof,N+1))*(np.sin(2*np.pi*np.linspace(0,1,N+1))**2)*2*2*np.pi*3*np.cos(2*np.pi*np.linspace(0,1,N+1))
			#print(phi_ex)
			#print(phi_ex)
			tau_s[j]=T*1.0/N
			
			#err_eucl[j]=np.abs(phi_ex[N]-	phi_sol_scal[N])
			#err_Linf[j]=np.max(np.abs(phi_ex-phi_sol_scal))

			#err_eucl[j]=np.linalg.norm(phi_sol[:,N]-phi_ex[:,N]) /dof
			err_mat=phi_sol[dof:2*dof]-phi_ex
			#print("err_mat:",err_mat[:4,:])
			err_mat=tau_s[j]*err_mat**2
			#print("err_mat_squared:",err_mat[:4,:])
			err_vec=np.sum(err_mat,axis=1)
			#print("err_vec:",err_vec)
			#err_Linf[i,j]=np.max(np.abs(phi_sol[:,N]-phi_ex[:,N]))
			err_Linf[i,j]=np.sqrt(np.max(err_vec))			
			#k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
			#print(err_eucl)
			print(err_Linf)
			#import matplotlib.pyplot as pyplot
			#pyplot.semilogy(np.abs(phi_sol[k,:N/2]-phi_ex[k,:N/2]),'b')
			#pyplot.plot(phi_ex[k,:N/2],'r')
			#pyplot.show()


	#k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
	#EX=6*(np.linspace(0,N,N+1)/(N))**2
	#import matplotlib.pyplot as pyplot
	#pyplot.plot(phi_sol[k,:],'b')
	#pyplot.plot(phi_ex[k,:],'r')
	#pyplot.show()
	#print(phi_ex[:,N]-phi_sol[:,N])
			

	return tau_s,QF_s,err_Linf
import cProfile, pstats, io
def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def Acoustic_neumann_error_plot(neumann_data,delta_func,T,Amount_time,Amount_space):
	import bempp.api
	import numpy as np

	bempp.api.global_parameters.hmat.eps=10**-8

	h_s=np.zeros(Amount_space)
	dof_s=np.zeros(Amount_space)
	err_Linf=np.zeros((Amount_space,Amount_time))
	#err_eucl=np.zeros(Amount)
	tau_s=np.zeros(Amount_time)
	#Define grid




	for i in range(0,Amount_space):

		print('Space_index:')			
		print(i)
		dx=2**(-i)
		print(dx)
		grid=bempp.api.shapes.sphere(h=dx)
		#Define space
		p1_space=bempp.api.function_space(grid, "P" ,1)
		dp0_space=bempp.api.function_space(grid,"P",1)
		dof=p1_space.global_dof_count


		dof_s[i]=dof
		print('DOF:')
		print(dof)

		for j in range(0,Amount_time):

			print('Time_index:')			
			print(j)
			N=2000*2**(j)
			

			OrderQF=10
			
			phi_sol,grid,p1_space,dof=Inverse_Acoustic_Neumann(grid,p1_space,dp0_space, neumann_data,delta_func,T,N,OrderQF)
		
			aver_1=np.sum(phi_sol[0:dof,:],axis=0)/dof
			aver_2=np.sum(phi_sol[dof:2*dof,:],axis=0)/dof
			#print(phi_sol_scal)
			#print(phi_sol[:,N])
			#start=time.time()
			#phi_sol_scal=Inverse_SCALAR(Dirichlet_data,delta_func,dx,Lap_f,T,N)
			#end=time.time()
			#print(end-start)
			
	
			
			## EVALUATIONS FOR PLOTTING 		######################################################################################################################################################
			#phi_ex=5.0/8.0*np.ones((dof,N+1))
			#phi_ex=8.0*np.ones((dof_const,N+1))
			#phi_ex=np.ones((dof_const,N+1))*16.0*(np.linspace(0,1,N+1))**7
			#phi_ex=np.ones((dof,N+1))*np.exp(-100*(np.linspace(0,1,N+1)-0.5*np.ones(N+1))**2)*-400*(np.linspace(0,1,N+1)-0.5*np.ones(N+1))

			
			phi_ex=np.ones((dof,N+1))*np.exp(-100*(np.linspace(0,1,N+1)-0.5*np.ones(N+1))**2)
			#phi_ex=np.ones((dof,N+1))*(np.sin(2*np.pi*np.linspace(0,1,N+1))**2)*2*2*np.pi*3*np.cos(2*np.pi*np.linspace(0,1,N+1))
			#print(phi_ex)
			#print(phi_ex)
			tau_s[j]=T*1.0/N
			h_s[i]=dx
			#err_eucl[j]=np.abs(phi_ex[N]-	phi_sol_scal[N])
			#err_Linf[j]=np.max(np.abs(phi_ex-phi_sol_scal))
			
			#err_eucl[j]=np.linalg.norm(phi_sol[:,N]-phi_ex[:,N]) /dof
			err_mat=phi_sol[0:dof,:]-phi_ex
			#print("err_mat:",err_mat[:4,:])
			err_mat=tau_s[j]*err_mat**2

			#print("err_mat_squared:",err_mat[:4,:])
			err_vec=np.sum(err_mat,axis=1)
		
			#print("err_vec:",err_vec)
			#err_Linf[i,j]=np.max(np.abs(phi_sol[:,N]-phi_ex[:,N]))
			err_Linf[i,j]=np.sqrt(np.max(err_vec))			
			#k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
			#print(err_eucl)
			print(err_Linf)
			import matplotlib.pyplot as pyplot
			#pyplot.plot(phi_sol[0,:],'b')
			#pyplot.plot(phi_ex[0,:],'r')
			#pyplot.show()
			#pyplot.plot(phi_sol[dof,:],'g')
			#pyplot.plot(phi_sol[0,:],'b')
			pyplot.plot(phi_ex[0,:],'r')
			pyplot.plot(aver_1,'g')	
			pyplot.plot(aver_2,'b')		
			pyplot.show()
	np.save('phi_ref_s',phi_sol)
	#k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
	#EX=6*(np.linspace(0,N,N+1)/(N))**2
	#import matplotlib.pyplot as pyplot
	#pyplot.plot(phi_sol[k,:],'b')
	#pyplot.plot(phi_ex[k,:],'r')
	#pyplot.show()
	#print(phi_ex[:,N]-phi_sol[:,N])
			

	return tau_s,h_s,dof_s,err_Linf











def Evaluate_phi_surf(phi_sol,T,delta_func,grid,p1_space):
	### This routine evaluates u on the boundary, for numerical tests
	import bempp.api
	import numpy as np
	dof=phi_sol[:,0].size
	N=phi_sol[0,:].size-1
	
	L,dt,tol,rho=GET_INTEGRATION_PARAMETERS(T,N)	

	# STEP 1
	#phi_scale=np.zeros((dof,L+1))
	#phi_fft=1j*np.ones((dof,L+1))
	#for j in range(0,dof):
	#	phi_scale[j,:]=np.concatenate((rho**(np.linspace(0,N,N+1))*phi_sol[j,:],np.zeros(N)),axis=0)		
	#	phi_fft[j,:]=np.fft.fft(phi_scale[j,:])
	phi_fft=SCALE_FFT(phi_sol,dof,N,L,rho)

	Zeta_vect=GET_ZETA_VECT(L,delta_func,rho,dt)
#################################################################################################	
	# STEP 2
	u_hat=1j*np.ones((dof,L+1))
	Half=int(np.ceil((L+1)/2))
	for j in range(0,Half+1):
		slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(p1_space,p1_space,p1_space,Zeta_vect[j])
		slp_discrete=slp.strong_form()
		#GridFun=bempp.api.GridFunction(p1_space,coefficients=phi_fft[:,j])
		u_hat[:,j]=slp_discrete*phi_fft[:,j]
		#u_hat[:,j]=(slp*GridFun).coefficients

	for j in range(Half+1 , L+1):
		u_hat[:,j]=np.conj(u_hat[:,L+1-j])
		

#################################################################################################	
	# STEP 3

	u_sol=RESCALE_IFFT(u_hat,dof,N,L,rho)

	
	#u_scaled=1j*np.ones((dof,L+1))
	#u_sol=np.zeros((dof,N+1))
	#for j in range(0,dof):
	
	#	u_scaled[j,:]=np.fft.ifft(u_hat[j,:])
	#	u_sol[j,:]=np.real(rho**(-np.linspace(0,N,N+1))*u_scaled[j,:N+1])


	u_exact=np.ones((dof,N+1))*np.linspace(0,1,N+1)**8
	err=np.max(np.abs(u_exact[:,N]-u_sol[:,N]))

	return err
#
#
#



def Error_plot_both(delta_func,Amount_time,Amount_space):
	import time
	T=1
	errors=np.zeros((Amount_space,Amount_time))	
	for i in range(0,Amount_space):
		for j in range(0,Amount_time): 
			grid=bempp.api.shapes.sphere(h=2**-i)
			p1_space=bempp.api.function_space(grid,"P",1)
			T1=time.time()
			phi_sol,grid,p1_space,dof_const=Inverse_Acoustic_Dirichlet(grid, p1_space,dirichlet_data, delta_func,T,2**j*16)
			
			T2=time.time()
			print("TIME INVERSE:",T2-T1)
			errors[i,j]=Evaluate_phi_surf(phi_sol,T,delta_func,grid,p1_space)
			T3=time.time()			
			print("TIME FORWARD:",T3-T2)
			print("COMPOSITION ERROR:",errors)
	return errors





def Acoustic_error_plot_FW(delta_func,Amount_time,Amount_space):
	import bempp.api
	import numpy as np

	h_s=np.zeros(Amount_space)
	dof_s=np.zeros(Amount_space)
	err_Linf=np.zeros((Amount_space,Amount_time))
	#err_eucl=np.zeros(Amount)
	tau_s=np.zeros(Amount_time)
	#Define grid

	T=1


	for i in range(0,Amount_space):
		print('Space_index_FW:')			
		print(i)
		
		for j in range(0,Amount_time):

			print('Time_index_FW:')			
			print(j)
			N=4*2**j
			dx=2**(-i)
			grid=bempp.api.shapes.sphere(h=dx)
			#Define space
			p1_space=bempp.api.function_space (grid, "P" , 1)
			

			dof=p1_space.global_dof_count

			phi_ex=np.ones((dof,N+1))*16*np.linspace(0,1,N+1)**7


			


			err_Linf[i,j]=Evaluate_phi_surf(phi_ex,T,delta_func,grid,p1_space)


#########################SAVING PARAMETERS
			dof_s[i]=dof
			tau_s[j]=T*1.0/N
			h_s[i]=dx


			print(err_Linf)



	#k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
	#EX=6*(np.linspace(0,N,N+1)/(N))**2
	#import matplotlib.pyplot as pyplot
	#pyplot.plot(phi_sol[k,:])
	#pyplot.plot(EX)
	#pyplot.show()
	#print(phi_ex[:,N]-phi_sol[:,N])
			

	return tau_s,h_s,dof_s,err_Linf



######################################################################################################
######################################################################################################
######################################################################################################


#def Evaluate_phi_surf(phi_sol,T,delta_func,grid,space):
#	### This routine evaluates u on the boundary, for numerical tests
#	import bempp.api
#	import numpy as np
#	dof=phi_sol[:,0].size
#	N=phi_sol[0,:].size-1
#	L=2*N
#
#	
#	dt=(1.0*T)/N
#
#	tol=10**(-16)
#
#	rho=tol**(1.0/(2*L))
#	# STEP 1
#	phi_scale=np.zeros((dof,L+1))
#	phi_fft=1j*np.ones((dof,L+1))
#	for j in range(0,dof):
#		phi_scale[j,:]=np.concatenate((rho**(np.linspace(0,N,N+1))*phi_sol[j,:],np.zeros(N)),axis=0)		
#		phi_fft[j,:]=np.fft.fft(phi_scale[j,:])
#	
#
#	##CREATING GRID TO PLOT
#	#plot_grid = np.mgrid[-1:1:n_plot*1j, -1:1:n_plot*1j]
#
#
#	#points = np.vstack((plot_grid[0].ravel(),
#	#			plot_grid[1].ravel(),
#	#			np.zeros(plot_grid[0].size)))
#
#	
#	u_hat=1j*np.ones((dof,L+1))
#
#	#Calculating the Unit Roots
#	Unit_Roots=np.exp(-1j*2*np.pi*np.linspace(0,L,L+1)/(L+1))
#	#Calculating zetavect
#	#Zeta_vect=1j*np.ones(L+1)
#	Zeta_vect=map(lambda y: delta_func( rho* y)/dt , Unit_Roots)
#
#	# STEP 2
#
#	for j in range(0,L+1):
#		slp = bempp.api.operators.boundary.modified_helmholtz.single_layer(space,space, space,Zeta_vect[j])

#		GridFun=bempp.api.GridFunction(space,coefficients=phi_fft[:,j])
#	
#		u_hat[:,j]=(slp*GridFun).coefficients
#		
#	
#	# STEP 3
#	u_scaled=1j*np.ones((dof,L+1))
#	u_sol=np.zeros((dof,N+1))
#	for j in range(0,dof):
#	
#		u_scaled[j,:]=np.fft.ifft(u_hat[j,:])
#		u_sol[j,:]=np.real(rho**(-np.linspace(0,N,N+1))*u_scaled[j,:N+1])
#
#
#	u_exact=np.ones((dof,N+1))*np.linspace(0,1,N+1)**4
#	err=np.max(np.abs(u_exact[:,N]-u_sol[:,N]))
#	return err




######################################################################################################
######################################################################################################
######################################################################################################

######################################################################################################
######################################################################################################
######################################################################################################


#def Inverse_Acoustic_Dirichlet(grid, p1_space, Dirichlet_data, delta_func,T,N):
#	import bempp.api
#	import numpy as np
#
#
#	# Setting parameters
#	dof=p1_space.global_dof_count
#	
#	L=2*N
#
#	dt=(T*1.0)/N
#
#	tol=10**(-16)
#
#	rho=tol**(1.0/(2*L))
#
#
#	# CALCULATING THE RHS
#
#	#Set up right hand sides
#################################################
#	#RHS_scal=np.zeros((1,N+1))
#	#RHS_scal=[CQ_RHS(x*1.0/N) for x in range(0,N+1)]
#	
#	RHS=np.zeros((dof,N+1))
#
#	for j in range(0,N+1):
#		tj=(T*1.0)*j*1.0/N
#		#RHS[:,j]=bempp.api.GridFunction(p1_space,fun=(lambda z : Dirichlet_data(z,tj)))
#		Dirichl=lambda x,n, domain_index,result :dirichlet_data(tj,x,n, domain_index,result)
#		RHS[:,j]=bempp.api.GridFunction(p1_space,fun=Dirichl).coefficients
#		
#	
#
#################################################################################################	
#	## SCALING AND FFT (STEP 1)
#	#RHS_scal=np.concatenate((rho**(np.linspace(0,N,N+1))*RHS_scal,np.zeros(L-N)),axis=0)
#	#print('RHS_scal')
#	#print(RHS_scal)
#	#RHS_scal_fft=np.fft.fft(RHS_scal)
######################################################################################################
#	
#
#	RHS_hat=1j*np.ones((dof,L+1))
#	RHS_fft=1j*np.ones((dof,L+1))
#	for j in range(0,dof):
#		RHS_hat[j,:]=np.concatenate((rho**(np.linspace(0,N,N+1))*RHS[j,:],np.zeros(L-N)),axis=0)
#		RHS_fft[j,:]=np.fft.fft(RHS_hat[j,:])
#
#	#Calculating the Unit Roots
#	Unit_Roots=np.exp(-1j*2*np.pi*np.linspace(0,L,L+1)/(L+1))
#
#	#Calculating zetavect
#	#Zeta_vect=1j*np.ones(L+1)
#	Zeta_vect=map(lambda y: delta_func( rho* y)/dt , Unit_Roots)
#	
#####################################################################################################
#	#Laps=map(Lap_f, Zeta_vect)
#	#SOLVING THE BLOCK DIAGONAL 
#	#phi_hat_scal=np.zeros(L+1)*1j
#	#for j in range(0,L+1):
#	#	phi_hat_scal[j]=RHS_scal_fft[j]/Laps[j] 				
#	#
#	#
#####################################################################################################	
#	## SOLVING THE BLOCK DIAGONAL SYSTEM (STEP 2)
#	phi_hat=1j*np.ones((dof,L+1))
#
#	from bempp.api.linalg.iterative_solvers import gmres
#	from bempp.api.linalg.iterative_solvers import cg
#
#	for j in range(0,L+1):
#		#Initializing the Operator
#		slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(p1_space,p1_space,p1_space,Zeta_vect[j])
#
#		#Initializing the Grid function on the right hand side
#		b=bempp.api.GridFunction(p1_space,coefficients=RHS_fft[:,j])
#		#Solving the system
#		if j == 0:
#			phi_grid,info = gmres(slp,b)
#		else:
#			phi_grid,info = gmres(slp,b)
#
#		#phi_grid,info = cg(slp,b,tol=10**(-16))
#		#Writing the computed coefficients in a new matrix u_hat
#		phi_hat[:,j]=phi_grid.coefficients
#
#	##RESCALING AND IFFT (STEP 3)
#########################################################################################################
# #Scalar case
	
#	#ift_phi_scal=1j*np.ones(L+1)
#	#phi_sol_scal=np.zeros(N+1)
#	
#	#ift_phi_scal=np.fft.ifft(phi_hat_scal)
#	#phi_sol_scal=np.real(rho**(-np.linspace(0,N,N+1))*ift_phi_scal[0:N+1])
#
#######################################################################################################
#	ift_phi=1j*np.ones((dof,L+1))
#	phi_sol=np.zeros((dof,N+1))
#	for j in range(0,dof):
#		ift_phi[j,:]=np.fft.ifft(phi_hat[j,:])
#		phi_sol[j,:]=np.real(rho**(-np.linspace(0,N,N+1))*ift_phi[j,0:N+1])
#
#	return phi_sol,grid,p1_space
######################################################################################################
######################################################################################################
######################################################################################################



#def Acoustic_error_plot(Dirichlet_data,delta_func,T):
#	import bempp.api
#	import numpy as np
#	import time
#	Amount_time=2
#	Amount_space=2
#	h_s=np.zeros(Amount_space)
#	dof_s=np.zeros(Amount_space)
#	err_Linf=np.zeros((Amount_space,Amount_time))
#	#err_eucl=np.zeros(Amount)
#	tau_s=np.zeros(Amount_time)
#	#Define grid
#
#
#
#
#	for i in range(0,Amount_space):
#		print('Space_index_INV:')			
#		print(i)
#		dx=2**(-i)
#		print(dx)
#		grid=bempp.api.shapes.sphere(h=dx)
#		#Define space
#		p1_space=bempp.api.function_space (grid, "P" , 1)
#		dof=p1_space.global_dof_count
#		dof_s[i]=dof
#		print('DOF:')
#		print(dof)
#
#		for j in range(0,Amount_time):
#
#			print('Time_index_INV:')			
#			print(j)
#			N=4*2**j
#
#
#
#			
#			phi_sol,grid,p1_space=Inverse_Acoustic_Dirichlet(grid,p1_space,Dirichlet_data,delta_func,T,N)
#			#print(phi_sol_scal)
#			#print(phi_sol[:,N])
#			#start=time.time()
#			#phi_sol_scal=Inverse_SCALAR(Dirichlet_data,delta_func,dx,Lap_f,T,N)
#			#end=time.time()
#			#print(end-start)
#			
#	
#			
#			## EVALUATIONS FOR PLOTTING 		#######################################################################################################################################################
#			#phi_ex=5.0/8.0*np.ones((dof,N+1))
#			phi_ex=8.0*np.ones((dof,N+1))
#			#phi_ex=(np.linspace(0,N,N+1)/(N))**3
#			#print(phi_ex)
#			tau_s[j]=T*1.0/N
#			h_s[i]=dx
#			#err_eucl[j]=np.abs(phi_ex[N]-	phi_sol_scal[N])
#			#err_Linf[j]=np.max(np.abs(phi_ex-phi_sol_scal))
#
#			#err_eucl[j]=np.linalg.norm(phi_sol[:,N]-phi_ex[:,N]) /dof
#			err_Linf[i,j]=np.max(np.abs(phi_sol[:,N]-phi_ex[:,N]))
#			k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
#			#print(err_eucl)
#			print(err_Linf)
#
#
#
#	#k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
#	#EX=6*(np.linspace(0,N,N+1)/(N))**2
#	#import matplotlib.pyplot as pyplot
#	#pyplot.plot(phi_sol[k,:])
#	#pyplot.plot(EX)
#	#pyplot.show()
#	#print(phi_ex[:,N]-phi_sol[:,N])
#			
#
#	return tau_s,h_s,dof_s,err_Linf












#################################################################################################################################################################
#import numpy as np
#delta_f1=lambda y : delta(y,1)
#delta_f2=lambda y : delta(y,2)
#dirichl=lambda y,t : t 
#Acoustic_solver(Dirichlet_data,delta_func,dx,T,N):

#dx=0.2
#T=1



#import bempp.api
#grid= bempp.api.shapes.sphere(h=dx)
#grid.plot()

#N=50
#u, plot_grid=Acoustic_solver(dirichlet_data,delta_f,dx,T,N)
#tau_s,err_Linf,err_eucl=Acoustic_error_plot(dirichlet_data,delta_f1,dx,T)
#tau_s,h_s,dof_s,err_Linf=Acoustic_error_plot(dirichlet_data,delta_f2,T)


#np.save('h_s_BIG',h_s)
#np.save('dof_s_BIG',dof_s)
#np.save('tau_s_BIG',tau_s)
#np.save('Err_space_Matrix',err_Linf)

#import matplotlib.pyplot as pyplot
#print(tau_s)
#print(err_Linf)
#print(err_eucl)
#pyplot.loglog(tau_s,err_Linf,basex=10, basey=10)
#pyplot.loglog(tau_s,err_Linf2,basex=10, basey=10)
#pyplot.loglog(tau_s,tau_s,basex=10, basey=10)
#pyplot.loglog(tau_s,tau_s**2,basex=10, basey=10)
#pyplot.show()









	

