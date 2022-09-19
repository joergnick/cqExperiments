from CQ_acoustic import *
import bempp.api
#from conv_op import *

class Conv_Model:
	import bempp.api
	import numpy as np
	OrderQF=5
	tol=10**(-8)
	#tol= np.finfo(float).eps
	bempp.api.global_parameters.hmat.eps=10**-6
	bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
	bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
	bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
	
	bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
	bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
	bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2


	bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
	bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3

	bempp.api.global_parameters.quadrature.double_singular = OrderQF
	

	def __init__(self ,grid,function_spaces,N,T,order):
		self.grid=grid
		self.function_spaces=function_spaces
		self.N=N
		self.T=T
		self.order=order
		self.delta=lambda zeta : delta(zeta,order)
		import bempp.api
		s=len(self.function_spaces)
		try:	
			dof=0
			self.dof_s=np.zeros(s)
			for index in range(0,s):
				self.dof_s[index]=(self.function_spaces[index]).global_dof_count
				dof+=(self.function_spaces[index]).global_dof_count
			self.dof=dof
		except:
			self.dof=0	
			print("No space attributed")
		

	@classmethod
	def from_values(cls ,dx,N,T,order,Amount_spaces):
		grid=bempp.api.shapes.sphere(h=dx)
		#Define space
		print("Create Function_space")
		p1_space=bempp.api.function_space(grid, "P" , 1)
		if Amount_spaces==1:
			spaces=[p1_space]
		else:
			spaces=[p1_space,p1_space]
		print("Call Constructor.")
		return cls(grid,spaces,N,T,order)

	def get_dof(self):
		import bempp.api
		dof=0

		s=len(self.function_spaces)
		
		for index in range(0,s):
			dof+=(self.function_spaces[index]).global_dof_count
		return dof

	def get_integration_parameters(self):
		N=self.N
		T=self.T
		tol=self.tol
		dt=(T*1.0)/N
		L=N
		rho=tol**(1.0/(2*L))
		return L,dt,tol,rho

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
					print("Multistep order not availible")

	def get_zeta_vect(self):
		L,dt,tol,rho=self.get_integration_parameters()
		import numpy as np
		#Calculating the Unit Roots
		Unit_Roots=np.exp(-1j*2*np.pi*(np.linspace(0,L,L+1)/(L+1)))
		#Calculating zetavect
		Zeta_vect=map(lambda y: self.delta( rho* y)/dt , Unit_Roots)
		return Zeta_vect

	def apply_convol(self,rhs):
		import numpy as np
		import bempp.api
		## Step 1
		print("In apply_convol")

	#	import matplotlib.pyplot as plt
	#	for j in range(0,len(rhs[:,0])):
	#		print("rhs : " , np.abs(rhs[j,:]))
	#		plt.plot(rhs[j,:])
	#	plt.show()

		rhs_fft=self.scale_fft(rhs)
		#print("rhs_fft:",rhs_fft)
		Zeta_vect=self.get_zeta_vect()
		#if np.abs(self.get_dof()-len(rhs_fft[:,0]))>0:
		#	raise Exception("Dimensions of operator and rhs do not agree.")
		dof=len(rhs[:,0])
		#print("dof="+str(dof))
		L,dt,tol,rho=self.get_integration_parameters()
		Half=int(np.ceil(float(L)/2.0))

		## Step 2

		phi_hat=1j*np.zeros((dof,L+1))

		normsRHS=np.ones(L+1)
		counter=0
		
		import matplotlib.pyplot as plt
	#	for j in range(0,dof):
	#		#print("rhs_fft : " , np.abs(rhs_fft[j,:]))
	#		plt.loglog(list(map(max,np.abs(rhs_fft[j,:]),np.ones(L+1)*10**-15)))
	#	plt.show()
		for j in range(0,Half+1):
			normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
			if normsRHS[j]>10**-8:
				counter=counter+1 
		#plt.loglog(normsRHS)
		#plt.show()
		print("Amount of Systems needed: "+ str(counter))
		#Timing the elliptic systems

		import time

		start=0
		end=0
		for j in range(0,Half+1):
			normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
		#	print("normRHS:",normsRHS[j])
			if normsRHS[j]>10**-8:
				print("j:",j,"L:",str(Half), "Time of previous iteration: " +str((end-start)/60), " MIN" )
				start=time.time()
				#print("NORM RHS: " ,normsRHS[j])			
				if j>0:
					phi_hat[:,j]=self.apply_elliptic_operator(Zeta_vect[j],rhs_fft[:,j],phi_hat[:,j-1])
				else:
					phi_hat[:,j]=self.apply_elliptic_operator(Zeta_vect[j],rhs_fft[:,j],np.zeros(dof))
				end=time.time()
		for j in range(Half+1,L+1):
			phi_hat[:,j]=np.conj(phi_hat[:,L+1-j])		

		## Step 3


		phi_sol=self.rescale_ifft(phi_hat)

		return phi_sol


	def apply_pot_convol(self,phi_sol,Points):
		import numpy as np
		import bempp.api
		## Step 1
		print("In apply pot")
		phi_fft=self.scale_fft(phi_sol)
		#print("rhs_fft:",rhs_fft)
		Zeta_vect=self.get_zeta_vect()
		#if np.abs(self.get_dof()-len(phi_fft[:,0]))>0:
		#	raise Exception("Dimensions of operator and rhs do not agree.")
		dof=len(phi_sol[:,0])
	
		L,dt,tol,rho=self.get_integration_parameters()
		Half=int(np.ceil(float(L)/2.0))
		## Step 2
		#print(Points)
		am_points=Points[0,:].size
		#print(am_points)
		phi_hat=1j*np.zeros((am_points,L+1))
		normsphi=np.ones(L+1)
		counter=0
		for j in range(0,Half+1):
			normsphi[j]=np.max(np.abs(phi_fft[:,j]))
			if normsphi[j]>10**-8:
				counter=counter+1
		print("Amount of Systems needed: "+ str(counter))

		for j in range(0,Half+1):
			normsphi[j]=np.max(np.abs(phi_fft[:,j]))

		#	print("normRHS:",normsRHS[j])
			if normsphi[j]>10**-8:
				print("j:",j,"L:",Half+1)
				phi_hat[:,j]=self.apply_pot_operator(Zeta_vect[j],phi_fft[:,j],Points)
		for j in range(Half+1,L+1):
			phi_hat[:,j]=np.conj(phi_hat[:,L+1-j])
			## Step 3


		phi_sol=self.rescale_ifft(phi_hat)

		return phi_sol

	def scale_fft(self,A):
		import numpy as np
		L,dt,tol,rho=self.get_integration_parameters()
		N=self.N
		n_rows=len(A[:,0])
		A_hat=1j*np.ones((n_rows,L+1))
		A_fft=1j*np.ones((n_rows,L+1))
		
		for j in range(0,n_rows):
			A_hat[j,:]=np.concatenate((rho**(np.linspace(0,N,N+1))*A[j,:],np.zeros(L-N)),axis=0)
			A_fft[j,:]=np.fft.fft(A_hat[j,:])
		return(A_fft)

	def rescale_ifft(self,A):
		import numpy as np
		L,dt,tol,rho=self.get_integration_parameters()
		N=self.N
		n_rows=len(A[:,0])
		ift_A=1j*np.ones((n_rows,L+1))
		A_sol=np.zeros((n_rows,N+1))
		for j in range(0,n_rows):
			ift_A[j,:]=np.fft.ifft(A[j,:])
			A_sol[j,:]=np.real(rho**(-np.linspace(0,N,N+1))*ift_A[j,0:N+1])
		return(A_sol)

	def apply_elliptic_operator(self,s,rhs,x0):
		raise NotImplementedError

	def apply_pot_operator(self,s,rhs,x0):
		raise NotImplementedError	

	def calc_err(self,num_sol,num_ex):	
		L,dt,tol,rho=self.get_integration_parameters()
		N=self.N
		err_mat=np.abs(num_sol-num_ex)
		
		err_mat=dt*(err_mat**2)
		err_vec=max(err_mat,axis=1)
		return np.sqrt(np.max(err_vec))

	def calc_L2_err(self,num_sol,num_ex,space):	
		print("In L2 err calculation")
		L,dt,tol,rho=self.get_integration_parameters()
		err_mat=np.abs(num_sol-num_ex)
		N_len=err_mat[0,:].size
		p1_space=self.function_spaces[0]
		id_op=bempp.api.operators.boundary.sparse.identity(space,space,space)
		id_disc=id_op.weak_form()

		L2_errs=np.zeros(N_len)
	
		for j in range(0,N_len):
			L2_errs[j]=np.dot(err_mat[:,j],id_disc*err_mat[:,j])
		L2_max = np.sqrt(max(L2_errs))
		L2_T = np.sqrt(L2_errs[N_len-1])
		return L2_max,L2_T

	def calc_H1_err(self,num_sol,num_ex,space):	
		L,dt,tol,rho=self.get_integration_parameters()
		err_mat=np.abs(num_sol-num_ex)
		N_len=err_mat[0,:].size
		p1_space=self.function_spaces[1]
		lb_op=bempp.api.operators.boundary.sparse.laplace_beltrami(space,space,space)
		lb_disc=lb_op.weak_form()

		H1_errs=np.zeros(N_len)	

		for j in range(0,N_len):
			H1_errs[j]=np.dot(err_mat[:,j],lb_disc*err_mat[:,j])

		H1_max=np.sqrt(max(H1_errs))
		H1_T=np.sqrt(H1_errs[N_len-1])	
		return H1_max,H1_T

class spherical_Incident_wave:
	import numpy as np
	import bempp.api
	C=-5
	t_0=3

#	def __init__(self):
		
	def get_params(self):
		return self.C,self.t_0

	def eval(self,t,x):
		C,t_0=self.get_params()
		x_trans=x
#		x_trans[0]=x[0]-0.5
#		x_trans[1]=x[1]-0.5
#		print(np.linalg.norm(x_trans))
		y=np.exp(C*(np.linalg.norm(x_trans)-t_0+t)**2)/np.linalg.norm(x_trans)
		return y
#	def eval_dot(self,t,x):
#		C,a,t_0=self.get_params()
#		y=np.exp(C*(np.linalg.norm(x)-t-t_0)**2)*2*C*(np.(a,x)-t-t_0)*-1
#		return y
#
#	def eval_ddot(self,t,x):
#		C,a,t_0=self.get_params()
#		y=np.exp(C*(np.dot(a,x)-t-t_0)**2)*(4*C**2*(np.dot(a,x)-t-t_0)**2+2*C)
#		return y
	
	def eval_dnormal(self,t,x,normal):
		import numpy as np
		C,t_0=self.get_params()
	#	halfs=1.0/2*np.ones((3))
	#	y=x-halfs
	#	i=np.argmax(np.abs(y))
		y=self.eval(t,x) 

		r=np.linalg.norm(x)
	#	print(r)
		z=(20*r-10*t*r-10*r**2-1)*y
		#z=(30*r-10*t*r-10*r**2-1)*y
	#	z=np.dot(normal,-x*(10*t*r-30*r+10*r**2+1)*1.0/r**2)*y
		#print("x=",x)
		#print("nu=",nu)
		return z


#	def eval_dnormal(self,t,x):
#		C,a,t_0=self.get_params()
#		ax=np.dot(a,x)	
#		y1=(ax-t-t_0)
#		y=ax*np.exp(C*y1**2)*2*C*y1
#		return y
	def get_gridfun(self,space,t):
		func_uinc=lambda x: self.eval(t,x)
		uinc_gridfun=bempp.api.GridFunction(p1_space,fun=func_uinc)
		return uinc_gridfun

class spherical_Model(Conv_Model):
	import numpy as np
	import bempp.api
	
	u_inc=spherical_Incident_wave()
	
	def apply_elliptic_operator(self,s,b,x0):
		#print("In apply_elliptic")
		#print(s)
		dp0_space=self.function_spaces[0]
		p1_space=self.function_spaces[1]
		
		dof0=dp0_space.global_dof_count
		dof1=p1_space.global_dof_count
		dof=dof0+dof1

		blocks=np.array([[None,None],[None,None]])
		## Definition of Operators

		slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
		dlp=bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,p1_space,s)
		adlp=bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,dp0_space,s)
		hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s,use_slp=True)

		ident_0=bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,dp0_space)
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		ident_10=bempp.api.operators.boundary.sparse.identity(p1_space,dp0_space,p1_space)
		
		## Bringing RHS into GridFuncion - type ;
		grid_rhs1=bempp.api.GridFunction(dp0_space,coefficients=b[:dof0])
		grid_rhs2=bempp.api.GridFunction(p1_space,coefficients=b[dof0:])
		#grid_imag=1j*bempp.api.GridFunction(p1_space,coefficients=np.imag(b[dof0:]))
		## Building Blocked System
		#grid_rhs2.plot()
	#	import matplotlib.pyplot as plt
	#	plt.plot(np.imag(b[dof0:]))
	#	plt.plot(np.imag(grid_rhs2.coefficients))
	#	plt.show()
		blocks=bempp.api.BlockedOperator(2,2)
		blocks[0,0] =(s*slp)
		blocks[0,1] = (dlp)-1.0/2*ident_1
		blocks[1,0] = (-adlp)+1.0/2*ident_0
		blocks[1,1] = (1.0/s*hslp+s**(-1.0/2)*ident_1)
		

		B_weak_form=blocks
		#B_weak_form=bempp.api.BlockedDiscreteOperator(np.array(blocks))
		#print("Weak_form_assembled")
		from bempp.api.linalg import gmres
		#from scipy.sparse.linalg import gmres
		sol,info=gmres(B_weak_form,[grid_rhs1,grid_rhs2],maxiter=300)
		#sol=B_weak_form*[grid_rhs1,grid_rhs2]
		if info>0:
			#res=np.linalg.norm(B_weak_form*sol-b)
			print("Linear Algebra warning")
		sol_ar=1j*np.zeros(dof0+dof1)
		sol_ar[:dof0]=sol[0].coefficients
		sol_ar[dof0:dof0+dof1]=sol[1].coefficients
		return sol_ar

	def apply_pot_operator(self,s,phi,Points):
		p1_space=self.function_spaces[0]
		dp0_space=self.function_spaces[1]	

		dof1=p1_space.global_dof_count
		dof0=dp0_space.global_dof_count
		
		dof=dof0+dof1

		slp_pot = bempp.api.operators.potential.modified_helmholtz.single_layer(dp0_space,Points,s)

		dlp_pot = bempp.api.operators.potential.modified_helmholtz.double_layer(p1_space, Points,s)
		
		varph=bempp.api.GridFunction(dp0_space,coefficients=phi[:dof0])
		psi=bempp.api.GridFunction(p1_space,coefficients=phi[dof0:dof])
		
		eval_slp=slp_pot*varph
		eval_dlp=dlp_pot*psi
		
		evaluated_elli_sol=(eval_slp+s**(-1)*eval_dlp)
		return evaluated_elli_sol


	def create_rhs(self):
		dof=self.dof
		dof1=int(self.dof_s[0])
		N=self.N
		Tend=self.T
		rhs=1j*np.zeros((dof,N+1))
		p1_space=self.function_spaces[0]

		u_inc=self.u_inc
		#u_sol=self.u_sol
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		u_incs=np.zeros((dof1,N+1))	
		pnu_incs=np.zeros((dof1,N+1))

		for j in range(0,N+1):
			t=j*Tend*1.0/N
			
			def u_inc_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval(t,x)
			def u_neu_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval_dnormal(t,x,normal)

			gridfun_inc=bempp.api.GridFunction(p1_space,fun=u_inc_fun)
			gridfun_neu=bempp.api.GridFunction(p1_space,fun=u_neu_fun)

			u_incs[:,j]=gridfun_inc.coefficients
			pnu_incs[:,j]=gridfun_neu.coefficients
		
		### Calculating rhs	
		#key
		def fract(s,b):
			return s**0.5*b 

		fract_der=Conv_Operator(fract)

		pt12_uinc=fract_der.apply_convol(u_incs,5,show_progress=False)

	#	import matplotlib.pyplot as plt
	#	for node in range(len(pnu_incs[:,0])):
	#		plt.plot(pt12_uinc[node,:])
	#	plt.show()
		rhs[int(self.dof_s[0]):dof,:]=-pt12_uinc+pnu_incs
#		rhs[int(self.dof_s[1]):dof,:]=pnu_incs

#		for j in range(0,N+1):
#			rhs[int(self.dof_s[1]):dof,j]=gridfun_rhs.coefficients
		
		return rhs

	def create_ex(self):
		dof=self.dof
		dof0=int(self.dof_s[0])
		dof1=int(self.dof_s[1])
		space_0=self.function_spaces[0]
		space_1=self.function_spaces[1]
		N=self.N
		Tend=self.T
		grid=self.grid
		u_inc=self.u_inc
		u_sol=self.u_sol
		phi_ex=np.ones((dof,N+1))

		for j in range(0,N+1):
			t=j*Tend*1.0/N
			def func_sol_dot(x,normal,domain_index,result):
				result[0]=  u_sol.eval_dot(t,x)
			def func_sol_mdnormal(x,normal,domain_index,result):
				result[0]= u_sol.eval_dnormal(t,x,normal)	
		
			gridfun_sol_dot=bempp.api.GridFunction(space_1,fun=func_sol_dot)
			gridfun_sol_mdnormal=bempp.api.GridFunction(space_1,fun=func_sol_mdnormal)

			phi_ex[0:dof0,j]=-gridfun_sol_mdnormal.coefficients
			phi_ex[dof0:dof,j]=gridfun_sol_dot.coefficients
			
		
		return phi_ex

	def calc_ref_psi(self):
		import scipy.io
		#workspace=scipy.io.loadmat('data/Ref_spherical.mat')
		
		N_ref=2**13
		T=5
		def combined_inverse(s,b):
			return (1+1.0/s+1.0/np.sqrt(s))**(-1)*b 
		
		def fract(s,b):
			return np.sqrt(s)*b 
		
		ons=np.ones(N_ref+1)
		tt=np.linspace(0,T,N_ref+1)
		
		u=np.exp(-5*(ons-(3*ons-tt))**2)
		pnu=(29*ons-10*tt-10*ons)*u
		fract_der=Conv_Operator(fract)
		
		rhs=-fract_der.apply_convol(u,T,show_progress=False)+pnu
		
		
		Aptm1=Conv_Operator(combined_inverse)
		ref_sol=Aptm1.apply_convol(rhs,T,show_progress=False)
		#ref_sol=workspace['ref_sol'][0]
		#N_ref=len(ref_sol)-1
		N=self.N

		speed=N_ref/N

		space_0=self.function_spaces[0]
		space_1=self.function_spaces[1]
		dof=self.dof_s[0]
		psi=np.ones((dof,N+1))
		
		for j in range(N+1):
			psi[:,j]=ref_sol[j*speed]*psi[:,j]
			
		return psi
		


class alternative_spherical_Model(Conv_Model):
	import numpy as np
	import bempp.api
	
	u_inc=spherical_Incident_wave()
	
	def apply_elliptic_operator(self,s,b,x0):
		#print("In apply_elliptic")
		#print(s)
		dp0_space = self.function_spaces[0]
		p1_space = self.function_spaces[1]
		
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
		grid_rhs1 = bempp.api.GridFunction(dp0_space,coefficients=b[:dof0])
		grid_rhs2 = bempp.api.GridFunction(p1_space,coefficients=b[dof0:dof])
	
		grid_ginc=grid_rhs2

##################### ABSORBING BC:
		#grid_ginc = -s**0.5*grid_rhs1+grid_rhs2
		## Building Blocked System

		blocks=bempp.api.BlockedOperator(2,2)
		blocks[0,0] =(s*slp)
		blocks[0,1] = (dlp)-1.0/2*ident_1
		blocks[1,0] = (-adlp)+1.0/2*ident_0
		#blocks[1,1] = (1.0/s*hslp+s**(-1.0/2)*ident_1)
		blocks[1,1] = 1.0/s*hslp	

		B_weak_form=blocks
		#B_weak_form=bempp.api.BlockedDiscreteOperator(np.array(blocks))
		#print("Weak_form_assembled")
		from bempp.api.linalg import gmres
		#from scipy.sparse.linalg import gmres
		sol,info=gmres(B_weak_form,[0*grid_rhs1,grid_ginc],maxiter=300)
		#sol=B_weak_form*[grid_rhs1,grid_rhs2]
		if info>0:
			#res=np.linalg.norm(B_weak_form*sol-b)
			print("Linear Algebra warning")
		sol_ar=1j*np.zeros(dof0+dof1)
		sol_ar[:dof0]=sol[0].coefficients
		sol_ar[dof0:dof0+dof1]=sol[1].coefficients
		return sol_ar

	def apply_pot_operator(self,s,phi,Points):
		p1_space=self.function_spaces[0]
		dp0_space=self.function_spaces[1]	

		dof1=p1_space.global_dof_count
		dof0=dp0_space.global_dof_count
		
		dof=dof0+dof1

		slp_pot = bempp.api.operators.potential.modified_helmholtz.single_layer(dp0_space,Points,s)

		dlp_pot = bempp.api.operators.potential.modified_helmholtz.double_layer(p1_space, Points,s)
		
		varph=bempp.api.GridFunction(dp0_space,coefficients=phi[:dof0])
		psi=bempp.api.GridFunction(p1_space,coefficients=phi[dof0:dof])
		
		eval_slp=slp_pot*varph
		eval_dlp=dlp_pot*psi
		
		evaluated_elli_sol=(eval_slp+s**(-1)*eval_dlp)
		return evaluated_elli_sol


	def create_rhs(self):
		dof=self.dof
		dof1=int(self.dof_s[0])
		N=self.N
		Tend=self.T
		rhs=np.zeros((dof,N+1))
		p1_space=self.function_spaces[0]

		u_inc=self.u_inc
		#u_sol=self.u_sol
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		u_incs=np.zeros((dof1,N+1))	
		pnu_incs=np.zeros((dof1,N+1))

		for j in range(0,N+1):
			t=j*Tend*1.0/N
			
			def u_inc_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval(t,x)
			def u_neu_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval_dnormal(t,x,normal)

			gridfun_inc=bempp.api.GridFunction(p1_space,fun=u_inc_fun)
			gridfun_neu=bempp.api.GridFunction(p1_space,fun=u_neu_fun)

			u_incs[:,j]=gridfun_inc.coefficients
			pnu_incs[:,j]=gridfun_neu.coefficients

			rhs[0:dof1,j]=gridfun_inc.coefficients
			rhs[dof1:2*dof1,j]=gridfun_neu.coefficients
		return rhs

	def create_ex(self):
		dof=self.dof
		dof0=int(self.dof_s[0])
		dof1=int(self.dof_s[1])
		space_0=self.function_spaces[0]
		space_1=self.function_spaces[1]
		N=self.N
		Tend=self.T
		grid=self.grid
		u_inc=self.u_inc
		u_sol=self.u_sol
		phi_ex=np.ones((dof,N+1))

		for j in range(0,N+1):
			t=j*Tend*1.0/N
			def func_sol_dot(x,normal,domain_index,result):
				result[0]=  u_sol.eval_dot(t,x)
			def func_sol_mdnormal(x,normal,domain_index,result):
				result[0]= u_sol.eval_dnormal(t,x,normal)	
		
			gridfun_sol_dot=bempp.api.GridFunction(space_1,fun=func_sol_dot)
			gridfun_sol_mdnormal=bempp.api.GridFunction(space_1,fun=func_sol_mdnormal)

			phi_ex[0:dof0,j]=-gridfun_sol_mdnormal.coefficients
			phi_ex[dof0:dof,j]=gridfun_sol_dot.coefficients
			
		
		return phi_ex

	def calc_ref_psi(self):
		import scipy.io
		#workspace=scipy.io.loadmat('data/Ref_spherical.mat')
		
		N_ref=2**13
		T=5
		def combined_inverse(s,b):
		#	return (1+1.0/s+1.0/np.sqrt(s))**(-1)*b 
			return (1+1.0/s)**(-1)*b	
		def fract(s,b):
	#		return np.sqrt(s)*b 
			return 0*b
		
		ons=np.ones(N_ref+1)
		tt=np.linspace(0,T,N_ref+1)
		
		u=np.exp(-5*(ons-(3*ons-tt))**2)
		pnu=(29*ons-10*tt-10*ons)*u
		fract_der=Conv_Operator(fract)
		
	#	rhs=-fract_der.apply_convol(u,T,show_progress=False)+pnu
		rhs=pnu	
		
		Aptm1=Conv_Operator(combined_inverse)
		ref_sol=Aptm1.apply_convol(rhs,T,show_progress=False)
		#ref_sol=workspace['ref_sol'][0]
		#N_ref=len(ref_sol)-1
		N=self.N

		speed=N_ref/N

		space_0=self.function_spaces[0]
		space_1=self.function_spaces[1]
		dof=self.dof_s[0]
		psi=np.ones((dof,N+1))
		
		for j in range(N+1):
			psi[:,j]=ref_sol[j*speed]*psi[:,j]
			
		return psi
		








class Incident_wave:
	import numpy as np
	import bempp.api
	def __init__(self,C,a,t_0):
		self.C=C
		self.a=a
		self.t_0=t_0
	def get_params(self):
		return self.C,self.a,self.t_0

	def eval(self,t,x):
		C,a,t_0=self.get_params()
		y=np.exp(C*(np.dot(a,x)-t-t_0)**2)
		return y
	def eval_dot(self,t,x):
		C,a,t_0=self.get_params()
		y=np.exp(C*(np.dot(a,x)-t-t_0)**2)*2*C*(np.dot(a,x)-t-t_0)*-1
		return y

	def eval_ddot(self,t,x):
		C,a,t_0=self.get_params()
		y=np.exp(C*(np.dot(a,x)-t-t_0)**2)*(4*C**2*(np.dot(a,x)-t-t_0)**2+2*C)
		return y

############## Cube#######################
#	def eval_dnormal(self,t,x,normal):
#		import numpy as np
#		C,a,t_0=self.get_params()
#		halfs=1.0/2*np.ones((3))
#		y=x-halfs
##		i=np.argmax(np.abs(y))
##		nu=np.zeros(3)
##		nu[i]=np.sign(y[i])
#		ax=np.dot(a,x)
#		y1=(ax-t-t_0)
#		#z=a[i]*np.sign(y[i])*np.exp(C*y1**2)*2*C*y1
#
#		z=np.dot(a,normal)*np.exp(C*y1**2)*2*C*y1
#		
#		#print("x=",x)
#		#print("nu=",nu)
#		return z


################# MAGNET##########################
	def eval_dnormal(self,t,x,normal):
#		if x[0]<10**-4:
#			normal[0]=-1
#			normal[1]=0
#			normal[2]=0
#		elif x[2]<10**-4:
#			normal[0]=0
#			normal[1]=0
#			normal[2]=-1
#		elif x[2]>1-10**-4:
#			normal[0]=0
#			normal[1]=0
#			normal[2]=1
#		elif np.abs(np.sqrt(x[0]**2+(x[1]-0.5)**2)-0.5)<10**-1:
#			normal[0]=x[0]
#			normal[1]=(x[1]-0.5)
#			normal[2]=0
#		elif np.abs(np.sqrt(x[0]**2+(x[1]-0.5)**2)-0.4)<10**-1:
#			normal[0]=-x[0]
#			normal[1]=-(x[1]-0.5)
#			normal[2]=0
#		else:
#			print("Some BS with Normal vector")
#		normal=normal/np.linalg.norm(normal)
		C,a,t_0=self.get_params()
		ax=np.dot(a,x)
		y1=(ax-t-t_0)

		z=np.dot(a,normal)*np.exp(C*y1**2)*2*C*y1
		return z		
	
		



#	def eval_nu(self,x):
#		halfs=1.0/2*np.ones((3))
#		y=x-halfs
#		i=np.argmax(np.abs(y))
#		nu=np.zeros(3)
#		nu[i]=np.sign(y[i])
#		return nu
#

#	def eval_dnormal(self,t,x):
#		C,a,t_0=self.get_params()
#		ax=np.dot(a,x)	
#		y1=(ax-t-t_0)
#		y=ax*np.exp(C*y1**2)*2*C*y1
#		return y
	def get_gridfun(self,space,t):
		func_uinc=lambda x: self.eval(t,x)
		uinc_gridfun=bempp.api.GridFunction(p1_space,fun=func_uinc)
		return uinc_gridfun

	def get_lbgridfun(self,space,t):
		lb=bempp.api.boundary.sparse.laplace_beltrami(space,space,space)
		gridfun=self.get_gridfun(space,t)
		return -lb*gridfun


class Dirichlet_Model(Conv_Model):

	#def __init__(self ,grid,function_spaces,N,T,order):
	#	Conv_Model.__init__(self ,grid,function_spaces,N,T,order)
	u_inc=Incident_wave(-50,np.array([-1,0,0]),-2.5)
	#u_inc=Incident_wave(-10,np.array([-1,0,0]),-2.5)

	def apply_elliptic_operator(self,s,b,x0):
		import bempp.api
		import scipy
		
		p1_space=(self.function_spaces[0])
		slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(p1_space,p1_space,p1_space,s)
		slp_discrete=slp.strong_form()
		sol,info=scipy.sparse.linalg.gmres(slp_discrete,b,tol=10**-8,maxiter=300,x0=x0)
		if info>0:
			print("Linear Algebra warning, info:", info)
		return sol
	def apply_pot_operator(self,s,phi,Points):
		p1_space=self.function_spaces[0]
	

		dof1=p1_space.global_dof_count

		
		dof=dof1

		slp_pot = bempp.api.operators.potential.modified_helmholtz.single_layer(p1_space,Points,s)


		
		varph=bempp.api.GridFunction(p1_space,coefficients=phi[:])

		eval_slp=slp_pot*varph

		return eval_slp

	def create_rhs(self):
		dof=self.dof
		N=self.N
		T=self.T
#		u_sol=self.u_sol
#		rhs=np.ones((dof,N+1))*np.exp(-100*(np.linspace(0,T,N+1)-0.5*np.ones(N+1))**2)
		rhs=np.zeros((dof,N+1))
		space=self.function_spaces[0]
		for j in range(0,N+1):
			t=T*1.0*j/N
	
			def u_func(x,normal,domain_index,result):
				result[0]=-self.u_inc.eval(t,x)			
			gridfun_u=bempp.api.GridFunction(space,fun=u_func)
			rhs[:,j]=gridfun_u.coefficients

		return rhs

	def create_ex(self):
		dof=self.dof
		N=self.N
		T=self.T	
		
		phi_ex=np.ones((dof,N+1))*np.exp(-100*(np.linspace(0,T,N+1)-0.5*np.ones(N+1))**2)*-400*(np.linspace(0,T,N+1)-0.5*np.ones(N+1))
		return phi_ex



class Test_Model(Conv_Model):

	u_sol=Incident_wave(-10,1.0/np.sqrt(3)*np.array([1,1,1]),-2.5)

	

	def apply_elliptic_operator(self,s,b,x0):
		p1_space=self.function_spaces[0]
		dp0_space=self.function_spaces[1]
		dof=self.dof
		dof0=p1_space.global_dof_count
		blocks=np.array([[None,None],[None,None]])
			
		slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
		dlp=bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,p1_space,s)
		adlp=bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,dp0_space,s)
		hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s)

		ident_0=bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,dp0_space)
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		lb=bempp.api.operators.boundary.sparse.laplace_beltrami(p1_space,p1_space,p1_space)


#		b[:dof0]=ident_1.weak_form()*b[:dof0]
#		b[dof0:dof]=ident_1.weak_form()*b[dof0:dof]
#
#		#b=rhs
##		blocks[0,0] =-(s*slp).weak_form()
#		blocks[0,1] = -(dlp).weak_form()
#		blocks[1,0] = -(-adlp).weak_form()
#		blocks[1,1] = -(1.0/s*hslp).weak_form()

#		B_weak_form=bempp.api.BlockedDiscreteOperator(blocks)

		grid_rhs1=bempp.api.GridFunction(dp0_space,coefficients=b[:dof0])
		grid_rhs2=bempp.api.GridFunction(p1_space,coefficients=b[dof0:dof])

		#b=rhs
		blocks[0,0] =-(s*slp)
		blocks[0,1] = -(dlp)
		blocks[1,0] = -(-adlp)
		blocks[1,1] = -(1.0/s*hslp)
		B_op=blocks
		#B_weak_form=bempp.api.BlockedDiscreteOperator(blocks)
		from scipy.sparse.linalg import gmres
		global it_count
		it_count = 0
		def iteration_counter(x):
			global it_count
			it_count += 1
		
		sol,info=gmres(B_weak_form,b,maxiter=300,tol=10**-8,callback=iteration_counter,x0=x0)
		print("Amount iterations:", it_count)
		
		#sol[:dof0],info=gmres(ident_0.weak_form(),sol[:dof0],maxiter=100)
		#sol[dof0:2*dof0],info=gmres(ident_1.weak_form(),sol[dof0:2*dof0],maxiter=100)		

		
		if info>0:
			print("Linear Algebra warning, info:", info)
		return sol
	def create_rhs(self):
		dof=self.dof
		dof0=int(self.dof_s[0])
		dof1=int(self.dof_s[1])
		N=self.N
		Tend=self.T
		rhs=np.zeros((dof,N+1))
		p1_space=self.function_spaces[0]
		space=self.function_spaces[0]
		u_sol=self.u_sol
		for j in range(0,N+1):
			t=Tend*1.0*j/N
	
			def u_func_dnormal(x,normal,domain_index,result):
				result[0]=u_sol.eval_dnormal(t,x,normal)			
			gridfun_udnormal=bempp.api.GridFunction(space,fun=u_func_dnormal)
			
			rhs[dof0:dof,j]=-1.0/2.0*gridfun_udnormal.coefficients

			def u_func_dot(x,normal,domain_index,result):
				result[0]=u_sol.eval_dot(t,x)			
			gridfun_udot=bempp.api.GridFunction(space,fun=u_func_dot)
			
			rhs[0:dof0,j]=1.0/2.0*gridfun_udot.coefficients

		return rhs

	def create_ex(self):
		dof=self.dof
		dof0=int(self.dof_s[0])
		dof1=int(self.dof_s[1])
		N=self.N
		Tend=self.T
		phi_ex=np.zeros((dof,N+1))
		p1_space=self.function_spaces[0]
		space=self.function_spaces[0]
		u_sol=self.u_sol
		for j in range(0,N+1):
			t=Tend*1.0*j/N

		

			def u_func_dnormal(x,normal,domain_index,result):
				result[0]=u_sol.eval_dnormal(t,x,normal)
			def u_func_dot(x,normal,domain_index,result):
				result[0]=u_sol.eval_dot(t,x)			
			gridfun_udot=bempp.api.GridFunction(space,fun=u_func_dot)
						
			gridfun_udnormal=bempp.api.GridFunction(space,fun=u_func_dnormal)
			
			phi_ex[0:dof0,j]=-gridfun_udnormal.coefficients

			phi_ex[dof0:dof,j]=gridfun_udot.coefficients

		return phi_ex



def assembly_operator(op):
	opweak_form=op.weak_form()
	return opweak_form

class Gderivative_Model(Conv_Model):
	 #corresponds to no Convolution
	def __init__(self, F,grid,function_spaces,N,T,order):
		self.F=F
		self.grid=grid
		self.function_spaces=function_spaces
		self.N=N
		self.T=T
		self.order=order
		self.delta=lambda zeta : delta(zeta,order)
		import bempp.api
		dof=0
		s=len(self.function_spaces)
		self.dof_s=np.zeros(s)
		for index in range(0,s):
			self.dof_s[index]=(self.function_spaces[index]).global_dof_count
			dof+=(self.function_spaces[index]).global_dof_count
		self.dof=dof

	def apply_elliptic_operator(self,s,b,x0):
		F=self.F
		sol=F(s)*b		
		return sol

	def create_rhs(self):
		import numpy as np
		N=self.N
		T=self.T
		rhs=np.zeros((N+1))	
	
		for j in range(0,N+1):
			t=j*T*1.0/N
			rhs[0,j]=t**2
		return rhs



class derivative_Model(Conv_Model):
	k=0 #corresponds to no Convolution
	def __init__(self, k,grid,function_spaces,N,T,order):
		self.k=k
		self.grid=grid
		self.function_spaces=function_spaces
		self.N=N
		self.T=T
		self.order=order
		self.delta=lambda zeta : delta(zeta,order)
		import bempp.api
		dof=0
		s=len(self.function_spaces)
		self.dof_s=np.zeros(s)
		for index in range(0,s):
			self.dof_s[index]=(self.function_spaces[index]).global_dof_count
			dof+=(self.function_spaces[index]).global_dof_count
		self.dof=dof

	def apply_elliptic_operator(self,s,b,x0):
		k=self.k
		sol=s**(k)*b		
		return sol

	def create_rhs(self):
		import numpy as np
		N=self.N
		T=self.T
		rhs=np.zeros((N+1))	
	
		for j in range(0,N+1):
			t=j*T*1.0/N
			rhs[0,j]=t**2
		return rhs



class FD_Model(Conv_Model):
	dof_s=np.zeros(1)
	def __init__(self, dof,grid,function_spaces,N,T,order):
		self.grid=grid
		self.function_spaces=function_spaces
		self.N=N
		self.T=T
		self.order=order
		self.delta=lambda zeta : delta(zeta,order)

		self.dof_s[0]=dof

		self.dof=dof

	def apply_elliptic_operator(self,s,b,x0):
		dof=len(b)
		A=1j*np.zeros((dof,dof))
		h=1.0/(dof)
		A[0,0]=s*1.0+s**(-1)*2.0/h**2
		for j in range(dof-1):
			A[j+1,j+1]=s*1.0+s**(-1)*2.0/h**2
			A[j+1,j]=-s**(-1)*1.0/h**2
			A[j,j+1]=-s**(-1)*1.0/h**2

		from scipy.sparse.linalg import gmres

		sol,info=gmres(s*A,b)

		return sol
	
	def create_rhs(self,dof):
		import numpy as np
		N=self.N
		T=self.T
		h=1.0/(dof-1)

		rhs=np.zeros((dof-2,N+1))	
	
		for j in range(0,N+1):
			t=j*T*1.0/N
			rhs[dof-3,j]=1.0/h**2*np.exp(-100*(1-t)**2)
		return rhs

class GIBC_Model(Conv_Model):
	import numpy as np
	import bempp.api
	
	disp_coeff=1	
	epsilon=10**(-1)
	u_inc=Incident_wave(-100,np.array([0,-1.0,0]),-3)
	dx=1	
	#u_inc=Incident_wave(-60,np.array([-1/np.sqrt(2),-1/np.sqrt(2),0]),-2.75)
	#u_sol=Incident_wave(-10,np.array([-1.0/np.sqrt(2),-1/np.sqrt(2),0]),-3)
	
	def apply_elliptic_operator(self,s,b,x0):
		#print("In apply_elliptic")
		#grid=bempp.api.shapes.cube(h=self.dx)
		#grid=bempp.api.import_grid('grids/magnet_h05_h01.msh')
		grid=self.grid
		dp0_space=bempp.api.function_space(grid, "P" ,1)
		p1_space=bempp.api.function_space(grid, "P" ,1)
		
		dof0=dp0_space.global_dof_count
		dof1=p1_space.global_dof_count
		dof=dof0+dof1

		blocks=np.array([[None,None],[None,None]])
#print("Definition of operators:")
		#print(np.real(s))

		slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
		#print("slp defined.")
		dlp=bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,p1_space,s)

		#print("dlp defined.")
		adlp=bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,dp0_space,s)
		#print("adlp defined.")
		#hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s)
		hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s,use_slp=True)

		ident_0=bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,dp0_space)
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		ident_10=bempp.api.operators.boundary.sparse.identity(p1_space,dp0_space,p1_space)
		lb=bempp.api.operators.boundary.sparse.laplace_beltrami(p1_space,dp0_space,p1_space)
		


		grid_rhs1=bempp.api.GridFunction(dp0_space,coefficients=b[:dof0])
		grid_rhs2=bempp.api.GridFunction(p1_space,coefficients=b[dof0:dof])
		#b[:dof0]=b[:dof0]
		#b[dof0:dof]=b[dof0:dof]
		#print("Begin Assembly:")
		#b=rhs
		blocks=bempp.api.BlockedOperator(2,2)
		blocks[0,0] =(s*slp)
		blocks[0,1] = (dlp)-1.0/2*ident_1
		blocks[1,0] =- (adlp)+1.0/2*ident_0
		blocks[1,1] = (1.0/s*hslp+self.epsilon*(s*ident_10+self.disp_coeff*1.0/s*lb))
		
#		blocks[0,0] =-(s*slp.weak_form())
#		blocks[0,1] = -(dlp).weak_form()-1.0/2*ident_1.weak_form()
#		blocks[1,0] = -(-adlp).weak_form()+1.0/2*ident_0.weak_form()
#		blocks[1,1] = -(1.0/s*hslp.weak_form()-self.epsilon*(s*ident_10.weak_form()-1.0/s*lb.weak_form()))




		B_weak_form=blocks
		#B_weak_form=bempp.api.BlockedDiscreteOperator(np.array(blocks))
		#print("Weak_form_assembled")
		from bempp.api.linalg import gmres
		#from scipy.sparse.linalg import gmres
		sol,info,it_count=gmres(B_weak_form,[grid_rhs1,grid_rhs2],return_iteration_count=True,restart=500,use_strong_form=True,maxiter=1000)
		print("System solved, amount iterations: "+str(it_count))
		if info>0:
			new_gridfuns=B_weak_form*sol
			diff=new_gridfuns[0]-grid_rhs1
			diff1=new_gridfuns[1]-grid_rhs2
#			res=np.linalg.norm(B_weak_form*sol-b)
			res=np.linalg.norm(diff.coefficients)+np.linalg.norm(diff1.coefficients)	
			res=np.linalg.norm(new_gridfuns[0].coefficients)+np.linalg.norm(new_gridfuns[1].coefficients)
			print("Linear Algebra warning"+str(res))
		#	B_weak=B_weak_form.weak_form()
		#	B_mat=bempp.api.as_matrix(B_weak)
		#	eigs=scipy.linalg.eig(B_mat)
			#for j in range(len(eigs)):
			#	print("Eigenvalue: ",eigs[j])
			
		sol_ar=1j*np.zeros((dof0+dof1))
		sol_ar[:dof0]=sol[0].coefficients
		sol_ar[dof0:dof0+dof1]=sol[1].coefficients		
		return sol_ar

	def apply_pot_operator(self,s,phi,Points):
		#grid=bempp.api.import_grid('grids/magnet_h05_h01.msh')
		grid=self.grid
		dp0_space=bempp.api.function_space(grid, "P" ,1)
		p1_space=bempp.api.function_space(grid, "P" ,1)
		dof1=p1_space.global_dof_count
		dof0=dp0_space.global_dof_count
		
		dof=dof0+dof1

		slp_pot = bempp.api.operators.potential.modified_helmholtz.single_layer(dp0_space,Points,s)

		dlp_pot = bempp.api.operators.potential.modified_helmholtz.double_layer(p1_space, Points,s)
		
		varph=bempp.api.GridFunction(dp0_space,coefficients=phi[:dof0])
		psi=bempp.api.GridFunction(p1_space,coefficients=phi[dof0:dof])
		
		eval_slp=slp_pot*varph
		eval_dlp=dlp_pot*psi
		
		evaluated_elli_sol=(eval_slp+s**(-1)*eval_dlp)
		return evaluated_elli_sol

	def create_rhs(self):
		grid=self.grid
		#grid=bempp.api.shapes.cube(h=self.dx)
		#grid=bempp.api.import_grid('grids/magnet_h05_h01.msh')
		dp0_space=bempp.api.function_space(grid, "P" ,1)
		p1_space=bempp.api.function_space(grid, "P" ,1)

		dof=dp0_space.global_dof_count+p1_space.global_dof_count
		dof1=p1_space.global_dof_count
		N=self.N
		Tend=self.T
		rhs=np.zeros((dof,N+1))

		epsilon=self.epsilon
		u_inc=self.u_inc
		#u_sol=self.u_sol
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		RHS=np.zeros((dof,N+1))	
		lb=bempp.api.operators.boundary.sparse.laplace_beltrami(p1_space,p1_space,p1_space)
		#lb_discrete=lb.strong_form()

		for j in range(0,N+1):
			t=j*Tend*1.0/N
			def func_fwlb(x,normal,domain_index,result):
				 result[0]= u_sol.eval_dnormal(t,x,normal)-u_inc.eval_dnormal(t,x,normal)-epsilon*(u_inc.eval_ddot(t,x)+u_sol.eval_ddot(t,x))

			def u_ddot_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval_ddot(t,x)
			def u_inc_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval(t,x)
			def u_neu_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval_dnormal(t,x,normal)

			gridfun_ddot=bempp.api.GridFunction(p1_space,fun=u_ddot_fun)
			gridfun_inc=bempp.api.GridFunction(p1_space,fun=u_inc_fun)
			gridfun_neu=bempp.api.GridFunction(p1_space,fun=u_neu_fun)

			#gridfun_rhs=-epsilon*(ident_1*gridfun_ddot+lb*gridfun_inc)+ident_1*gridfun_neu
			gridfun_rhs=-epsilon*(ident_1*gridfun_ddot+self.disp_coeff*lb*gridfun_inc)+ident_1*gridfun_neu
			
			#from bempp.api.linalg import gmres			
			#gridfun_rhs,info=gmres(ident_1,gridfun_rhs,maxiter=100)			
			rhs[dof1:dof,j]=gridfun_rhs.coefficients
		
		return rhs

	def create_ex(self):
		dof=self.dof
		dof0=int(self.dof_s[0])
		dof1=int(self.dof_s[1])
		space_0=self.function_spaces[0]
		space_1=self.function_spaces[1]
		N=self.N
		Tend=self.T
		grid=self.grid
		u_inc=self.u_inc
		u_sol=self.u_sol
		phi_ex=np.ones((dof,N+1))

		for j in range(0,N+1):
			t=j*Tend*1.0/N
			def func_sol_dot(x,normal,domain_index,result):
				result[0]=  u_sol.eval_dot(t,x)
			def func_sol_mdnormal(x,normal,domain_index,result):
				result[0]= u_sol.eval_dnormal(t,x,normal)	
		
			gridfun_sol_dot=bempp.api.GridFunction(space_1,fun=func_sol_dot)
			gridfun_sol_mdnormal=bempp.api.GridFunction(space_1,fun=func_sol_mdnormal)

			phi_ex[0:dof0,j]=-gridfun_sol_mdnormal.coefficients
			phi_ex[dof0:dof,j]=gridfun_sol_dot.coefficients
			
		
		return phi_ex

class GIBC_inner_Model(Conv_Model):
	import numpy as np
	import bempp.api
	
	epsilon=0
	u_inc=Incident_wave(-50,np.array([-1.0,0,0]),3)
	#u_inc=spherical_Incident_wave()
	#u_inc=Incident_wave(-60,np.array([-1/np.sqrt(2),-1/np.sqrt(2),0]),-2.75)
	#u_sol=Incident_wave(-10,np.array([-1.0/np.sqrt(2),-1/np.sqrt(2),0]),-3)
	
	def apply_elliptic_operator(self,s,b,x0):
		#print("In apply_elliptic")
		#print(s)
		dp0_space=self.function_spaces[0]
		p1_space=self.function_spaces[1]
		
		dof0=dp0_space.global_dof_count
		dof1=p1_space.global_dof_count
		dof=dof0+dof1

		blocks=np.array([[None,None],[None,None]])

		#print("Definition of operators:")
		#print(np.real(s))

		slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
		#print("slp defined.")
		dlp=bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,p1_space,s)

		#print("dlp defined.")
		adlp=bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,dp0_space,s)
		#print("adlp defined.")
		#hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s)
		hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s,use_slp=True)

		ident_0=bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,dp0_space)
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		ident_10=bempp.api.operators.boundary.sparse.identity(p1_space,dp0_space,p1_space)
		lb=bempp.api.operators.boundary.sparse.laplace_beltrami(p1_space,dp0_space,p1_space)
		


		grid_rhs1=bempp.api.GridFunction(dp0_space,coefficients=b[:dof0])
		grid_rhs2=bempp.api.GridFunction(p1_space,coefficients=b[dof0:dof])

		blocks=bempp.api.BlockedOperator(2,2)
		blocks[0,0] =(s*slp)
		blocks[0,1] = (-dlp)-1.0/2*ident_1
		blocks[1,0] = (adlp)+1.0/2*ident_0
		blocks[1,1] = (1.0/s*hslp+self.epsilon*(s*ident_10+1.0/s*lb))
		



		B_weak_form=blocks
		#B_weak_form=bempp.api.BlockedDiscreteOperator(np.array(blocks))
		#print("Weak_form_assembled")
		from bempp.api.linalg import gmres
		#from scipy.sparse.linalg import gmres
		sol,info=gmres(B_weak_form,[grid_rhs1,grid_rhs2],maxiter=500)
		if info>0:
			new_gridfuns=B_weak_form*sol
			diff=new_gridfuns[0]-grid_rhs1
			diff1=new_gridfuns[1]-grid_rhs2
#			res=np.linalg.norm(B_weak_form*sol-b)
			res=np.linalg.norm(diff.coefficients)+np.linalg.norm(diff1.coefficients)	
			res=np.linalg.norm(new_gridfuns[0].coefficients)+np.linalg.norm(new_gridfuns[1].coefficients)
			print("Linear Algebra warning"+str(res))

		sol_ar=1j*np.zeros((dof0+dof1))
		sol_ar[:dof0]=sol[0].coefficients
		sol_ar[dof0:dof0+dof1]=sol[1].coefficients		
		return sol_ar

	def apply_pot_operator(self,s,phi,Points):
		p1_space=self.function_spaces[0]
		dp0_space=self.function_spaces[1]	

		dof1=p1_space.global_dof_count
		dof0=dp0_space.global_dof_count
		
		dof=dof0+dof1

		slp_pot = bempp.api.operators.potential.modified_helmholtz.single_layer(dp0_space,Points,s)

		dlp_pot = bempp.api.operators.potential.modified_helmholtz.double_layer(p1_space, Points,s)
		
		varph=bempp.api.GridFunction(dp0_space,coefficients=phi[:dof0])
		psi=bempp.api.GridFunction(p1_space,coefficients=phi[dof0:dof])
		
		eval_slp=slp_pot*varph
		eval_dlp=dlp_pot*psi
		
		evaluated_elli_sol=(eval_slp-s**(-1)*eval_dlp)
		return evaluated_elli_sol

		
#	def create_rhs(self):
#		dof=self.dof
#		dof1=int(self.dof_s[0])
#		N=self.N
#		Tend=self.T
#		rhs=np.zeros((dof,N+1))
#		p1_space=self.function_spaces[0]
#
#		epsilon=self.epsilon
#		u_inc=self.u_inc
#		u_sol=self.u_sol
#		ident_1=bempp.api.operators.boundarypp.api..sparse.identity(p1_space,p1_space,p1_space)
#		RHS=np.zeros((dof,N+1))	
#		lb=bempp.api.operators.boundary.sparse.laplace_beltrami(p1_space,p1_space,p1_space)
#		#lb_discrete=lb.strong_form()
#
#		for j in range(0,N+1):
#			t=j*Tend*1.0/N
#			def func_fwlb(x,normal,domain_index,result):
#				 result[0]= u_sol.eval_dnormal(t,x)-u_inc.eval_dnormal(t,x)-epsilon*(u_inc.eval_ddot(t,x)+u_sol.eval_ddot(t,x))
#
#
#
##			func_fwlb=lambda (x,normal,domain_index,result): u_sol.eval_dnormal(t,x)-u_inc.eval_dnormal(t,x)-epsilon*(u_inc.eval_ddot(t,x)+u_sol.eval_ddot(t,x))
#
#			def func_sum(x,normal,domain_index,result):
#				result[0]=  epsilon*(u_inc.eval(t,x)+u_sol.eval(t,x))
#
#			gridfun_fwlb=bempp.api.GridFunction(p1_space,fun=func_fwlb)
#			
#			gridfun_sum=bempp.api.GridFunction(p1_space,fun=func_sum)
#			gridfun_f=gridfun_fwlb+lb*gridfun_sum
#			
#
#			def func_half_rhs(x,normal,domain_index,result):
#				result[0]=  -epsilon*u_inc.eval_ddot(t,x)+u_inc.eval_dnormal(t,x)
#
#			def func_dnormal(x,normal,domain_index,result):
#				result[0]=  u_inc.eval_dnormal(t,x)
#	
#			gridfun_half_rhs=bempp.api.GridFunction(p1_space,fun=func_half_rhs)
#			#gridfun_half_rhs.plot()
#			gridfun_dnormal= bempp.api.GridFunction(p1_space,fun=func_dnormal)
#			
#			gridfun_rhs=ident_1*gridfun_half_rhs+epsilon*lb*gridfun_dnormal-ident_1*gridfun_f
#			#BullshitTEST!
#			gridfun_rhs=ident_1*gridfun_half_rhs
##			#diff_eps=lambda(x):-epsilon*(u_inc.eval_ddot(t,x)+u_sol.eval_ddot(t,x))
##
##			gridfun_lb=lb*bempp.api.GridFunction(p1_space,fun=func_sum)
##
##			gridfun_f=bempp.api.GridFunction(p1_space,fun=func_rhs)+gridfun_lb
##			gridfun_inh=bempp.api.GridFunction(p1_space,fun=func_rhs)
#			#gridfun_rhs.plot()
#			rhs[dof1:dof,j]=gridfun_rhs.coefficients
#			#print("RHSnormCREATION:",np.linalg.norm(rhs[dof1:dof,j]))
#		return rhs
	def create_rhs(self):
		dof=self.dof
		dof1=int(self.dof_s[0])
		N=self.N
		Tend=self.T
		rhs=np.zeros((dof,N+1))
		p1_space=self.function_spaces[0]

		epsilon=self.epsilon
		u_inc=self.u_inc
		#u_sol=self.u_sol
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		RHS=np.zeros((dof,N+1))	
		lb=bempp.api.operators.boundary.sparse.laplace_beltrami(p1_space,p1_space,p1_space)
		#lb_discrete=lb.strong_form()

		for j in range(0,N+1):
			t=j*Tend*1.0/N
			def func_fwlb(x,normal,domain_index,result):
				 result[0]= u_sol.eval_dnormal(t,x,normal)-u_inc.eval_dnormal(t,x,normal)-epsilon*(u_inc.eval_ddot(t,x)+u_sol.eval_ddot(t,x))

			def u_ddot_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval_ddot(t,x)
			def u_inc_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval(t,x)
			def u_neu_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval_dnormal(t,x,normal)

			gridfun_ddot=bempp.api.GridFunction(p1_space,fun=u_ddot_fun)
			gridfun_inc=bempp.api.GridFunction(p1_space,fun=u_inc_fun)
			gridfun_neu=bempp.api.GridFunction(p1_space,fun=u_neu_fun)

			#gridfun_rhs=-epsilon*(ident_1*gridfun_ddot+lb*gridfun_inc)+ident_1*gridfun_neu
			gridfun_rhs=-epsilon*(ident_1*gridfun_ddot+lb*gridfun_inc)-ident_1*gridfun_neu
			
			#from bempp.api.linalg import gmres			
			#gridfun_rhs,info=gmres(ident_1,gridfun_rhs,maxiter=100)			
			rhs[int(self.dof_s[1]):dof,j]=gridfun_rhs.coefficients
				
		return rhs

	def create_ex(self):
		dof=self.dof
		dof0=int(self.dof_s[0])
		dof1=int(self.dof_s[1])
		space_0=self.function_spaces[0]
		space_1=self.function_spaces[1]
		N=self.N
		Tend=self.T
		grid=self.grid
		u_inc=self.u_inc
		u_sol=self.u_sol
		phi_ex=np.ones((dof,N+1))

		for j in range(0,N+1):
			t=j*Tend*1.0/N
			def func_sol_dot(x,normal,domain_index,result):
				result[0]=  u_sol.eval_dot(t,x)
			def func_sol_mdnormal(x,normal,domain_index,result):
				result[0]= u_sol.eval_dnormal(t,x,normal)	
		
			gridfun_sol_dot=bempp.api.GridFunction(space_1,fun=func_sol_dot)
			gridfun_sol_mdnormal=bempp.api.GridFunction(space_1,fun=func_sol_mdnormal)

			phi_ex[0:dof0,j]=-gridfun_sol_mdnormal.coefficients
			phi_ex[dof0:dof,j]=gridfun_sol_dot.coefficients
			
		
		return phi_ex

class General_Model(Conv_Model):
	import numpy as np
	import bempp.api
	
	u_inc=Incident_wave(-100,np.array([0,-1.0,0]),-3)
#	u_inc=spherical_Incident_wave()
	#u_inc=Incident_wave(-60,np.array([-1/np.sqrt(2),-1/np.sqrt(2),0]),-2.75)
	#u_sol=Incident_wave(-10,np.array([-1.0/np.sqrt(2),-1/np.sqrt(2),0]),-3)
	def __init__(self, F,grid,function_spaces,N,T,order):
		self.F=F
		self.grid=grid
		self.function_spaces=function_spaces
		self.N=N
		self.T=T
		self.order=order
		self.delta=lambda zeta : delta(zeta,order)
		import bempp.api
		dof=0
		s=len(self.function_spaces)
		self.dof_s=np.zeros(s)
		for index in range(0,s):
			self.dof_s[index]=(self.function_spaces[index]).global_dof_count
			dof+=(self.function_spaces[index]).global_dof_count
		self.dof=dof


	def apply_elliptic_operator(self,s,b,x0):
		#print("In apply_elliptic")
		#print(s)
		dp0_space=self.function_spaces[0]
		p1_space=self.function_spaces[1]
		
		dof0=dp0_space.global_dof_count
		dof1=p1_space.global_dof_count
		dof=dof0+dof1

		blocks=np.array([[None,None],[None,None]])

		#print("Definition of operators:")
		#print(np.real(s))

		slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
		#print("slp defined.")
		dlp=bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,p1_space,s)

		#print("dlp defined.")
		adlp=bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,dp0_space,s)
		#print("adlp defined.")
		#hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s)
		hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s,use_slp=True)

		ident_0=bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,dp0_space)
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		ident_10=bempp.api.operators.boundary.sparse.identity(p1_space,dp0_space,p1_space)
		lb=bempp.api.operators.boundary.sparse.laplace_beltrami(p1_space,dp0_space,p1_space)
		


		grid_rhs1=bempp.api.GridFunction(dp0_space,coefficients=b[:dof0])
		grid_rhs2=bempp.api.GridFunction(p1_space,coefficients=b[dof0:dof])
		#b[:dof0]=b[:dof0]
		#b[dof0:dof]=b[dof0:dof]
		#print("Begin Assembly:")
		#b=rhs
		blocks=bempp.api.BlockedOperator(2,2)
		blocks[0,0] =(s*slp)
		blocks[0,1] = (dlp)-1.0/2*ident_1
		blocks[1,0] =- (adlp)+1.0/2*ident_0
		blocks[1,1] = (1.0/s*hslp+self.F(s)*1.0/s*ident_0)


#		blocks[0,0] =-(s*slp.weak_form())
#		blocks[0,1] = -(dlp).weak_form()-1.0/2*ident_1.weak_form()
#		blocks[1,0] = -(-adlp).weak_form()+1.0/2*ident_0.weak_form()
#		blocks[1,1] = -(1.0/s*hslp.weak_form()-self.epsilon*(s*ident_10.weak_form()-1.0/s*lb.weak_form()))




		B_weak_form=blocks
		#B_weak_form=bempp.api.BlockedDiscreteOperator(np.array(blocks))
		#print("Weak_form_assembled")
		from bempp.api.linalg import gmres
		#from scipy.sparse.linalg import gmres
		sol,info,it_count=gmres(B_weak_form,[grid_rhs1,grid_rhs2],return_iteration_count=True,restart=500,use_strong_form=True,maxiter=1000)
		print("System solved, amount iterations: "+str(it_count))
		if info>0:
			new_gridfuns=B_weak_form*sol
			diff=new_gridfuns[0]-grid_rhs1
			diff1=new_gridfuns[1]-grid_rhs2
#			res=np.linalg.norm(B_weak_form*sol-b)
			res=np.linalg.norm(diff.coefficients)+np.linalg.norm(diff1.coefficients)	
			res=np.linalg.norm(new_gridfuns[0].coefficients)+np.linalg.norm(new_gridfuns[1].coefficients)
			print("Linear Algebra warning"+str(res))
		#	B_weak=B_weak_form.weak_form()
		#	B_mat=bempp.api.as_matrix(B_weak)
		#	eigs=scipy.linalg.eig(B_mat)
			#for j in range(len(eigs)):
			#	print("Eigenvalue: ",eigs[j])
			
		sol_ar=1j*np.zeros((dof0+dof1))
		sol_ar[:dof0]=sol[0].coefficients
		sol_ar[dof0:dof0+dof1]=sol[1].coefficients		
		return sol_ar
	

	def apply_pot_operator(self,s,phi,Points):
		p1_space=self.function_spaces[0]
		dp0_space=self.function_spaces[1]	

		dof1=p1_space.global_dof_count
		dof0=dp0_space.global_dof_count
		
		dof=dof0+dof1

		slp_pot = bempp.api.operators.potential.modified_helmholtz.single_layer(dp0_space,Points,s)

		dlp_pot = bempp.api.operators.potential.modified_helmholtz.double_layer(p1_space, Points,s)
		
		varph=bempp.api.GridFunction(dp0_space,coefficients=phi[:dof0])
		psi=bempp.api.GridFunction(p1_space,coefficients=phi[dof0:dof])
		
		eval_slp=slp_pot*varph
		eval_dlp=dlp_pot*psi
		
		evaluated_elli_sol=(eval_slp+s**(-1)*eval_dlp)
		return evaluated_elli_sol


	def create_rhs(self):
		dof=self.dof
		dof1=int(self.dof_s[0])
		N=self.N
		Tend=self.T
		rhs=np.zeros((dof,N+1))
		p1_space=self.function_spaces[0]

		u_inc=self.u_inc
		#u_sol=self.u_sol
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		u_incs=np.zeros((dof1,N+1))	
		pnu_incs=np.zeros((dof1,N+1))

		lb=bempp.api.operators.boundary.sparse.laplace_beltrami(p1_space,p1_space,p1_space)
		#lb_discrete=lb.strong_form()

		for j in range(0,N+1):
			t=j*Tend*1.0/N
			def func_fwlb(x,normal,domain_index,result):
				 result[0]= u_sol.eval_dnormal(t,x,normal)-u_inc.eval_dnormal(t,x,normal)-epsilon*(u_inc.eval_ddot(t,x)+u_sol.eval_ddot(t,x))

			#def u_ddot_fun(x,normal,domain_index,result):
			#	result[0]=u_inc.eval_ddot(t,x)
			def u_inc_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval(t,x)
			def u_neu_fun(x,normal,domain_index,result):
				result[0]=u_inc.eval_dnormal(t,x,normal)

			#gridfun_ddot=bempp.api.GridFunction(p1_space,fun=u_ddot_fun)
			gridfun_inc=bempp.api.GridFunction(p1_space,fun=u_inc_fun)
			gridfun_neu=bempp.api.GridFunction(p1_space,fun=u_neu_fun)

			u_incs[:,j]=gridfun_inc.coefficients
			pnu_incs[:,j]=gridfun_neu.coefficients
			
		#	gridfun_rhs=ident_1*gridfun_neu
		F_mod=Gderivative_Model(self.F,self.grid,[self.function_spaces[0]],N,Tend,2)
		
		Fpt_uinc=F_mod.apply_convol(u_incs)
		rhs[int(self.dof_s[1]):dof,:]=-Fpt_uinc+pnu_incs
#		rhs[int(self.dof_s[1]):dof,:]=pnu_incs

#		for j in range(0,N+1):
#			rhs[int(self.dof_s[1]):dof,j]=gridfun_rhs.coefficients
		
		return rhs

	def create_ex(self):
		dof=self.dof
		dof0=int(self.dof_s[0])
		dof1=int(self.dof_s[1])
		space_0=self.function_spaces[0]
		space_1=self.function_spaces[1]
		N=self.N
		Tend=self.T
		grid=self.grid
		u_inc=self.u_inc
		u_sol=self.u_sol
		phi_ex=np.ones((dof,N+1))

		for j in range(0,N+1):
			t=j*Tend*1.0/N
			def func_sol_dot(x,normal,domain_index,result):
				result[0]=  u_sol.eval_dot(t,x)
			def func_sol_mdnormal(x,normal,domain_index,result):
				result[0]= u_sol.eval_dnormal(t,x,normal)	
		
			gridfun_sol_dot=bempp.api.GridFunction(space_1,fun=func_sol_dot)
			gridfun_sol_mdnormal=bempp.api.GridFunction(space_1,fun=func_sol_mdnormal)

			phi_ex[0:dof0,j]=-gridfun_sol_mdnormal.coefficients
			phi_ex[dof0:dof,j]=gridfun_sol_dot.coefficients
			
		
		return phi_ex







## OLD STUFF ###########################################################

class Neumann_Model(Conv_Model):
	import numpy as np
	import bempp.api
	#u_inc=Incident_wave(-10,np.array([1,0,0]),-2.5)3
	
	u_sol=Incident_wave(-10,np.array([1,0,0]),-2)

	def apply_elliptic_operator(self,s,b,x0):	
		dp0_space=self.function_spaces[0]
		p1_space=self.function_spaces[1]
		dof=self.dof
		dof0=int(self.dof_s[0])
		dof1=int(self.dof_s[1])

		blocks=np.array([[None,None],[None,None]])
			
		slp=bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,p1_space,dp0_space,s)
		dlp=bempp.api.operators.boundary.modified_helmholtz.double_layer(p1_space,p1_space,dp0_space,s)
		adlp=bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(dp0_space,dp0_space,p1_space,s)
		hslp=bempp.api.operators.boundary.modified_helmholtz.hypersingular(p1_space,dp0_space,p1_space,s)

		ident_0=bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,p1_space)
		ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,dp0_space)

		ident_00=bempp.api.operators.boundary.sparse.identity(dp0_space,dp0_space,dp0_space)
		ident_11=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
		lb=bempp.api.operators.boundary.sparse.laplace_beltrami(p1_space,p1_space,p1_space)



		b[:dof1]=ident_11.weak_form()*b[:dof1]
		b[dof1:dof]=ident_00.weak_form()*b[dof1:dof]

		#b=rhs
		blocks[0,0] =-(s*slp).weak_form()
		blocks[0,1] = -(dlp).weak_form()-1.0/2*ident_1.weak_form()
		blocks[1,0] = -(-adlp).weak_form()+1.0/2*ident_0.weak_form()
		blocks[1,1] = -(1.0/s*hslp).weak_form()

		#print(bempp.api.as_matrix(blocks[0,0]).shape)
		#print(bempp.api.as_matrix(blocks[0,1]).shape)
		#print(bempp.api.as_matrix(blocks[1,0]).shape)	
		#print(bempp.api.as_matrix(blocks[1,1]).shape)



		B_weak_form=bempp.api.BlockedDiscreteOperator(blocks)
		print("Weak_form_assembled")
		global it_count
		it_count = 0 
		def iteration_counter(x):
			global it_count
			it_count += 1
		
		from scipy.sparse.linalg import gmres
		sol,info=gmres(B_weak_form,b,maxiter=500,tol=10**-8,callback=iteration_counter,x0=x0)
		print("System solved, Amount iterations:", it_count)
		print("x_0-x_k",np.linalg.norm(sol-x0))
		#sol[:dof0],info=gmres(ident_0.weak_form(),sol[:dof0],maxiter=100)
		#sol[dof0:2*dof0],info=gmres(ident_1.weak_form(),sol[dof0:2*dof0],maxiter=100)		
		
		
		if info>0:
			res=np.linalg.norm(B_weak_form*sol-b)
			print("Linear Algebra warning, res:", res)
		return sol			
	
	def create_rhs(self):
		dof=self.dof
		dof0=int(self.dof_s[0])
		dof1=int(self.dof_s[1])
		N=self.N
		Tend=self.T
		rhs=np.zeros((dof,N+1))
		dp0_space=self.function_spaces[0]
		p1_space=self.function_spaces[1]
		u_sol=self.u_sol
		for j in range(0,N+1):
			t=Tend*1.0*j/N
	
			def u_func(x,normal,domain_index,result):
				result[0]=-u_sol.eval_dnormal(t,x,normal)			
			gridfun_u=bempp.api.GridFunction(dp0_space,fun=u_func)
			
			rhs[dof1:dof,j]=gridfun_u.coefficients
		return rhs

	def create_ex(self):
		dof=self.dof
		dof0=int(self.dof_s[0])
		dof1=int(self.dof_s[1])
		N=self.N
		Tend=self.T
		phi_ex=np.zeros((dof,N+1))
		dp0_space=self.function_spaces[0]
		p1_space=self.function_spaces[1]
		u_sol=self.u_sol
		for j in range(0,N+1):
			t=Tend*1.0*j/N
	
			def u_func_dnormal(x,normal,domain_index,result):
				result[0]=u_sol.eval_dnormal(t,x,normal)			
			gridfun_udnormal=bempp.api.GridFunction(dp0_space,fun=u_func_dnormal)
			
			phi_ex[0:dof0,j]=-gridfun_udnormal.coefficients

			def u_func_dot(x,normal,domain_index,result):
				result[0]=u_sol.eval_dot(t,x)			
			gridfun_udot=bempp.api.GridFunction(p1_space,fun=u_func_dot)
			
			phi_ex[dof0:dof,j]=gridfun_udot.coefficients

		return phi_ex


