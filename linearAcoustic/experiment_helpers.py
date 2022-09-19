from acoustic_models import *
import bempp.api
from conv_op import *


def create_ref_err(Amount_time,dx,boundary_cond):
	#import bempp.api
	errn=np.zeros(Amount_time)
	errd=np.zeros(Amount_time)
	err=np.zeros(Amount_time)
	tau_s=np.zeros(Amount_time)

	T=5
	#Creating reference solution
	MaxN=4*2**(Amount_time)
	N=MaxN
	#N=4097
	u_inc=spherical_Incident_wave()
	tau_small=T*1.0/MaxN
	if boundary_cond=="Dirichlet":
		mod=Dirichlet_Model.from_values(dx,N,T,2,1)
	elif boundary_cond=="GIBC":
		#mod=GIBC_Model.from_values(dx,N,T,2,2)
	        #from_values(cls ,dx,N,T,order,Amount_spaces) 
		#mod.OrderQF=np.max(6,mod.OrderQF-i)
		#mod=Neumann_Model.from_values(dx,N,T,2,2)
		order=2
		grid=bempp.api.shapes.cube(h=dx)
		#Define space
		dp0_space=bempp.api.function_space(grid,"P",1)
		print("dof0:"+str(dp0_space.global_dof_count))
		p1_space=bempp.api.function_space(grid, "P" ,1)
		print("dof1:"+str(p1_space.global_dof_count))
		spaces=[dp0_space,p1_space]

		mod=GIBC_Model(grid,spaces,N,T,order)

	elif boundary_cond=="Neumann":
		#mod=Neumann_Model.from_values(dx,N,T,2,2)
		order=2
		grid=bempp.api.shapes.sphere(h=dx)
		#Define space
		dp0_space=bempp.api.function_space(grid,"P",1)
		p1_space=bempp.api.function_space(grid, "P" ,1)
		spaces=[p1_space,p1_space]
		mod=Neumann_Model(grid,spaces,N,T,order)
	elif boundary_cond=="Test":
		mod=Test_Model.from_values(dx,N,T,2,2)
	print("Model created.")

	rhs=mod.create_rhs()
		
	phi_ex=mod.apply_convol(rhs)

	import matplotlib.pyplot as plt
#	plt.plot(phi_ex[2,:],label='ref_sol')
#	plt.plot(phi_ex[20,:],label='ref_sol2')
#	plt.legend()
#	plt.show()
	#phi_ex=np.load('phi_ref_T5_h05_N1024.npy')
	print("Size phi_ex:"+str(phi_ex[0,:].size))
	tau_small=T*1.0/(phi_ex[0,:].size-1)
	#print(phi_ex)
	np.save('data/phi_ref_T5_h05_N2h15',phi_ex)
	#solving the system with other timesteps
	for j in range(0,Amount_time):
		print('Time_index: '+str(j))			
		N=4*2**(j)
		tau_s[j]=T*1.0/N
		if boundary_cond=="Dirichlet":
			mod=Dirichlet_Model.from_values(dx,N,T,2,1)
		elif boundary_cond=="GIBC":
			order=2
			#print("create mod")
			#mod=GIBC_Model.from_values(dx,N,T,2,2)
			#print("mod created")
			#mod.OrderQF=np.max(6,mod.OrderQF-i)
			#p1_space=mod.function_spaces[1]
			#dp0_space=mod.function_spaces[0]
			mod=GIBC_Model(grid,spaces,N,T,order)
		elif boundary_cond=="Neumann":
			#mod=Neumann_Model.from_values(dx,N,T,2,2)
			order=2
			grid=bempp.api.shapes.sphere(h=dx)
			#Define space
			dp0_space=bempp.api.function_space(grid,"DP",0)
			p1_space=bempp.api.function_space(grid, "P" ,1)
			spaces=[dp0_space,p1_space]
			mod=Neumann_Model(grid,spaces,N,T,order)
		elif boundary_cond=="Test":
			mod=Test_Model.from_values(dx,N,T,2,2)
		print("Model created.")
		rhs=mod.create_rhs()
		print("RHS created.")			
		phi_sol=mod.apply_convol(rhs)
		print("Convolution applied.")
		dof=mod.dof

		dof0=int(mod.dof_s[0])
		rearrange=[None]*(N+1)

		for i in range(0,N+1):
			rearrange[i]=int(i*tau_s[j]/tau_small)


		EndN=N
		dp0_space=mod.function_spaces[0]
		#p1_space=mod.function_spaces[1]
		errn[j]=mod.calc_L2_err(phi_sol[0:dof0,:],phi_ex[0:dof0,rearrange],dp0_space)
		errd[j]=mod.calc_H1_err(phi_sol[dof0:dof,:],phi_ex[dof0:dof,rearrange],p1_space)
		#err[i,j]=mod.calc_L2_err(phi_sol,phi_ex)
		print("errn",errn)
		
		print("errd",errd)
		np.save('data/tau_s_h01e',tau_s)
		np.save('data/Err_h01_ne',errn)
		np.save('data/Err_h01_de',errd)
		
		#plt.plot(phi_sol[3,:],label='num_sol')
		#plt.plot(phi_ex[3,rearrange],label='ex_sol')
		#plt.legend()
		#plt.show()



	err_mat=(np.abs(phi_sol-phi_ex[:,rearrange]))**2
	err_vec=np.sum(err_mat,axis=1)
	
	k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
	print("k",k)
	import matplotlib.pyplot as plt
	#print(rhs)
	#plt.plot(phi_sol[:,N-1],label='num_dnormal')
	#plt.plot(phi_ex[:,N-1],label='ex_dnormal')
	#plt.legend()
	#plt.show()
	
	plt.plot(phi_sol[k,:],label='num_sol')
	plt.plot(phi_ex[k,rearrange],label='ex_sol')
	plt.legend()
#	plt.show()


	#plt.plot(phi_sol[int(k+3),:],label='num_sol')
	#plt.plot(phi_ex[int(k+3),rearrange],label='ex_sol')

	plt.legend()
#	plt.show()

	
	#plt.plot(phi_sol[k,:int(N/2)],label='num_sol')
	#plt.plot(phi_ex[k,:int(N/2)],label='ex_sol')
	#plt.legend()
	#plt.show()


	return tau_s,errd, errn








def create_err_mat(Amount_time,Amount_space,boundary_cond):
	#import bempp.api
	h_s=np.zeros(Amount_space)
	dof_s=np.zeros(Amount_space)
	errn=np.zeros((Amount_space,Amount_time))
	errd=np.zeros((Amount_space,Amount_time))
	err=np.zeros((Amount_space,Amount_time))
	tau_s=np.zeros(Amount_time)

	T=5
	for i in range(0,Amount_space):
		print('Space_index:')			
		print(i)
		dx=np.sqrt(2)**(-i)
		h_s[i]=dx

		print("h=",dx)

		for j in range(0,Amount_time):

			print('Time_index:')			
			print(j)
			N=4*2**(j+1)
			tau_s[j]=T*1.0/N
			if boundary_cond=="Dirichlet":
				mod=Dirichlet_Model.from_values(dx,N,T,2,1)
				
	
			elif boundary_cond=="GIBC":
				mod=GIBC_Model.from_values(dx,N,T,2,2)

				#mod.OrderQF=np.max(6,mod.OrderQF-i)
			elif boundary_cond=="Neumann":
				#mod=Neumann_Model.from_values(dx,N,T,2,2)
				
				order=2
				grid=bempp.api.shapes.sphere(h=dx)
				#Define space
				dp0_space=bempp.api.function_space(grid,"P",1)
				p1_space=bempp.api.function_space(grid, "P" ,2)

				spaces=[p1_space,p1_space]

				mod=Neumann_Model(grid,spaces,N,T,order)
			elif boundary_cond=="Test":
				mod=Test_Model.from_values(dx,N,T,2,2)
			elif boundary_cond=="spherical":
				order=2
				grid=bempp.api.shapes.sphere(h=dx)
				#Define space
				dp0_space=bempp.api.function_space(grid,"P",1)
				p1_space=bempp.api.function_space(grid, "P" ,1)

				spaces=[p1_space,p1_space]

				mod=alternative_spherical_Model(grid,spaces,N,T,order)
			

			print("Model created.")
			rhs=mod.create_rhs()

			#### Plotting RHS
	#		import matplotlib.pyplot as plt

	#		for node in range(len(rhs[:,0])):
	#			plt.plot(rhs[node,:])
	#		plt.show()
			print("RHS created.")
			phi_sol=mod.apply_convol(rhs)
			print("Convolution applied.")
			
			#phi_ex=mod.create_ex()
			psi_ex=mod.calc_ref_psi()
			dof_s[i]=mod.dof
			dof=mod.dof
			print('DOF:')
			print(dof_s[i])
			dof0=int(mod.dof_s[0])
			#import matplotlib.pyplot as plt
#			for k in range(0,dof0-1):
#				plt.plot(phi_sol[dof0+k,:])
#			plt.plot(psi_ex[3,:], 'bo')
			#plt.show()	
			EndN=N
			dp0_space=mod.function_spaces[0]
			p1_space=mod.function_spaces[1]
			errn[i,j]=mod.calc_H1_err(phi_sol[dof0:2*dof0,:],psi_ex,dp0_space)
			errd[i,j]=mod.calc_L2_err(phi_sol[dof0:2*dof0,:],psi_ex,dp0_space)
		#	errd[i,j]=mod.calc_L2_err(phi_sol[dof0:dof,:EndN],phi_ex[dof0:dof,:EndN],p1_space)
			#err[i,j]=mod.calc_L2_err(phi_sol,phi_ex)
			print("errn",errn)
			
			print("errd",errd)
			#np.save('data/h_s_psi_max',h_s)
			#np.save('data/dof_s_psi_max',dof_s)
			#np.save('data/tau_s_psi_max',tau_s)
			#np.save('data/Err_psi_L2_max',errd)
			#np.save('data/Err_psi_H1_max',errn)
			import scipy.io
			#scipy.io.savemat('data/Err_plot_data.mat', dict(dof_s=dof_s, Err_H1=errn,Err_L2=errd,h_s=h_s,tau_s=tau_s))

#	err_mat=(np.abs(phi_sol-phi_ex))**2
#	err_vec=np.sum(err_mat,axis=1)
#	
#	k=np.argmax(np.abs(phi_sol[:,N]-phi_ex[:,N]))
#	print("k",k)
#	import matplotlib.pyplot as plt
#	#print(rhs)
#	plt.plot(phi_sol[:,N-1],label='num_dnormal')
#	plt.plot(phi_ex[:,N-1],label='ex_dnormal')
#	plt.legend()
##	plt.show()
#	
#	plt.plot(phi_sol[k,:N],label='num_sol')
#	plt.plot(phi_ex[k,:N],label='ex_sol')
#	plt.legend()
#	plt.show()


#	plt.plot(phi_sol[int(k+3),:N],label='num_sol')
#	plt.plot(phi_ex[int(k+3),:N],label='ex_sol')
#	plt.legend()
#	plt.show()

	
	#plt.plot(phi_sol[k,:int(N/2)],label='num_sol')
	#plt.plot(phi_ex[k,:int(N/2)],label='ex_sol')
	#plt.legend()
	#plt.show()


	return tau_s,h_s,dof_s,errn

#import bempp.api

#grid=bempp.api.shapes.sphere(h=0.5)
#Define space
#p1_space=bempp.api.function_space(grid, "P" , 1)

#dir_mod=Dirichlet_Model(grid,[p1_space],1000,1,2)

#rhs=dir_mod.create_rhs()
#print("rhs:",rhs[0,:])
#phi_sol=dir_mod.apply_convol(rhs)
#phi_ex=dir_mod.create_ex()
#import matplotlib.pyplot as plt
#plt.plot(phi_sol[0,:])
#plt.plot(phi_ex[0,:])
#plt.show()
#print(np.max(np.abs(phi_ex-phi_sol)))

#create_err_mat(1,1,"Neumann")



def scatter_waves_plot(N_per_tp,n_grid_points,time_points,dx,boundary_cond):
	lth_sqr=4
	N=N_per_tp
	T=np.max(time_points)
	#Creating solution
	if boundary_cond=="Dirichlet":
		#mod=Dirichlet_Model.from_values(dx,N,T,2,1)
		order=2
		grid=bempp.api.shapes.cube(h=dx)
		p1_space=bempp.api.function_space(grid, "P" ,1)
		spaces=[p1_space]
		mod=Dirichlet_Model(grid,spaces,N,T,order)
	elif boundary_cond=="GIBC":
		#mod=GIBC_Model.from_values(dx,N,T,2,2)
		order=2
		#grid=bempp.api.shapes.cube(h=dx)
	#	grid=bempp.api.import_grid('magnet_finer_03.msh')
	#	grid=bempp.api.import_grid('magnet_h05_h1.msh')
	#	grid.plot(/grids/)
		grid=bempp.api.import_grid('grids/magnet_h05_h01.msh')
		#grid=bempp.api.shapes.sphere(h=dx)
	    #Define space
		#grid.plot()
		dp0_space=bempp.api.function_space(grid,"P",1)
		p1_space=bempp.api.function_space(grid, "P" ,1)
		print(p1_space.global_dof_count)
		spaces=[dp0_space,p1_space]
		print("N:"+str(N))
		mod=GIBC_Model(grid,spaces,N,T,order)

	elif boundary_cond=="GIBC_inner":
		#mod=GIBC_Model.from_values(dx,N,T,2,2)
		order=2
		grid=bempp.api.shapes.cube(h=dx)
		#grid=bempp.api.shapes.sphere(h=dx)
	    #Define space
		dp0_space=bempp.api.function_space(grid,"P",1)
		p1_space=bempp.api.function_space(grid, "P" ,1)
		print(p1_space.global_dof_count)
		spaces=[dp0_space,p1_space]
		print("N:"+str(N))
		mod=GIBC_inner_Model(grid,spaces,N,T,order)
	elif boundary_cond=="spherical":
		#mod=Neumann_Model.from_values(dx,N,T,2,2)
		order=2
		grid=bempp.api.shapes.sphere(h=dx)
	
		#Define space
		dp0_space=bempp.api.function_space(grid,"P" , 1)
		p1_space=bempp.api.function_space(grid, "P" , 1)
		spaces=[dp0_space,p1_space]
		mod=alternative_spherical_Model(grid,spaces,N,T,order)
    	elif boundary_cond=="Test":
        	mod=Test_Model.from_values(dx,N,T,2,2)

	elif boundary_cond=="Acoustic":
		def F_acc(s):
			return (s+1+s**(-1))**(-1)*s
		print("Acoustic boundary condition")	
		order=2
	#	grid=bempp.api.shapes.sphere(h=dx)
		grid=bempp.api.import_grid('grids/magnet_h05_h01.msh')
		#grid.plot()
		#Define space
		dp0_space=bempp.api.function_space(grid,"P" , 1)
		p1_space=bempp.api.function_space(grid, "P" , 1)
		spaces=[dp0_space,p1_space]
		mod=General_Model(F_acc,grid,spaces,N,T,order)
 
	elif boundary_cond=="Absorbing":
		def F_acc(s):
			return s**(1.0/2)/0.1
		print("Absorbing boundary condition")
#######################Acoustic boundary condition
		order=2
	#	grid=bempp.api.shapes.sphere(h=dx)
		grid=bempp.api.import_grid('grids/magnet_h05_h01.msh')
		#grid.plot()
		#Define space
		dp0_space=bempp.api.function_space(grid,"P" , 1)
		p1_space=bempp.api.function_space(grid, "P" , 1)
		spaces=[dp0_space,p1_space]
		mod=General_Model(F_acc,grid,spaces,N,T,order)
 
	print("Model created.")
	rhs=mod.create_rhs()
	phi_sol=mod.apply_convol(rhs)
	

	#import matplotlib.pyplot as plt
## For plotting Scattering from square
#	plot_grid = np.mgrid[-(lth_sqr):(lth_sqr+1):1j*n_grid_points, -lth_sqr:(lth_sqr+1):1j*n_grid_points]

## For plotting Scattering from magnet
##	x_a=-2
##	x_b=2
##	y_a=-2
##	y_b=2

########DRAFT MAGNET PICTURE DATA:
	x_a=-0.75
	x_b=0.75
	y_a=-0.25
	y_b=1.25
###############################################
	plot_grid = np.mgrid[y_a:y_b:1j*n_grid_points, x_a:x_b:1j*n_grid_points]
#	plot_grid = np.mgrid[-0.5:1:1j*n_grid_points, -1.5:1.5:1j*n_grid_points]
	#print(plot_grid)
	Points = np.vstack( ( plot_grid[0].ravel() , plot_grid[1].ravel() , 0.25*np.ones(plot_grid[0].size) ) )
	#Convolution with potential operators
	u_eval=mod.apply_pot_convol(phi_sol,Points)
	Frames=N
	speed=int(N*1.0/Frames)
	u_ges=np.zeros((n_grid_points**2,N/speed))

	import matplotlib
	matplotlib.rcParams['figure.figsize'] = (5.0, 4.0)
	#
	from matplotlib import pylab as plt
	#plt.plot(phi_sol[0,:])
	#plt.plot(phi_sol[3,:])
	A=plot_grid[0]
	B=plot_grid[1]
	max_grid=np.zeros((n_grid_points,n_grid_points))

	for j in range(n_grid_points):
		for i in range(n_grid_points):
			max_grid[i,j]=max(np.abs(A[i,j]-0.5),np.abs(B[i,j]-0.5))
	zrs=np.zeros((n_grid_points,n_grid_points))
	#zrs[max_grid<0.50]=np.nan

	
	plt.imshow(zrs, extent=(x_a,x_b,y_a,y_b))
	plt.title('Computed solution')
	plt.clim(-1,1)
	plt.colorbar()
	

	for indt in range(0,int(N/speed)):
		u_tp=u_eval[:,indt*speed]
		print("Plotting, Frame "+str(indt)+" from " +str(int(N/speed)) )

	#####################################################################################

		uinc=np.zeros(n_grid_points**2)
		uinc_wave=mod.u_inc


		for k in range(n_grid_points**2):
			uinc[k]=uinc_wave.eval(T*indt*speed*1.0/N,Points[:,k])
		uinc_rs=uinc.reshape((n_grid_points,n_grid_points))
		u_ges[:,indt]=u_tp+uinc

	#######################################################################################
		u_tp_rs = np.real(u_tp.reshape((n_grid_points,n_grid_points)))
		u_tp_rs=u_tp_rs+uinc_rs
#		u_tp_rs=uinc_rs	
		radius=np.sqrt(plot_grid[0]**2+plot_grid[1]**2)
		u_tp_rs[radius<1]=np.nan
		#u_tp_rs[max_grid<0.50]=np.nan

			## Plot the image

		#matplotlib.rcParams['figure.figsize'] = (5.0, 4.0)
	#
		from matplotlib import pylab as plt
	#
	#	fname ="/home/nick/Python_code/Anim/wave_imag/Fractional_h05_dof10000n{}.png".format(indt)
		
		plt.imshow(u_tp_rs, extent=(x_a,x_b,y_a,y_b))

		plt.clim(-1,1)

		fname ="data/wave_images/"+boundary_cond+"_n{}.png".format(indt)
	#	plt.savefig(fname)
		plt.close
		
		#plt.imshow(u_tp_rs, extent=(-(lth_sqr),(lth_sqr+1),-(lth_sqr),(lth_sqr+1)))
		#plt.plot(u_tp_rs[n_grid_points/2,:])

		#plt.ylim(-1,1)

		#plt.savefig(fname)
		#plt.clf()
		#plt.close

	import scipy.io

	scipy.io.savemat('data/'+boundary_cond+'.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,Points=Points))


	#np.save('u_vals2',u_ges)
	return 0
####
##N_per_tp=50
##time_points=np.array([5])
###
###test_mat=np.array([[ 1 ,2,3 ],[4,5,6]])
##dx=1
###import time
###boundary_cond="General"
###boundary_cond="GIBC"
####boundary_cond="Dirichlet"
##n_grid_points=20
###
####scatter_waves_plot(N_per_tp,n_grid_points,time_points,dx,boundary_cond)
###
###Amount_time=4
###Amount_space=1
##boundary_cond="spherical"
###create_err_mat(Amount_time,Amount_space,boundary_cond)
###
##
##scatter_waves_plot(N_per_tp,n_grid_points,time_points,dx,boundary_cond)
