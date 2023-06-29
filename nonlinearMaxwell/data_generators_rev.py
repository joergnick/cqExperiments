import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')

import numpy as np
import bempp.api
import scipy.io
## Own inclusions
from linearcq import Conv_Operator
from customOperators import precompMM,sparseWeightedMM,applyNonlinearity,sparseMM
from newtonStepperRev import NewtonIntegrator

OrderQF =16
#print(bempp.api.global_parameters.quadrature.near.max_rel_dist)
bempp.api.global_parameters.quadrature.near.max_rel_dist = 2

bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
bempp.api.global_parameters.quadrature.double_singular = OrderQF
bempp.api.global_parameters.assembly.boundary_operator_assembly_type = 'dense'
bempp.api.global_parameters.assembly.enable_interpolation_for_oscillatory_kernels = False
bempp.api.global_parameters.assembly.interpolation_points_per_wavelength = 5000
#bempp.api.global_parameters.hmat.eps=10**-5
bempp.api.global_parameters.hmat.admissibility='strong'
space_string = "RT"
nrspace_string = "NC"

tshift = 0
variance = 2
def calc_gtH(rk,grid,N,T):
    m = len(rk.c)
    tau = T*1.0/N
    RT_space=bempp.api.function_space(grid, space_string,0)
    #RT_space=bempp.api.function_space(grid, "RT",0)
    dof = RT_space.global_dof_count
    gTE = np.zeros((dof,m*N))
    curls = np.zeros((dof,m*N))
    gtHinc_closed = np.zeros((dof,m*N))
    for j in range(N):
        for stageInd in range(m):
            t = tau*j+tau*rk.c[stageInd] 
            t += tshift
            def func_rhs(x,n,domain_index,result):
                Einc =  np.array([np.exp(-variance*(x[2]+t-4)**2), 0. * x[2], 0. * x[2]])    
                #tang = np.cross(n,np.cross(inc, n))
                PT_Einc = np.cross(n,np.cross(Einc, n))
                result[:] = PT_Einc
            gt_Einc_grid = bempp.api.GridFunction(RT_space,fun = func_rhs,dual_space = RT_space)
            gTE[:,j*rk.m+stageInd] = gt_Einc_grid.coefficients
           # def func_curls(x,n,domain_index,result):
           #     curlU=np.array([ 0. * x[2],-2*variance*(x[2]+t-2)*np.exp(-variance*(x[2]+t-2)**2), 0. * x[2]])
           #     result[:] = np.cross(curlU,n)
            def func_Hinc(x,n,domain_index,result):
                Hinc =  -np.array([0. * x[2], np.sin(20*(x[2]+t-4))*np.exp(-variance*(x[2]+t-4)**2), 0. * x[2]])    
                result[:] = np.cross(Hinc,n)
            gtHinc_closed[:,j*rk.m+stageInd] = bempp.api.GridFunction(RT_space,fun = func_Hinc,dual_space = RT_space).coefficients
           # curlfun_inc = bempp.api.GridFunction(RT_space,fun = func_curls,dual_space = RT_space) 
           # curls[:,j*rk.m+stageInd]  = curlfun_inc.coefficients
   # def sinv(s,b):
   #     return s**(-1)*b
   # IntegralOperator = Conv_Operator(sinv)
   # gTH = np.real(-IntegralOperator.apply_RKconvol(curls,T,method="RadauIIA-"+str(m),show_progress=False,first_value_is_t0=False))
 
   # print("MAX ||Hinc-Hincapprox||", max(np.linalg.norm(gTH-gtHinc_closed,axis = 0)))
    gTH = gtHinc_closed
    gTH = np.concatenate((np.zeros((dof,1)),gTH),axis = 1)
    gTE = np.concatenate((np.zeros((dof,1)),gTE),axis = 1)
    #rhs[0:dof,:]=np.real(gTH)-rhs[0:dof,:]
    return gTH,gTE

def load_grid(gridfilename):
    "Loads grids from filename, accounts for different indices and orientation of bempp (underlying gmsh) and distmesh."
    load_success = False
    if gridfilename[-3:] == 'mat':
        mat_contents=scipy.io.loadmat(gridfilename)
        Nodes=np.array(mat_contents['Nodes']).T
        rawElements=mat_contents['Elements']
        ## Switching orientation
        for j in range(len(rawElements)):
            betw=rawElements[j][0]
            rawElements[j][0]=rawElements[j][1]
            rawElements[j][1]=betw
        Elements=np.array(rawElements).T
        ## Subtraction due to different conventions of distmesh and bempp, grid starts from 0 instead of 1
        Elements=Elements-1
        load_success = True
    if gridfilename[-3:] == 'npy':
        #mat_contents = np.load(gridfilename).item()
        mat_contents = np.load(gridfilename,allow_pickle=True).item()
        Nodes        = mat_contents['Nodes']
        Elements     = mat_contents['Elements']
        load_success = True
    if not load_success:
        raise ValueError("Filename of grid: "+gridfilename+" does not have .mat or .npy ending.")
    grid=bempp.api.grid_from_element_data(Nodes,Elements)
    return grid
 

def compute_densities(alpha,N,gridfilename,T,rk,debug_mode=False,sigma = 0):
    "Computes densities for scattering problem with nonlinear absorbing b.c. for given gridfilename and a plane wave as incoming wave."
    grid = load_grid(gridfilename)
    #grid = bempp.api.shapes.cube(h=1)
    RT_space=bempp.api.function_space(grid, space_string,0)

    print("GLOBAL DOF: ",RT_space.global_dof_count)
    #RT_space=bempp.api.function_space(grid, "RT",0)
    gridfunList,neighborlist,domainDict = precompMM(RT_space)
    id_op=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
    id_weak = id_op.weak_form()
    gtH ,dummy    = calc_gtH(rk,grid,N,T)
    class ScatModel(NewtonIntegrator):
        alpha = -1.0 ## invalid value
        debug_mode = False
        def __init__(self,alpha=1.0):
            NewtonIntegrator.__init__(self)
            self.newton_dense = None
            #print("Parameter alpha has been set to: "+str(alpha)+".")
            if (alpha<=0) or (alpha>1):
                print("Parameter alpha has been set to: "+str(alpha)+".")
                raise ValueError("The parameter alpha must be in the interval (0,1].")
            self.alpha = alpha
        def a(self,x):
            cutoff_space = 10**(-15)
            normx = np.linalg.norm(x)
            if normx <10**(-50):
                direc_x = 1.0/np.sqrt(3)*np.ones(3)
            else:
                direc_x = x*1.0/normx
            if normx<cutoff_space:
                x=cutoff_space*direc_x
            #return 0*np.linalg.norm(x)**(1-self.alpha)*x
            return np.linalg.norm(x)**(self.alpha-1)*x
        def Da(self,x):
            cutoff_space = 10**(-15)
            normx = np.linalg.norm(x)
            if normx <10**(-50):
                direc_x = 1.0/np.sqrt(3)*np.ones(3)
            else:
                direc_x = x*1.0/normx
            if normx<cutoff_space:
                x=cutoff_space*direc_x
            #return np.eye(3)
            return ((self.alpha-1)*np.linalg.norm(x)**(self.alpha-3)*np.outer(x,x)+np.linalg.norm(x)**(self.alpha-1)*np.eye(3))
          #  return -0.5*np.linalg.norm(x)**(-2.5)*np.outer(x,x)+np.linalg.norm(x)**(-0.5)*np.eye(3)
        def precompute_system(self,m,dof,W0,rk):
            if self.newton_dense is not None:
                return self.newton_dense
            def newton_func(x_dummy):
                x_mat    = x_dummy.reshape(m,dof).T
                x_diag   = rk.diagonalize(x_mat)
                grad_mat = 1j*np.zeros((dof,m))
                Bs_mat   = 1j*np.zeros((dof,m))
                for m_index in range(m):
                    Bs_mat[:,m_index]   = self.harmonic_forward(rk.delta_eigs[m_index],x_diag[:,m_index],precomp = W0[m_index])
                res_mat  = rk.reverse_diagonalize(Bs_mat)
                new_res =  res_mat.T.ravel()
                #print("||IM(res)|| = ",np.linalg.norm(np.imag(new_res)))
                #return new_res
                return np.real(new_res)
            id_mat = np.eye(m*dof)
            newton_dense = np.array([newton_func(id_mat[:,i]) for i in range(m*dof)])            
            self.newton_dense = newton_dense
            return newton_dense


        def precomputing(self,s):
            s = s + sigma
            #NC_space=bempp.api.function_space(grid, "NC",0)
            NC_space=bempp.api.function_space(grid, nrspace_string,0)
            #RT_space=bempp.api.function_space(grid, "RT",0)
            RT_space=bempp.api.function_space(grid, space_string,0)
            elec = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*s)
            magn = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*s)
            identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
            identity2=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
            blocks=np.array([[None,None], [None,None]])
            blocks[0,0] = -elec.weak_form()
            blocks[0,1] =  magn.weak_form()-1.0/2*identity.weak_form()
            blocks[1,0] = -magn.weak_form()-1.0/2*identity.weak_form()
            blocks[1,1] = -elec.weak_form()
            return [bempp.api.BlockedDiscreteOperator(blocks),identity2]
        def harmonic_forward(self,s,b,precomp = None):
            return precomp[0]*b
        def calc_jacobian(self,x,t,time_index):
            dof = int(np.round(len(x)/2))
            weightphiGF = bempp.api.GridFunction(RT_space,coefficients = x[:dof])
            weightIncGF = bempp.api.GridFunction(RT_space,coefficients = gtH[:,time_index])
            jacobsparse =sparseWeightedMM(RT_space,np.exp(sigma*t)*weightphiGF+weightIncGF,self.Da,gridfunList,neighborlist,domainDict)
            #normJ = np.linalg.norm(jacob.todense())
            #print("normJ = ",normJ)
            #jacob = sparseMM(RT_space,gridfunList,neighborlist,domainDict)
            dof = int(np.round(len(x)/2))
            jacob = np.zeros((2*dof,2*dof))
            jacob[:dof,:dof] = jacobsparse.todense()
            return jacob
        def apply_jacobian(self,jacob,x):
            dof = int(np.round(len(x)/2))
           # jx = 1j*np.zeros(2*dof)
           # jx[:dof] = jacob*x[:dof]
            return jacob.dot(x)
        def nonlinearity(self,coeff,t,time_index):
            dof = int(np.round(len(coeff)/2))
            phiGridFun = bempp.api.GridFunction(RT_space,coefficients=coeff[:dof]) 
            gTHFun     = bempp.api.GridFunction(RT_space,coefficients = gtH[:,time_index])
            agridFun   = np.exp(-sigma*t)*applyNonlinearity(np.exp(sigma*t)*phiGridFun+gTHFun,self.a,gridfunList,domainDict)
            result     = np.zeros(2*dof) 
            #result[:dof] = id_weak*(phiGridFun+gTHFun).coefficients
            result[:dof] = id_weak*agridFun.coefficients
            return result
    
        def righthandside(self,t,time_index,history=None):
            def func_rhs(x,n,domain_index,result):
                inc  = np.array([np.sin(20*(x[2]+t-4))*np.exp(-variance*(x[2]+t-4)**2), 0. * x[2], 0. * x[2]])    
                tang = np.cross(np.cross(inc, n),n)
                result[:] = np.exp(-sigma*t)*tang
                
            gridfunrhs = bempp.api.GridFunction(RT_space,fun = func_rhs,dual_space = RT_space)
            dof = RT_space.global_dof_count
            rhs = np.zeros(dof*2)
            #rhs[:dof] = gridfunrhs.coefficients
            rhs[:dof] = id_weak*gridfunrhs.coefficients
            #print("||rhs|| = ",np.linalg.norm(rhs)," ||gridfun|| = ",np.linalg.norm(gridfunrhs.coefficients))
            return rhs
    model = ScatModel(alpha = alpha)
    dof = RT_space.global_dof_count
    sol ,counters  = model.integrate(T,N, method = rk.method_name,max_evals_saved=16,debug_mode=debug_mode,same_rho = False)
    exp_sigma_tt = np.exp(sigma*rk.get_time_points(T))
    return exp_sigma_tt*sol

def extract_densities(filename):
    resDict = np.load(filename,allow_pickle=True).item()
    sol = resDict["sol"]
    T   = resDict["T"]
    m   = resDict["m"]
    return sol,T,m
def evaluate_densities(filename,gridfilename):
    "Evaluates the densities saved at the points by convolution quadrature with $m$-stages."
    #points=np.array([[0],[0],[0]])
    #points=np.array([[1.5],[0],[0]])
    points= 1.2*np.array([[1,-1,0,0,0,0],[0,0,1,-1,0,0],[0,0,0,0,1,-1]])
    n_points = len(points[0])
    grid = load_grid(gridfilename)
    RT_space=bempp.api.function_space(grid, space_string,0) 
    dof = RT_space.global_dof_count
    #resDict = np.load(filename).item()
    resDict = np.load(filename,allow_pickle=True).item()
    rhs = resDict["sol"]
    T   = resDict["T"]
    m   = resDict["m"]
    def th_potential_evaluation(s,b):
        slp_pot = bempp.api.operators.potential.maxwell.electric_field(RT_space, points, s*1j)
        dlp_pot = bempp.api.operators.potential.maxwell.magnetic_field(RT_space, points, s*1j)
        phigrid=bempp.api.GridFunction(RT_space,coefficients=b[0:dof],dual_space=RT_space)
        psigrid=bempp.api.GridFunction(RT_space,coefficients=b[dof:2*dof],dual_space=RT_space)
        #print("Evaluate field : ")  
        scattered_field_data = -slp_pot * phigrid+dlp_pot*psigrid
        if np.isnan(scattered_field_data).any():
            #print("NAN Warning, s = ", s)
            scattered_field_data=np.zeros(np.shape(scattered_field_data))
        return scattered_field_data.reshape(3*n_points,1)[:,0]
    td_potential = Conv_Operator(th_potential_evaluation) 
    evaluated_data = td_potential.apply_RKconvol(rhs,T,cutoff = 10**(-8),method = "RadauIIA-"+str(m),show_progress= False,first_value_is_t0=False)
    sol_points = np.zeros((len(evaluated_data[:,0]),len(evaluated_data[0,::m])+1))
    sol_points[:,1::] = np.real(evaluated_data[:,::m])
    return sol_points,T,dof
