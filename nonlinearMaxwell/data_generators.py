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
from customOperators import precompMM,sparseWeightedMM,applyNonlinearity
from newtonStepper import NewtonIntegrator

OrderQF = 14
bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
bempp.api.global_parameters.quadrature.double_singular = OrderQF
bempp.api.global_parameters.hmat.eps=10**-10
bempp.api.global_parameters.hmat.admissibility='strong'

def calc_gtH(rk,grid,N,T):
    m = len(rk.c)
    tau = T*1.0/N
    RT_space=bempp.api.function_space(grid, "RT",0)
    dof = RT_space.global_dof_count
    gTE = np.zeros((dof,m*N))
    curls = np.zeros((dof,m*N))
    for j in range(N):
        for stageInd in range(m):
            t = tau*j+tau*rk.c[stageInd] 
            def func_rhs(x,n,domain_index,result):
                Einc =  np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])    
                #tang = np.cross(n,np.cross(inc, n))
                PT_Einc = np.cross(n,np.cross(Einc, n))
                result[:] = PT_Einc
            gt_Einc_grid = bempp.api.GridFunction(RT_space,fun = func_rhs,dual_space = RT_space)
            gTE[:,j*rk.m+stageInd] = gt_Einc_grid.coefficients
            def func_curls(x,n,domain_index,result):
                curlU=np.array([ 0. * x[2],-100*(x[2]-t+2)*np.exp(-50*(x[2]-t+2)**2), 0. * x[2]])
                result[:] = np.cross(curlU,n)
            curlfun_inc = bempp.api.GridFunction(RT_space,fun = func_curls,dual_space = RT_space) 
            curls[:,j*rk.m+stageInd]  = curlfun_inc.coefficients
    def sinv(s,b):
        return s**(-1)*b
    IntegralOperator = Conv_Operator(sinv)
    gTH = np.real(-IntegralOperator.apply_RKconvol(curls,T,method="RadauIIA-"+str(m),show_progress=False))
    print("MAX ||Hinc||", max(np.linalg.norm(gTH,axis = 0)))
    gTH = np.concatenate((np.zeros((dof,1)),gTH),axis = 1)
    gTE = np.concatenate((np.zeros((dof,1)),gTE),axis = 1)
    #rhs[0:dof,:]=np.real(gTH)-rhs[0:dof,:]
    return gTH,gTE




def compute_densities(alpha,N,gridfilename,T,rk,debug_mode=True):
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
        mat_contents = np.load(gridfilename,allow_pickle=True).item()
        Nodes        = mat_contents['Nodes']
        Elements     = mat_contents['Elements']
        load_success = True
    if not load_success:
        raise ValueError("Filename of grid: "+gridfilename+" does not have .mat or .npy ending.")

    grid=bempp.api.grid_from_element_data(Nodes,Elements)
    RT_space=bempp.api.function_space(grid, "RT",0)
    gridfunList,neighborlist,domainDict = precompMM(RT_space)
    id_op=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
    id_weak = id_op.weak_form()
    gtH ,dummy    = calc_gtH(rk,grid,N,T)
    class ScatModel(NewtonIntegrator):
        alpha = -1.0 ## invalid value
        debug_mode = False
        def __init__(self,alpha=1.0):
            NewtonIntegrator.__init__(self)
            if (alpha<=0) or (alpha>1):
                raise ValueError("The parameter alpha must be in the interval (0,1].")
            self.alpha = alpha
        def a(self,x):
#            return x
            #return 0*np.linalg.norm(x)**(1-self.alpha)*x
            return 0*np.linalg.norm(x)**(1-self.alpha)*x
        def Da(self,x):
        #    if np.linalg.norm(x)<10**(-15):
        #        x=10**(-15)*np.ones(3)
           # return np.eye(3)
            return 0*((self.alpha-1)*np.linalg.norm(x)**(self.alpha-3)*np.outer(x,x)+np.linalg.norm(x)**(self.alpha-1)*np.eye(3))
          #  return -0.5*np.linalg.norm(x)**(-2.5)*np.outer(x,x)+np.linalg.norm(x)**(-0.5)*np.eye(3)
        def precomputing(self,s):
            NC_space=bempp.api.function_space(grid, "NC",0)
            RT_space=bempp.api.function_space(grid, "RT",0)
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
            weightphiGF = bempp.api.GridFunction(RT_space,coefficients = x[:dof])
            weightIncGF = bempp.api.GridFunction(RT_space,coefficients = gtH[:,time_index])
            jacob = sparseWeightedMM(RT_space,weightphiGF+weightIncGF,self.Da,gridfunList,neighborlist,domainDict)
            return jacob
        def apply_jacobian(self,jacob,x):
            dof = int(np.round(len(x)/2))
            jx = 1j*np.zeros(2*dof)
            jx[:dof] = jacob*x[:dof]
            return jx
        def nonlinearity(self,coeff,t,time_index):
            dof = int(np.round(len(coeff)/2))
            phiGridFun = bempp.api.GridFunction(RT_space,coefficients=coeff[:dof]) 
            gTHFun     = bempp.api.GridFunction(RT_space,coefficients = gtH[:,time_index])
            agridFun   = applyNonlinearity(phiGridFun+0*gTHFun,self.a,gridfunList,domainDict)
            result     = np.zeros(2*dof) 
            result[:dof] = id_weak*agridFun.coefficients
            return result
    
        def righthandside(self,t,history=None):
            def func_rhs(x,n,domain_index,result):
                inc  = np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])    
                tang = np.cross(np.cross(inc, n),n)
                result[:] = tang
                #return np.cross(curlU,n)
            gridfunrhs = bempp.api.GridFunction(RT_space,fun = func_rhs,dual_space = RT_space)
            dof = RT_space.global_dof_count
            rhs = np.zeros(dof*2)
            #rhs[:dof] = gridfunrhs.coefficients
            rhs[:dof] = id_weak*gridfunrhs.coefficients
            return rhs
    model = ScatModel(alpha = alpha)
    dof = RT_space.global_dof_count
    print("GLOBAL DOF: ",dof)
    print("Finished RHS.")
    sol ,counters  = model.integrate(T,N, method = rk.method_name,max_evals_saved=100000,debug_mode=debug_mode,same_rho = False)
    dof = RT_space.global_dof_count
    return sol
