import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')

from cqStepper import AbstractIntegrator
from cqDirectStepper import AbstractIntegratorDirect
from scipy.sparse.linalg import gmres
#from bempp.api import gmres
from customOperators import precompMM,sparseWeightedMM,applyNonlinearity
import numpy as np
from rkmethods import RKMethod
from linearcq import Conv_Operator
import bempp.api
OrderQF = 9
bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
bempp.api.global_parameters.quadrature.double_singular = OrderQF
#bempp.api.global_parameters.assembly.boundary_operator_assembly_type = 'dense'
#bempp.api.global_parameters.assembly.enable_interpolation_for_oscillatory_kernels = False
#bempp.api.global_parameters.assembly.interpolation_points_per_wavelength = 20000
bempp.api.global_parameters.hmat.eps=10**-5
bempp.api.global_parameters.hmat.admissibility='strong'

def calc_gtH(rk,grid,N,T):
    m = len(rk.c)
    tau = T*1.0/N
    #RT_space=bempp.api.function_space(grid, "BC",0)
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

def compute_linear_densities_direct(N,gridfilename,T,rk,debug_mode=True):
    load_success = False
    if gridfilename[-3:] == 'mat':
        mat_contents = scipy.io.loadmat(gridfilename)
        Nodes=np.array(mat_contents['Nodes']).T
        rawElements  = mat_contents['Elements']
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
    #######################################################
    grid=bempp.api.shapes.cube(h=1)
    RT_space=bempp.api.function_space(grid, "RT",0)
    #RT_space=bempp.api.function_space(grid, "BC",0)
    NC_space=bempp.api.function_space(grid, "NC",0)
    #NC_space=bempp.api.function_space(grid, "RBC",0)
    id_op=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
    id_op2=-bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
    id_weak = id_op.weak_form()
    id_weak2 = id_op2.weak_form()
    gtH,gtE = calc_gtH(rk,grid,N,T)
    class ScatModel2(AbstractIntegratorDirect):
        def time_step(self,W0,j,rk,history,w_star_sol_j,tolsolver = 0):
            tau = rk.tau
            t   = tau*j 
            rhs = np.zeros(w_star_sol_j.shape)
            #rhs[:dof] = gridfunrhs.coefficients
            for i in range(rk.m):
                rhs[:,i] = -w_star_sol_j[:,i] + self.righthandside(t+rk.c[i]*tau)
            rhs = rk.diagonalize(rhs)
            sol = 1j*0*rhs
            for i in range(rk.m):
                sol[:,i],info = gmres(W0[i],rhs[:,i],tol = 10**(-8),maxiter = 1000)
                if info>0:
                    print("WARNING,INFO >0.")
            sol = np.real(rk.reverse_diagonalize(sol))
            return sol
        def harmonic_forward(self,s,b,precomp = None):
            return precomp*b
        def precomputing(self,s,is_W0 = False):
            elec = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*s)
            magn = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*s)
            identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
            identity2=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
            blocks=np.array([[None,None], [None,None]])
            blocks[0,0] = -elec.weak_form()+identity2.weak_form()
            blocks[0,1] =  magn.weak_form()-1.0/2*identity.weak_form()
            blocks[1,0] = -magn.weak_form()-1.0/2*identity.weak_form()
            blocks[1,1] = -elec.weak_form()
            return bempp.api.BlockedDiscreteOperator(blocks)
        def righthandside(self,t,history=None):
            tshift = 0.0
            t = t+tshift
            def func_rhs(x,n,domain_index,result):
                inc  = np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])    
                tang = np.cross(np.cross(inc, n),n)
                result[:] = tang
            def gt_E(x,n,domain_index,result):
                inc  = np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])
                tang = np.cross(np.cross(inc, n),n)
                result[:] = tang
#            gridfunrhs2 = bempp.api.GridFunction(RT_space,fun = gt_E,dual_space = NC_space)
            gridfunrhs = bempp.api.GridFunction(RT_space,fun = func_rhs,dual_space = RT_space)
            rhs        = np.zeros(RT_space.global_dof_count*2)
            rhs[:dof]  = id_weak*gridfunrhs.coefficients
#            print("NormRhs = ",np.linalg.norm(rhs))
            #rhs[:dof] = 100*id_weak2*gridfunrhs.coefficients
            #print("||rhs|| = ",np.linalg.norm(rhs), " ||gridfunrhs|| = ",np.linalg.norm(gridfunrhs.coefficients))
            return rhs

    model = ScatModel2()
    dof = RT_space.global_dof_count
    sol ,counters  = model.integrate(T,N, method = rk.method_name,max_evals_saved=10000,debug_mode=debug_mode,same_rho = False)
    return sol
