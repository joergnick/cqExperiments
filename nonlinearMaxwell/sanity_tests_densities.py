import numpy as np
import bempp.api
import sys
sys.path.append('cqToolbox')
#sys.path.append('../cqToolbox')
sys.path.append('data')
#sys.path.append('../data')
import scipy.io
#mat_contents=scipy.io.loadmat("data/grids/TorusDOF340.mat")
#h=1.0
h=0.5
gridfilename='data/grids/sphere_python3_h'+str(np.round(h,3))+'.npy'
mat_contents = np.load(gridfilename,allow_pickle=True).item()
Nodes        = mat_contents['Nodes']
Elements     = mat_contents['Elements']

#Nodes=np.array(mat_contents['Nodes']).T
#rawElements=mat_contents['Elements']
#for j in range(len(rawElements)):
#    betw=rawElements[j][0]
#    rawElements[j][0]=rawElements[j][1]
#    rawElements[j][1]=betw
#Elements=np.array(rawElements).T
#Elements=Elements-1
grid=bempp.api.grid_from_element_data(Nodes,Elements)
#grid = bempp.api.shapes.sphere(h=1)

RT_space = bempp.api.function_space(grid,"RT",0)
dof = int(RT_space.global_dof_count)
#N   = 16
N   = 128
m = 2
density_filename = 'data/density_sphere_h_'+str(np.round(h,3)) +'_N_'+str(N)+'_m_'+str(m)+ '.npy'
#solDict = np.load('data/sphereDOF72.npy').item()
solDict = np.load(density_filename,allow_pickle=True).item()
sol = solDict["sol"]
######################################################################
#import matplotlib.pyplot as plt
#plt.plot(np.linalg.norm(sol,axis=0))
#plt.show()
#import matplotlib.pyplot as plt
#print(len(sol[:,0]),len(sol[0,:]))
#plt.plot([np.linalg.norm(sol[:,k]) for k in range(len(sol[0,:]))] )
#plt.show()
#raise ValueError("End of plot")
#######################################################################
print("Does sol contain Nan values? Answer: "+str(np.isnan(sol).any()))
m = solDict["m"]
T = solDict["T"]

#N = (len(sol[0,:])-1)/2
from rkmethods import RKMethod
rk = RKMethod("RadauIIA-"+str(m),T*1.0/N)
print("N= ",N)
dof = int(np.round(len(sol[:,0])/2))

id_op=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
id_weak = id_op.weak_form()

RT_space=bempp.api.function_space(grid, "RT",0)
NC_space=bempp.api.function_space(grid, "NC",0)
idrot= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
idrot_weak = idrot.weak_form()
def a(x):
    if np.linalg.norm(x)<10**(-6):
        return 0*x
    return np.linalg.norm(x)**(-0.5)*x
from linearcq import Conv_Operator
def calc_gtH(rk,grid,N,T):
    m = len(rk.c)
    tau = T*1.0/N
    RT_space=bempp.api.function_space(grid, "RT",0)
    dof = int(RT_space.global_dof_count)
    gTE = np.zeros((dof,m*N))
    curls = np.zeros((dof,m*N))
    for j in range(N):
        for stageInd in range(m):
            t = tau*j+tau*rk.c[stageInd] 
            def func_rhs(x,n,domain_index,result):
                Einc =  np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])    
                #tang = np.cross(n,np.cross(inc, n))
                gT_Einc = np.cross(Einc, n)
                result[:] = gT_Einc
            gt_Einc_grid = bempp.api.GridFunction(RT_space,fun = func_rhs,dual_space = RT_space)
            def func_curls(x,n,domain_index,result):
                curlU=np.array([ 0. * x[2],-100*(x[2]-t+2)*np.exp(-50*(x[2]-t+2)**2), 0. * x[2]])
                result[:] = np.cross(curlU,n)
            curlfun_inc = bempp.api.GridFunction(RT_space,fun = func_curls,dual_space = RT_space) 
            curls[:,j*rk.m+stageInd]  = curlfun_inc.coefficients
            gTE[:,j*rk.m+stageInd] = gt_Einc_grid.coefficients
    def sinv(s,b):
        return s**(-1)*b
    IntegralOperator = Conv_Operator(sinv)
    gTH = -IntegralOperator.apply_RKconvol(curls,T,method="RadauIIA-"+str(m),show_progress=False)
    gTH = np.concatenate((np.zeros((dof,1)),gTH),axis = 1)
    gTE = np.concatenate((np.zeros((dof,1)),gTE),axis = 1)
    return gTH,gTE
gTH_inc, gTE_inc = calc_gtH(rk,grid,N,T)
print("Computed incident traces.")
#def nonlinearity(self,coeff,t,time_index):
from customOperators import precompMM,sparseWeightedMM,applyNonlinearity
gridfunList,neighborlist,domainDict = precompMM(RT_space)
print("FINISHED PRECOMPUTATION FOR CUSTOM_OPERATOR.")
for time_index in range(len(sol[0,:])):
    phi_j = sol[:dof,time_index]
    psi_j = sol[dof:,time_index]
    gTHFun     = bempp.api.GridFunction(RT_space,coefficients = gTH_inc[:,time_index]-phi_j)
    agridFun   = applyNonlinearity(gTHFun,a,gridfunList,domainDict)
    projection_aH_tot = id_weak*agridFun.coefficients
    gTE_tot_coefficients = -psi_j-gTE_inc[:,time_index]
    projection_gTE_tot   = idrot_weak*gTE_tot_coefficients
    print("NORM gtE_tot: ",np.linalg.norm(projection_gTE_tot),"  NORM aH_tot: ", np.linalg.norm(projection_aH_tot)," NORM addition: ", np.linalg.norm(projection_gTE_tot+projection_aH_tot), " NORM difference: ", np.linalg.norm(projection_gTE_tot-projection_aH_tot))
#grid_fun = bempp.api.GridFunction(RT_space,coefficients=gTE_inc[:,time_index])
#grid_fun.plot()
#    #result     = np.zeros(2*dof) 
#    #result[:dof] = id_weak*agridFun.coefficients
#    lhs        = (-psi_j+agridFun.coefficients)
#    #lhs_grid = bempp.api.GridFunction(RT_space,coefficients=lhs)
#    
#    #gtE_grid      = bempp.api.GridFunction(RT_space,coefficients=gTE[:,time_index])
#    residual      = id_weak*lhs -idrot_weak*
#    #residual_grid = lhs_grid-
#    print("Norm lhs: ",np.linalg.norm(-sol[dof:,time_index]+agridFun.coefficients)," Norm rhs: ",np.linalg.norm(gTE[:,time_index]), " Norm error: ", np.linalg.norm(())-gTE[:,time_index]))
##    return result
#
