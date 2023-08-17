import bempp.api
import numpy as np
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

def extract_densities(filename):
    resDict = np.load(filename,allow_pickle=True).item()
    sol = resDict["sol"]
    T   = resDict["T"]
    m   = resDict["m"]
    return sol,T,m

def save_densities(filename,sol,T,m,N):
    resDict = dict()
    resDict["sol"] = sol
    resDict["T"] = T
    resDict["m"] = m
    resDict["N"] = N
    np.save(filename,resDict)
    return None

def evaluate_densities(filename,gridfilename):
    "Evaluates the densities saved at the points by convolution quadrature with $m$-stages."
    #points=np.array([[0],[0],[0]])
    points=np.array([[2],[0],[0]])
    #points=np.array([[0],[0],[2]])
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
        return scattered_field_data.reshape(3,1)[:,0]
    td_potential = Conv_Operator(th_potential_evaluation) 
    evaluated_data = td_potential.apply_RKconvol(rhs,T,cutoff = 10**(-8),method = "RadauIIA-"+str(m),show_progress= False,first_value_is_t0=False)
    sol_points = np.zeros((len(evaluated_data[:,0]),len(evaluated_data[0,::m])+1))
    sol_points[:,1::] = evaluated_data[:,::m]
    return sol_points,T,dof
