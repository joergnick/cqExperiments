import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')

import bempp.api
import numpy as np

from data_generators import load_grid
#for j in range(12):
#    h = np.round(2**(-j*1.0/2),3)
#    grid = bempp.api.shapes.sphere(h=h)
#    Nodes = grid.leaf_view.vertices
#    Elements = grid.leaf_view.elements
#    dof      = len(Nodes[0,:])
#    print("Generate grid, number of nodes: "+str(dof))
#    filename = 'data/grids/sphere_python3_h' +str(h)+'.npy'
#    #filename = 'data/grids/sphereDOF'+str(dof)+'h' +str(h)+'.npy'
#    grid_data = dict()
#    grid_data["Nodes"] = Nodes
#    grid_data["Elements"] = Elements
#    np.save(filename,grid_data)
#    print("Generated grid file : "+filename)

#gridfilename = 'data/grids/angle_oriented.npy'
for space_ind in range(9):    
    h=2**(-0.5*(space_ind))
    grid = bempp.api.shapes.cube(h=h)
    #grid = bempp.api.import_grid('data/grids/angle_oriented.msh')
    #grid = bempp.api.import_grid('data/grids/angle_transformed.msh')
    nodes_left = grid.leaf_view.vertices
    elements_left = grid.leaf_view.elements
    nodes_right = 1.0*nodes_left
    
    am_nodes = len(nodes_left[0,:])
    elements_right = elements_left+am_nodes
    nodes_right[0,:] = nodes_right[0,:]+0.25*np.ones(am_nodes)
    nodes_left[0,:] = nodes_left[0,:]-1.25*np.ones(am_nodes)
    nodes = np.concatenate((nodes_left,nodes_right),axis = 1)
    elements = np.concatenate((elements_left,elements_right),axis = 1)
    dof      = len(nodes[0,:])
    print("Generate grid, number of nodes: "+str(dof))
    #filename = 'data/grids/sphere_python3_h' +str(h)+'.npy'
    #filename = 'data/grids/sphereDOF'+str(dof)+'h' +str(h)+'.npy'
    #filename = 'data/grids/angle_transformed.npy'
    filename = 'data/grids/two_cubes_h_'+str(np.round(h,3))+'.npy'
    grid_data = dict()
    grid_data["Nodes"] = nodes
    grid_data["Elements"] = elements
    np.save(filename,grid_data)
    print("Generated grid file : "+filename)
    
    #gridfilename = 'data/grids/angle_transformed.npy'
    
    #grid = bempp.api.shapes.sphere(1)
    
    grid = load_grid(filename)
    #grid.plot()
    def normal_x1(x,n,domain_index,result):
        #v = np.array([0,1,0]) 
        #result[:] = np.dot(v,n)
        result[:] = x[0]
    space = bempp.api.function_space(grid,'P',1)
    dspace = bempp.api.function_space(grid,'DP',0)
    n1 = bempp.api.GridFunction(space,fun = normal_x1,dual_space = dspace)
    n1.plot()
