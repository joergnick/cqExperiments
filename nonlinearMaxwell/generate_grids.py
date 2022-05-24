import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')

import bempp.api
import numpy as np

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
grid = bempp.api.import_grid('data/grids/angle_transformed.msh')
Nodes = grid.leaf_view.vertices
Elements = grid.leaf_view.elements
dof      = len(Nodes[0,:])
print("Generate grid, number of nodes: "+str(dof))
#filename = 'data/grids/sphere_python3_h' +str(h)+'.npy'
#filename = 'data/grids/sphereDOF'+str(dof)+'h' +str(h)+'.npy'
filename = 'data/grids/angle_transformed.npy'
grid_data = dict()
grid_data["Nodes"] = Nodes
grid_data["Elements"] = Elements
np.save(filename,grid_data)
print("Generated grid file : "+filename)
