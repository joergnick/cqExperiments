import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')
import bempp.api
from data_generators import load_grid
import numpy as np
#gridfilename = 'data/grids/angle.npy'
gridfilename = 'data/grids/angle_oriented.npy'
#gridfilename = 'data/grids/angle_transformed.npy'

#grid = bempp.api.shapes.sphere(1)

grid = load_grid(gridfilename)
#grid = bempp.api.shapes.sphere(1)
#grid.plot()

def normal_x1(x,n,domain_index,result):
    v = np.array([0,0,1]) 
    result[:] = np.dot(v,n)
space = bempp.api.function_space(grid,'P',1)
dspace = bempp.api.function_space(grid,'DP',0)
n1 = bempp.api.GridFunction(space,fun = normal_x1,dual_space = dspace)
n1.plot()

def normal_x1(x,n,domain_index,result):
    v = np.array([0,1,0]) 
    result[:] = np.dot(v,n)
space = bempp.api.function_space(grid,'P',1)
dspace = bempp.api.function_space(grid,'DP',0)
n1 = bempp.api.GridFunction(space,fun = normal_x1,dual_space = dspace)
n1.plot()

def normal_x1(x,n,domain_index,result):
    v = np.array([1,0,0]) 
    result[:] = np.dot(v,n)
space = bempp.api.function_space(grid,'P',1)
dspace = bempp.api.function_space(grid,'DP',0)
n1 = bempp.api.GridFunction(space,fun = normal_x1,dual_space = dspace)
n1.plot()

