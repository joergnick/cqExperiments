import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')

from scipy.sparse.linalg import gmres
from customOperators import precompMM,sparseMM,sparseWeightedMM,applyNonlinearity
import bempp.api
import numpy as np

grid = bempp.api.shapes.sphere(h=1)
rt_space = bempp.api.function_space(grid,"BC",0)
gridfunList,neighborlist,domainDict = precompMM(rt_space)
id_custom = sparseMM(rt_space,gridfunList,neighborlist,domainDict)
id_bempp = bempp.api.operators.boundary.sparse.identity(rt_space,rt_space,rt_space).weak_form()
def Da(x):
    return 2*np.eye(3)
weightGF = bempp.api.GridFunction.from_zeros(rt_space)
id_custom2 = sparseWeightedMM(rt_space,weightGF ,Da ,gridfunList,neighborlist,domainDict)
id_mat = bempp.api.as_matrix(id_bempp)


print("Max difference = ",np.max(np.abs(2*id_mat- id_custom2.toarray())))