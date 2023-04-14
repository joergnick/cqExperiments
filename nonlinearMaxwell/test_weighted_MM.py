import numpy as np
from customOperators import precompMM,sparseWeightedMM
space_string = "RT"
nrspace_string = "NC"
def Da(self,x):
    if np.linalg.norm(x)<10**(-15):
        x=10**(-15)*np.ones(3)
    #return np.eye(3)
    return ((self.alpha-1)*np.linalg.norm(x)**(self.alpha-3)*np.outer(x,x)+np.linalg.norm(x)**(self.alpha-1)*np.eye(3))

