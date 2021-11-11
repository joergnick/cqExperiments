import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')
#from cqtoolbox import CQModel
from newtonStepper import NewtonIntegrator
from linearcq import Conv_Operator
from rkmethods import RKMethod
from data_generators import compute_densities
import numpy as np

gridfilename='data/grids/sphereh1.0.npy'
T = 8
N = 2**4
tau = T*1.0/N
rk = RKMethod("RadauIIA-3",tau)
sol = compute_densities(N,gridfilename,T,rk,)
filename = 'data/sphereDOF' + str(len(sol[:,0])/2) +'N'+str(N)+ '.npy'
resDict = dict()
resDict["sol"] = sol
resDict["T"] = T
resDict["m"] = rk.m
resDict["N"] = N
np.save(filename,resDict)
