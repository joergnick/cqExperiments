import numpy as np
import sys
import warnings
warnings.filterwarnings("error")
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
from linearcq import Conv_Operator
from cqtoolbox import CQModel
from newtonStepper import NewtonIntegrator
from rkmethods     import RKMethod

class LinearScatModelInt2(NewtonIntegrator):
    def precomputing(self,s):
        return s**(-2)
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,history = None):
        return 1.0/5*t**5
    def nonlinearity(self,x,t,time_index):
        return 0*x
    def ex_sol(self,ts):
        return 4*ts**3



modelL       = LinearScatModelInt2()
m = 1
N = 100
T = 1
tau = T*1.0/N
sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
rk = RKMethod("RadauIIA-"+str(m),tau)
rhs = 1.0/5*rk.get_time_points(T)**5
def deriv2(s,b):
    return s**1*b
td2_op = Conv_Operator(deriv2)
sol2 = td2_op.apply_RKconvol(rhs[1:],T)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot(np.real(sol2[0,::m]))
plt.plot(exSol,linestyle='dashed',color='red')
#plt.semilogy(np.abs(sol[0,::m]-exSol),linestyle='dashed',color='red')
plt.savefig('temp.png')
err          = max(np.abs(sol[0,::m]-exSol))