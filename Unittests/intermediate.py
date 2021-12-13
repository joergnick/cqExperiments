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

def deriv2(s,b):
    return s**2*b

td2_op = Conv_Operator(deriv2)


m = 3
T = 1

Am = 12
errs = np.zeros(Am)
taus = np.zeros(Am)
for j in range(6):
    N = 4*2**j
    tau = T*1.0/N
    taus[j] = tau
    rk = RKMethod("RadauIIA-"+str(m),tau)
    rhs = rk.get_time_points(T)**5
    ex_sol = 5*4*rk.get_time_points(T)**3
    sol2 = td2_op.apply_RKconvol(rhs[1:],T,method = rk.method_name)
    errs[j] = np.max(np.abs(sol2[0,::m]-ex_sol[1::m]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.loglog(taus,taus,linestyle='dashed')
plt.loglog(taus,errs)

#plt.plot(np.real(sol2[0,::m]))
#plt.plot(exSol,linestyle='dashed',color='red')
#plt.semilogy(np.abs(sol[0,::m]-exSol),linestyle='dashed',color='red')
plt.savefig('temp.png')
#err          = max(np.abs(sol[0,::m]-exSol))