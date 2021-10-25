from cqStepper import AbstractIntegrator
import numpy as np
from rkmethods import RKMethod
class ImplicitEuler(AbstractIntegrator):
    def time_step(self,W0,j,tau,method,history,w_star_sol_j):
        t   = tau*j 
        rk = RKMethod(method,tau=tau)
        rhs = np.zeros((1,m))
        for stageInd in range(m):
            rhs[:,stageInd] = 3*(t+tau*rk.c[stageInd])**2-w_star_sol_j[:,stageInd]
        rhs = rk.diagonalize(rhs)
        sol = np.zeros((1,m))
        for stageInd in range(m):
            sol[:,stageInd] = W0[stageInd]**(-1)*(rhs[:,stageInd])
        sol = np.real(rk.reverse_diagonalize(sol))
        return sol ,0
    def harmonicForward(self,s,b,precomp = None):
        return precomp*b
    def precomputing(self,s):
        return s    


arr = np.array([1,2,3,4,5,6])
print("SHAPE: ", arr.shape)
arr = arr.reshape(2,3)
print("SHAPE: ", arr.shape)
print(arr)
print(arr.T.ravel())
test = RKMethod("RadauIIA-1") 
print(test.m)
int_der = ImplicitEuler()
Am = 8
m  = 2
err = np.zeros(Am)
Ns  = np.zeros(Am)
for j in range(Am):
    N   = 4*2**j
    Ns[j] = N
    sol = int_der.integrate(1,N,method = "RadauIIA-"+str(m))
    
    err[j] = max(np.abs(sol[0,::m]-np.linspace(0,1,N+1)**3))
import matplotlib.pyplot as plt
print(err)
#plt.loglog(Ns**(-1),Ns**(-1),linestyle='dashed')
#plt.loglog(Ns**(-1),err)
##plt.plot(sol[0])
##plt.plot(np.linspace(0,1,N+1)**3,linestyle='dashed')
#plt.show()
