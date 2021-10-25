from cqStepper import AbstractIntegrator
import numpy as np
from rkmethods import RKMethod
class ImplicitEuler(AbstractIntegrator):
    def time_step(self,W0,j,rk,history,w_star_sol_j):
        tau = rk.tau
        t   = tau*j 
        rhs = np.zeros((1,m))
        for stageInd in range(m):
            rhs[:,stageInd] = 3*(t+tau*rk.c[stageInd])**2-w_star_sol_j[:,stageInd]
        rhs = rk.diagonalize(rhs)
        sol = 1j*np.zeros((1,m))
        for stageInd in range(m):
            sol[:,stageInd] = W0[stageInd]**(-1)*(rhs[:,stageInd])
        sol = np.real(rk.reverse_diagonalize(sol))
        return sol ,0

    def harmonicForward(self,s,b,precomp = None):
        return precomp*b
    def precomputing(self,s):
        return s    


test = RKMethod("RadauIIA-1",1) 
print(test.m)
int_der = ImplicitEuler()
Am = 8
m  = 2
T  = 1
err = np.zeros(Am)
Ns  = np.zeros(Am)
for j in range(Am):
    N   = 4*2**j
    Ns[j] = N
    rk = RKMethod("RadauIIA-"+str(m),T*1.0/N)
    sol = int_der.integrate(T,rk)
    
    err[j] = max(np.abs(sol[0,::m]-np.linspace(0,1,N+1)**3))
import matplotlib.pyplot as plt
print(err)
#plt.loglog(Ns**(-1),Ns**(-1),linestyle='dashed')
#plt.loglog(Ns**(-1),err)
##plt.plot(sol[0])
##plt.plot(np.linspace(0,1,N+1)**3,linestyle='dashed')
#plt.show()
