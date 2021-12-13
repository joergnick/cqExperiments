from cqStepper import AbstractIntegrator
import numpy as np
from rkmethods import RKMethod
class ImplicitEuler(AbstractIntegrator):
    def time_step(self,W0,j,rk,history,w_star_sol_j,tolsolver = 0):
        tau = rk.tau
        t   = tau*j 
        rhs = np.zeros((1,m))
        for stageInd in range(m):
            rhs[:,stageInd] = (t+tau*rk.c[stageInd])**3-w_star_sol_j[:,stageInd]
        rhs = rk.diagonalize(rhs)
        sol = 1j*np.zeros((1,m))
        for stageInd in range(m):
            sol[:,stageInd] = W0[stageInd]**(-1)*(rhs[:,stageInd])
        sol = np.real(rk.reverse_diagonalize(sol))
        return sol 
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def precomputing(self,s):
        return s**(-1)


int_der = ImplicitEuler()
Am = 11
m  = 2
T  = 1
err = np.zeros(Am)
Ns  = np.zeros(Am)
for j in range(Am):
    N   = 4*2**j
    Ns[j] = N
    print(Ns)
    rk = RKMethod("RadauIIA-"+str(m),T*1.0/N)
    sol,counters = int_der.integrate(T,N,method = rk.method_name)
    err[j] = max(np.abs(sol[0,::m]-3*np.linspace(0,1,N+1)**2))
    print(err)
import matplotlib.pyplot as plt
print(Ns)
print(err)
#plt.loglog(Ns**(-1),Ns**(-1),linestyle='dashed')
#plt.loglog(Ns**(-1),err)
##plt.plot(sol[0])
##plt.plot(np.linspace(0,1,N+1)**3,linestyle='dashed')
#plt.show()
