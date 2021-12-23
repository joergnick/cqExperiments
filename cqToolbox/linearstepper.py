from cqStepper import AbstractIntegrator
import numpy as np
from rkmethods import RKMethod
import math
import random
class ImplicitEuler(AbstractIntegrator):
    def time_step(self,W0,j,rk,history,w_star_sol_j,tolsolver = 0):
        tau = rk.tau
        t   = tau*j 
        rhs = np.zeros((1,m))
        for stageInd in range(m):
            rhs[:,stageInd] = 1.0/5*(t+tau*rk.c[stageInd])**5-w_star_sol_j[:,stageInd]
        rhs = rk.diagonalize(rhs)
        sol = 1j*np.zeros((1,m))
        for stageInd in range(m):
            sol[:,stageInd] = W0[stageInd]**(-1)*(rhs[:,stageInd])
        sol = np.real(rk.reverse_diagonalize(sol))
        return np.real(sol)
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def precomputing(self,s):
        return s**(-2)*(1+10**(-10)*random.uniform(-1,1))

int_der = ImplicitEuler()
Am = 12
m  = 2
T  = 1
err = np.zeros(Am)
Ns  = np.zeros(Am)
for j in range(Am):
    #N   = 10000
    N   = 4*2**j+1
    Ns[j] = N
    rk = RKMethod("RadauIIA-"+str(m),T*1.0/N)
    sol,counters = int_der.integrate(T,N,method = rk.method_name,factor_laplace_evaluations = 2,max_evals_saved = 100000)
    err[j] = max(np.abs(sol[0,::m]-4*np.linspace(0,T,N+1)**3))
    print(Ns)
    print(err)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot(sol[0,::m])
plt.plot(4*np.linspace(0,T,N+1)**3,linestyle='dashed',color='red')
#plt.semilogy(np.abs(sol[0,::]-4*rk.get_time_points(T)**3))
#plt.semilogy(np.abs(sol[0,::m]-4*np.linspace(0,T,N+1)**3))
#print(Ns)
#print(err)

#plt.loglog(Ns**(-1),Ns**(-2),linestyle='dashed')
#plt.loglog(Ns**(-1),err)

plt.savefig('temp.png')
##plt.plot(sol[0])
##plt.plot(np.linspace(0,1,N+1)**3,linestyle='dashed')
#plt.show()
