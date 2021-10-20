from cqStepper import AbstractIntegrator
import numpy as np
class ImplicitEuler(AbstractIntegrator):
    def time_step(self,W0,j,tau,c_RK,history,w_star_sol_j):
        t   = tau*j
        sol = W0[0]**(-1)*(3*(t+tau*c_RK[0])**2-w_star_sol_j)
        return sol ,0
    def harmonicForward(self,s,b,precomp = None):
        return precomp*b
    def precomputing(self,s):
        return s    
int_der = ImplicitEuler()
N=100
sol = int_der.integrate(1,N,method = "RadauIIA-1")
import matplotlib.pyplot as plt
plt.plot(sol[0])
plt.plot(np.linspace(0,1,N+1)**3,linestyle='dashed')
plt.show()
