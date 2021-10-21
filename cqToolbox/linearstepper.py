from cqStepper import AbstractIntegrator
import numpy as np
class ImplicitEuler(AbstractIntegrator):
    def time_step(self,W0,j,tau,method,history,w_star_sol_j):
        t   = tau*j 
        [A_RK,b_RK,c_RK,m] = self.tdForward.get_method_characteristics(methoo
    d) 
        rhs = np.zeros((1,m))
        for stageInd in range(m):
            rhs[:,stageInd] = 3*(t+tau*c_RK[stageInd])**2-w_star_sol_j[stageInd]
        Tdiag = 
        Tinv = np.linalg.inv(Tdiag)
        rhs = np.matmul(stageRhs,Tinv.T)
        sol = W0[0]**(-1)*(3*(t+tau*c_RK[0])**2-w_star_sol_j)
        return sol ,0
    def harmonicForward(self,s,b,precomp = None):
        return precomp*b
    def precomputing(self,s):
        return s    

int_der = ImplicitEuler()
Am = 8
err = np.zeros(Am)
Ns  = np.zeros(Am)
for j in range(Am):
    N   = 4*2**j
    Ns[j] = N
    sol = int_der.integrate(1,N,method = "RadauIIA-1")
    err[j] = max(np.abs(sol[0]-np.linspace(0,1,N+1)**3))
import matplotlib.pyplot as plt
plt.loglog(Ns**(-1),Ns**(-1),linestyle='dashed')
plt.loglog(Ns**(-1),err)
#plt.plot(sol[0])
#plt.plot(np.linspace(0,1,N+1)**3,linestyle='dashed')
plt.show()
