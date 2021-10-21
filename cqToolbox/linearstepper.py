from cqStepper import AbstractIntegrator
import numpy as np
class ImplicitEuler(AbstractIntegrator):
    def time_step(self,W0,j,tau,method,history,w_star_sol_j):
        t   = tau*j 
        [A_RK,b_RK,c_RK,m] = self.tdForward.get_method_characteristics(method)
        rhs = np.zeros((1,m))
        for stageInd in range(m):
            rhs[:,stageInd] = 3*(t+tau*c_RK[stageInd])**2-w_star_sol_j[:,stageInd]
        S0 = np.linalg.inv(A_RK)/tau
        deltaEigs,Tdiag = np.linalg.eig(S0)
        Tinv = np.linalg.inv(Tdiag)
        rhs = np.matmul(rhs,Tinv.T)
        sol = np.zeros((1,m))
        for stageInd in range(m):
            sol[:,stageInd] = W0[stageInd]**(-1)*(rhs[:,stageInd])
        sol = np.real(np.matmul(sol,Tdiag.T))
        return sol ,0
    def harmonicForward(self,s,b,precomp = None):
        return precomp*b
    def precomputing(self,s):
        return s    

class RKMethod():
    "Collects data and methods corresponding to a Runge-Kutta method."
    method = ""
    m      = 0
    Tdiag  = 0
    Tinv   = 0 
    c_RK   = 0
    A_RK   = 0
    b_RK   = 0
    def __init__(self,method):
        if (method is "RadauIIA-1") or (method is "Implicit Euler") or (method is "BDF-1"):
            m     = 1
            Tdiag = 1
            Tinv  = 1
            c_RK  = [1]
            A_RK  = [[1]]
            b_RK  = [1]

test = RKMethod("RadauIIA-1") 
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
plt.loglog(Ns**(-1),Ns**(-1),linestyle='dashed')
plt.loglog(Ns**(-1),err)
#plt.plot(sol[0])
#plt.plot(np.linspace(0,1,N+1)**3,linestyle='dashed')
plt.show()
