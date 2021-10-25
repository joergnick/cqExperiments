import numpy as np
import math
class RKMethod():
    "Collects data and methods corresponding to a Runge-Kutta multistage method."
    method_name = ""
    c,A,b,m = 0,0,0,0
    delta_eigs,Tdiag,Tinv,delta_zero,tau  = 0,0,0,0,0
    def __init__(self,method,tau):
        if (method =="RadauIIA-1") or (method =="BDF-1") or (method == "Implicit Euler"):
            self.A=np.array([[1]])
            self.c=np.array([1])
            self.b=np.array([1])
        elif (method == "RadauIIA-2"):        
            self.A=np.array([[5.0/12,-1.0/12],
                           [3.0/4,1.0/4]])
            self.c=np.array([1.0/3,1])    
            self.b=np.array([[3.0/4,1.0/4]])  
        elif (method == "RadauIIA-3"):
            self.A=np.array([[11.0/45-7*math.sqrt(6)/360, 37.0/225-169.0*math.sqrt(6)/1800 , -2.0/225+math.sqrt(6)/75],
                           [37.0/225+169.0*math.sqrt(6)/1800,11.0/45+7*math.sqrt(6)/360,-2.0/225-math.sqrt(6)/75],
                           [4.0/9-math.sqrt(6)/36,4.0/9+math.sqrt(6)/36,1.0/9]])
            self.c=np.array([2.0/5-math.sqrt(6)/10,2.0/5+math.sqrt(6)/10,1])
            self.b=np.array([4.0/9-math.sqrt(6)/36,4.0/9+math.sqrt(6)/36,1.0/9])
        else:
            raise ValueError("Given method "+method+" not implemented.")
        self.method_name = method
        self.m    = len(self.c)
        self.tau  = tau
        self.delta_zero = np.linalg.inv(self.A)/tau
        self.delta_eigs,self.Tdiag  = np.linalg.eig(self.delta_zero)
        self.Tinv = np.linalg.inv(self.Tdiag)
    def diagonalize(self,x):
       # if len(x.shape)==1:
       #     if len(x) % self.m !=0:
       #         raise ValueError("Vector dimensions of Input does not allow diagonalization.")
       #     x.reshape((len(x)/self.m,self.m))
        return np.matmul(x,self.Tinv.T)

    def reverse_diagonalize(self,b_dof_x_m):
        return np.matmul(b_dof_x_m,self.Tdiag.T)
