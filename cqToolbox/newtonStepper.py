#import psutil
from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov
import numpy as np
from linearcq import Conv_Operator
import sys
class newtonModel(CQStepper):
    def time_step(self,S0,W0,j,N,sol_history,w_star_sol):
        raise NotImplementedError("No time stepping given.") 
        ## Methods supplied by user:
    def nonlinearity(self,x,t):
        raise NotImplementedError("No nonlinearity given.")
    def righthandside(self,t,history=None):
        return 0
        #raise NotImplementedError("No right-hand side given.")
        ## Optional method supplied by user:
    def precomputing(self,s):
        raise NotImplementedError("No precomputing given.")

    def extrapolCoefficients(self,p):
            coeffs = np.ones(p+1)
            for j in range(p+1):
                    for m in range(p+1):
                            if m != j:
                                    coeffs[j]=coeffs[j]*(p+1-m)*1.0/(j-m)
            return coeffs

    def calc_jacobian(self,x0,t,phi=0):
        taugrad = 10**(-8)
        dof = len(x0)
        idMat = np.identity(dof)
        jacoba = np.zeros((dof,dof))
        for i in range(dof):
            y_plus =  self.nonlinearity(x0+taugrad*idMat[:,i],t,phi)
            y_minus = self.nonlinearity(x0-taugrad*idMat[:,i],t,phi)
            jacoba[:,i] = (y_plus-y_minus)/(2*taugrad)
        return jacoba

    def applyJacobian(J,x):
        try:
            return J*b
        except:
             raise NotImplementedError("Given Jacobian does not support *-Operator and the method applyJacobian has not been overwritten by user.")

    def time_step(self,S0,t,history,conv_history,x0):

    def time_step(self,t,tau,c_RK,deltaEigs,rhs,W0,Tdiag, x0,rhsInhom,tolsolver = 10**(-4),debugmode=False,coeff = 1):
            x0pure = x0
            dof = len(rhs)
            m = len(W0)
            for stageInd in range(m):
                for j in range(dof):
                    if np.abs(x0[j,stageInd])<10**(-10):
                        x0[j,stageInd] = 10**(-10)
            Tinv = np.linalg.inv(Tdiag)
            jacobList = [self.calcJacobian(x0[:,k],t,rhsInhom[:,k]) for k in range(m)]
            stageRHS = x0+1j*np.zeros((dof,m))
            ## Calculating right-hand side
            stageRHS = np.matmul(stageRHS,Tinv.T)
            for stageInd in range(m):
                stageRHS[:,stageInd] = self.harmonicForward(deltaEigs[stageInd],stageRHS[:,stageInd],precomp=W0[stageInd])
            stageRHS = np.matmul(stageRHS,Tdiag.T)
            #rhsNewton = [stageRHS[:,k]+self.nonlinearity(x0[:,k])-rhs[:,k] for k in range(m)]
            ax0 = np.zeros((dof,m))
    
            for stageInd in range(m):
                ax0[:,stageInd] = self.nonlinearity(x0[:,stageInd],t+tau*c_RK[stageInd],rhsInhom[:,stageInd])
            rhsNewton = stageRHS+ax0-rhs
            ## Solving system W0y = b
            rhsLong = 1j*np.zeros(m*dof)
    
            for stageInd in range(m):
                rhsLong[stageInd*dof:(stageInd+1)*dof] = rhsNewton[:,stageInd]
                x0pureLong[stageInd*dof:(stageInd+1)*dof] = x0pure[:,stageInd]
    
            def NewtonFunc(xdummy):
                idMat  = np.identity(dof)
                Tinvdof = np.kron(Tinv,idMat)
                Tdiagdof = np.kron(Tdiag,idMat)
                ydummy = 1j*np.zeros(dof*m)
                BsTxdummy = 1j*np.zeros(dof*m)
                Daxdummy = 1j*np.zeros(dof*m)
                Txdummy = Tinvdof.dot(xdummy)
                for j in range(m):
                    BsTxdummy[j*dof:(j+1)*dof] = self.harmonicForward(deltaEigs[j],Txdummy[j*dof:(j+1)*dof],precomp = W0[j])
                    Daxdummy[j*dof:(j+1)*dof] =self.applyJacobian(jacobList[j],xdummy[j*dof:(j+1)*dof])
                ydummy = Tdiagdof.dot(BsTxdummy)+Daxdummy
                return ydummy
            NewtonLambda = lambda x: NewtonFunc(x)
            from scipy.sparse.linalg import LinearOperator
            NewtonOperator = LinearOperator((m*dof,m*dof),NewtonLambda)
            dxlong,info = gmres(NewtonOperator,rhsLong,restart = 2*m*dof,maxiter = 2*dof,x0=x0pureLong,tol=1e-5)
   f info != 0:
            print("GMRES Info not zero, Info: ", info)
        dx = 1j*np.zeros((dof,m))
        for stageInd in range(m):
            dx[:,stageInd] = dxlong[dof*stageInd:dof*(stageInd+1)]
        x1 = x0-coeff*dx
        #print("np.linalg.norm(dx) = ",np.linalg.norm(dx))
        if coeff*np.linalg.norm(dx)/dof<tolsolver:
            info = 0
        else:
            info = coeff*np.linalg.norm(dx)/dof
        return np.real(x1),info
 
