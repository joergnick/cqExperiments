#import psutil
from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov
import numpy as np
from linearcq import Conv_Operator
import sys
class newtonModel(CQStepper):
        ## Methods supplied by user:
    def nonlinearity(self,x,t):
        raise NotImplementedError("No nonlinearity given.")
    def righthandside(self,t,history=None):
        return 0
        #raise NotImplementedError("No right-hand side given.")
        ## Optional method supplied by user:
    def precomputing(self,s):
        raise NotImplementedError("No precomputing given.")

    def extrapol(self,u,p):
        if len(u[0,:])<=p+1:
            u = np.concatenate((np.zeros((len(u[:,0]),p+1-len(u[0,:]))),u),axis=1)
        extrU = np.zeros(len(u[:,0]))
        gammas = self.extrapol_coefficients(p)
        for j in range(p+1):
            extrU = extrU+gammas[j]*u[:,-p-1+j]
        return extrU
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

    def apply_jacobian(J,x):
        try:
            return J.dot(x)
        except:
             raise NotImplementedError("Given Jacobian does not support .dot method and the method apply_jacobian has not been overwritten by user.")

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

    def time_step(self,j,rk,sol_history,w_star_sol_j):
        x0pure = x0
        dof = len(rhs)
        m = len(W0)
        for stageInd in range(m):
            for j in range(dof):
                if np.abs(x0[j,stageInd])<10**(-10):
                    x0[j,stageInd] = 10**(-10)
        Tinv = np.linalg.inv(Tdiag)

        jacobList = [self.calc_jacobian(x0[:,k],t,rhsInhom[:,k]) for k in range(m)]

        stageRHS = rk.diagonalize(x0+1j*np.zeros((dof,m)))
        ## Calculating right-hand side
        for stageInd in range(m):
            stageRHS[:,stageInd] = self.harmonicForward(deltaEigs[stageInd],stageRHS[:,stageInd],precomp=W0[stageInd])
        stageRHS = rk.reverse_diagonalize(stageRHS)
        #rhsNewton = [stageRHS[:,k]+self.nonlinearity(x0[:,k])-rhs[:,k] for k in range(m)]
        ax0 = np.zeros((dof,m))
        
        for stageInd in range(m):
            ax0[:,stageInd] = self.nonlinearity(x0[:,stageInd],t+tau*c_RK[stageInd],rhsInhom[:,stageInd])
        rhsNewton = stageRHS+ax0-rhs
        ## Solving system W0y = b
        rhsLong = 1j*np.zeros(m*dof)
        x0pureLong = 1j*np.zeros(m*dof)
        for stageInd in range(m):
            rhsLong[stageInd*dof:(stageInd+1)*dof] = rhsNewton[:,stageInd]
            x0pureLong[stageInd*dof:(stageInd+1)*dof] = x0pure[:,stageInd]
        def NewtonFunc(x_dummy):
            x_mat    = x_dummy.reshape(m,dof).T
            x_diag   = rk.diagonalize(x_mat)

            grad_mat = 1j*np.zeros((dof,m))
            Bs_mat   = 1j*np.zeros((dof,m))
            for j in range(m):
                grad_mat[:,j] = self.apply_jacobian(jacobList[j],x_mat[:,j])
                Bs_mat[:,j]   = self.harmonic_forward(deltaEigs[j],x_diag[:,j],precomp = W0[j])
            res_mat  = rk.reverse_diagonalize(Bs_mat) + grad_mat
            new_res =  res_mat.T.ravel()
            return new_res
