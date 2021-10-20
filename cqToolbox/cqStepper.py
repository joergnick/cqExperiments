#import psutil
from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov
import numpy as np
from linearcq import Conv_Operator
import sys
class CQModel:
    def __init__(self):
        self.tdForward = Conv_Operator(self.forwardWrapper)
        self.freqObj = dict()
        self.freqUse = dict()
        self.countEv = 0
    def time_step(self,s0,t,history,conv_history,x0):
        raise NotImplementedError("No time stepping given.") 
        ## Methods supplied by user:
    def nonlinearity(self,x,t):
        raise NotImplementedError("No nonlinearity given.")
    def harmonicForward(self,s,b,precomp = None):
        raise NotImplementedError("No time-harmonic forward operator given.")
    def harmonicBackward(self,s,b,precomp = None):
        raise NotImplementedError("No time-harmonic backward operator given.")
    def righthandside(self,t,history=None):
        return 0
        #raise NotImplementedError("No right-hand side given.")
        ## Optional method supplied by user:
    def precomputing(self,s):
        raise NotImplementedError("No precomputing given.")

        ## Methods provided by class
    def forwardWrapper(self,s,b):
        if s in self.freqObj:
            self.freqUse[s] = self.freqUse[s]+1
        else:
            self.freqObj[s] = self.precomputing(s)
            self.freqUse[s] = 1
        return self.harmonicForward(s,b,precomp=self.freqObj[s])

    def createFFTLengths(self,N):
        lengths = [1]
        it = 1
        while len(lengths)<=N:
            lengths.append( 2**it)
            lengths.extend(lengths[:-1][::-1])
            it = it+1
        return lengths

    def extrapolCoefficients(self,p):
            coeffs = np.ones(p+1)
            for j in range(p+1):
                    for m in range(p+1):
                            if m != j:
                                    coeffs[j]=coeffs[j]*(p+1-m)*1.0/(j-m)
            return coeffs

    def extrapol(self,u,p):
        if len(u[0,:])<=p+1:
            u = np.concatenate((np.zeros((len(u[:,0]),p+1-len(u[0,:]))),u),axis=1)
        extrU = np.zeros(len(u[:,0]))
        gammas = self.extrapolCoefficients(p)
        for j in range(p+1):
            extrU = extrU+gammas[j]*u[:,-p-1+j]
        return extrU
    
    def integrate(self,T,N,method = "RadauIIA-2",tolsolver = 10**(-5),reUse=True,debugMode=False):
        tau = T*1.0/N
        ## Initializing right-hand side:
        lengths = self.createFFTLengths(N)
        try:
            dof = len(self.righthandside(0))
        except:
            dof = 1
        ## Actual solving:
        [A_RK,b_RK,c_RK,m] = self.tdForward.get_method_characteristics(method)
        S00 = np.linalg.inv(A_RK)/tau
        deltaEigs,Tdiag =np.linalg.eig(charMatriS0) 
        W0 = []
        for j in range(m):
            W0.append(self.precomputing(deltaEigs[j]))

        rhs = np.zeros((dof,m*N+1))
        sol = np.zeros((dof,m*N+1))
        extr = np.zeros((dof,m))
        if rhsInhom is None:
            rhsInhom = np.zeros((dof,m*N+1))
        for j in range(0,N):
            if debugMode:
                print(j*1.0/N)
            ## Calculating solution at timepoint tj
            tj = tau*j
            for i in range(m):
                rhs[:,j*m+i+1] = rhs[:,j*m+i+1] + self.righthandside(tj+c_RK[i]*tau,history=sol[:,:j*m])
                if j >=1:
                    extr[:,i] = self.extrapol(sol[:,i+1:j*m+i+1:m],m+1)
                else:
                    extr[:,i] = np.zeros(dof)

            sol[:,j*m+1:(j+1)*m+1],info = self.timeStep(j,tau,c_RK,deltaEigs,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,extr,rhsInhom[:,j*m+1:(j+1)*m+1])
            ## Solving Completed #####################################
            ## Calculating Local History:
            currLen = lengths[j]
            localHist = np.concatenate((sol[:,m*(j+1)+1-m*currLen:m*(j+1)+1],np.zeros((dof,m*currLen))),axis=1)
            if len(localHist[0,:])>=1:
                localconvHist = np.real(self.tdForward.apply_RKconvol(localHist,(len(localHist[0,:]))*tau/m,method = method,show_progress=False))
            else:
                break
            ## Updating Global History: 
            currLenCut = min(currLen,N-j-1)
            rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m] = rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m]-localconvHist[:,currLen*m:currLen*m+currLenCut*m]
            if not reUse:
                self.freqUse = dict()
                self.freqObj = dict()
        return sol 
 
#    def integrate(self,T,N,method = "RadauIIA-2",tolsolver = 10**(-8)):
#        tau = T*1.0/N
#        ## Initializing right-hand side:
#        lengths = self.createFFTLengths(N)
#        try:
#            dof = len(self.righthandside(0))
#        except:
#            dof = 1
#        ## Actual solving:
#        [A_RK,b_RK,c_RK,m] = self.tdForward.get_method_characteristics(method)
#        charMatrix0 = np.linalg.inv(A_RK)/tau
#        deltaEigs,Tdiag =np.linalg.eig(charMatrix0) 
#    #   print(np.matmul(Tdiag,np.linalg.inv(Tdiag)))
#    #   print(np.matmul(np.matmul(np.linalg.inv(Tdiag),charMatrix0),Tdiag))
#        W0 = []
#        for j in range(m):
#            W0.append(self.precomputing(deltaEigs[j]))
#        #zeta0 = self.tdForward.delta(0)/tau
#        #W0 = self.precomputing(zeta0)
#        rhs = np.zeros((dof,m*N+1))
#        sol = np.zeros((dof,m*N+1))
#        extr = np.zeros((dof,m))
#        for j in range(0,N):
#            ## Calculating solution at timepoint tj
#            tj       = tau*j
#	    print(j)
#            #print("NEW STEP : ",j, "ex_ sol: ",[(tj+c*tau)**3 for c in c_RK])
#            for i in range(m):
#                rhs[:,j*m+i+1] = rhs[:,j*m+i+1] + self.righthandside(tj+c_RK[i]*tau,history=sol[:,:j*m])
#                if j >=1:
#                    extr[:,i] = self.extrapol(sol[:,i+1:j*m+i+1:m],m+1)
#                else:
#                    extr[:,i] = np.zeros(dof)
#   #         ###  Use simplified Weighted Newon's method ######
#            sol[:,j*m+1:(j+1)*m+1] = extr
#            sol[:,j*m+1:(j+1)*m+1] = self.time_step(W0,tj,,rhs[:,j*m+1:(j+1)*m+1],sol[:,j*m+1:(j+1)*m+1],Tdiag,charMatrix0)
#            #sol[:,j*m+1:(j+1)*m+1],grada,info = self.time_step(W0,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,sol[:,j*m+1:(j+1)*m+1],charMatrix0)
#            ## Solving Completed #####################################
#            ## Calculating Local History:
#            currLen = lengths[j]
#            localHist = np.concatenate((sol[:,m*(j+1)+1-m*currLen:m*(j+1)+1],np.zeros((dof,m*currLen))),axis=1)
#            if len(localHist[0,:])>=1:
#                localconvHist = np.real(self.tdForward.apply_RKconvol(localHist,(len(localHist[0,:]))*tau/m,method = method,show_progress=False))
#            else:
#                break
#            ## Updating Global History: 
#            currLenCut = min(currLen,N-j-1)
#            rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m] = rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m]-localconvHist[:,currLen*m:currLen*m+currLenCut*m]
#        self.freqUse = dict()
#        self.freqObj = dict()
#        return sol 
#


