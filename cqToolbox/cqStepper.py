#import psutil
from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov
import numpy as np
from linearcq import Conv_Operator
import sys
from rkmethods import RKMethod
class AbstractIntegrator:
    def __init__(self):
        self.tdForward = Conv_Operator(self.forwardWrapper)
        self.freqObj = dict()
        self.freqUse = dict()
        self.countEv = 0

        ## Methods supplied by user:
    def time_step(self,s0,W0,t,history,conv_history,x0):
        raise NotImplementedError("No time stepping given.") 
    def harmonicForward(self,s,b,precomp = None):
        raise NotImplementedError("No time-harmonic forward operator given.")
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

    def integrate(self,T,N,method = "RadauIIA-2",tolsolver = 10**(-5),re_use=True,debug_mode=False):
        tau = T*1.0/N
        rk = RKMethod(method,tau)
        m = rk.m
        ## Initializing right-hand side:
        lengths = self.createFFTLengths(N)
        try:
            dof = len(self.righthandside(0))
        except:
            dof = 1
        ## Actual solving:
        W0 = []
        for j in range(m):
            W0.append(self.precomputing(rk.delta_eigs[j]))
        conv_hist = np.zeros((dof,m*N+1))
        sol = np.zeros((dof,m*N+1))
        counters = np.zeros(N)
        for j in range(0,N):
            if debug_mode:
                print(j*1.0/N)
            ## Calculating solution at timepoint tj
            sol[:,j*m+1:(j+1)*m+1] = self.time_step(W0,j,rk,sol[:,:rk.m*(j)+1],conv_hist[:,j*m+1:(j+1)*m+1])
            ## Calculating Local History:
            currLen = lengths[j]
            localHist = np.concatenate((sol[:,m*(j+1)+1-m*currLen:m*(j+1)+1],np.zeros((dof,m*currLen))),axis=1)
            if len(localHist[0,:])>=1:
                localconvHist = np.real(self.tdForward.apply_RKconvol(localHist,(len(localHist[0,:]))*tau/m,method = method,show_progress=False))
            else:
                break
            ## Updating Global History: 
            currLenCut = min(currLen,N-j-1)
            conv_hist[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m] += localconvHist[:,currLen*m:currLen*m+currLenCut*m]
            if not re_use:
                self.freqUse = dict()
                self.freqObj = dict()
        return sol ,counters
#    def integrate(self,T,rk,reUse=True,debugMode=False):
#        tau = rk.tau
#        m   = rk.m
#        N   = int(T/tau)
#        ## Initializing right-hand side:
#        lengths = self.createFFTLengths(N)
#        try:
#            dof = len(self.righthandside(0))
#        except:
#            dof = 1
#        ## Actual solving:
#        S0 = np.linalg.inv(rk.A)/rk.tau
#        delta_eigs,T_diag =np.linalg.eig(S0) 
#        W0 = []
#        for j in range(m):
#            W0.append(self.precomputing(delta_eigs[j]))
#        sol       = np.zeros((dof,m*N+1))
#        conv_hist = np.zeros((dof,m*N+1))
#        for j in range(0,N):
#            if debugMode: print(j*1.0/N)
#            ## Calculating solution at timepoint tj
#            tj = tau*j
#            next_sol,info = self.time_step(W0,j,rk,sol[:,:j*m+1],conv_hist[:,j*m+1:(j+1)*m+1])
#            sol[:,j*m+1:(j+1)*m+1] = next_sol
#            ## Solving Completed #####################################
#            ## Calculating Local History:
#            curr_len = lengths[j]
#            local_hist = np.concatenate((sol[:,m*(j+1)+1-m*curr_len:m*(j+1)+1],np.zeros((dof,m*curr_len))),axis=1)
#            if len(local_hist[0,:])>=1:
#                local_conv_hist = np.real(self.tdForward.apply_RKconvol(local_hist,(len(local_hist[0,:]))*tau/m,method = rk.method_name,show_progress=False))
#            else:
#                break
#            ## Updating Global History: 
#            curr_len_cut = min(curr_len,N-j-1)
#            conv_hist[:,(j+1)*m+1:(j+1)*m+1+curr_len_cut*m] +=local_conv_hist[:,curr_len*m:curr_len*m+curr_len_cut*m]
#            if not reUse:
#                self.freqUse = dict()
#                self.freqObj = dict()
#        return sol 
#
#