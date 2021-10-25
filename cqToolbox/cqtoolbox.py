#import psutil
from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov
import numpy as np
from rkmethods import RKMethod
from linearcq import Conv_Operator
import sys
class CQModel:
    def __init__(self):
        self.tdForward = Conv_Operator(self.forward_wrapper)
        self.freqObj = dict()
        self.freqUse = dict()
        self.countEv = 0
    def time_step(self,s0,t,history,conv_history,x0):
        raise NotImplementedError("No time stepping given.") 
        ## Methods supplied by user:
    def nonlinearity(self,x,t,phi):
        raise NotImplementedError("No nonlinearity given.")
    def nonlinearityInverse(self,x):
        raise NotImplementedError("No inverse to nonlinearity given.")
    def harmonic_forward(self,s,b,precomp = None):
        raise NotImplementedError("No time-harmonic forward operator given.")
    def harmonicBackward(self,s,b,precomp = None):
        raise NotImplementedError("No time-harmonic backward operator given.")
#    def calc_jacobian(self,x,t,rhs):
#        raise NotImplementedError("No gradient given.")
    def applyJacobian(self,Jacobian,b):
        try: 
            return Jacobian.dot(b)
        except:
            raise NotImplementedError("Gradient has no custom applyGradient method, however * is not supported.") 
    def righthandside(self,t,history=None):
        return 0
        #raise NotImplementedError("No right-hand side given.")
        ## Optional method supplied by user:
    def precomputing(self,s):
        raise NotImplementedError("No precomputing given.")
    def preconditioning(self,precomp):
        raise NotImplementedError("No preconditioner given.")
        ## Methods provided by class
    def forward_wrapper(self,s,b):
        if s in self.freqObj:
            self.freqUse[s] = self.freqUse[s]+1
        else:
            self.freqObj[s] = self.precomputing(s)
            self.freqUse[s] = 1
        return self.harmonic_forward(s,b,precomp=self.freqObj[s])


    def calc_jacobian(self,x0,t,time_index):
        taugrad = 10**(-8)
        dof = len(x0)
        idMat = np.identity(dof)
        jacoba = np.zeros((dof,dof))
        for i in range(dof):
            y_plus =  self.nonlinearity(x0+taugrad*idMat[:,i],t,time_index)
            y_minus = self.nonlinearity(x0-taugrad*idMat[:,i],t,time_index)
            jacoba[:,i] = (y_plus-y_minus)/(2*taugrad)
        return jacoba

    def newton_solver(self,j,rk,rhs,W0, x0,history,tolsolver = 10**(-4),debugmode=False,coeff = 1):
        x0 = np.zeros(rhs.shape)

        print("Historyshape: ",history.shape)


        for i in range(rk.m):
            rhs[:,i] = rhs[:,i] + self.righthandside(j*rk.tau+rk.c[i]*rk.tau,history=history)
        counter = 0
        thresh = 3
        x = x0
        info = 1
        while info >0:
                if counter <=thresh:
                    scal = 1 
                else:
                    scal = 0.9
                x,info = self.newton_iteration(j,rk,rhs,W0,x,history,coeff=scal**(counter-thresh))
                if debugmode:
                    print("INFO AFTER {} STEP: ".format(counter),info)
                if np.linalg.norm(x)>10**5:
                    print("Warning, setback after divergence in Newton's method.")
                    x = x0
                    break
                counter = counter+1
        return x,counter

    def newton_iteration(self,j,rk,rhs,W0,x0,history,tolsolver = 10**(-6),debugmode=False,coeff = 1):
        t = j*rk.tau
        m = rk.m
       # for i in range(rk.m):
       #     rhs[:,i] = rhs[:,i] + self.righthandside(t+rk.c[i]*rk.tau,history=history)
        x0_pure = x0
        dof = len(rhs)
        for stage_ind in range(m):
            for j in range(dof):
                if np.abs(x0[j,stage_ind])<10**(-10):
                    x0[j,stage_ind] = 10**(-10)
        jacob_list = [self.calc_jacobian(x0[:,k],t+rk.tau*rk.c[k],j*m+k) for k in range(m)]
        stage_rhs = rk.diagonalize(x0+1j*np.zeros((dof,m)))
        ## Calculating right-hand side
        for stage_ind in range(m):
            stage_rhs[:,stage_ind] = self.harmonic_forward(rk.delta_eigs[stage_ind],stage_rhs[:,stage_ind],precomp=W0[stage_ind])
        stage_rhs = rk.reverse_diagonalize(stage_rhs)

        ax0 = np.zeros((dof,m))
        for stage_ind in range(m):
            ax0[:,stage_ind] = self.nonlinearity(x0[:,stage_ind],t+rk.tau*rk.c[stage_ind],j*m+stage_ind)
        rhs_newton = stage_rhs+ax0-rhs
        ## Solving system W0y = b
        rhs_long = 1j*np.zeros(m*dof)
        x0_pure_long = 1j*np.zeros(m*dof)
        for stage_ind in range(m):
            rhs_long[stage_ind*dof:(stage_ind+1)*dof] = rhs_newton[:,stage_ind]
            x0_pure_long[stage_ind*dof:(stage_ind+1)*dof] = x0_pure[:,stage_ind]
        def newton_func(x_dummy):
            x_mat    = x_dummy.reshape(m,dof).T
            x_diag   = rk.diagonalize(x_mat)
            grad_mat = 1j*np.zeros((dof,m))
            Bs_mat   = 1j*np.zeros((dof,m))
            for j in range(m):
                grad_mat[:,j] = self.applyJacobian(jacob_list[j],x_mat[:,j])
                Bs_mat[:,j]   = self.harmonic_forward(rk.delta_eigs[j],x_diag[:,j],precomp = W0[j])
            res_mat  = rk.reverse_diagonalize(Bs_mat) + grad_mat
            new_res =  res_mat.T.ravel()
            return new_res

        newton_lambda = lambda x: newton_func(x)
        from scipy.sparse.linalg import LinearOperator
        newton_operator = LinearOperator((m*dof,m*dof),newton_lambda)
        dx_long,info = gmres(newton_operator,rhs_long,restart = 2*m*dof,maxiter = 2*dof,x0=x0_pure_long,tol=1e-5)
        #dxlong,info = gmres(NewtonOperator,rhsLong,restart = 2*dof,maxiter = 2*dof,x0=x0pureLong,tol=1e-5)
        if info != 0:
            print("GMRES Info not zero, Info: ", info)
        dx = 1j*np.zeros((dof,m))
        for stageInd in range(m):
            dx[:,stageInd] = dx_long[dof*stageInd:dof*(stageInd+1)]
        x1 = x0-coeff*dx
        #print("np.linalg.norm(dx) = ",np.linalg.norm(dx))
        if coeff*np.linalg.norm(dx)/dof<tolsolver:
            info = 0
        else:
            info = coeff*np.linalg.norm(dx)/dof
        return np.real(x1),info

    def createFFTLengths(self,N):
        lengths = [1]
        it = 1
        while len(lengths)<=N:
            lengths.append( 2**it)
            lengths.extend(lengths[:-1][::-1])
            it = it+1
        return lengths
    def extrapol_coefficients(self,p):
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
        gammas = self.extrapol_coefficients(p)
        for j in range(p+1):
            extrU = extrU+gammas[j]*u[:,-p-1+j]
        return extrU
    
    def simulate(self,T,N,method = "RadauIIA-2",tolsolver = 10**(-5),reUse=True,debugMode=False):
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
        rhs = np.zeros((dof,m*N+1))
        sol = np.zeros((dof,m*N+1))
        extr = np.zeros((dof,m))
        counters = np.zeros(N)
        for j in range(0,N):
            if debugMode:
                print(j*1.0/N)
            ## Calculating solution at timepoint tj
            for i in range(rk.m):
                if j >=1:
                    extr[:,i] = self.extrapol(sol[:,i+1:j*rk.m+i+1:rk.m],1)
                else:
                    extr[:,i] = np.zeros(len(rhs[:,0]))
   #         ###  Use simplified Weighted Newon's method ######
            sol[:,j*m+1:(j+1)*m+1],info = self.newton_solver(j,rk,rhs[:,j*m+1:(j+1)*m+1],W0,extr,sol[:,:rk.m*j])
            #print("First Newton step finished. Info: ",info, "Norm of solution: ", np.linalg.norm(sol[:,j*m+1:(j+1)*m+1]))

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
            rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m] += -localconvHist[:,currLen*m:currLen*m+currLenCut*m]
            if not reUse:
                self.freqUse = dict()
                self.freqObj = dict()
        return sol ,counters
 
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


