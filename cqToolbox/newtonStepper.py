#import psutil
from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov
import numpy as np
from cqStepper import AbstractIntegrator
from rkmethods import RKMethod
from linearcq import Conv_Operator
class gmres_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
class NewtonIntegrator(AbstractIntegrator):
    def __init__(self):
        self.tdForward = Conv_Operator(self.forward_wrapper)
        self.freqObj = dict()
        self.freqUse = dict()
        self.countEv = 0
        self.debug_mode = False
        ## Methods supplied by user:
    def nonlinearity(self,x,t,time_index):
        raise NotImplementedError("No nonlinearity given.")
    def harmonic_forward(self,s,b,precomp = None):
        raise NotImplementedError("No time-harmonic forward operator given.")
    def apply_jacobian(self,jacobian,b):
        try: 
            return jacobian.dot(b)
        except:
            raise NotImplementedError("Gradient has no custom applyGradient method, however * is not supported.") 
    def righthandside(self,t,history=None):
        return 0
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

    def time_step(self,W0,j,rk,history,w_star_sol_j):
        x0  = np.zeros(w_star_sol_j.shape)
        rhs = np.zeros(w_star_sol_j.shape)
        for i in range(rk.m):
            rhs[:,i] = -w_star_sol_j[:,i] + self.righthandside(j*rk.tau+rk.c[i]*rk.tau,history=history)
            if j >=1:
                x0[:,i] = self.extrapol(history[:,i+1:j*rk.m+i+1:rk.m],1)
            else:
                x0[:,i] = np.zeros(len(w_star_sol_j[:,0]))
        counter = 0
        thresh = 10
        x = x0
        info = 1
        while info >0:
            if counter <=thresh:
                scal = 1 
            else:
                scal = 0.5
            x,info = self.newton_iteration(j,rk,rhs,W0,x,history,tolsolver = 10**(-5),coeff=scal**(counter-thresh))
            if self.debug_mode:
                print("INFO AFTER {} STEP: ".format(counter),info)
            if np.linalg.norm(x)>10**5:
                print("Warning, setback after divergence in Newton's method.")
                x = x0
                break
            counter = counter+1
            print("NEWTON ITERATION COUNTER: ",counter, " info: ",info)
        return x

    def newton_iteration(self,j,rk,rhs,W0,x0,history,tolsolver = 10**(-5),coeff = 1):
        t = j*rk.tau
        m = rk.m
        x0_pure = x0
        dof = len(rhs)
        for stage_ind in range(m):
            for dof_index in range(dof):
                if np.abs(x0[dof_index,stage_ind])<10**(-10):
                    x0[dof_index,stage_ind] = 10**(-10)
        jacob_list = [self.calc_jacobian(x0[:,k],t+rk.tau*rk.c[k],j*m+k+1) for k in range(m)]
        stage_rhs = rk.diagonalize(x0+1j*np.zeros((dof,m)))
        ## Calculating right-hand side
        for stage_ind in range(m):
            stage_rhs[:,stage_ind] = self.harmonic_forward(rk.delta_eigs[stage_ind],stage_rhs[:,stage_ind],precomp=W0[stage_ind])
        stage_rhs = rk.reverse_diagonalize(stage_rhs)

        ax0 = np.zeros((dof,m))
        for stage_ind in range(m):
            ax0[:,stage_ind] = self.nonlinearity(x0[:,stage_ind],t+rk.tau*rk.c[stage_ind],j*m+stage_ind+1)
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
            for m_index in range(m):
                grad_mat[:,m_index] = self.apply_jacobian(jacob_list[m_index],x_mat[:,m_index])
                Bs_mat[:,m_index]   = self.harmonic_forward(rk.delta_eigs[m_index],x_diag[:,m_index],precomp = W0[m_index])
            res_mat  = rk.reverse_diagonalize(Bs_mat) + grad_mat
            new_res =  res_mat.T.ravel()
            return new_res

        newton_lambda = lambda x: newton_func(x)
        from scipy.sparse.linalg import LinearOperator
        newton_operator = LinearOperator((m*dof,m*dof),newton_lambda)
        counterObj = gmres_counter()
        print("Residual: ",np.linalg.norm(rhs_long))
        dx_long,info = gmres(newton_operator,rhs_long,maxiter = 100,callback = counterObj,tol=1e-3)
        print("Residual after GMRES: ",np.linalg.norm(rhs_long-newton_func(dx_long)))
        print("COUNT GMRES: ",counterObj.niter)
        print("NORM dx_long: ",np.linalg.norm(dx_long))
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