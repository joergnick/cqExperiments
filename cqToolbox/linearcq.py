import numpy as np
from rkmethods import Extrapolator,RKMethod
class Conv_Operator():
    tol=10**-15
    #tol=10**-20
    external_N   = -1
    external_rho = None
    external_L = None
    factor_laplace_evaluations= -1
    def __init__(self,apply_elliptic_operator,order=2):
        self.order=order
        self.delta=lambda zeta : self.char_functions(zeta,order)
        self.apply_elliptic_operator=apply_elliptic_operator
    def get_integration_parameters(self,N,T):
        tol=self.tol
        dt=(T*1.0)/N
        if self.external_L:
            L = self.external_L
        else:
            if self.external_N<0:
                self.external_N = N
            L=max(4*int(N),1)
            ###################### BEST WORKING PARAMETERS KNOWN: 
            #L=4*int(N)
        #L=3.0/2*N
        #rho=tol**(1.0/(2*L))
        if self.external_rho:
            rho = self.external_rho
            print("WARNING, EXTERNAL RHO WAS SET.")
        else:
            rho=tol**(1.0/(3*N))
            #rho=tol**(1.0/((3.0/2*L)))
            ###################### BEST WORKING PARAMETERS KNOWN: 
            #rho=tol**(1.0/(4*N))
            #rho=tol**(1.0/(L))
            #rho=tol**(1.0/(L))
            #rho=tol**(1.0/(2*N))
            #rho=tol**(1.0/(3*N))
            #rho=tol**(1.0/(2*self.external_N))
        return L,dt,tol,rho

    def char_functions(self,zeta,order):
        if order==1:
            return 1-zeta
        else:
            if order==2:
                return 1.5-2.0*zeta+0.5*zeta**2
            else:
                if order ==3:
                    return (1-zeta)+0.5*(1-zeta)**2+1.0/3.0*(1-zeta)**3
                else:
                    print("Multistep order not availible")

    def get_frequencies(self,N,T):
        import numpy as np
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        Unit_Roots=np.exp(-1j*2*np.pi*(np.linspace(0,L-1,L)/(L)))
        return rho*Unit_Roots

    def get_method_characteristics(self,method):
        import numpy as np
        import math
        rk = RKMethod(method,1)
        return rk.A,rk.b,rk.c,rk.m
       
    def format_rhs(self,rhs,m):
        try:
            N=int(round((len(rhs[0,:]))//m))
        except:
            N=int(round((len(rhs))/m))
            rhs_mat=np.zeros((1,m*N))
            if (len(rhs)==m*N+1):
                ## Assume first value is zero:
                print("Assume first value is zero value, rhs[0] = ",rhs[0])
                rhs = rhs[1:]
            rhs_mat[0,:]=rhs
            rhs=rhs_mat 
        return rhs,N

    def apply_RKconvol(self,rhs,T,show_progress=True,method="RadauIIA-2",factor_laplace_evaluations=2,cutoff=10**(-15),prolonge_by = 0,external_rho = None,external_L = None):
        if external_rho:
            self.external_rho = external_rho
        if external_L:
            self.external_L = external_L
        m      = RKMethod(method,1).m
        [rhs,N]=self.format_rhs(rhs,m)

#        if N>10:
#            L,dt,tol,rho=self.get_integration_parameters(N,T)
#            print("N= ",N)
#            print(rho)
#            print(rhs)
#            raise ValueError("Hi")

        tau = T*1.0/N ## tau stays the same after prolongation
        rk   = RKMethod(method,tau) ## rk also stays the same, as tau is equal.
        prolonged_flag = (prolonge_by>0)
        ### Prolongation
        if (N>10) and prolonged_flag : 
            extr = Extrapolator()
            rhs  = extr.prolonge_towards_0(rhs,prolonge_by,rk)    
        ## Step 1
        self.factor_laplace_evaluations = max(1,factor_laplace_evaluations)
        [A_RK,b_RK,c_RK,m]=self.get_method_characteristics(method)
        self.m=m    
        [rhs,N]=self.format_rhs(rhs,m)
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        rhs_fft=1j*np.zeros((len(rhs[:,0]),m*L))

        for stageInd in range(m):
            rhs_fft[:,stageInd:m*L:m]=self.scale_fft(rhs[:,stageInd:m*N:m],N,T)
        #Initialising important parameters for the later stage  
        s_vect=self.get_frequencies(N,T)
        dof=len(rhs[:,0])
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        Half=int(np.ceil(float(L)/2.0))
        ## Step 2
        normsRHS=np.ones(m*L)
        counter=0
        for j in range(0,m*L):
            normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
            if normsRHS[j]>cutoff:
                counter=counter+1
        if normsRHS[0]>10**200:
            raise ValueError("Fourier coefficients have exploded.")


        HalfL= int(np.ceil(float(L)/2.0))
        if show_progress:
            print("Amount of Systems needed: "+ str(counter))

        ##Timing the elliptic systems
        import time
        start=0
        end=0

        rhsStages=1j*np.zeros((len(rhs_fft[:,0]),m))
        invA_RK = np.linalg.inv(A_RK)
        for j in range(0,HalfL+1):
            s=s_vect[j]
            deltaMatrix=np.linalg.inv(A_RK+s*1.0/(1-s)*np.ones((m,1))*b_RK)
            deltaEigs,T =np.linalg.eig(deltaMatrix)
            if np.linalg.cond(T)>200:
                print("CONDITION LARGE", np.linalg.cond(T))
            #res = np.linalg.norm(deltaMatrix-T.dot(np.diag(deltaEigs)).dot(np.linalg.inv(T)))
            #if np.linalg.cond(deltaMatrix)>1000:
            #    print("s = "+str(s)+"CONDITION deltaMatrix: ", np.linalg.cond(deltaMatrix))
            #    print(deltaEigs)
            deltaEigs=deltaEigs/dt
            Tinv=np.linalg.inv(T)
#           print("NormT: ",np.linalg.norm(T))
            if show_progress:
                print("j:",j,"L:",L, "Time of previous iteration: " +str((end-start)/60), " MIN" )
            start=time.time()
            if j>0:
                relevantChange=False
                rhsStages=1j*rhsStages*0
                lhsStages=1j*lhsStages*0
                for stageInd in range(m):
                    for sumInd in range(m):
                        rhsStages[:,stageInd]=rhsStages[:,stageInd]+Tinv[stageInd,sumInd]*rhs_fft[:,m*j+sumInd]
                    maxRHS=np.max(np.abs(rhsStages[:,stageInd]))
                    #if (maxRHS>cutoff or j<20 ) and (j<400):
                    #if j<= first_minimum:
                    if maxRHS>cutoff :
                        relevantChange=True
                        lhsStages[:,stageInd]=self.apply_elliptic_operator(deltaEigs[stageInd],rhsStages[:,stageInd])       
            else:
                relevantChange=True
                for sumInd in range(m):
                    rhsStages[:,0]=rhsStages[:,0]+Tinv[0,sumInd]*rhs_fft[:,m*j+sumInd]
                first_eval=self.apply_elliptic_operator(deltaEigs[0],rhsStages[:,0])
                phi_hat=1j*np.zeros((len(first_eval),m*L))
                lhsStages=1j*np.zeros((len(phi_hat[:,0]),m))
                lhsStages[:,0]=first_eval
                for stageInd in range(1,m):
                    for sumInd in range(m):
                        rhsStages[:,stageInd]=rhsStages[:,stageInd]+Tinv[stageInd,sumInd]*rhs_fft[:,m*j+sumInd]
                    lhsStages[:,stageInd]=self.apply_elliptic_operator(deltaEigs[stageInd],rhsStages[:,stageInd])       
            if relevantChange:  
                for stageInd in range(m):
                    for sumInd in range(m):
                        phi_hat[:,m*j+stageInd]=phi_hat[:,m*j+stageInd]+T[stageInd,sumInd]*lhsStages[:,sumInd]
            end=time.time()
        ## Mirroring the second part of the frequencies by complex conjugation
        for freqInd in range(HalfL,L):  
            for stageInd in range(m):
                phi_hat[:,freqInd*m+stageInd]=np.conj(phi_hat[:,m*(L-freqInd)+stageInd])

        ## Step 3, Postprocessing
        for stageInd in range(m):
            phi_hat[:,stageInd:m*L:m]=self.rescale_ifft(phi_hat[:,stageInd:m*L:m],N,T)
        phi_sol=phi_hat[:,:m*N]
        if prolonged_flag:
            return extr.cut_back(phi_sol)
        else:
            return phi_sol
    def get_zeta_vect(self,N,T):
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        import numpy as np
        Unit_Roots=np.exp(-1j*2*np.pi*(np.linspace(0,L,L+1)/(L+1)))
        Zeta_vect=self.delta( rho* Unit_Roots)/dt 
        return Zeta_vect



    def apply_convol(self,rhs,T,show_progress=False,method="BDF2",cutoff=10**(-8)):
        ## Step 1
        if not len(rhs[0]):
            raise ValueError("Right hand side provided is empty.")
        try:
            N=len(rhs[0,:])-1
        except:
            N=len(rhs)-1    
            rhs_mat=np.zeros((1,N+1))
            rhs_mat[0,:]=rhs
            rhs=rhs_mat

        L,dt,tol,rho=self.get_integration_parameters(N,T)
        normsRHS=np.ones(N+1)
        for j in range(0,N+1):
            normsRHS[j]=np.max(np.abs(rhs[:,j]))
####################### Scaling +FFT of RHS ##################################  
        n_rows=len(rhs[:,0])
        n_columns=len(rhs[0,:])
        rhs=rho**(np.linspace(0,n_columns-1,N+1))*rhs
        rhs_fft=np.fft.fft(np.concatenate((rhs,np.zeros((n_rows,L+1-n_columns))),axis=1))
#######################     
        #Initialising important parameters for the later stage  
        Zeta_vect=self.get_zeta_vect(N,T)
        dof=len(rhs[:,0])
        Half=int(np.ceil(float(L)/2.0))
        ## Step 2
        #phi_hat=1j*np.zeros((dof,L+1))
        normsRHS=np.ones(L+1)
        counter=0
        for j in range(0,Half+1):
            normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
            if normsRHS[j]>cutoff:
                counter=counter+1
        #Timing the elliptic systems
        import time
        start=0
        end=0
        first_eval=self.apply_elliptic_operator(Zeta_vect[0],rhs_fft[:,0])
        phi_hat=1j*np.zeros((len(first_eval),L+1))
        phi_hat[:,0]=first_eval

        for j in range(1,Half+1):
            normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
            #print("normRHS:",normsRHS[j])
            if normsRHS[j]>cutoff:
                if show_progress:
                    print("j:",j,"L:",str(Half), "Time of previous iteration: " +str((end-start)/60), " MIN" )
                start=time.time()
                phi_hat[:,j]=self.apply_elliptic_operator(Zeta_vect[j],rhs_fft[:,j])
                end=time.time()
        for j in range(Half+1,L+1):
            phi_hat[:,j]=np.conj(phi_hat[:,L+1-j])      
        ## Step 3

        phi_sol=self.rescale_ifft(phi_hat,N,T)
        phi_sol=phi_sol[:,:N+1]
        return phi_sol


    def scale_fft(self,A,N,T):
        import numpy as np
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        n_rows=len(A[:,0])
        n_columns=len(A[0,:])
        A=rho**(np.linspace(0,n_columns-1,n_columns))*A
        A_fft=np.fft.fft(np.concatenate((A,np.zeros((n_rows,L-n_columns))),axis=1))
        return A_fft

    def rescale_ifft(self,A,N,T):
        import numpy as np
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        n_rows=len(A[:,0])
        n_columns=len(A[0,:])

        A_ift=np.real(np.fft.ifft(A))
        A_sol=rho**(-np.linspace(0,n_columns-1,n_columns))*A_ift[:,0:n_columns]
        return(A_sol)

