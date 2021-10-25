import unittest
import numpy as np
import sys

sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
from cqtoolbox import CQModel
from newtonStepper import NewtonIntegrator

## Problems with analytic solutions 
class LinearScatModel(NewtonIntegrator):
    def precomputing(self,s):
        return s**2
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,history = None):
        return 8*7*t**6
    def nonlinearity(self,x,t,phi):
        return 0*x
    def ex_sol(self,ts):
        return ts**8
class LinearScatModelSimple(NewtonIntegrator):
    def precomputing(self,s):
        return s**1
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,history = None):
        return 3*t**2
    def nonlinearity(self,x,t,phi):
        return 0*x
    def ex_sol(self,ts):
        return ts**3


class NonlinearScatModel(NewtonIntegrator):
    def precomputing(self,s):
        return s**1
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,history = None):
        return 3*t**2+t**9
    def nonlinearity(self,x,t,phi):
        return x**3
    def ex_sol(self,ts):
        return ts**3

class NonlinearScatModel2Components(NewtonIntegrator):
    def precomputing(self,s):
        return np.array([s**1,s**2])
    def harmonic_forward(self,s,b,precomp = None):
        return np.array([precomp[0]*b[0],precomp[1]*b[1]])
    def righthandside(self,t,history = None):
        return np.array([3*t**2+t**9 +t**4,4*3*t**2+t**4])
    def nonlinearity(self,x,t,phi):
        return np.array([x[0]**3+x[1],x[1]])
    def ex_sol(self,ts):
        return np.array([ts**3,ts**4])


## Test cases, two for each of the predefined models
## above.
class TestCQMethods(unittest.TestCase):
    def test_linear_RadauIIA_1Simple(self):
        modelL       = LinearScatModelSimple()
        m = 1
        N = 30
        T = 1
        sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-1))


    def test_linear_RadauIIA_2(self):
        modelL       = LinearScatModel()
        m = 2
        N = 8
        T = 1
        sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-2))

    def test_linear_RadauIIA_3(self):
        modelL       = LinearScatModel()
        m = 3
        N = 9
        T = 1
        sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-3))

    def test_nonlinear_RadauIIA_2(self):
        modelN       = NonlinearScatModel()
        m = 2
        N = 11
        T = 2
        sol,counters = modelN.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-3))
    def test_nonlinear_RadauIIA_2_components(self):
        modelN       = NonlinearScatModel2Components()
        m = 2
        N = 15
        T = 2
        sol,counters = modelN.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = np.max(np.max(np.abs(sol[:,::m]-exSol)))
        self.assertLess(np.abs(err),10**(-3))


    def test_nonlinear_RadauIIA_3(self):
        modelN       = NonlinearScatModel()
        m = 3
        N = 7
        T = 2
        sol,counters = modelN.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-7))
    
    def test_extrapolation_p1(self):
        modelN       = NonlinearScatModel()
        self.assertTrue((modelN.extrapol_coefficients(1)==[-1,2]).all())
    def test_extrapolation_p2(self):
        modelN       = NonlinearScatModel()
        self.assertTrue((modelN.extrapol_coefficients(2)==[1,-3,3]).all())


if __name__ == '__main__':
    unittest.main()

