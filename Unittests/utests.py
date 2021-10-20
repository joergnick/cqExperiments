import unittest
import numpy as np
import sys
sys.path.append('../cqToolbox')
#from .. import nonlinear1ddamping
from cqtoolbox import CQModel


class LinearScatModel(CQModel):
    def precomputing(self,s):
        return s**2
    def harmonicForward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,history = None):
        return 8*7*t**6
    def nonlinearity(self,x,t,phi):
        return 0*x
    def ex_sol(self,ts):
        return ts**8

class NonlinearScatModel(CQModel):
    def precomputing(self,s):
        return s**1
    def harmonicForward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,history = None):
        return 3*t**2+t**9
    def nonlinearity(self,x,t,phi):
        return x**3
    def ex_sol(self,ts):
        return ts**3


class TestCQMethods(unittest.TestCase):
#    def setUp(self):
#        self.model = LinearScatModel()
#        self.modelN = NonlinearScatModel() 
    def test_LinearRadauIIA_2(self):
        modelL       = LinearScatModel()
        m = 2
        N = 8
        T = 1
        sol,counters = modelL.simulate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-2))

    def test_LinearRadauIIA_3(self):
        modelL       = LinearScatModel()
        m = 3
        N = 9
        T = 1
        sol,counters = modelL.simulate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-3))

    def testNonlinearRadauIIA_2(self):
        modelN       = NonlinearScatModel()
        m = 2
        N = 11
        T = 2
        sol,counters = modelN.simulate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-3))

    def testNonlinearRadauIIA_3(self):
        modelN       = NonlinearScatModel()
        m = 3
        N = 7
        T = 2
        sol,counters = modelN.simulate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-7))



if __name__ == '__main__':
    unittest.main()

