import numpy as np
class DirectCQ:
    tol = 10**(-15)
    N       = -1
    weights = []
    def __init__(self,laplace_evals,N):
        self.N       = N
        self.laplace_evals = laplace_evals
        self.weights = self.calc_weights(N)
    def calc_weights(self,N):
        L = 2*N
        rho = self.tol**(1.0/(2*L))
        
        return range(N)
    def forward_convolution(self,weights,g):
        if( len(weights) != len(g)):
            raise ValueError('Lengths are different.')
        N = len(weights) - 1
        w_star_g = np.zeros(N+1) 
        for j in range(N+1):
            for i in range(j+1):
                w_star_g[j] += weights[i]*g[j-i]
        return w_star_g
def laplace_evals(s):
    return s
cq = DirectCQ(laplace_evals,5)
w = [1, 0.5,1]
g = [0, 1,2]
print(cq.forward_convolution(w,g))