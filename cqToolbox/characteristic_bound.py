from rkmethods import RKMethod
import numpy as np

N = 1000000000
T = 1
tau = T*1.0/N
rk = RKMethod("RadauIIA-3",tau)

invA = np.linalg.inv(rk.A)
def Delta(zeta):
    return invA-zeta*invA.dot(np.ones((rk.m,1))).dot(rk.b).dot(invA)

def invDelta(zeta):
    return rk.A+zeta*1.0/(1-zeta)*np.ones((rk.m,1)).dot(rk.b)

tt = np.linspace(0,2*np.pi,10000)
points_inner_circle = 0.9999999999*np.exp(1j*tt)
evals = np.zeros(len(tt))
for index,point in enumerate(points_inner_circle):
    #evals[index] = np.linalg.norm(  Delta(np.conj(point)).T.dot(invDelta(point)))
    #evals[index] = np.linalg.norm(  np.conj(invDelta(point)).T.dot(Delta(point)))
    evals[index] = np.linalg.norm(  np.conj(invDelta(point)).T.dot(Delta(point)))

print("MAX EVALS: ",max(evals))
print(Delta(0.99).dot(invDelta(0.99)))
print(np.ones((rk.m,1)).dot(rk.b))
vals = np