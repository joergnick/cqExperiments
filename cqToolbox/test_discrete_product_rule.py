import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')


from linearcq import Conv_Operator
from rkmethods import RKMethod
import numpy as np

T = 1
N = 10
tau = T*1.0/N
rk = RKMethod("RadauIIA-2",tau)

tp = rk.get_time_points(T)
tp = tp[1:]
uu = tp**3
vv = tp**2
uv = uu*vv
def deriv(s,b,precomp=None):
    return s*b

td_deriv = Conv_Operator(deriv)

pt_uv = td_deriv.apply_RKconvol(uv,T,method = rk.method_name)
pt_uu = td_deriv.apply_RKconvol(uu,T,method = rk.method_name)
pt_vv = td_deriv.apply_RKconvol(vv,T,method = rk.method_name)
#
m = rk.m

shifted_uu = uu*0
for j in range(N-1):
    shifted_uu[(j+1)*m:(j+2)*m] = uu[j*m:(j+1)*m]

invA = np.linalg.inv(rk.A)
rule_mat = np.matmul(invA.dot(np.ones((rk.m,1))),rk.b)
#

print("Residual of initial CQ: ",np.linalg.norm(tau*pt_uv[0,:m]-invA.dot(uv[:m])))
rhs = invA.dot(uv[m:2*m])-np.matmul(rule_mat,invA).dot(uv[:m])
lhs = tau*pt_uv[0,m:2*m]
print("One step later: ",np.linalg.norm(lhs-rhs))
rhs = invA.dot(uu[m:2*m])-np.matmul(rule_mat,invA).dot(uu[:m])
lhs = tau*pt_uu[0,m:2*m]

print("Same for u ",np.linalg.norm(lhs-rhs))
#print(invA.dot(uv[:m]))
##print(uv[:m])
##print(vv[1:m+1])
#print(len(vv))
#print(len(pt_uu[0,:]))
#print("LHS: ",pt_uv[0,:m], " RHS : ",pt_uu[0,:m]*vv[0:m])
#res = pt_uv[0,:m]-pt_uu[0,:m]*vv[0:m]

#res = pt_uu[1:]-pt_uu*vv[1:]-rule_mat.dot(shifted_uu[1:]*pt_vv)
#print(res)

