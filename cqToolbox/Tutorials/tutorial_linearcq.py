import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')

import numpy as np

from linearcq import Conv_Operator
from rkmethods import RKMethod
from linearcq import Conv_Operator

T = 1
N = 10
method = "RadauIIA-3"

## Task: Integrate the function t^8


rk = RKMethod(method,T*1.0/N)
rhs = rk.get_time_points(T)**8


def th_integral(s,b):
    return s**(-1)*b
integral = Conv_Operator(th_integral)

sol = integral.apply_RKconvol(rhs,T,method = method,first_value_is_t0=True)
ex  = 1.0/9*rk.get_time_points(T)**9
print(np.max(np.abs(sol-ex)))
#print(rhs)
