#import numpy as np
#import bempp.api
#import math
#from RKconv_op import *
import bempp.api
import numpy as np
import math
#from RKconv_op import *
import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
sys.path.append('data')
sys.path.append('../data')
sys.path.append('..')
from linearcq import Conv_Operator
from rml_main import compute_densities
import math

import time

T=8


import time
am_space = 3
am_time  = 9
diffs = np.zeros(am_time)
norms_direct = np.zeros(am_time)
diffs_direct = np.zeros(am_time)
m = 2
import time
for space_index in range(am_space):
    for time_index in range(am_time):
        h   = 2**(-(space_index+6)*1.0/2)
        N   = int(np.round(8*2**(time_index)))
        #### MAX DIFFERENCE IS 0.012 for N:
        #N   = 255*2**time_index
        #STILL WORKS until at least 85% : N   = 600*2**time_index
        tau = T*1.0/N
        start = time.time() 
        compute_densities(h,N,T,m,use_sphere=True)
        end = time.time()
        print("h = "+str(h)+", N = "+str(N)+ " end - start = "+str(end-start))
