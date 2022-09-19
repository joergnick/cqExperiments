from conv_op import *
import bempp.api
import numpy as np
import matplotlib.pyplot as plt






def fract(s,b):
	return s**(0.5)*b 

deriv=Conv_Operator(fract)

T=2
gt=np.linspace(0,T,N+1)**3

gd=deriv.apply_convol(gt,T)















#
#print("gt : ",gt)
#print("gd : ",gd)
#plt.plot(np.linspace(0,2,N+1),gt,'bo')
#plt.plot(np.linspace(0,2,N+1),gd)
#plt.show()
##print(np.abs(gd[0,N]-12))
