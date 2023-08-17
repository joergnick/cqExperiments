import numpy as np


for j in range(50):
    s = 0.01-5j+j*1.0j
    #print("s = ",s," np.sqrt(s) = ",np.sqrt(s), " np.emath.sqrt(s) = ",np.emath.sqrt(s))
    print("s = ",s," np.sqrt(s) = ",np.real(s/(1+np.sqrt(s))))
