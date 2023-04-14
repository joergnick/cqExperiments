import numpy as np


for j in range(10):
    s = 1.0-5j+j*1.0j
    print("s = ",s," np.sqrt(s) = ",np.sqrt(s), " np.emath.sqrt(s) = ",np.emath.sqrt(s))
