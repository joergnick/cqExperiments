import numpy as np
from experiment_helpers import *

N_per_tp=400
time_points=np.array([5])
dx=1
import time
#boundary_cond="GIBC"
#boundary_cond="Dirichlet"
n_grid_points=200

boundary_cond="Absorbing"
#scatter_waves_plot(N_per_tp,n_grid_points,time_points,dx,boundary_cond)

print("Absorbing b.c. completed!")
boundary_cond="Acoustic"
#scatter_waves_plot(N_per_tp,n_grid_points,time_points,dx,boundary_cond)

print("Acoustic b.c. completed!")
boundary_cond="GIBC"
scatter_waves_plot(N_per_tp,n_grid_points,time_points,dx,boundary_cond)

print("Thin-layer b.c. completed!")
