clear all
load('../data/ERR_DATA_ACOUSTIC_dp0.mat')
%load('data/ERR_DATA.mat')
%load('data/Ref_Err_plot_data.mat')
%Plot_error_matrices_final( Err_H1,Err_L2,h_s,dof_s/2,tau_s,"Absorbing b.c.",2,1)

Plot_errors(ERR,h_s,tau_s, "Absorbing b.c.", 3,1)
