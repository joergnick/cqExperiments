clear all
%load('../data/Err_data_delta10.mat')
% load('../data/err_data_nonlinear_server2.mat')
load('../data/error_m_2_space_II.mat')
errors(3,:) = [];
errors_II = errors;
load('../data/error_m_2_space_I.mat')
errors(3,:) = [];
errors_I = errors;
hs(3) = [];
h_s = hs;
%load('../data/error_m_2_time_I.mat')
load('../data/error_multiple_m_time_II.mat')
tau_s = taus;

%load('data/Err_data_delta0p1.mat')
%load('data/Ref_Err_plot_data.mat')
%Plot_error_matrices_final( Err_H1,Err_L2,h_s,dof_s/2,tau_s,"Absorbing b.c.",2,1)

% ERR2=ERR;
% load('../data/Err_data_delta01.mat')
%load('data/Err_data_newsigns2.mat')
% ERR = errors;
% ERR(3,:) = [];
% h_s = hs;
% h_s(3) = [];
% tau_s = taus;
% ERR(:,1:length(ERR(1,:)))
% h_s
% tau_s
semiDisNonlinear(errors_I,errors_II,errors,h_s,tau_s, "Absorbing b.c.", 3,1)
