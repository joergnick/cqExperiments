function [] = semiDisNonlinear(space_errs_I,space_errs_II,time_errs_II,mesh_size,step_size,string1,s,figurenumber)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
%% Creating time convergence plots
h1=figure(figurenumber);
%subplot(1,2,1)
set(h1,'Position',[10 10 561 420])
h_s=mesh_size;
N_s=3./step_size;

 %   style_list={'-d','-o','-^','-h','-*','-+','-x','p','.','r','b','y'};


start_time=1;
end_time=9;
%% Creating the plots
start_space= 1;
end_space  =2;




err=time_errs_II(start_space:end_space,start_time:end_time);
[n1 n2]=size(err);

vect=step_size(start_time:end_time);

symbols='sox*d+^.v><';
ms=[6 6 6 8 6 6 6 6 6 6 6];
gr=(linspace(.66,0,n1))';
colors=[gr gr gr];
for jj=1:n1
    loglog(vect,err(jj,:), ...
           'LineWidth',1,...
           'Marker',symbols(jj),...
           'MarkerSize',ms(jj),...
           'Color', colors(jj,:));%'Color', 'black'); %
    if jj==1
        hold on;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 xlim([10^(-2.3) 10^(-0.8)])  
 ylim([10^(-3) 10^(-0)])

%% Plotting the reference line

  %loglog((step_size),5*step_size,'--k')
  loglog((step_size),5*step_size,'--k')
  loglog((step_size),50*step_size.^2,'--k')
 % loglog((step_size),10^3*step_size.^3,'--k')
 %% Title

title("Time convergence",'Interpreter','Latex')
    hold off
%% Labelling the axis
xlabel('step size $\tau$','Interpreter','Latex')
ylabel('Maximal Error in $\textit{\textbf{P}}=(0,0,0)$','Interpreter','Latex')

hold off
legend(strcat('$m=1$'), strcat('$m=2$') ,'$\mathcal O(\tau)$','$\mathcal O(\tau^2)$' ,'location','southeast');  
  
%% SPACE CONVERGENCE PLOT 1
%% Comment in from here for space convergence plot
saveas(gcf,'Plots/semidis_time_conv','epsc')  

h2=figure(figurenumber+1)
set(h2,'Position',[10 10 900 420])
subplot(1,2,1)

start_time=1;
end_time=1;
%% Creating the plots
start_space=1;
end_space=6;

vect = mesh_size(start_space:end_space);
err=space_errs_I(start_space:end_space,start_time:end_time)';
[n1 n2]=size(err);

symbols='sox*d+^.v><';
ms=[6 6 6 8 6 6 6 6 6 6 6];
gr=(linspace(.66,0,n1))';
colors=[gr gr gr];

for jj=1:n1
    loglog(vect,err(jj,:), ...
           'LineWidth',1,...
           'Marker',symbols(jj),...
           'MarkerSize',ms(jj),...
           'Color', colors(jj,:));%'Color', 'black'); %
    if jj==1
        hold on;
    end
end


%% Plotting the reference line

 loglog(mesh_size,0.7*mesh_size.^1,'--k')
 % loglog(mesh_size,0.7*mesh_size.^1.5,'--k')
 %% Title
% strL2=strcat(string1,' s= ',num2str(s),' L2 time conv.');
%strL2=strcat(string1,' L^2-norm time conv.');
%title(strL2)
title("Space convergence I", 'Interpreter','Latex')
    hold off
 xlim([10^(-1.0) 10^(0.1)])  
 ylim([10^(-2) 10^(-0.)])
    
%% Creating the legend
  l=legend({strcat('$N = 32$'),'$\mathcal O(h)$'},'Interpreter','Latex','location','southeast');

%% Labelling the axis
xlabel('mesh width $h$','Interpreter','Latex')
ylabel('Maximal Error in $\textit{\textbf{P}}=(0,0,0)$','Interpreter','Latex')
%ylabel('H^1-norm error')
hold off

%% SPACE CONVERGENCE PLOT 2
%% Comment in from here for space convergence plot

subplot(1,2,2)

start_time=1;
end_time=1;
%% Creating the plots
start_space=1;
end_space=6;

%vect=step_size(start_time:end_time);

vect = mesh_size(start_space:end_space);
err=space_errs_II(start_space:end_space,start_time:end_time)'
[n1 n2]=size(err);

symbols='sox*d+^.v><';
ms=[6 6 6 8 6 6 6 6 6 6 6];
gr=(linspace(.66,0,n1))';
colors=[gr gr gr];



for jj=1:n1
    loglog(vect,err(jj,:), ...
           'LineWidth',1,...
           'Marker',symbols(jj),...
           'MarkerSize',ms(jj),...
           'Color', colors(jj,:));%'Color', 'black'); %
    if jj==1
        hold on;
    end
end




% loglog(step_size(start_time:end_time),H1_err(1,start_time:end_time),'-d')

%xlim([10^(-1.5) 10^(-1)])  
%ylim([10^(-3) 10^(-1)])

%  
%loglog(step_size(start_time:end_time),H1_err(2,start_time:end_time),'-o')
% % % % marksize=8;
% % % % colormap gray(30)
% % % % %loglog(step_size(start_time:end_time),H1_err(3,start_time:end_time),'-^','MarkerSize',marksize)
% % % % loglog(step_size(start_time:end_time),H1_err(4,start_time:end_time),'-h','MarkerSize',marksize)
% % % % xlim([10^(-1.5) 10^(0)]) 
% % % % ylim([10^(-2) 10^(0)]) 
% % % % 
% % % % hold on
% % % % 
% % % % loglog(step_size(start_time:end_time),H1_err(5,start_time:end_time),'-*','MarkerSize',marksize)
% % % % loglog(step_size(start_time:end_time),H1_err(6,start_time:end_time),'-d','MarkerSize',marksize)
% % % % loglog(step_size(start_time:end_time),H1_err(7,start_time:end_time),'-o','MarkerSize',marksize)
% % % % loglog(step_size(start_time:end_time),H1_err(8,start_time:end_time),'-s','MarkerSize',marksize)
%% Plotting the reference line

 loglog(mesh_size,0.8*mesh_size.^1,'--k')
 xlim([10^(-1.0) 10^(0.1)])  
 ylim([10^(-2) 10^(-0.)])
 %% Title
% strL2=strcat(string1,' s= ',num2str(s),' L2 time conv.');
%strL2=strcat(string1,' L^2-norm time conv.');
%title(strL2)
title("Space convergence II", 'Interpreter','Latex')
    hold off
% xlim([10^(-1.2) 10^(0)])  
% ylim([10^(-2.5) 10^(0)])
    
%% Creating the legend
%legend({strcat('dof = ',num2str(dof_s(4))),strcat('dof = ',num2str(dof_s(5))),...
 %  strcat('dof = ',num2str(dof_s(6))),strcat('dof = ',num2str(dof_s(7))),strcat('dof = ',num2str(dof_s(8))),'$\mathcal{O}(\tau^2)$'},'location','southeast' ,'FontSize',14 );

 
%  l=legend({strcat('$N$ = ',num2str(N_s(3))),strcat('$N$ = ',num2str(N_s(4))),...
%      strcat('$N$ = ',num2str(N_s(5))),strcat('$N$ = ',num2str(N_s(6))),...
%     strcat('$N$ = ',num2str(N_s(7))),strcat('$N$ = ',num2str(N_s(7))),'$\mathcal O(h)$','$\mathcal O(h^{3/2})$'},'Interpreter','Latex','location','southeast');
 l=legend({strcat('$N = 32$'),'$\mathcal O(h)$'},'Interpreter','Latex','location','southeast');

%% Labelling the axis
xlabel("mesh width $h$",'Interpreter','Latex')
ylabel("Maximal Error in $\textit{\textbf{P}}=(0,0,0)$",'Interpreter','Latex')
%ylabel('H^1-norm error')
hold off

saveas(gcf,'Plots/semidis_space_conv','epsc')  
end

