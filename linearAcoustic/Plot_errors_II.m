function [] = Plot_errors_II( ERRS,mesh_size,step_size,string1,s,figurenumber)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
%% Creating time convergence plots
figure(figurenumber)
%subplot(1,2,1)

h_s=mesh_size;
N_s=4./step_size;

 %   style_list={'-d','-o','-^','-h','-*','-+','-x','p','.','r','b','y'};


start_time=1;
end_time=7;
%% Creating the plots
start_space=2;
end_space=6;

vect=step_size(start_time:end_time);
err=ERRS(start_space:end_space,start_time:end_time);

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
%% TIME BOUNDS
 xlim([10^(-2.2) 10^(-0.2)])  
 ylim([10^(-4) 10^(-0.7)])

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

 loglog((step_size),0.8*step_size.^s,'--k')
 %% Title
% strL2=strcat(string1,' s= ',num2str(s),' L2 time conv.');
%strL2=strcat(string1,' L^2-norm time conv.');
%title(strL2)
title("Time convergence II",'Interpreter','Latex')
    hold off

    
%% Creating the legend
%legend({strcat('dof = ',num2str(dof_s(4))),strcat('dof = ',num2str(dof_s(5))),...
 %  strcat('dof = ',num2str(dof_s(6))),strcat('dof = ',num2str(dof_s(7))),strcat('dof = ',num2str(dof_s(8))),'$\mathcal{O}(\tau^2)$'},'location','southeast' ,'FontSize',14 );

 % legend(strcat('h = ',num2str(dof_s(2))),strcat('h = ',num2str(dof_s(3))),...
%    strcat('h = ',num2str(dof_s(4))),strcat('h = ',num2str(dof_s(5))),...
%    strcat('h = ',num2str(dof_s(6))),strcat('h = ',num2str(dof_s(7))),'O(\tau^2)' );
%% Labelling the axis
xlabel('step size $\tau$','Interpreter','Latex')
ylabel('Maximal Error in $\textit{\textbf{P}}=(2,0,0)$','Interpreter','Latex')

% strH1=strcat(string1,' H^1-norm of error  ');
%title(strH1)

%ylabel('H^1-norm error')
hold off
%   legend({strcat('$h$ = ',num2str(h_s(1))),strcat('$h$  = ',num2str(h_s(2))),strcat('$h$  = ',num2str(h_s(3))),...
%     strcat('$h$  = ',num2str(h_s(4))),strcat('$h$  = ',num2str(h_s(5))),...
%     strcat('$h$  = ',num2str(h_s(6))),strcat('$h$  = ',num2str(h_s(7))),strcat('$h$  = ',num2str(h_s(8))),'$\mathcal O(\tau^2)$' },'location','southeast','FontSize',9);
% 
%   legend({'$h = 2^{0}$','$h = 2^{-1/2}$','$h = 2^{-1}$',...
%     '$h = 2^{-3/2}$','$h = 2^{-2}$',...
%     '$h = 2^{-5/2}$','$h = 2^{-3}$','$h = 2^{-7/2}$','$\mathcal O(\tau^2)$' },'location','southeast','FontSize',9);

  legend({'$h = 2^{-1}$','$h = 2^{-2}$',...
    '$h = 2^{-3}$','$h = 2^{-4}$','$h = 2^{-5}$',...
     '$\mathcal O(\tau^3)$' },'location','southeast','FontSize',9);
saveas(gcf,'../plottingScripts/Plots/TimeConvergenceAcoustic_II','epsc')
%% SPACE CONVERGENCE PLOT
%% Comment in from here for space convergence plot


%subplot(1,2,2)
figure(figurenumber+1)

 %   style_list={'-d','-o','-^','-h','-*','-+','-x','p','.','r','b','y'};


start_time=1;
end_time=7;
%% Creating the plots
start_space=1;
end_space=6;

%vect=step_size(start_time:end_time);
vect = mesh_size(start_space:end_space);
err=ERRS(start_space:end_space,start_time:end_time)';
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

 loglog(mesh_size,1/20*mesh_size.^1.5,'--k')
 %% Title
% strL2=strcat(string1,' s= ',num2str(s),' L2 time conv.');
%strL2=strcat(string1,' L^2-norm time conv.');
%title(strL2)
title("Space convergence II",'Interpreter','Latex')
    hold off
 xlim([10^(-1.6) 10^(0.1)])  
 ylim([10^(-4) 10^(-0.5)])
    
%% Creating the legend
%legend({strcat('dof = ',num2str(dof_s(4))),strcat('dof = ',num2str(dof_s(5))),...
 %  strcat('dof = ',num2str(dof_s(6))),strcat('dof = ',num2str(dof_s(7))),strcat('dof = ',num2str(dof_s(8))),'$\mathcal{O}(\tau^2)$'},'location','southeast' ,'FontSize',14 );

 
 l=legend({strcat('$N$ = ',num2str(N_s(1))),strcat('$N$ = ',num2str(N_s(2))),strcat('$N$ = ',num2str(N_s(3))),strcat('$N$ = ',num2str(N_s(4))),...
    strcat('$N$ = ',num2str(N_s(5))),strcat('$N$ = ',num2str(N_s(6))),strcat('$N$ = ',num2str(N_s(7))),'$\mathcal O(h^{3/2})$'},'Interpreter','Latex','location','southeast','Fontsize',9);
 
 % legend(strcat('h = ',num2str(dof_s(2))),strcat('h = ',num2str(dof_s(3))),...
%    strcat('h = ',num2str(dof_s(4))),strcat('h = ',num2str(dof_s(5))),...
%    strcat('h = ',num2str(dof_s(6))),strcat('h = ',num2str(dof_s(7))),'O(\tau^2)' );
%% Labelling the axis
xlabel('mesh width $h$','Interpreter','Latex')
ylabel('Maximal Error in $\textit{\textbf{P}}=(2,0,0)$','Interpreter','Latex')
%ylabel('H^1-norm error')
hold off




% % 
% % 
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %% Creating the space convergence plot
% % 
% % 
% %  %   style_list={'-d','-o','-^','-h','-*','-+','-x','p','.','r','b','y'};
% %  hold off
% % figure(figurenumber+1)
% % %subtitle('Time convergence of X')
% % 
% % %loglog(mesh_sizes,L2_err(:,1),'-d')
% % 
% % %loglog(mesh_sizes,H1_err(:,2),'-o')
% % loglog(mesh_sizes,H1_err(:,3),'-^')
% %  ylim([10^(-2) 10^(1)])
% % hold on
% % 
% % loglog(mesh_sizes,H1_err(:,4),'-h')
% % loglog(mesh_sizes,H1_err(:,5),'-*')
% % loglog(mesh_sizes,H1_err(:,6),'-+')
% % loglog(mesh_sizes,H1_err(:,7),'-+')
% % 
% % loglog(mesh_sizes,mesh_sizes.^1*H1_err(1,1),'--k')
% % loglog(mesh_sizes,mesh_sizes.^2*H1_err(1,1),'.--')
% % hold off
% % %% Labeling the axis
% % xlabel('mesh size h')
% % ylabel('H^1-norm error')
% % 
% % step_size=5*step_size.^(-1);
% % % %% Legend
% % %% Legend
% % % legend(strcat('\tau = ',num2str(step_size(1))),strcat('\tau = ',num2str(step_size(2))),...
% % %    strcat('\tau = ',num2str(step_size(3))),strcat('\tau = ',num2str(step_size(4))),...
% % %    strcat('\tau = ',num2str(step_size(5))),strcat('\tau = ',num2str(step_size(6))),'O(h^1)','O(h^2)');
% % 
% % 
% %     
% %     %% Initializing titles
% % 
% % strL2=strcat('quadr. ESFEM',' L^2-norm space conv.');
% % strH1=strcat('quadr. ESFEM',' H^1-seminorm space conv.');
% % %% Plotting the legend, title, axis labels for the first subplot
% % 
% % 
% % 
% % 
% % %% Plotting the legend, title, axis labels for the second subplot
% % 
% % 
% % title(strH1)
% % 
% % hold off
saveas(gcf,'../plottingScripts/Plots/SpaceConvergenceAcoustic_II','epsc')  
end

