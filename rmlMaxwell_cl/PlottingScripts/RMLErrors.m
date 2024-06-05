function [] = RMLErrors( ERRS,mesh_size,step_size,string1,s,figurenumber)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
%% Creating time convergence plots
h1=figure(figurenumber);
%subplot(1,2,1)
set(h1,'Position',[10 10 561 420])
h_s=mesh_size;
N_s=6./step_size;

 %   style_list={'-d','-o','-^','-h','-*','-+','-x','p','.','r','b','y'};


start_time=1;
end_time=8;
%% Creating the plots
start_space=2;
end_space=8;

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xlim([10^(-2.25) 10^(0.1)])  
ylim([10^(-4) 10^(-0.0)])

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


  loglog((step_size),2*step_size.^3,'--k')
 %% Title
% strL2=strcat(string1,' s= ',num2str(s),' L2 time conv.');
%strL2=strcat(string1,' L^2-norm time conv.');
%title(strL2)
title("Time convergence",'Interpreter','Latex')
    hold off

    
%% Creating the legend
%legend({strcat('dof = ',num2str(dof_s(4))),strcat('dof = ',num2str(dof_s(5))),...
 %  strcat('dof = ',num2str(dof_s(6))),strcat('dof = ',num2str(dof_s(7))),strcat('dof = ',num2str(dof_s(8))),'$\mathcal{O}(\tau^2)$'},'location','southeast' ,'FontSize',14 );

 % legend(strcat('h = ',num2str(dof_s(2))),strcat('h = ',num2str(dof_s(3))),...
%    strcat('h = ',num2str(dof_s(4))),strcat('h = ',num2str(dof_s(5))),...
%    strcat('h = ',num2str(dof_s(6))),strcat('h = ',num2str(dof_s(7))),'O(\tau^2)' );
%% Labelling the axis
xlabel('step size $\tau$','Interpreter','Latex')
ylabel('Maximal Error in the origin $\textit{\textbf{P}}=(0,0,0)$','Interpreter','Latex')

% strH1=strcat(string1,' H^1-norm of error  ');
%title(strH1)

%ylabel('H^1-norm error')
hold off
  legend(strcat('$h=2^{-1/2}$'),strcat('$h=2^{-1}$'),...
    strcat('$h=2^{-3/2}$'),strcat('$h=2^{-2}$'),...
    strcat('$h=2^{-5/2}$'),strcat('$h=2^{-3}$'),strcat('$h=2^{-7/2}$'),...
    '$\mathcal O(\tau^3)$' ,'location','southeast');  
% legend(strcat('$h$  = ',num2str(h_s(2))),strcat('$h$  = ',num2str(h_s(3))),...
%     strcat('$h$  = ',num2str(h_s(4))),strcat('$h$  = ',num2str(h_s(5))),...
%     strcat('$h$  = ',num2str(h_s(6))),strcat('$h$  = ',num2str(h_s(7))),num2str(h_s(8)),...
%     '$\mathcal O(\tau^3)$' ,'location','southeast');

%% SPACE CONVERGENCE PLOT 1
%% Comment in from here for space convergence plot
saveas(gcf,'../../plottingScripts/Plots/TimeConvergenceLinearMaxwell','epsc')  
h2=figure(figurenumber+1)
set(h1,'Position',[10 10 561 420])



 %   style_list={'-d','-o','-^','-h','-*','-+','-x','p','.','r','b','y'};


start_time=1;
end_time=8;
%% Creating the plots
start_space=1;
end_space=8;

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

  loglog(mesh_size,0.1*mesh_size.^1.5,'--k')
 %% Title
% strL2=strcat(string1,' s= ',num2str(s),' L2 time conv.');
%strL2=strcat(string1,' L^2-norm time conv.');
%title(strL2)
title("Space convergence", 'Interpreter','Latex')
    hold off
xlim([10^(-1.1) 10^(0.05)])  
ylim([10^(-4) 10^(0)])
    
%% Creating the legend
%legend({strcat('dof = ',num2str(dof_s(4))),strcat('dof = ',num2str(dof_s(5))),...
 %  strcat('dof = ',num2str(dof_s(6))),strcat('dof = ',num2str(dof_s(7))),strcat('dof = ',num2str(dof_s(8))),'$\mathcal{O}(\tau^2)$'},'location','southeast' ,'FontSize',14 );

 
%  l=legend({strcat('$N$ = ',num2str(N_s(3))),strcat('$N$ = ',num2str(N_s(4))),...
%      strcat('$N$ = ',num2str(N_s(5))),strcat('$N$ = ',num2str(N_s(6))),...
%     strcat('$N$ = ',num2str(N_s(7))),'$\mathcal O(h)$','$\mathcal O(h^{3/2})$'},'Interpreter','Latex','location','southeast');
  l=legend({strcat('$N = 2^3$'),strcat('$N = 2^4$'),...
     strcat('$N = 2^5$'),strcat('$N = 2^6$'),...
    strcat('$N = 2^7$'),strcat('$N = 2^8$'),strcat('$N = 2^9$'),strcat('$N = 2^{10}$'),'$\mathcal O(h^{3/2})$'},'Interpreter','Latex','location','southeast');
 % legend(strcat('h = ',num2str(dof_s(2))),strcat('h = ',num2str(dof_s(3))),...
%    strcat('h = ',num2str(dof_s(4))),strcat('h = ',num2str(dof_s(5))),...
%    strcat('h = ',num2str(dof_s(6))),strcat('h = ',num2str(dof_s(7))),'O(\tau^2)' );
%% Labelling the axis
xlabel('mesh width $h$','Interpreter','Latex')
ylabel('Maximal Error in the origin $\textit{\textbf{P}}=(0,0,0)$','Interpreter','Latex')
%ylabel('H^1-norm error')
hold off


saveas(gcf,'../../plottingScripts/Plots/SpaceConvergenceLinearMaxwell','epsc')  
end

