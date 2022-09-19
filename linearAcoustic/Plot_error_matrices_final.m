function [] = Plot_error_matrices_final( L2_err,H1_semi_err,mesh_sizes,dof_s,step_size,string1,s,figurenumber)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
%% Creating time convergence plots
figure(figurenumber)

  dof_s;

 %   style_list={'-d','-o','-^','-h','-*','-+','-x','p','.','r','b','y'};
H1_err=sqrt(L2_err.^2 +H1_semi_err.^2);

start_time=1;
end_time=6;
%% Creating the plots
start_space=3;
end_space=6;

vect=step_size(start_time:end_time);
err=H1_err(start_space:end_space,start_time:end_time);
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
xlim([10^(-1.5) 10^(0)])  
ylim([10^(-2) 10^(0)])
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

 loglog((step_size),2*step_size.^s,'--k')
 %% Title
% strL2=strcat(string1,' s= ',num2str(s),' L2 time conv.');
%strL2=strcat(string1,' L^2-norm time conv.');
%title(strL2)
title("Time convergence absorbing b.c.")
    hold off

    
%% Creating the legend
legend({strcat('dof = ',num2str(dof_s(4))),strcat('dof = ',num2str(dof_s(5))),...
   strcat('dof = ',num2str(dof_s(6))),strcat('dof = ',num2str(dof_s(7))),strcat('dof = ',num2str(dof_s(8))),'$\mathcal{O}(\tau^2)$'},'location','southeast' ,'FontSize',14 );
% legend(strcat('h = ',num2str(dof_s(2))),strcat('h = ',num2str(dof_s(3))),...
%    strcat('h = ',num2str(dof_s(4))),strcat('h = ',num2str(dof_s(5))),...
%    strcat('h = ',num2str(dof_s(6))),strcat('h = ',num2str(dof_s(7))),'O(\tau^2)' );
%% Labelling the axis
xlabel('step size \tau')
ylabel('H^1-norm error')

 strH1=strcat(string1,' H^1-norm of error  ');
%title(strH1)
xlabel('step size \tau')
ylabel('H^1-norm error')
hold off

%% Comment in from here for space convergence plot


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
% % legend({strcat('N = ',num2str(step_size(3))),strcat('N = ',num2str(step_size(4))),...
% %    strcat('N = ',num2str(step_size(5))),strcat('N = ',num2str(step_size(6))),strcat('N = ',num2str(step_size(7))),'O(h)','O(h^2)'},'location','southeast');
% % 
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

end

