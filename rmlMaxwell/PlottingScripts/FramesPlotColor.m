
clear all
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

load('../data/delta01_dof896')
X_mesh(:,:)=plot_grid(1,:,:);
Y_mesh(:,:)=plot_grid(2,:,:);
n_grid=sqrt(length(u_ges(:,1))) ; 

[scatterer ] = create_torus(X_mesh,Y_mesh);
u_sq=zeros(n_grid,n_grid);
n_grid=sqrt(length(u_ges(:,1)));
 [u_ges ] = draw_scatterer( u_ges, scatterer);
frames=[10,15,20,25,30,35];
%%%%frames=[20,30,40,50,60,70];
%frames=[80,100,120,140,160,180];
 figure('Position',[200 -2000 600 800])

for n=1:length(frames)

   subplot(3,2,n)
    j=frames(n);
    
   limit_colors=[0,1.2];
   limit_height=[0,2];
   limit_x=[-1.5,1.5];
   limit_y=[-1.5,1.5];
%    [X,Y] = hatch_coordinates(limit_x , limit_y , 1 , 0.05 , false ) ;
%    plot(X,Y,'Color',[.7 .7 .7],'linewidth',0.1,'LineStyle',':');
   hold on
   width=0.213405797101449;
  
   am_colors=50;
   mymap=zeros(am_colors,3);
   for colIndex=0:am_colors-1
       mymap(colIndex+1,:)=ones(1,3)-colIndex/(am_colors-1)*ones(1,3);
   end
   colormap(jet(256))
   %colormap(mymap);
  u_long=u_ges(:,j);

    for i=1:n_grid
        
        u_sq(:,i)=u_long((i-1)*n_grid+1:i*n_grid);
        
    end
 
    startp=1;
    endp=n_grid;
       
    surf(X_mesh(startp:endp,:),Y_mesh(startp:endp,:),u_sq(startp:endp,:)','edgecolor','none')
    %ylabel(strcat("t= ", num2str(4/400*frames(n))))
  
    caxis(limit_colors)
   % view(180,0)
    %view(0,90)
     view(2)
    xlim(limit_x)
    ylim(limit_y)
    zlim(limit_height)
    
    startp2=1;
    %endp2=n_grid2;
      title(strcat('t= ',num2str(4/100*frames(n))),'interpreter','latex')

%    hsp1 = get(gca, 'Position') ;      
   %% Position Subplot 1    
%   set(gca, 'Position', [hsp1(1)-0.05 hsp1(2) width+0.03 hsp1(4)])

   %set(gca, 'Color','k')
 %  set(gcf,'InvertHardCopy','Off');
 hold off
end
 hp6 = get(subplot(3,2,6),'Position');
 colorbar('position',[hp6(1)+hp6(3)+0.01  hp6(2)+0.55*hp6(3) 0.03 hp6(2)+hp6(3)*1.0 ])
saveas(gcf,'Plots/MaxwellFrames_1','epsc')  
%frames=[40,55,70,85,100,115];
frames=[40,45,50,55,60,65];
%%%%frames=[20,30,40,50,60,70];
%frames=[80,100,120,140,160,180];
%  figure('Position',[200 -2000 400 800])
  figure('Position',[200 -2000 600 800])
for n=1:length(frames)
n
   subplot(3,2,n)
    j=frames(n);
    
   limit_colors=[0,1.2];
   limit_height=[0,2];
   limit_x=[-1.5,1.5];
   limit_y=[-1.5,1.5];
   
   width=0.213405797101449;
   
  
   am_colors=50;
   mymap=zeros(am_colors,3);
   for colIndex=0:am_colors-1
       mymap(colIndex+1,:)=ones(1,3)-colIndex/(am_colors-1)*ones(1,3);
   end
   colormap(jet(256))
   %colormap(mymap);
  u_long=u_ges(:,j);

    for i=1:n_grid
        
        u_sq(:,i)=u_long((i-1)*n_grid+1:i*n_grid);
        
    end
 
    startp=1;
    endp=n_grid;
       
    surf(X_mesh(startp:endp,:),Y_mesh(startp:endp,:),u_sq(startp:endp,:)','edgecolor','none')
    %ylabel(strcat("t= ", num2str(4/400*frames(n))))
  
    caxis(limit_colors)
   % view(180,0)
    %view(0,90)
     view(2)
    xlim(limit_x)
    ylim(limit_y)
    zlim(limit_height)
    
    startp2=1;
    %endp2=n_grid2;
      title(strcat('t= ',num2str(4/100*frames(n))),'interpreter','latex')

%    hsp1 = get(gca, 'Position') ;      
   %% Position Subplot 1    
 %  set(gca, 'Position', [hsp1(1)-0.05 hsp1(2) width+0.03 hsp1(4)]) 
%   set(gca, 'Color','k')
%   set(gcf,'InvertHardCopy','Off');
end
 hp6 = get(subplot(3,2,6),'Position');
 colorbar('position',[hp6(1)+hp6(3)+0.01  hp6(2)+0.55*hp6(3) 0.03 hp6(2)+hp6(3)*1.0 ])
saveas(gcf,'Plots/MaxwellFrames_2','epsc')  