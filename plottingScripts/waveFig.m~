
clear all
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

%load('../data/DonutFieldDataDOF896N200')

load('../data/AngleTransformedFieldsN128')
N=64;
X_mesh(:,:)=plot_grid(1,:,:);
Y_mesh(:,:)=plot_grid(2,:,:);
n_grid=sqrt(length(u_ges(:,1))) ; 


[scatterer ] = create_cubes(X_mesh,Y_mesh);
% figure(11)
 %spy(scatterer)
u_sq=zeros(n_grid,n_grid);
n_grid=sqrt(length(u_ges(:,1)));
%[scatterer] = create_scatterer(X_mesh,Y_mesh);
% 
%%%%%%%%%%%%%%%%%%%%
 [u_ges ] = draw_scatterer( u_ges, scatterer);
 
% norms_u = zeros(1,200);
% for j=1:N+1
%     norms_u(j) = norm(abs(u_ges[:,j]));
% end

%%%%%%%%%%%%%%%%%%%5
% load('data/Absorbing')

% [u_ges2 ] = draw_scatterer( u_ges2, scatterer2);
% 
% load('data/GIBC')
% 
% 
% n_grid=sqrt(length(u_ges(:,1)));
% u_sq=zeros(n_grid,n_grid);
% 
% X_mesh=zeros(n_grid,n_grid);
% 
% Y_mesh=zeros(n_grid,n_grid);
% 
% 
% X_mesh(:,:)=plot_grid(1,:,:);
% Y_mesh(:,:)=plot_grid(2,:,:);
%   
% % X_mesh=double_resolution(X_mesh);
% % Y_mesh=double_resolution(Y_mesh);
% %u_ges=double_resolution(u_ges);
% 
% [scatterer ] = create_scatterer(X_mesh,Y_mesh);
% 
% [u_ges ] = draw_scatterer( u_ges, scatterer);
% 
% %u_ges=normalize_u(u_ges);
% %u_ges=draw_magnet(plot_grid,u_ges);


%frames=[40,55,70,85,100,115];
%frames=[20,30,40,50,60,70];
frames = [1:N+1];
%frames=[80,100,120,140,160,180];
%  figure('Position',[200 -2000 400 800])
%   video_object = VideoWriter('testvideoN128_angle');
%   video_object.Quality = 95;
%   video_object.FrameRate = 5;
 % open(video_object);
  frames = [16,24,32,40,48,56];
  figure('Position',[200 -2000 400 800]);
  
for n=1:length(frames)
   % figure(n);
   j=frames(n);
   subplot(3,2,n)
   limit_colors=[0,1];
   limit_height=[0,2];
   limit_x=[-1.5,1.5];
   limit_y=[-0.5,1.5];
%    limit_x=[-1.5,1.5];
% %    limit_y=[-1.5,1.5];
    colormap(jet(256))
   width=0.213405797101449;
%    am_colors=50;
%    mymap=zeros(am_colors,3);
%    for colIndex=0:am_colors-1
%        mymap(colIndex+1,:)=ones(1,3)-colIndex/(am_colors-1)*ones(1,3);
%    end
%    colormap(mymap);

   u_long=u_ges(:,j);
   for i=1:n_grid        
       u_sq(:,i)=u_long((i-1)*n_grid+1:i*n_grid);   
   end
   
 
   startp=1;
   endp=n_grid;
       
   surf(X_mesh(startp:endp,:),Y_mesh(startp:endp,:),u_sq(startp:endp,:)','edgecolor','none')
   %ylabel(strcat("t= ", num2str(4/400*frames(n))))
%   colorbar();
   caxis(limit_colors)
   % view(180,0)
    %view(0,90)
   view(2)
   xlim(limit_x)
   ylim(limit_y)
   zlim(limit_height)
    frame = getframe(gcf);
%    writeVideo(video_object,frame);
   
   %endp2=n_grid2;
    title(strcat('t= ',num2str(3/128*frames(n))),'interpreter','latex')

    hsp1 = get(gca, 'Position') ;      
   %% Position Subplot 1    
    set(gca, 'Position', [hsp1(1)-0.05 hsp1(2) width+0.03 hsp1(4)]) 
    set(gca, 'Color','k')
  % drawnow;
% %     subplot(1,3,2)
% %      
% %        
% %     surf(X_mesh2(startp2:endp2,:),Y_mesh2(startp2:endp2,:),u_sq2(startp2:endp2,:)','edgecolor','none')
% %  
% %     %colorbar
% %     caxis(limit_colors)
% %    % view(180,0)
% %     view(2)
% %     xlim(limit_x)
% %     ylim(limit_y)
% %     zlim(limit_height)
% %     
% %     startp3=1;
% %     endp3=n_grid3;
% %     
% %     hsp2 = get(gca, 'Position') ;
% %        %% Position Subplot 2 
% %     set(gca, 'Position', [hsp2(1)-0.04 hsp1(2) width+0.03 hsp1(4)]) 
% %          title('(B1) Highly absorbing b.c.')
% % 
% %      subplot(1,3,3)
% %      hsp3=get(gca,'Position');
% %     surf(X_mesh3(startp3:endp3,:),Y_mesh3(startp3:endp3,:),u_sq3(startp3:endp3,:)','edgecolor','none')
% %     %colorbar
% %     
% %     
% %     caxis(limit_colors)
% %     view(0,180)
% %     view(2)
% %     xlim(limit_x)
% %     ylim(limit_y)
% %     zlim(limit_height)
% % 
% %    title('(C) Acoustic b.c.')
% %      hsp2(3)
% %       %% Position Subplot 3  
% %      set(gca, 'Position', [hsp3(1)-0.03 hsp3(2) width+0.02 hsp1(4)]) 
% %     cb=  colorbar;
% %    hcb=get(cb,'position');
% %    set(cb,'position',[hcb(1)+0.05 hcb(2)+0.025 hcb(3) hcb(4)-0.05] )
% %      
  %   drawnow
     %saveas(gcf,strcat('Framenumber',num2str(frames(n))),'epsc')

end

saveas(gcf,'Plots/NonlinearMaxwellFrames','epsc')  