
clear all
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');



load('../data/DonutFieldDataDOF896N200')
alphas = [0.25,0.5,0.75];

%load('../data/AngleTransformedFieldsN256')
N=256;
X_mesh(:,:)=plot_grid(1,:,:);
Y_mesh(:,:)=plot_grid(2,:,:);
n_grid=sqrt(length(u_ges(:,1))) ; 
[scatterer ] = create_cubes(X_mesh,Y_mesh);
u_sq=zeros(n_grid,n_grid);
n_grid=sqrt(length(u_ges(:,1)));
[u_ges ] = draw_scatterer( u_ges, scatterer);
%frames=[64,96,128,160,192,192+32];
frames=[64,96,128,160,192,192+32];
figure('Position',[200 -2000 400 800])

for n=1:length(frames)
   j=frames(n);
%   title
   subplot(3,2,n)
   
   limit_colors=[0,1];
   limit_height=[0,2];
   limit_x=[-1.5,1.5];
   limit_y=[-0.5,1.5];
   colormap(jet(256))
   width=0.213405797101449;
   u_long=u_ges(:,j);
   for i=1:n_grid        
       u_sq(:,i)=u_long((i-1)*n_grid+1:i*n_grid);   
   end
   startp=1;
   endp=n_grid;
   surf(X_mesh(startp:endp,:),Y_mesh(startp:endp,:),u_sq(startp:endp,:)','edgecolor','none')
   caxis(limit_colors)
   % view(180,0)
    %view(0,90)
   view(2)
   xlim(limit_x)
   ylim(limit_y)
   zlim(limit_height)
   title(strcat('t= ',num2str(3/256*frames(n))),'interpreter','latex')
end
%close(video_object);
saveas(gcf,'Plots/MaxwellFrames','epsc')  