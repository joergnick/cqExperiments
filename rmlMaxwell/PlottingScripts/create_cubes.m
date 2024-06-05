function [scatterer ] = create_cubes(X_mesh,Y_mesh)
%load('GIBCe0T5N200')


n_grid=length(X_mesh(:,1));
ny_grid = length(Y_mesh(:,1));

 
scatterer=zeros(n_grid,n_grid);

for ind_x=1:n_grid
    for ind_y=1:n_grid
        x=X_mesh(ind_x,ind_y);
        y=Y_mesh(ind_x,ind_y);
        if (-1.26 <= x) && (x<=1.26) && (-0.01<=y) && (y <=1.01)
            %% DIFFERENT ORIENTATION, due to simulation data !
            scatterer(ind_y,ind_x)=1;
        end
        if abs(x)<0.25
            %% Cutting out the tunnel
            scatterer(ind_y,ind_x)=0;
        end
        
    end
    
end

%surf(X_mesh,Y_mesh,scatterer)


end
