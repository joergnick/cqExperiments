function [scatterer ] = create_scatterer(X_mesh,Y_mesh)
%load('GIBCe0T5N200')
n_grid=length(X_mesh(:,1));

  


scatterer=zeros(n_grid,n_grid);


for ind_x=1:n_grid
    for ind_y=1:n_grid
        
        x=X_mesh(ind_x,ind_y);

        y=Y_mesh(ind_x,ind_y);
        r=sqrt((x-0.5)^2+(y)^2);
        if x<0.5
            
            if 0.39<r && r<0.51
            
            scatterer(ind_x,ind_y)=1;
            end
            
        end
        
        
    end
    
end

%surf(X_mesh,Y_mesh,scatterer)


end

