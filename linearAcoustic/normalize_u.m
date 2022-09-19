function [u_ges] = normalize_u(u_ges )

for j=1:length(u_ges(1,:))
    for i=1:length(u_ges(:,1))
        
        u_ges(i,j)=min(1,u_ges(i,j));
        u_ges(i,j)=max(-1,u_ges(i,j));
        
    end
end


end

