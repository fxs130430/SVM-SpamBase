function K = gaussian_kernel( X,sigma )
[n,d] = size(X);
 K = ones(n,n);
 for i=1:n
     for j=1:n
         K(i,j) = exp( -norm(X(i,:)-X(j,:))/(2*sigma*sigma));
         %K(j,i) = K(i,j);
     end
 end

end

