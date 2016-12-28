function [ X,Y ] = init_data( file_URL )

data = load(file_URL);
[m n] = size(data);
X = data(:,1:n-1);
Y = data(:,n);
Y(Y(:,1)==0)=-1;

end

