function [W,b] = svm_primal(X,Y, C)
[row,col] = size(X);
H = diag([ones(1,col),zeros(1,row+1)]);
f= [zeros(col + 1,1); C*ones(row,1)];
A = -1*[X.*(Y*ones(1,col)) Y eye(row)];
b = -1*ones(row,1);
lb = [-inf*ones(col + 1,1); zeros(row,1)];
opts = optimset('Algorithm','interior-point-convex','Display','off');
WB = quadprog(H,f,A,b,[],[],lb,[],[],opts);
W = WB(1:col,1);
b =WB(col+1,1);
end


