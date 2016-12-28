function [lambda,b] = svm_dual(X,y,C,sigma)
    
[row col] = size(X);
K = gaussian_kernel(X,sigma);


YY = y*y';
H = K.*YY;
f = -ones(row,1);
Aeq = y';
beq = 0;

lb = zeros(row,1);
ub = C*ones(row,1);

opts = optimset('Algorithm','interior-point-convex','display','off');
[lambda,fval,e,ignore,mults] = quadprog(H,f',[],[],Aeq,beq,lb,ub,[],opts);

b = mults.eqlin;

end