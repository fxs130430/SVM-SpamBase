function [accuracy] = svm_dual_accuracy(X_train,Y_train, X_validation,Y_validation, sigma, lambda, b)
[row,col] = size(X_train);
[row1,col1] = size(X_validation);

kernel=zeros(row1,row);
wTx = zeros(row1,1);
for i=1:row1
    for j=1:row
        kernel(j,i) = exp(-norm(X_train(j,:)-X_validation(i,:))/(2*sigma*sigma));       
        wTx(i) = wTx(i) + lambda(j)*Y_train(j)*kernel(j,i);
    end
end


h = -1*ones(row1,1);
for i=1:row1
    if (wTx(i)+b) > 0 
        h(i) = 1;
    end
end

accuracy = sum(h==Y_validation)*100/row1;

end