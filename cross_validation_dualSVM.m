function [c_best,sigma_best] =cross_validation_dualSVM(file_URL_train,file_URL_validate,file_URL_test)

[X_train Y_train] = init_data(file_URL_train);
[X_validate Y_validate] = init_data(file_URL_validate);
[X_test Y_test] = init_data(file_URL_test);


[m n]=size(X_train);

c = [1 10 100 1000 10000];
sigma = [0.001 0.01 0.1 1 10 100];
lambda = zeros(m,1,5,6);
b = zeros(1,5,6);
accuracy = zeros(5,6);

for i=1:5
    for j=1:6
        [lambda(:,:,i,j) b(:,i,j)] = svm_dual(X_train,Y_train,c(i),sigma(j));
        accuracy(i,j) = svm_dual_accuracy(X_train,Y_train,X_train,Y_train,sigma(j),lambda(:,:,i,j),b(:,i,j));
        disp(sprintf('c=%f,sig=%d, accuracy=%f',c(1,i),sigma(1,j),accuracy(i,j)));
    end
end

[max_acc max_indx]=max(accuracy(:));

disp(max_indx);
disp(max_acc);
%disp(sprintf('accuracy on test = %f',accuracy_test));





end