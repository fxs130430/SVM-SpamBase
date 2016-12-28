function [c_best] =cross_validation_primalSVM(file_URL_train,file_URL_validate,file_URL_test)

[X_train_orig Y_train] = init_data(file_URL_train);
[X_validate_orig Y_validate] = init_data(file_URL_validate);
[X_test_orig Y_test] = init_data(file_URL_test);


%--> pca on spambase dataset
X_train_orig_m = bsxfun(@minus,X_train_orig,mean(X_train_orig));
%X_train_orig_m = bsxfun(@rdivide,X_train_orig_m,std(X_train_orig));
X_validate_orig_m = bsxfun(@minus,X_validate_orig,mean(X_train_orig));
%X_validate_orig_m = bsxfun(@rdivide,X_validate_orig_m,std(X_train_orig ));
X_test_orig_m = bsxfun(@minus,X_test_orig,mean(X_train_orig));
%X_test_orig_m = bsxfun(@rdivide,X_test_orig_m,std(X_train_orig));

[Dpca Wpca] = pca(X_train_orig_m);


X_train = project(X_train_orig_m,Wpca(:,1:6));
X_validate = project(X_validate_orig_m,Wpca(:,1:6));
X_test = project(X_test_orig_m,Wpca(:,1:6));

eigs = cumsum(Dpca)/sum(Dpca);
disp(eigs);
%<-- pca on spambase dataset

[m n]=size(X_train);

c = [1 10 100 1000 10000];
w = zeros(5,n);
b = zeros(5,1);
accuracy = zeros(1,5);

for i=1:5
    [w(i,:) b(i)]=svm_primal(X_train, Y_train,c(1,i)); 
    % accuracy(1,i) = svm_primal_accuracy(X_validate ,Y_validate,w(i,:),b(i));
    accuracy(1,i) = svm_primal_accuracy([X_validate Y_validate],w(i,:),b(i));
    disp(sprintf('c=%f, accuracy=%f\n',c(1,i),accuracy(1,i)));
end
[max_acc max_indx]=max(accuracy);

c_best = c(1,max_indx);
disp(sprintf('the best c = %f',c_best));

[w b] = svm_primal(X_train, Y_train,c_best);

% accuracy_test = svm_primal_accuracy(X_test, Y_test,w',b);
accuracy_test = svm_primal_accuracy([X_test Y_test],w,b);
disp(sprintf('accuracy on test = %f',accuracy_test));

end