%hw_1_559
clear;
% synthetic1.mat 
load('synthetic1.mat');
%error rate of training data
[output, w] = perceptron_classifier(feature_train, feature_train,label_train);
error = error_rate(label_train, output);
disp('Classification error rate on training data of synthetic1.mat'); disp(error)
% classification and error rate of testing data
[output, w] = perceptron_classifier(feature_train, feature_test,label_train);
error = error_rate(label_test, output);
disp('Classification error rate on testing data of synthetic1.mat'); disp(error)
disp('Weight vector for synthetic1.mat is: ');
disp(w);
plotDecBoundaries(feature_train, label_train, w);

clear;
% synthetic2.mat 
load('synthetic2.mat');
%error rate of training data
[output, w] = perceptron_classifier(feature_train, feature_train,label_train);
error = error_rate(label_train, output);
disp('Classification error rate on training data of synthetic2.mat'); disp(error)
% classification and error rate of testing data
[output, w] = perceptron_classifier(feature_train, feature_test,label_train);
error = error_rate(label_test, output);
disp('Classification error rate on testing data of synthetic2.mat'); disp(error)
disp('Weight vector for synthetic2.mat is: ');
disp(w);
plotDecBoundaries(feature_train, label_train, w);
clear;
% synthetic3.mat 
load('synthetic3.mat');
%error rate of training data
[output, w] = perceptron_classifier(feature_train, feature_train,label_train);
error = error_rate(label_train, output);
disp('Classification error rate on training data of synthetic3.mat'); disp(error)
% classification and error rate of testing data
[output, w] = perceptron_classifier(feature_train, feature_test,label_train);
error = error_rate(label_test, output);
disp('Classification error rate on testing data of synthetic3.mat'); disp(error)
disp('Weight vector for synthetic3.mat is: ');
disp(w);
plotDecBoundaries(feature_train, label_train, w);

