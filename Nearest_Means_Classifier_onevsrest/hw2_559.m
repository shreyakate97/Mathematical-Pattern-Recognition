
% wine.mat features 1 and 2
load('wine.mat');

[output,m,m1vsrest] = nearest_means_classifier_1vsrest(feature_train, feature_test, label_train);
m1 = [ m(1,:); m1vsrest(1,:)];
plotDecBoundaries(feature_train, label_train, m1);

m2 = [ m(2,:); m1vsrest(2,:)];
plotDecBoundaries(feature_train, label_train, m2 );

m3 = [ m(3,:); m1vsrest(3,:)];
plotDecBoundaries(feature_train, label_train,m3 );

plotDecBoundaries_new(feature_train, label_train,m,m1vsrest);

%error rate of training data
[output,m,m1vsrest] = nearest_means_classifier_1vsrest(feature_train, feature_train, label_train);
accuracy1 = accuracy(label_train, output);
disp('Classification accuracy on training data of wine.mat'); disp(accuracy1)
% classification and error rate of testing data
[output,m,m1vsrest] = nearest_means_classifier_1vsrest(feature_train, feature_test, label_train);
accuracy1 = accuracy(label_test, output);
disp('Classification accuracy on testing data of wine.mat'); disp(accuracy1)