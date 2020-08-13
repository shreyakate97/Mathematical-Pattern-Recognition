%hw_1_559

clear;
% synthetic1.mat 
load('synthetic1.mat');
%error rate of training data
[output,m] = nearest_means_classifier(feature_train, feature_train, label_train);
error = error_rate(label_train, output);
disp('Classification error rate on training data of synthetic1.mat'); disp(error)
% classification and error rate of testing data
[output,m] = nearest_means_classifier(feature_train, feature_test, label_train);
error = error_rate(label_test, output);
disp('Classification error rate on testing data of synthetic1.mat'); disp(error)
plotDecBoundaries(feature_train, label_train, m);

clear;
% synthetic2.mat 
load('synthetic2.mat');
%error rate of training data
[output,m] = nearest_means_classifier(feature_train, feature_train, label_train);
error = error_rate(label_train, output);
disp('Classification error rate on training data of synthetic2.mat'); disp(error)
% classification and error rate of testing data
[output,m] = nearest_means_classifier(feature_train, feature_test, label_train);
error = error_rate(label_test,output);
disp('Classification error rate on testing data of synthetic2.mat'); disp(error)

plotDecBoundaries(feature_train, label_train, m);

clear;
% wine.mat features 1 and 2
load('wine.mat');
%error rate of training data
[output,m] = nearest_means_classifier(feature_train, feature_train, label_train);
error = error_rate(label_train, output);
disp('Classification error rate on training data of wine.mat'); disp(error)
% classification and error rate of testing data
[output,m] = nearest_means_classifier(feature_train, feature_test, label_train);
error = error_rate(label_test, output);
disp('Classification error rate on testing data of wine.mat'); disp(error)

plotDecBoundaries(feature_train, label_train, m);

% error rates of all feature combinations
ltrain = label_train(:,1);
ltest = label_test(:,1);
minval= 1000; min_i = 0; min_j = 0;
maxval = -1; max_i = 0; max_j = 0;

for i = 1:13
    for j = i+1:13
        if i ~= j
            ftrain(:,1:2) = [feature_train(:,i) feature_train(:,j)];
            ftest(:,1:2) = [feature_test(:,i) feature_test(:,j)];
            
            [output,m] = nearest_means_classifier(ftrain, ftest, ltrain);

            error(i,j) = error_rate(ltest, output);
            if error(i,j) < minval
                minval = error(i,j);
                min_i = i;
                min_j = j;
%             elseif error(i,j) > maxval
%                 maxval = error(i,j);
%                 max_i = i;
%                 max_j = j;
            end       
        end
    end
end
disp(minval); disp(min_i); disp(min_j);
% disp(maxval); disp(max_i); disp(max_j);

% wine.mat features with minimum error
ftrain(:,1:2) = [feature_train(:,min_i) feature_train(:,min_j)];
ftest(:,1:2) = [feature_test(:,min_i) feature_test(:,min_j)];
% error rate of training data 
[output,m] = nearest_means_classifier(ftrain, ftrain, ltrain);
error(i,j) = error_rate(ltrain, output);
disp('classification error on training data of wine.mat (minimum error)'); disp(error(min_i,min_j))
% classification and error rate of testing data 
[output,m] = nearest_means_classifier(ftrain, ftest, ltrain);
error(i,j) = error_rate(ltest, output);
disp('classification error on test data of wine.mat (minimum error)'); disp(error(min_i,min_j))
plotDecBoundaries(ftrain, ltrain, m);