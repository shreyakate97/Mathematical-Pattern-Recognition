addpath('C:\Users\shrey\Downloads\prtools');
load('wine.mat');
% (b)
m=zeros(1,13); std_dev = zeros(1,13);
%mean and std deviation
for i = 1:13
        m(i) = sum(feature_train(:,i))/89;
        std_dev(i) = sqrt((1/89)*sum((feature_train(:,i)-m(i)).^2));
end

%standardize the data
std_data_train = zeros(89,13); std_data_test = zeros(89,13);
for j = 1:13
    for i = 1:89
    std_data_train(i,j) = (feature_train(i,j) - m(j)) / std_dev(j);
    std_data_test(i,j) = (feature_test(i,j) - m(j)) / std_dev(j);
    end
end

%rescaling
for i = 1:13
        min_train(i) = min(feature_train(:,i));
        max_train(i) = max(feature_train(:,i));
        X_train(:,i) = (feature_train(:,i)-min_train(i))/(max_train(i)-min_train(i));
        min_test(i) = min(feature_test(:,i));
        max_test(i) = max(feature_test(:,i));
        X_test(:,i) = (feature_test(:,i)-min_test(i))/(max_test(i)-min_test(i));
end

% addpath('C:\Users\shrey\Downloads\prtools');
% PERCEPTRON LEARNING ALGORITHM
% perceptron weight vectors and accuracy for 2 features
% (d)
prXtrain2 = prdataset(X_train(:,1:2),label_train);
prXtest2 = prdataset(X_test(:,1:2),label_test);
w_ini = [];
w2 = perlc(prXtrain2,1000,0.1,[],'seq');

acc_2_pertrain = 1 - testc(prXtrain2,w2);
acc_2_pertest = 1 - testc(prXtest2,w2);
disp('resulting weight vectors for 2 features of perceptron:');
disp(getWeightsFromPrmapping(w2));
disp('classification accuracy for training set using perceptron:');
disp(acc_2_pertrain);
disp('classification accuracy for test set using perceptron:');
disp(acc_2_pertest);
% perceptron weight vectors and accuracy for 13 features
prXtrain13 = prdataset(X_train,label_train);
prXtest13 = prdataset(X_test,label_test);
w_ini = [];
w13 = perlc(prXtrain13,1000,0.1,[],'seq');

acc_13_pertrain = 1 - testc(prXtrain13,w13);
acc_13_pertest = 1 - testc(prXtest13,w13);
disp('resulting weight vectors for 13 features of perceptron:');
disp(getWeightsFromPrmapping(w13));
disp('classification accuracy for training set using perceptron:');
disp(acc_13_pertrain);
disp('classification accuracy for test set using perceptron:');
disp(acc_13_pertest);

%best performance on training set for 2 features
% (e)
best_acc_pertrain2=0;
for i = 1:100
    w_ini = [];
    w2 = perlc(prXtrain2,1000,0.1,[],'seq');
    if (1 - testc(prXtrain2,w2))> best_acc_pertrain2
         best_acc_pertrain2 = 1 - testc(prXtrain2,w2);
         best_acc_pertest2 = 1 - testc(prXtest2,w2);
         best_weight_per2 = w2;
    end
end

disp('best final weight vector for 2 feature perceptron');
disp(getWeightsFromPrmapping(best_weight_per2));
disp('best training accuracy');
disp(best_acc_pertrain2);
disp('best test accuracy');
disp(best_acc_pertest2);

%best performance on training set for 13 features
best_acc_pertrain13=0;
for i = 1:100
    w_ini = [];
    w13 = perlc(prXtrain13,1000,0.1,[],'seq');
    if (1 - testc(prXtrain13,w13))> best_acc_pertrain13
         best_acc_pertrain13 = 1 - testc(prXtrain13,w13);
         best_acc_pertest13 = 1 - testc(prXtest13,w13);
         best_weight_per13 = w13;
    end
end

disp('best final weight vector for 13 feature perceptron');
disp(getWeightsFromPrmapping(best_weight_per13));
disp('best training accuracy');
disp(best_acc_pertrain13);
disp('best test accuracy');
disp(best_acc_pertest13);

% MSE Classifier
% (g) use unnormalized data
A2 = prdataset(feature_train(:,1:2),label_train);
A2_test = prdataset(feature_test(:,1:2),label_test);
w2mse = fisherc(A2);
acc_2_msetest = 1 - testc(A2_test,w2mse);

disp('classification test set accuracy for 2 features using MSE (unnormalized data)');
disp(acc_2_msetest);

A13 = prdataset(feature_train,label_train);
A13_test = prdataset(feature_test,label_test);
w13mse = fisherc(A13);
acc_13_msetest = 1 - testc(A13_test,w13mse);

disp('classification test set accuracy for 13 features using MSE(unnormalized data)');
disp(acc_13_msetest);

% (g) use standardized data

w2mse = fisherc(prXtrain2);

acc_2_msetest_std = 1 - testc(prXtest2,w2mse);

disp('classification test set accuracy for 2 features using MSE(standardized data)');
disp(acc_2_msetest_std);

% (h)
w13mse = fisherc(prXtrain13);
acc_13_msetest_std = 1 - testc(prXtest13,w13mse);

disp('classification test set accuracy for 13 features using MSE (standardized data)');
disp(acc_13_msetest_std);



