% nearest-means classifier, using euclidean distance (L-2 norm)
function [output,m,m1vsrest] = nearest_means_classifier_1vsrest(feature_train, feature_test, label_train)

%number of training and testing data points
train_d = 89; %length(feature_train);
test_d = 89;%length(feature_test); 
output = zeros(test_d,1);
no_classes = 3;%length(unique(label_train));
add = zeros(no_classes,2); c = zeros(no_classes,1); m = zeros(no_classes,2); d = zeros(no_classes,2);;

for i = 1:no_classes %1:3
    for j = 1:train_d %1:89
        if i == label_train(j)
            add(i,1) = add(i,1) + feature_train(j,1);
            add(i,2) = add(i,2) + feature_train(j,2);
            c(i) = c(i) + 1;
        end
    end
    m(i,1) = add(i,1)/c(i);
    m(i,2) = add(i,2)/c(i);
end
% disp('sample mean:'); disp(m);

m1vsrest(2,1) = (m(1,1)+ m(3,1))/2 ; m1vsrest(1,2) = (m(1,2)+ m(3,2))/2 ;
m1vsrest(1,1) = (m(2,1) + m(3,1))/2; m1vsrest(2,2) = (m(2,2) + m(3,2))/2; 
m1vsrest(3,1) = (m(2,1) + m(1,1))/2; m1vsrest(3,2) = (m(2,2) + m(1,2))/2;

% disp('sample mean of 23 13 and 12:'); disp(m1vsrest);

for i = 1: test_d % 1:89
    d(1,1)= sqrt((feature_test(i,1)-m(1,1))^2 + (feature_test(i,2)-m(1,2))^2);
    d(1,2) = 1;
    d(2,1)= sqrt((feature_test(i,1)-m1vsrest(1,1))^2 + (feature_test(i,2)-m1vsrest(1,2))^2);
    d(2,2) = 0;
    sorted = sortrows(d,1);
    
    output1(i,:) = sorted(2,:);


    d(1,1)= sqrt((feature_test(i,1)-m(2,1))^2 + (feature_test(i,2)-m(2,2))^2);
    d(1,2) = 2;
    d(2,1)= sqrt((feature_test(i,1)-m1vsrest(2,1))^2 + (feature_test(i,2)-m1vsrest(2,2))^2);
    d(2,2) = 0;
    sorted = sortrows(d,1);
    
    output2(i,:) = sorted(2,:);

    d(1,1)= sqrt((feature_test(i,1)-m(3,1))^2 + (feature_test(i,2)-m(3,2))^2);
    d(1,2) = 3;
    d(2,1)= sqrt((feature_test(i,1)-m1vsrest(3,1))^2 + (feature_test(i,2)-m1vsrest(3,2))^2);
    d(2,2) = 0;
    sorted = sortrows(d,1);
    
    output3(i,:) = sorted(2,:);
    
    temp = [output1(i,:); output2(i,:); output3(i,:)];
    sorted = sortrows(temp,1);
    output(i) = sorted(1,2);
end

