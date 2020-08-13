% nearest-means classifier, using euclidean distance (L-2 norm)
function [output,m] = nearest_means_classifier(feature_train, feature_test, label_train)

%number of training and testing data points
train_d = length(feature_train);
test_d = length(feature_test);
output = zeros(test_d,1);
no_classes = length(unique(label_train));
add = zeros(no_classes,2); c = zeros(no_classes,1); m = zeros(no_classes,2); d = zeros(no_classes,2);;
for i = 1:no_classes
    for j = 1:train_d
        if i == label_train(j)
            add(i,1) = add(i,1) + feature_train(j,1);
            add(i,2) = add(i,2) + feature_train(j,2);
            c(i) = c(i) + 1;
        end
    end
    m(i,1) = add(i,1)/c(i);
    m(i,2) = add(i,2)/c(i);
end
disp('sample mean:'); disp(m);

for i = 1: test_d % 1:89
    for j = 1: no_classes % 1:3
    d(j,1)= sqrt((feature_test(i,1)-m(j,1))^2 + (feature_test(i,2)-m(j,2))^2);
    d(j,2) = j;
    end
    sorted = sortrows(d,1);
    
    output(i) = sorted(1,2);
end