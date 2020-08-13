% nearest-means classifier, using euclidean distance (L-2 norm)
function output = knn_classifier(feature_train, feature_test, label_train)
k = 11;
%number of training and testing data points
train_d = length(feature_train);
test_d = length(feature_test);
output = zeros(test_d,1);

for j = 1:test_d
    euclidean_distance = zeros(train_d,2);
    for i = 1: train_d
        euclidean_distance(i,1) = sqrt((feature_test(j,1)-feature_train(i,1))^2 + (feature_test(j,2)-feature_train(i,2))^2);
        euclidean_distance(i,2) = label_train(i,1);
    end

    euclidean_distance_sorted = sortrows(euclidean_distance,1);
    
    class1 = 0; class2 = 0;
    for i = 1:k
        if euclidean_distance_sorted(i,2) == 1
            class1 = class1 + 1;
        else
            class2 = class2 + 1;
        end
    end
    if class1 > class2 
        output(j) = 1;
    else
        output(j) = 2;
    end
end

end

