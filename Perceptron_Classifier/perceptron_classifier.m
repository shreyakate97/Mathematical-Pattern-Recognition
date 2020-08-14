% perceptron classifier

function [output,w] = perceptron_classifier(b, c,label_train)

f_train = [b, ones(length(b),1)];
f_test = [c, ones(length(c),1)];
zn = zeros(length(b),1);
for i = 1:length(b)
    if label_train(i) == 1
        zn(i) = 1;  
    else
        zn(i) = -1;
    end
end
%reflected data points
f_train = zn.*f_train;
%weight vector
w = 0.1*ones(1,3);
sum = 0; correct = 0;
r = randperm(length(b));
for i = 1:1000
    for j = 1:length(b)
        if i == 1 && j == 1
            continue;
        end
        sum = f_train(r(j),1)*w(1) + f_train(r(j),2)*w(2) + f_train(r(j),3)*w(3);
          if sum <= 0
                w = [f_train(r(j),1)+w(1) f_train(r(j),2)+w(2) f_train(r(j),3)+w(3)];
                correct = 0;
          else
                correct = correct +1;
            end

        if correct == length(b)
            break;
        end
        sum = 0;
    end
end

for i = 1:length(c)
    sum = f_test(i,1)*w(1) + f_test(i,2)*w(2) + f_test(i,3)*w(3);
    if sum > 0
        output(i) = 1;
    else
        output(i) = 2;
    end
end
end
