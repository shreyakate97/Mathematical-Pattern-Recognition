% error rate function

function error = error_rate(label_test, output)
e1 = 0;
for i = 1: length(label_test)
    if label_test(i) ~= output(i)
        e1 = e1 + 1;
    end
end
error = e1/length(label_test); 
end
