% error rate function

function error = error_rate(a,output)
e1 = 0;
for i = 1: length(a)
    if a(i) ~= output(i)'
        e1 = e1 + 1;
    end
end
error = e1/length(a); 
end
