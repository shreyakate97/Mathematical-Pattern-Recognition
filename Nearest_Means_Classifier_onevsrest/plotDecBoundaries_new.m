function [] = plotDecBoundaries_new(training, label_train, one_mean, rest_mean)
%Plot the decision boundaries and data points for minimum distance to
%class mean classifier

% training: traning data
% label_train: class lables correspond to training data
% sample_mean: mean vector for each class

% Total number of classes
nclass =  max(unique(label_train));

% Set the feature range for ploting
max_x = ceil(max(training(:, 1))) + 1;
min_x = floor(min(training(:, 1))) - 1;
max_y = ceil(max(training(:, 2))) + 1;
min_y = floor(min(training(:, 2))) - 1;

xrange = [min_x max_x];
yrange = [min_y max_y];

% step size for how finely you want to visualize the decision boundary.
inc = 0.005;

% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

% size of the (x, y) image, which will also be the size of the
% decision boundary image that is used as the plot background.
image_size = size(x);
xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.

% distance measure evaluations for each (x,y) pair.
dist_mat = pdist2(xy, one_mean);
[~, pred_label] = min(dist_mat, [], 2);

% reshape the idx (which contains the class label) into an image.
decisionmap = reshape(pred_label, image_size);

figure;
 
%show the image, give each coordinate a color according to its class label
% imagesc(xrange,yrange,decisionmap);
hold on;
% set(gca,'ydir','normal');
 
% % colormap for the classes:
% cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1;];
% colormap(cmap);
 
% plot the class training data.
plot(training(label_train == 1,1),training(label_train == 1,2), 'rx');
plot(training(label_train == 2,1),training(label_train == 2,2), 'go');
if nclass == 3
    plot(training(label_train == 3,1),training(label_train == 3,2), 'b*');
end

% include legend for training data
if nclass == 3
    legend('Class 1', 'Class 2', 'Class 3', ...
    'Location','northoutside','Orientation', 'horizontal');
else
    legend('Class 1', 'Class 2', ...
    'Location','northoutside','Orientation', 'horizontal');
end

% plot the class mean vector.
mean1 = plot(one_mean(1,1),one_mean(1,2), 'rd', ...
             'MarkerSize', 8, 'MarkerFaceColor', 'r');
mean2 = plot(one_mean(2,1),one_mean(2,2), 'gd', ...
             'MarkerSize', 8, 'MarkerFaceColor', 'g');
if nclass == 3
    mean3 = plot(one_mean(3,1),one_mean(3,2), 'bd',...
                'MarkerSize', 8, 'MarkerFaceColor', 'b');
end
p1x = one_mean(1,1);
p1y = one_mean(1,2);
p2x = rest_mean(1,1);
p2y = rest_mean(1,2);

grid on;

% Find midpoint
midX = mean([p1x, p2x]);
midY = mean([p1y, p2y]);
hold on;
plot(midX, midY, 'r', 'LineWidth', 1, 'MarkerSize', 1);
% Get the slope
slope = (p2y-p1y) / (p2x-p1x);
% For perpendicular line, slope = -1/slope
slope = -1/slope;
% Point slope formula (y-yp) = slope * (x-xp)
% y = slope * (x - midX) + midY
% Compute y at some x, for example at x=300
x = 11;
y = slope * (x - midX) + midY;
plot([x, midX], [y, midY], 'r-', 'LineWidth', 1);
x = 15;
y = slope * (x - midX) + midY;
plot([x, midX], [y, midY], 'r-', 'LineWidth', 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p1x = one_mean(2,1);
p1y = one_mean(2,2);
p2x = rest_mean(2,1);
p2y = rest_mean(2,2);

grid on;

% Find midpoint
midX = mean([p1x, p2x]);
midY = mean([p1y, p2y]);
hold on;

% Get the slope
slope = (p2y-p1y) / (p2x-p1x);
% For perpendicular line, slope = -1/slope
slope = -1/slope;
% Point slope formula (y-yp) = slope * (x-xp)
% y = slope * (x - midX) + midY
% Compute y at some x, for example at x=300
x = 11;
y = slope * (x - midX) + midY;
plot([x, midX], [y, midY], 'g-', 'LineWidth', 1);
x = 15;
y = slope * (x - midX) + midY;
plot([x, midX], [y, midY], 'g-', 'LineWidth', 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p1x = one_mean(3,1);
p1y = one_mean(3,2);
p2x = rest_mean(3,1);
p2y = rest_mean(3,2);

grid on;

% Find midpoint
midX = mean([p1x, p2x]);
midY = mean([p1y, p2y]);
hold on;

% Get the slope
slope = (p2y-p1y) / (p2x-p1x);
% For perpendicular line, slope = -1/slope
slope = -1/slope;
% Point slope formula (y-yp) = slope * (x-xp)
% y = slope * (x - midX) + midY
% Compute y at some x, for example at x=300
x = 11;
y = slope * (x - midX) + midY;
plot([x, midX], [y, midY], 'b-', 'LineWidth', 1);
x = 15;
y = slope * (x - midX) + midY;
plot([x, midX], [y, midY], 'b-', 'LineWidth', 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create a new axis for lengends of class mean vectors
ah=axes('position',get(gca,'position'),'visible','off');

% include legend for class mean vector
if nclass == 3
    legend(ah, [mean1, mean2, mean3], {'Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'}, ...
    'Location','northoutside','Orientation', 'horizontal');
else
    legend(ah, [mean1, mean2], {'One Mean', 'Rest Mean'},  ...
    'Location','northoutside','Orientation', 'horizontal');
end

% label the axes.
xlabel('featureX');
ylabel('featureY');

end
