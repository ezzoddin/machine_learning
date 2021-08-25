function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

h = X * theta; %multiply X and theta to give prediciton 'h'

error = h - y; %compute error (difference between h and y)

error_sqr = error.^2; %compute element-wise squared error

error_sqr_sum = sum(error_sqr);

J = (1/(2*m)) * error_sqr_sum; %cacluate cost


% =========================================================================

end