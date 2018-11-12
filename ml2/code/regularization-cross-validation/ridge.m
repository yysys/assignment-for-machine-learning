function w = ridge(X, y, lambda)
%RIDGE Ridge Regression.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE

[P, N] = size(X);
w = zeros(P+1, 1);
x = [ones(1, N); X];

w = pinv(x * x' + lambda * eye(P+1)) * (x * y');

end
