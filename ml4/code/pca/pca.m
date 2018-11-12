function [eigvector, eigvalue] = PCA(data)
%PCA	Principal Component Analysis
%
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of PCA eigen-problem.
%

% YOUR CODE HERE

[N, P] = size(data);

data_mean = mean(data);

X = data - repmat(data_mean, N, 1);

covMatrix = cov(X);

[vector, value] = eig(covMatrix);

value = diag(value);

[eigvalue, id] = sort(value, 'descend');

eigvector = vector(:, id(:));

end