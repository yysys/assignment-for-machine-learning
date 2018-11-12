function [idx, ctrs, iter_ctrs] = kmeans(X, K)
%KMEANS K-Means clustering algorithm
%
%   Input: X - data point features, n-by-p maxtirx.
%          K - the number of clusters
%
%   OUTPUT: idx  - cluster label
%           ctrs - cluster centers, K-by-p matrix.
%           iter_ctrs - cluster centers of each iteration, K-by-p-by-iter
%                       3D matrix.

% YOUR CODE HERE

[N, P] = size(X);

id = randperm(N);
id = id(1 : K);
ctrs = X(id(:), :);
iter_ctrs = [];
idx = zeros(1, N);
dis = 1;
eps = 0.1;
cnt = 1;

while (dis > eps)
    
    iter_ctrs(:, :, cnt) = ctrs;
    cnt = cnt + 1;
    dist = zeros(N, K);
    
%     for i = 1 : N
%         for j = 1 : K
%             dist(i, j) = norm(X(i, :) - ctrs(j, :));
%         end
%     end
    
    dist = pdist2(X, ctrs);
    
    [D, idx] = min(dist');
    
    center = zeros(K, P);
    
    for i = 1 : K
        center(i,:) = mean(X(idx == i, :));
    end
    
    dis = norm(center - ctrs);
    
    ctrs = center;
end

end
