function W = knn_graph(X, k, threshold)
%KNN_GRAPH Construct W using KNN graph
%   Input: X - data point features, n-by-p maxtirx.
%          k - number of nn.
%          threshold - distance threshold.
%
%   Output:W - adjacency matrix, n-by-n matrix.

% YOUR CODE HERE

[N, P] = size(X);
W = zeros(N, N);

for i = 1 : N
    [D, I] = pdist2(X, X(i, :), 'euclidean', 'Smallest', k);
   
    for j = 1 : k
        if D(j) > threshold
            W(i, I(j)) = 0;
            W(I(j), i) = 0;
        else 
            W(i, I(j)) = 1;
            W(I(j), i) = 1;
        end
    end
end

for i = 1 : N
    W(i, i) = 0;
end

end
