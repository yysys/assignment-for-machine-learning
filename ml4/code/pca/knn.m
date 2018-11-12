function y = knn(X, X_train, y_train, K)
%KNN k-Nearest Neighbors Algorithm.
%
%   INPUT:  X:         testing sample features, P-by-N_test matrix.
%           X_train:   training sample features, P-by-N matrix.
%           y_train:   training sample labels, 1-by-N row vector.
%           K:         the k in k-Nearest Neighbors
%
%   OUTPUT: y    : predicted labels, 1-by-N_test row vector.
%

% YOUR CODE HERE

[P, N1] = size(X);
N2 = size(X_train, 1);
y = zeros(1, N1);

[D, I] = pdist2(X_train', X', 'euclidean', 'Smallest', K);

for i = 1 : N1
    yy = y_train(I(:, i));
    
    t = yy(:);
    t = sort(t);
    d = diff([t;max(t)+1]);
    count = diff(find([1;d]));
    tmp =[t(find(d)) count];
    
    mx = 0;
    for j = 1 : size(tmp, 1)
        if mx < tmp(j, 2)
            mx = tmp(j, 2);
            y(i) = tmp(j, 1);
        end
    end
end


end

