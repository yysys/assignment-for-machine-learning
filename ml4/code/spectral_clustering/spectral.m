function idx = spectral(W, k)
%SPECTRUAL spectral clustering
%   Input:
%     W: Adjacency matrix, N-by-N matrix
%     k: number of clusters
%   Output:
%     idx: data point cluster labels, n-by-1 vector.

% YOUR CODE HERE

D = sum(W);
D = diag(D);

L = D - W;

L = D^(-1.0/2.0) * L * D^(-1.0/2.0);

% [eigVector, eigValue] = eig(L);
% 
% eigValue = diag(eigValue)';

opt = struct('issym', true, 'isreal', true);
[eigvector, eigvalue] = eigs(L, D, k, 'SM', opt);

% [~, id] = sort(eigValue);
% 
% X = [];
% 
% for i = 1 : k
%     X = [X, eigVector(:, id(i))]
% end

idx = litekmeans(eigvector, k);

end
