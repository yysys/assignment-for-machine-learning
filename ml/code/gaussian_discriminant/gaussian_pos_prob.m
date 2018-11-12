function p = gaussian_pos_prob(X, Mu, Sigma, Phi)
%GAUSSIAN_POS_PROB Posterior probability of GDA.
%   p = GAUSSIAN_POS_PROB(X, Mu, Sigma) compute the posterior probability
%   of given N data points X using Gaussian Discriminant Analysis where the
%   K gaussian distributions are specified by Mu, Sigma and Phi.
%
%   Inputs:
%       'X'     - M-by-N matrix, N data points of dimension M.
%       'Mu'    - M-by-K matrix, mean of K Gaussian distributions.
%       'Sigma' - M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of
%                   K Gaussian distributions.
%       'Phi'   - 1-by-K matrix, prior of K Gaussian distributions.
%
%   Outputs:
%       'p'     - N-by-K matrix, posterior probability of N data points
%                   with in K Gaussian distributions.

N = size(X, 2);
K = length(Phi);
P = zeros(N, K);

% Your code HERE


p = zeros(N, K);

for i = 1 : N
    for k = 1 : K
        P(i:i, k:k) = 1.0 / (2 * pi * sqrt(det(Sigma(:, :, k)))) * exp(-1.0 / 2 * (X(:,i) - Mu(:, k))' / Sigma(:, :, k) * (X(:,i) - Mu(:, k)));
    end
end

for i = 1 : N
    for k = 1 : K
        p(i:i, k:k) = P(i:i, k:k) * Phi(k);
    end
    p(i,:) = p(i,:) / sum(p(i,:));
end


    

