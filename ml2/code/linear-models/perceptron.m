function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE

[P, N] = size(X);
tmp = ones(1, N);
x = [tmp; X];
w = zeros(P+1, 1);

count = 0;
iter = 0;

while count ~= N

    iter = iter+1;
    count = 0;
    for i = 1 : N
        if w' * x(:, i) * y(i) > 0
            count = count + 1;
        else
            w = w + x(:, i) * y(i);
        end
    end
end

end
