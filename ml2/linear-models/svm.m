function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE

[P, N] = size(X);
w = zeros(P+1, 1);
x = [ones(1, N); X];
b = -ones(N, 1);
H = eye(P);
H = [zeros(1,P);H];
H = [zeros(P+1,1), H];
f = zeros(P+1, 1);
A = [];

for i = 1 : N
    A = [A; -y(i) * x(:, i)'];
end

w = quadprog(H, f, A, b);

num = 0;

for i = 1 : N
    if y(i) * w' * x(:, i) <= 1.0001
        num = num+1;
    end
end

end
