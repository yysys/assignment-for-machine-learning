function w = logistic_r(X, y, lambda)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE

[P, N] = size(X);
w = zeros(P+1, 1);
neww = zeros(P+1, 1);
x = [ones(1, N); X];

study_rate = 0.1;
eps = 0.1;
step = 1;

while(step > eps)
    
    w = neww;
    
    delta = -study_rate * x * (y' .* exp((w' * x)' .* -y') ./ (ones(N, 1) + exp((w' * x)' .* -y')));
    %delta = delta - study_rate * lambda * w;
    neww = w - delta / N;
    step = norm(delta / N);
%     
%     for j = 1 : P+1
%         delta = 0;
%         for i = 1 : N
%             delta = delta - study_rate * y(i) * x(j, i) * exp(-y(i) * w' * x(:, i)) / (1 + exp(-y(i) * w' * x(:, i)));
%         end
%         
%         delta = delta - lambda * w(j);
%         neww(j) = w(j) - delta;
%     end
    
end

end
