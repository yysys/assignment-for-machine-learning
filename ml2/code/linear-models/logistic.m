function w = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE

[P, N] = size(X);
w = zeros(P+1, 1);
neww = zeros(P+1, 1);
x = [ones(1, N); X];

study_rate = 0.001;
eps = 0.001;
step = 1;

while(step > eps)
    delta = -study_rate * x * (y' .* exp((w' * x)' .* -y') ./ (ones(N, 1) + exp((w' * x)' .* -y')));
    
    neww = w - delta;
    
%     for j = 1 : P+1
%         delta = 0;
%         for i = 1 : N
%             
%             delta = delta - study_rate * y(i) * x(j, i) * exp(-y(i) * w' * x(:, i)) / (1 + exp(-y(i) * w' * x(:, i)));
%         end
%         
%         neww(j) = w(j) - delta;
%     end
    
    step = norm(neww - w);
    
    w = neww;
    
end




end