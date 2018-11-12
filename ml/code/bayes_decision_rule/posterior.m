function p = posterior(x)
%POSTERIOR Two Class Posterior Using Bayes Formula
%
%   INPUT:  x, features of different class, C-By-N vector
%           C is the number of classes, N is the number of different feature
%
%   OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
%

[C, N] = size(x);
l = likelihood(x);
total = sum(sum(x));
%TODO

p = zeros(C, N);

prior = zeros(1, C);

sum2 = zeros(1, C);

for i = 1 : C
    for j = 1 : N
        sum2(i) = sum2(i) + x(i:i, j:j)
    end
    
    prior(i) = sum2(i) / total;
end


for i = 1 : C
    for j = 1 : N
        p(i:i, j:j) = l(i:i, j:j) * prior(i:i);
    end
end

for i = 1 : N
    p(:, i) = p(:, i) / sum(p(:, i)); 
end

end
