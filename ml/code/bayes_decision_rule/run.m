% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%%load data
load('data');
all_x = cat(2, x1_train, x1_test, x2_train, x2_test);
range = [min(all_x), max(all_x)];
train_x = get_x_distribution(x1_train, x2_train, range);
test_x = get_x_distribution(x1_test, x2_test, range);

%% Part1 likelihood: 
l = likelihood(train_x);

% bar(range(1):range(2), l');
% xlabel('x');
% ylabel('P(x|\omega)');
% axis([range(1) - 1, range(2) + 1, 0, 0.5]);

%TODO
%compute the number of all the misclassified x using maximum likelihood decision rule

misclassified = 0;
[C, N] = size(l);

for i = 1 : N
    if l(1:1, i:i) > l(2:2, i:i) 
        misclassified = misclassified + test_x(2:2, i:i);
    else
        misclassified = misclassified + test_x(1:1, i:i);
    end
end

%% Part2 posterior:
p = posterior(train_x);

bar(range(1):range(2), p');
xlabel('x');
ylabel('P(\omega|x)');
axis([range(1) - 1, range(2) + 1, 0, 1.2]);

%TODO
%compute the number of all the misclassified x using optimal bayes decision rule

misclassified2 = 0;
[C, N] = size(l);

for i = 1 : N
    if p(1:1, i:i) > p(2:2, i:i) 
        misclassified2 = misclassified2 + test_x(2:2, i:i);
    else
        misclassified2 = misclassified2 + test_x(1:1, i:i);
    end
end

%% Part3 risk:
risk = [0, 1; 2, 0];
%TODO
%get the minimal risk using optimal bayes decision rule and risk weights

minRisk = 0;

% for i = 1 : N
%     if (risk(2, 1) - risk(1, 1)) * p(1, i) > (risk(1, 2) - risk(2, 2)) * p(2, i)
%         minRisk = minRisk + risk(1, 2) * test_x(2, i);
%     else 
%         minRisk = minRisk + risk(2, 1) * test_x(1, i);
%     end
% end



for f = 1 : N
    if risk(1, 2) * p(2, f) < risk(2, 1) * p(1, f)
        minRisk = minRisk + risk(1, 2) * p(2, f) * sum(test_x(:, f));
    else 
        minRisk = minRisk + risk(2, 1) * p(1, f) * sum(test_x(:, f));
    end
end

fprintf('misclassied1 = %d\n', misclassified);
fprintf('misclassied2 = %d\n', misclassified2);
fprintf('minRisk = %d\n', minRisk);

