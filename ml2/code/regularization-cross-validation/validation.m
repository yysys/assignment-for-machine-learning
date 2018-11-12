%% Ridge Regression
load('digit_train', 'X', 'y');

% Do feature normalization
for i = 1 : size(X, 2)
    X(:, i) = (X(:, i) - mean(X(:, i))) / (std(X(:, i)));
end
% ...

% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;

mi_E_val = 10000;
for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        X_ = X; y_ = y; % take point j out of X
        X_(:, j) = [];
        y_(j) = [];
        w = ridge(X_, y_, lambdas(i));
        if y(j) * w' * [1; X(:, j)] < 0
            E_val = E_val + 1;
        end
    end
    %Update lambda according validation error
    if (E_val < mi_E_val) 
        lambda = lambdas(i);
        mi_E_val = E_val;
    end
end

w0 = ridge(X, y, 0);
wSquare0 = norm(w0) * norm(w0);

E_train0 = 0;
for i = 1 : size(X, 2)
    if y(i) * w0' * [1; X(:, i)] < 0
        E_train0 = E_train0 + 1;
    end
end

w1 = ridge(X, y, lambda);
% Compute training error
E_train1 = 0;
for i = 1 : size(X, 2)
    if y(i) * w1' * [1; X(:, i)] < 0
        E_train1 = E_train1 + 1;
    end
end

wSquare1 = norm(w1) * norm(w1);

E_train0 = E_train0 / size(X, 2);
E_train1 = E_train1 / size(X, 2);

fprintf('wSquare0 = %d\n', wSquare0);
fprintf('wSquare1 = %d\n', wSquare1);
fprintf('E_train0 = %d\n', E_train0);
fprintf('E_train1 = %d\n', E_train1);

load('digit_test', 'X_test', 'y_test');

% Do feature normalization
for i = 1 : size(X_test, 2)
    X_test(:, i) = (X_test(:, i) - mean(X_test(:, i))) / (std(X_test(:, i)));
end
% ...
% Compute test error

E_test0 = 0;
for i = 1 : size(X_test, 2)
    if y_test(i) * w0' * [1; X_test(:, i)] < 0
        E_test0 = E_test0 + 1;
    end
end

E_test1 = 0;
for i = 1 : size(X_test, 2)
    if y_test(i) * w1' * [1; X_test(:, i)] < 0
        E_test1 = E_test1 + 1;
    end
end

E_test0 = E_test0 / size(X_test, 2);
E_test1 = E_test1 / size(X_test, 2);

fprintf('E_test0 = %d\n', E_test0);
fprintf('E_test1 = %d\n', E_test1);

%% Logistic

load('digit_train', 'X', 'y');

% Do feature normalization
for i = 1 : size(X, 2)
    X(:, i) = (X(:, i) - mean(X(:, i))) / (std(X(:, i)));
end
% ...

% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;

mi_E_val = 10000;
for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        X_ = X; y_ = y; % take point j out of X
        X_(:, j) = [];
        y_(j) = [];
        w = logistic_r(X_, y_, lambdas(i));
        if y(j) * w' * [1; X(:, j)] < 0
            E_val = E_val + 1;
        end
    end
    %Update lambda according validation error
    if (E_val <= mi_E_val) 
        lambda = lambdas(i);
        mi_E_val = E_val;
    end
end

w0 = logistic_r(X, y, 0);
wSquare0 = norm(w0) * norm(w0);

E_train0 = 0;
for i = 1 : size(X, 2)
    if y(i) * w0' * [1; X(:, i)] < 0
        E_train0 = E_train0 + 1;
    end
end

w1 = logistic_r(X, y, lambda);
% Compute training error
E_train1 = 0;
for i = 1 : size(X, 2)
    if y(i) * w1' * [1; X(:, i)] < 0
        E_train1 = E_train1 + 1;
    end
end

wSquare1 = norm(w1) * norm(w1);

E_train0 = E_train0 / size(X, 2);
E_train1 = E_train1 / size(X, 2);

fprintf('wSquare0 = %d\n', wSquare0);
fprintf('wSquare1 = %d\n', wSquare1);
fprintf('E_train0 = %d\n', E_train0);
fprintf('E_train1 = %d\n', E_train1);

load('digit_test', 'X_test', 'y_test');

% Do feature normalization
for i = 1 : size(X_test, 2)
    X_test(:, i) = (X_test(:, i) - mean(X_test(:, i))) / (std(X_test(:, i)));
end
% ...
% Compute test error

E_test0 = 0;
for i = 1 : size(X_test, 2)
    if y_test(i) * w0' * [1; X_test(:, i)] < 0
        E_test0 = E_test0 + 1;
    end
end

E_test1 = 0;
for i = 1 : size(X_test, 2)
    if y_test(i) * w1' * [1; X_test(:, i)] < 0
        E_test1 = E_test1 + 1;
    end
end

E_test0 = E_test0 / size(X_test, 2);
E_test1 = E_test1 / size(X_test, 2);

fprintf('E_test0 = %d\n', E_test0);
fprintf('E_test1 = %d\n', E_test1);

%% SVM with slack variable
