% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%% Part1: Preceptron
nRep = 1000; % number of replicates
nTrain = 100; % number of training data

nTest = 10 * nTrain;
iterSum = 0;
E_train = 0;
E_test = 0;
for i = 1:nRep
    [tX, ty, w_f] = mkdata(nTrain + nTest);
    x_train = tX(:, 1:nTrain);
    x_test = tX(:, nTrain+1: nTrain + nTest);
    
    y_train = ty(:, 1:nTrain);
    y_test = ty(:, nTrain+1: nTrain + nTest);
    
    [w_g, iter] = perceptron(x_train, y_train);
    % Compute training, testing error
    
    numTrainError = 0;
    numTestError = 0;
    
    for j = 1 : size(x_train, 2)
        if w_g' * [1; x_train(:, j)] * y_train(j) < 0
            numTrainError = numTrainError + 1;
        end
    end
    
    for j = 1 : size(x_test, 2)
        if w_g' * [1; x_test(:, j)] * y_test(j) < 0
            numTestError = numTestError + 1;
        end
    end
    
    E_train = E_train + numTrainError / nTrain;
    E_test = E_test + numTestError / nTest;
    
    % Sum up number of iterations
    iterSum = iterSum + iter;
end

avgIter = iterSum / nRep
E_train = E_train / nRep;
E_test = E_test / nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('Average number of iterations is %d.\n', avgIter);
plotdata(x_train, y_train, w_f, w_g, 'Pecertron');

%% Part2: Preceptron: Non-linearly separable case
nTrain = 100; % number of training data
[X, y, w_f] = mkdata(nTrain, 'noisy');
[w_g, iter] = perceptron(X, y)


%% Part3: Linear Regression
nRep = 1000; % number of replicates
nTrain = 100; % number of training data

nTest = 10 * nTrain;
E_train = 0;
E_test = 0;

for i = 1:nRep
    [tX, ty, w_f] = mkdata(nTrain + nTest);
    x_train = tX(:, 1:nTrain);
    x_test = tX(:, nTrain+1: nTrain + nTest);
    
    y_train = ty(:, 1:nTrain);
    y_test = ty(:, nTrain+1: nTrain + nTest);
    
    w_g = linear_regression(x_train, y_train);
    % Compute training, testing error
    numTrainError = 0;
    numTestError = 0;
    
    for j = 1 : size(x_train, 2)
        if w_g' * [1; x_train(:, j)] * y_train(j) < 0
            numTrainError = numTrainError + 1;
        end
    end
    
    for j = 1 : size(x_test, 2)
        if w_g' * [1; x_test(:, j)] * y_test(j) < 0
            numTestError = numTestError + 1;
        end
    end
    
    E_train = E_train + numTrainError / nTrain;
    E_test = E_test + numTestError / nTest;
    
end

E_train = E_train / nRep;
E_test = E_test / nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(x_train, y_train, w_f, w_g, 'Linear Regression');

%% Part4: Linear Regression: noisy

nRep = 1000; % number of replicates
nTrain = 100; % number of training data

nTest = 10 * nTrain;
E_train = 0;
E_test = 0;

for i = 1:nRep
    [tX, ty, w_f] = mkdata(nTrain + nTest, 'noisy');
    x_train = tX(:, 1:nTrain);
    x_test = tX(:, nTrain+1: nTrain + nTest);
    
    y_train = ty(:, 1:nTrain);
    y_test = ty(:, nTrain+1: nTrain + nTest);
    
    w_g = linear_regression(x_train, y_train);
    % Compute training, testing error
    numTrainError = 0;
    numTestError = 0;
    
    for j = 1 : size(x_train, 2)
        if w_g' * [1; x_train(:, j)] * y_train(j) < 0
            numTrainError = numTrainError + 1;
        end
    end
    
    for j = 1 : size(x_test, 2)
        if w_g' * [1; x_test(:, j)] * y_test(j) < 0
            numTestError = numTestError + 1;
        end
    end
    
    E_train = E_train + numTrainError / nTrain;
    E_test = E_test + numTestError / nTest;
    
end

E_train = E_train / nRep;
E_test = E_test / nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(x_train, y_train, w_f, w_g, 'Linear Regression: noisy');

%% Part5: Linear Regression: poly_fit
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');
w = linear_regression(X, y)
% Compute training, testing error

E_train = 0;
E_test = 0;
for i = 1 : size(X, 2)
    if (w' * [1; X(:, i)] * y(i) < 0)
        E_train = E_train + 1;
    end
end

for i = 1 : size(X_test, 2)
    if (w' * [1; X_test(:, i)] * y_test(i) < 0)
        E_test = E_test + 1;
    end
end

E_train = E_train / size(X, 2);
E_test = E_test / size(X_test, 2);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);

% poly_fit with transform
X_t = [X; X(1,:).*X(2,:);  X(1,:).*X(1,:);  X(2,:).*X(2,:)]; % CHANGE THIS LINE TO DO TRANSFORMATION
X_test_t = [X_test; X_test(1,:).*X_test(2,:);  X_test(1,:).*X_test(1,:);  X_test(2,:).*X_test(2,:)]; % CHANGE THIS LINE TO DO TRANSFORMATION
w = linear_regression(X_t, y)
% Compute training, testing error

E_train = 0;
E_test = 0;
for i = 1 : size(X_t, 2)
    if (w' * [1; X_t(:, i)] * y(i) < 0)
        E_train = E_train + 1;
    end
end

for i = 1 : size(X_test_t, 2)
    if (w' * [1; X_test_t(:, i)] * y_test(i) < 0)
        E_test = E_test + 1;
    end
end

E_train = E_train / size(X, 2);
E_test = E_test / size(X_test, 2);
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);


%% Part6: Logistic Regression
nRep = 100; % number of replicates
nTrain = 100; % number of training data

nTest = 10 * nTrain;
E_train = 0;
E_test = 0;

for i = 1:nRep
    [tX, ty, w_f] = mkdata(nTrain + nTest);
    x_train = tX(:, 1:nTrain);
    x_test = tX(:, nTrain+1: nTrain + nTest);
     
    y_train = ty(:, 1:nTrain);
    y_test = ty(:, nTrain+1: nTrain + nTest);
     
    w_g = logistic(x_train, y_train);
    % Compute training, testing error
    
    numTrainError = 0;
    numTestError = 0;
    
    for j = 1 : size(x_train, 2)
        if w_g' * [1; x_train(:, j)] * y_train(j) < 0
            numTrainError = numTrainError + 1;
        end
    end
    
    for j = 1 : size(x_test, 2)
        if w_g' * [1; x_test(:, j)] * y_test(j) < 0
            numTestError = numTestError + 1;
        end
    end
    
    E_train = E_train + numTrainError / nTrain;
    E_test = E_test + numTestError / nTest;
    
end

E_train = E_train / nRep;
E_test = E_test / nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(x_train, y_train, w_f, w_g, 'Logistic Regression');

Part7: Logistic Regression: noisy
nRep = 100; % number of replicates
nTrain = 100; % number of training data

nTest = 10 * nTrain;
E_train = 0;
E_test = 0;

for i = 1:nRep
    [tX, ty, w_f] = mkdata(nTrain + nTest, 'noisy');
    x_train = tX(:, 1:nTrain);
    x_test = tX(:, nTrain+1: nTrain + nTest);
     
    y_train = ty(:, 1:nTrain);
    y_test = ty(:, nTrain+1: nTrain + nTest);
     
    w_g = logistic(x_train, y_train);
    %Compute training, testing error
    
    numTrainError = 0;
    numTestError = 0;
    
    for j = 1 : size(x_train, 2)
        if w_g' * [1; x_train(:, j)] * y_train(j) < 0
            numTrainError = numTrainError + 1;
        end
    end
    
    for j = 1 : size(x_test, 2)
        if w_g' * [1; x_test(:, j)] * y_test(j) < 0
            numTestError = numTestError + 1;
        end
    end
    
    E_train = E_train + numTrainError / nTrain;
    E_test = E_test + numTestError / nTest;
    
end

E_train = E_train / nRep;
E_test = E_test / nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(x_train, y_train, w_f, w_g, 'Logistic Regression: noisy');
 
Part8: SVM
nRep = 1000; % number of replicates
nTrain = 100; % number of training data

nTest = 10 * nTrain;
E_train = 0;
E_test = 0;
num = 0;

for i = 1:nRep
    [tX, ty, w_f] = mkdata(nTrain + nTest);
    x_train = tX(:, 1:nTrain);
    x_test = tX(:, nTrain+1: nTrain + nTest);
     
    y_train = ty(:, 1:nTrain);
    y_test = ty(:, nTrain+1: nTrain + nTest);
    
    [w_g, num_sc] = svm(x_train, y_train);
    % Compute training, testing error
    
    numTrainError = 0;
    numTestError = 0;
    
    for j = 1 : size(x_train, 2)
        if w_g' * [1; x_train(:, j)] * y_train(j) < 0
            numTrainError = numTrainError + 1;
        end
    end
    
    for j = 1 : size(x_test, 2)
        if w_g' * [1; x_test(:, j)] * y_test(j) < 0
            numTestError = numTestError + 1;
        end
    end
    
    E_train = E_train + numTrainError / nTrain;
    E_test = E_test + numTestError / nTest;
    
    % Sum up number of support vectors
    
    num = num + num_sc;
end

E_train = E_train / nRep;
E_test = E_test / nRep;
num = num / nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(x_train, y_train, w_f, w_g, 'SVM');