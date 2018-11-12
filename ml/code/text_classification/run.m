%ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
%spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');
%N is the size of vocabulary.
N = size(ham_train, 2);
%There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;
num_spam_train = 3372;
%Do smoothing
x = [ham_train;spam_train] + 1;

%ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;
%spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;

%TODO
%Implement a ham/spam email classifier, and calculate the accuracy of your classifier

prior = [(num_ham_train / (num_ham_train + num_spam_train)) (num_spam_train / (num_ham_train + num_spam_train))];

l = likelihood(x);

ratio = zeros(1, N);
for i = 1 : N
    ratio(i) = l(2, i) / l(1, i);
end

[sortRatio, id] = sort(ratio);
for i = N-9 : N
    fprintf('id = %d\n', id(i));
end

num_ham_test = 3011;
num_spam_test = 1124;

TP = 0;
FN = 0;
FP = 0;
TN = 0;

for s = 1 : num_ham_test
    vec = ham_test(s, 1:N);
    
    [u, v, S] = find(vec);
    [tmp, cnt] = size(u);
    p1 = log(prior(1));
    p2 = log(prior(2));
    for i = 1 : cnt
        p1 = p1 + log(l(1, v(i))) * S(i);
        p2 = p2 + log(l(2, v(i))) * S(i);
    end
    
    if (p1 > p2) 
        TN = TN + 1;
    else
        FP = FP + 1;
    end
end

for s = 1 : num_spam_test
    vec = spam_test(s, 1:N);
    
    [u, v, S] = find(vec);
    [tmp, cnt] = size(u);
    p1 = log(prior(1));
    p2 = log(prior(2));
    for i = 1 : cnt
        p1 = p1 + log(l(1, v(i))) * S(i);
        p2 = p2 + log(l(2, v(i))) * S(i);
    end
    
    if (p1 > p2) 
        FN = FN + 1;
    else
        TP = TP + 1;
    end
end

accuracy = (TP + TN) / (TP + TN + FP + FN);
percision = TP / (TP + FP);
recall = TP / (TP + FN);

fprintf('accuracy = %.6f\n', accuracy);
fprintf('percision = %.6f\n', percision);
fprintf('recall = %.6f\n', recall);