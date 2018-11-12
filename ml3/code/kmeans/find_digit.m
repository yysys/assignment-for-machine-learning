load('digit_data');
K = 10;

[idx, ctrs, iter_ctrs] = kmeans(X, K);
show_digit(ctrs);


