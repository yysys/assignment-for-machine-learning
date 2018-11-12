% YOUR CODE HERE
load('ORL_data', 'fea_Train', 'gnd_Train', 'fea_Test', 'gnd_Test');


fea_Train = zscore(fea_Train);
fea_Test = zscore(fea_Test);
[eigvector, eigvalue] = pca(fea_Train);

K = 8;

fea = eigvector(:, 1:K)' * fea_Train';

fea =  (eigvector(:, 1:K) * fea)';

show_face(fea);
