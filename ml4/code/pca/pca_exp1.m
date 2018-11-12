load('ORL_data', 'fea_Train', 'gnd_Train', 'fea_Test', 'gnd_Test');
 
% YOUR CODE HERE
 
% 1. Feature preprocessing
% 2. Run PCA
% 3. Visualize eigenface
% 4. Project data on to low dimensional space
% 5. Run KNN in low dimensional space
% 6. Recover face images form low dimensional space, visualize them
 
fea_Train = zscore(fea_Train);
fea_Test = zscore(fea_Test);
[eigvector, eigvalue] = pca(fea_Train);
 
show_face(eigvector');

%fea_Train = fea_Train * eigvector;
%fea_Test = fea_Test * eigvector;

cnt = 0;
K = [8, 16, 32, 64, 128];

for j = 1 : size(K, 2)
    cnt = 0;

    knn_fea_Train = (eigvector(:, 1:K(j))' * fea_Train')';
    knn_fea_Test = (eigvector(:, 1:K(j))' * fea_Test')';

    for i = 1 : 200
        t = knn(knn_fea_Test(i, :)', knn_fea_Train', gnd_Train', 1);

        if t ~= gnd_Test(i)
            cnt = cnt+1;
        end
    end
    error = cnt / 200.0
end
