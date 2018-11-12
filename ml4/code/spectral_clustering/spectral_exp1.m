load('cluster_data', 'X');

% Choose proper parameters
k_in_knn_graph = 130;
threshold = 0.2;

W = knn_graph(X, k_in_knn_graph, threshold);

% imshow(W);
idx = spectral(W, 2);
cluster_plot(X, idx);

figure;
idx = kmeans(X, 2);
cluster_plot(X, idx);
