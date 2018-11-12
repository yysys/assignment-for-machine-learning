load('kmeans_data');

K = 2;

[idx, ctrs, iter_ctrs] = kmeans(X, K);
%kmeans_plot(X, idx, ctrs, iter_ctrs)

mi_sum = 100000;
mi_X = [];
mi_idx = [];
mi_ctrs = [];
mi_iter_ctrs = [];

mx_sum = 0;
mx_X = [];
mx_idx = [];
mx_ctrs = [];
mx_iter_ctrs = [];

for i = 1 : 1000
    
    [idx, ctrs, iter_ctrs] = kmeans(X, K);
    sum = 0;
    for j = 1 : K
        dis = (X(idx == j, :) - ctrs(j, :));
        sum = sum + norm(dis);
    end
    
    if (sum < mi_sum)
        mi_X = X;
        mi_sum = sum;
        mi_idx = idx;
        mi_ctrs = ctrs;
        mi_iter_ctrs = iter_ctrs;
    end
    
    if (sum > mx_sum)
        mx_X = X;
        mx_sum = sum;
        mx_idx = idx;
        mx_ctrs = ctrs;
        mx_iter_ctrs = iter_ctrs;
    end
    
end

kmeans_plot(mi_X, mi_idx, mi_ctrs, mi_iter_ctrs)
kmeans_plot(mx_X, mx_idx, mx_ctrs, mx_iter_ctrs)
