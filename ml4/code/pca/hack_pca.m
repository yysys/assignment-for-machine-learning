function img = hack_pca(filename)
% Input: filename -- input image file name/path
% Output: img -- image without rotation

img_r = double(imread(filename));

% YOUR CODE HERE

X = [];
for i = 1 : size(img_r, 1)
    for j = 1 : size(img_r, 2)
        if img_r(i, j) ~= 255
            tmp = [i, j];
            X = [X; tmp];
        end
    end
end

[eigvector, eigvalue] = pca(X);

tt = [2, 1];
eigvector = eigvector(:, tt);
eigvector(:, 1) = -eigvector(:, 1);

point = X * eigvector;

x_mi = min(point(:, 1)) - 1;
y_mi = min(point(:, 2)) - 1;

point = point + repmat([-x_mi, -y_mi], size(point, 1), 1);

x_mx = ceil(max(point(:, 1)));
y_mx = ceil(max(point(:, 2)));

img = repmat(255, x_mx, y_mx);

for i = 1 : size(point)
    t1 = round(point(i, 1));
    t2 = round(point(i, 2));
    img(t1, t2) = img_r(X(i, 1), X(i, 2));
end

end