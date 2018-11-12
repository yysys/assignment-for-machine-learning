img = imread('sample0.jpg');
fea = double(reshape(img, size(img, 1)*size(img, 2), 3));
% YOUR (TWO LINE) CODE HERE
[idx, ctrs, iter_ctrs] = kmeans(fea, 64);
fea(:, :) = ctrs(idx(:), :);
imshow(uint8(reshape(fea, size(img))));

img1 = uint8(reshape(fea, size(img)));
imwrite(img1, 'vq0.jpg');

info = imfinfo('sample0.jpg')
info1 = imfinfo('vq0.jpg')