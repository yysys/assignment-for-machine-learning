X_train = [];
y_train = [];

p = 'C:\Users\54295\Desktop\ml3\code\knn\KnnData';

for i = 1 : 250
    s = strcat(p, '\data', num2str(i), '.jpg');
    X = extract_image(s);
    X_train = [X_train, X];
end

[id,label] = textread('label.txt','%d%d');

for i = 1 : 250
    
    t = 10000;
    for j = 1 : 5
        l = floor(label(i) / t);
        label(i) = mod(label(i), t);
        t = t / 10;
        y_train = [y_train, l];
    end
end

save('hack_data.mat', 'X_train', 'y_train');