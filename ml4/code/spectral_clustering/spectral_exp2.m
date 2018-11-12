load('TDT2_data', 'fea', 'gnd');

% YOUR CODE HERE

rep = 10;

MIhat_avg1 = 0;
MIhat_avg2 = 0;
for i = 1 : rep
    threshold = 10;

    options = [];
    options.NeighborMode = 'KNN';
    options.k = 20;
    options.WeightMode = 'Binary';
    options.t = 1;
    W = constructW(fea,options);
    W = full(W);

    res = spectral(W, 5);

    res = bestMap(gnd,res);
    %=============  evaluate AC: accuracy ==============
    AC = length(find(gnd == res))/length(gnd);
    %=============  evaluate MIhat: nomalized mutual information =================
    MIhat = MutualInfo(gnd,res);
    MIhat_avg1 = MIhat_avg1 + MIhat;
    
    res = litekmeans(fea, 5);
    
    res = bestMap(gnd,res);
    %=============  evaluate AC: accuracy ==============
    AC = length(find(gnd == res))/length(gnd);
    %=============  evaluate MIhat: nomalized mutual information =================
    MIhat = MutualInfo(gnd,res);
    MIhat_avg2 = MIhat_avg2 + MIhat;
    
end

MIhat_avg1 = MIhat_avg1 / rep;
MIhat_avg2 = MIhat_avg2 / rep;

MIhat_avg1
MIhat_avg2

