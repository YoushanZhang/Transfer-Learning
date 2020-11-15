function [Acc,acc_iter,Beta,Yt_pred] = MDA(Xs,Ys,Xt,Yt,options) % Cls
%% Zhang, Youshan, and Brian D. Davison. 
% "Impact of ImageNet Model Selection on Domain Adaptation."
% In Proceedings of the IEEE Winter Conference on Applications of Computer Vision Workshops, pp. 173-182. 2020.

%% Zhang, Youshan, and Brian D. Davison.
% "Modified distribution alignment for domain adaptation with pre-trained inception resnet." 
% arXiv preprint arXiv:1904.02322 (2019).

%% Modified from:
% Jindong Wang et al. Visual Domain Adaptation with Manifold Embedded Distribution
%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d      :  dimension after manifold feature learning (default: 20)
%%%%% options.T      :  number of iteration (default: 10)
%%%%% options.lambda :  lambda in the paper (default: 10)
%%%%% options.eta    :  eta in the paper (default: 0.1)
%%%%% options.rho    :  rho in the paper (default: 1.0)
%%%%% options.base   :  base classifier for soft labels (default: NN)

%% Outputs:
%%%% Acc      :  Final accuracy value
%%%% acc_iter :  Accuracy value list of all iterations, T * 1
%%%% Beta     :  Cofficient matrix
%%%% Yt_pred  :  Prediction labels for target domain

    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'eta')
        options.eta = 0.1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 1.0;
    end
    if ~isfield(options,'rho')
        options.rho = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'d')
        options.d = 20;
    end

% Using Deep feature
    Xs = double(Xs');
    Xt = double(Xt');


% Pre-processing data

    X = [Xs,Xt];
    n = size(Xs,2);
    m = size(Xt,2);
    C = length(unique(Ys));
    acc_iter = [];
    
    YY = [];
    for c = 1 : C
        YY = [YY,Ys==c];
    end
    YY = [YY;zeros(m,C)];

    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

    %% Construct graph Laplacian
    if options.rho > 0
        opt.k = options.p;
        opt.Metric = 'Cosine';
        opt.NeighborMode = 'KNN';
        opt.WeightMode = 'Cosine';
        W = lapgraph(X',opt);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n + m) - Dw * W * Dw;
    else
        L = 0;
    end

    % Generate soft labels for the target domain
    knn_model = fitcknn(X(:,1:n)',Ys,'NumNeighbors',1); 
    Cls = knn_model.predict(X(:,n + 1:end)');

    % Construct rbf kernel
    n1sq = sum(X.^2,1);
    n1 = size(X,2);
    D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
    K = exp(-D/(2*(sqrt(sum(sum(X .^ 2).^0.5)/(n + m)))^2)); 
            
            
    E = diag(sparse([ones(n,1);zeros(m,1)]));

    for t = 1 : options.T
        % Estimate mu
        mu = estimate_mu(Xs',Ys,Xt',Cls);
        % Construct MMD matrix
        e = [1 / n * ones(n,1); -1 / m * ones(m,1)];
        M = e * e' * length(unique(Ys));
        N = 0;
        for c = reshape(unique(Ys),1,length(unique(Ys)))
            e = zeros(n + m,1);
            e(Ys == c) = 1 / length(find(Ys == c));
            e(n + find(Cls == c)) = -1 / length(find(Cls == c));
            e(isinf(e)) = 0;
            N = N + e * e';
        end
        M = (1 - mu) * M + mu * N;
        M = M / norm(M,'fro');

        % Compute coefficients vector Beta
        Beta = ((E + options.lambda * M + options.rho * L) * K + options.eta * speye(n + m,n + m)) \ (E * YY);
        F = K * Beta;
        [~,Cls] = max(F,[],2);

        %% Compute accuracy
        Acc = numel(find(Cls(n+1:end)==Yt)) / m *100;
        Cls = Cls(n+1:end);
        acc_iter = [acc_iter;Acc];
    end
    Yt_pred = Cls;
end

function [mu,adist_m,adist_c] = estimate_mu(Xs,Ys,Xt,Yt)
    C = length(unique(Ys));
    list_adist_c = [];
    epsilon = 1e-3;
    for i = 1 : C
        index_i = Ys == i;
        Xsi = Xs(index_i,:);
        index_j = Yt == 1;
        Xtj = Xt(index_j,:);
        adist_i = adist(Xsi,Xtj);
        list_adist_c = [list_adist_c;adist_i];
    end
    adist_c = mean(list_adist_c);
    
    adist_m = adist(Xs,Xt);
    mu = adist_c / (adist_c + adist_m);
    if mu > 1    % Theoretically mu <= 1, but calculation may be over 1
        mu = 1;
    elseif mu <= epsilon
        mu = 0;  
    end
end

function dist = adist(Xs,Xt)
    Yss = ones(size(Xs,1),1);
    Ytt = ones(size(Xt,1),1) * 2;
   
    model_linear = fitclinear([Xs;Xt],[Yss;Ytt],'learner','svm');
    ypred = model_linear.predict([Xs;Xt]);
    error = mae([Yss;Ytt],ypred);
    dist = 2 * (1 - 2 * error);
end