%% Zhang, Youshan, and Brian D. Davison. 
% "Impact of ImageNet Model Selection on Domain Adaptation."
% In Proceedings of the IEEE Winter Conference on Applications of Computer Vision Workshops, pp. 173-182. 2020.

%% Zhang, Youshan, and Brian D. Davison.
% "Modified distribution alignment for domain adaptation with pre-trained inception resnet." 
% arXiv preprint arXiv:1904.02322 (2019).

%%
clear
% tic
%% Datasets and domains
datasets={'Office_caltech_10'};
str_domains_Office_caltech_10 = {'caltech', 'amazon', 'webcam', 'dslr'}; 

%% parameters
options.rho = 1.0;
options.p = 10;
options.lambda = 10.0;
options.eta = 0.1;
options.T = 10;
%% Load extracted NasNetLarge features
task=1;
for ii = 1:size(str_domains_Office_caltech_10,2)
    for jj = 1:size(str_domains_Office_caltech_10,2)
        if ii == jj
            continue; 
        end
        disp([str_domains_Office_caltech_10{ii},'  to  ', str_domains_Office_caltech_10{jj}])
        src = str_domains_Office_caltech_10{ii};
        tgt = str_domains_Office_caltech_10{jj};
       
        load(['./',datasets{1},'/',src, '_nasnetlarge_global_average_pooling2d.mat']);     % source domain 
        Xs = double(features);                 clear features
        Ys = double(labels);           clear labels
        
       
        load(['./',datasets{1},'/',tgt, '_nasnetlarge_global_average_pooling2d.mat']);     % target domain 
        Xt = double(features);                 clear features
        Yt = double(labels);           clear labels
        
        
%%      Modified Distribution Alignment (MDA) Classification
        [Acc,~,~,~] = MDA(Xs,Ys,Xt ,Yt,options);

        disp(num2str(roundn(Acc,-1)))
        all_tasks(task)=roundn(Acc,-1);
        task=task+1;
%         toc   
    end
end


%% Add mean at last column
all_tasks(task)=mean(all_tasks);
disp(['Mean accuracy is ', num2str(roundn(all_tasks(end),-1))])
      


