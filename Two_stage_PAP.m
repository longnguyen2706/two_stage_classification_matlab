% SIFT + BoW + RO-SVM Script

clear all;close all;clc;

pyramid = [1,2,4];                % spatial block structure for the SPM
knn = 5;                          % # of visual words for each LLC coding 
nRounds = 1;                     % # of experiments

NoFeature = false;                % True: Build BoW model from SIFT features; False: Load saved BoW model;
NoData = false;                   % True: Need to fetch saved BoW features from folders. False: use the saved .mat file to read BoW features. 

%*******SVM?Package? Liblinear *********************

addpath('liblinear-2.1\liblinear-2.1\matlab'); 

%*******Directories*******************

img_dir = 'image/PAP';       % directory for the image database                             
data_dir = 'data/PAP';       % directory for saving SIFT descriptors
fea_dir = 'features/PAP';    % directory for saving BoW features

% retrieve the directory of the database and load the codebook
database = retr_database_dir(data_dir);

if isempty(database),
    error('Data directory error!');
end

%******Load a BoW dictionary**********************
Bpath = ['dictionary/dictionary_1000_PAP.mat'];
load(Bpath);
B = dictionary';
nCodebook = size(B, 2);     


dFea = sum(nCodebook*pyramid.^2);   %dimensionality of BoW features
nFea = length(database.path);       % Number of images

%*********BoW feature building************************

fdatabase = struct;
fdatabase.path = cell(nFea, 1);         % path for each image feature
fdatabase.label = zeros(nFea, 1);       % class label for each image feature

if(NoFeature) % True: Build BoW model from SIFT features; False: Load saved BoW model;

for iter1 = 1:nFea  
    if ~mod(iter1, 5),
       fprintf('.');
    end
    if ~mod(iter1, 100),
        fprintf(' %d images processed\n', iter1);
    end
    fpath = database.path{iter1};
    flabel = database.label(iter1);
    
    load(fpath);
    [rtpath, fname] = fileparts(fpath);
    feaPath = fullfile(fea_dir, num2str(flabel), [fname '.mat']);
    
 
    fea = LLC_pooling(feaSet, B, pyramid, knn);
    label = database.label(iter1);

    if ~isdir(fullfile(fea_dir, num2str(flabel))),
        mkdir(fullfile(fea_dir, num2str(flabel)));
    end      
    save(feaPath, 'fea', 'label');
  
    fdatabase.label(iter1) = flabel;
    fdatabase.path{iter1} = feaPath;
end

save('FeaInfo_PAP.mat','fdatabase','fdatabase');
else
    load('FeaInfo_PAP.mat');
end

clabel = unique(fdatabase.label);   % Class labels
nclass = length(clabel);            % # of classes
%**************Start experiments**************************
if (NoData)  % True: Need to fetch saved BoW features from folders. False: use the saved .mat file to read BoW features. 
        
        total_data = [];
        total_label = [];

        for jj = 1:nclass,
            idx_label = find(fdatabase.label == clabel(jj));

            for kk = 1:length(idx_label)
                fpath = fdatabase.path{idx_label(kk)};
                load(fpath);
                total_data = [total_data; fea'];
                total_label = [total_label;label];
            end
        end
       
        save ('PAP_data.mat','total_data','total_label');
    else
        load ('PAP_data.mat');
end


tr_ratio = 0.8;                     % Training image ratio, e.g., 80%
acc_all = [];

for ii = 1:nRounds
    fprintf('Round: %d...\n', ii);
    
    %***********Ramdomly select training image index (tr_idx) and testing image
    %index (ts_idx)**********
    
    tr_idx = [];
    ts_idx = [];
    for jj = 1:nclass,
        idx_label = find(fdatabase.label == clabel(jj));
        num = length(idx_label);
        
        tr_num = floor(num*tr_ratio);
        idx_rand = randperm(num);
        
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
    end
    
    fprintf('Training number: %d\n', length(tr_idx));
    fprintf('Testing number:%d\n', length(ts_idx));
    
    %*************************%********Parameters************
    validation_ratio = 0.25;
    nclass = length(unique(total_label));
    

    %****************************

    %********training data*****************
    ts_data = total_data(ts_idx,:);
    ts_label = total_data(ts_idx,:);
    
    tr_data = total_data(tr_idx,:);
    tr_label = total_label(tr_idx,:);

    %*******Generate validation data and training data without validation data

    v_idx = [];
    tr_without_v_idx = [];

     for jj = 1:nclass
            idx_label = find(tr_label == jj);
            num = length(idx_label);

            validation_num = floor(num*validation_ratio);
            idx_rand = randperm(num);

            v_idx = [v_idx; idx_label(idx_rand(1:validation_num))];
            tr_without_v_idx = [tr_without_v_idx; idx_label(idx_rand(validation_num+1:end))];
     end

     tr_without_v_data = tr_data(tr_without_v_idx,:);
     tr_without_v_label = tr_label(tr_without_v_idx);

     v_data = tr_data(v_idx,:);
     v_label = tr_label(v_idx,:);
    %***************************
    rejectionRate_thr = 0.2;
    
    [t_opt, SVM_model] = Build_RO_SVM(tr_data, tr_label, tr_without_v_data, tr_without_v_label, v_data, v_label, rejectionRate_thr); % Expected Rejection rate is 0.1 by default, can be set within this function (var: rejectionRate_thr)
    [reject_index,stage1_predict_label] = Stage1_Classification(t_opt,SVM_model,tr_data, tr_label, ts_data, ts_label);
   
    acc1 = length(find(stage1_predict_label == total_label(ts_idx)))/length(total_label(ts_idx));
    fprintf('Stage 1 Round: %d, Accuracy: %.4f\n', ii, acc1);
    
    
    [stage2_predic_ts, stage2_predict_reject] = Stage2_Classification(total_data, total_label, tr_idx, ts_idx, reject_index);
    acc2 = length(find(stage2_predict_ts == total_label(ts_idx)))/length(total_label(ts_idx));
    fprintf('Stage 2 Round: %d, Accuracy: %.4f\n', ii, acc2);
    
    
    
%     acc_all = [acc_all; acc]; % Record each round of the results of Stage 1 on all the test samples
end

pause;

