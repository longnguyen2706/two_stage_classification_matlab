% SIFT + BoW + RO-SVM Script

clear all;close all;clc;
nRounds = 1;                     % # of experiments

NoFeature = true;                % True: Build BoW model from SIFT features; False: Load saved BoW model;
NoData = true;                   % True: Need to fetch saved BoW features from folders. False: use the saved .mat file to read BoW features. 

%*******SVM?Package? Liblinear *********************

addpath('liblinear-2.1\liblinear-2.1\matlab'); 

%*******Directories*******************

img_dir = 'image/PAP';       % directory for the image database                             
data_dir = 'data/PAP';       % directory for saving SIFT descriptors
bow_fea_dir = 'gen_feature_matlab/bow/PAP';    % directory for saving BoW features
cnn_fea_dir = 'gen_feature_matlab/cnn/PAP';
% retrieve the directory of the database and load the codebook
database = retr_database_dir(data_dir);

if isempty(database)
    error('Data directory error!');
end

%****************************************

[total_data_s1, total_label_s1, nclass_s1, fdatabase_bow] = load_bow_features(database, bow_fea_dir, false, false);
[total_data_s2, total_label_s2, nclass_s2, fdatabase_cnn] = load_cnn_features(database, cnn_fea_dir);

fprintf ('nclass1, nclass2 %d %d', nclass_s1, nclass_s2);



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
    
    %*************************%********Train/test/val split************
    validation_ratio = 0.25;
    
    
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
    
    [t_opt, SVM_model] = Build_RO_SVM(tr_data, tr_label, tr_without_v_data, tr_without_v_label, v_data, v_label, nclass,rejectionRate_thr); 

    [reject_index, stage1_predict_ts] = Stage1_Classification(t_opt,SVM_model,tr_data, tr_label, ts_data, ts_label, nclass);
   
    acc1 = length(find(stage1_predict_ts == total_label(ts_idx)))/length(total_label(ts_idx));
    fprintf('Stage 1 Round: %d, Accuracy: %.4f\n', ii, acc1);
    
    
    [stage2_predict_ts, stage2_predict_reject] = Stage2_Classification(total_data, total_label, tr_idx, ts_idx, reject_index);
    acc2 = length(find(stage2_predict_ts == total_label(ts_idx)))/length(total_label(ts_idx));
    fprintf('Stage 2 Round: %d, Accuracy: %.4f\n', ii, acc2);
    
    
    
%     acc_all = [acc_all; acc]; % Record each round of the results of Stage 1 on all the test samples
end

pause;
