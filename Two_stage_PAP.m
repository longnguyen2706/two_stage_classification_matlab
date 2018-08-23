% SIFT + BoW + RO-SVM Script

clear all;close all;clc;
nRounds = 30;    % # of experiments
validation_ratio = 0.25;  


tr_ratio = 0.8;                     % Training image ratio, e.g., 80%
 rejectionRate_thr = 0.3;

 genFeature = true;                % True: Build BoW model from SIFT features; False: Load saved BoW model;
genData = true;                 % True: Need to fetch saved BoW features from folders. False: use the saved .mat file to read BoW features. 

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

[total_data_s1, total_label_s1, nclass_s1, clabel_s1, fdatabase_s1] = load_bow_features(database, bow_fea_dir, genFeature, genData);
[total_data_s2, total_label_s2, nclass_s2, clabel_s2, fdatabase_s2] = load_cnn_features(database, cnn_fea_dir);

fprintf ('nclass1, nclass2 %d %d \n', nclass_s1, nclass_s2);
%fprintf ('is total label 2 stage equal %s \n', isequal(total_label_s1, total_label_s2));

acc_all = [];

nclass = nclass_s1;
clabel = clabel_s1;
fdatabase = fdatabase_s1;

all_acc_s1 = [];
all_acc_s2 = [];
all_acc_2_stage = [];
for ii = 1:nRounds
    fprintf('Round: %d...\n', ii);
    
    %*************************%********Train/test/val split************
    
    tr_idx = [];
    ts_idx = [];
    for jj = 1:nclass
        idx_label = find(fdatabase.label == clabel(jj));
        num = length(idx_label);
        
        tr_num = floor(num*tr_ratio);
        idx_rand = randperm(num);
        
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
    end
    
    fprintf('Training number: %d\n', length(tr_idx));
    fprintf('Testing number:%d\n', length(ts_idx));
    

    %*******Generate validation data and training data without validation data

    v_idx = [];
    tr_without_v_idx = [];
    [tr_data, tr_label] = label_data_from_idx(total_data_s1, total_label_s1, tr_idx);

     for jj = 1:nclass
            idx_label = find(tr_label == jj);
            num = length(idx_label);

            validation_num = floor(num*validation_ratio);
            idx_rand = randperm(num);

            v_idx = [v_idx; idx_label(idx_rand(1:validation_num))];
            tr_without_v_idx = [tr_without_v_idx; idx_label(idx_rand(validation_num+1:end))];
     end

    %***************************


    [t_opt, SVM_model] = Build_RO_SVM(total_data_s1, total_label_s1, tr_idx, ts_idx, tr_without_v_idx, v_idx, nclass,rejectionRate_thr); 

    [reject_index, stage1_predict_ts] = Stage1_Classification(t_opt,SVM_model,total_data_s1, total_label_s1, tr_idx, ts_idx,nclass);
   
    total_label = total_label_s1;
    
    acc_s1 = length(find(stage1_predict_ts == total_label(ts_idx)))/length(total_label(ts_idx));
    fprintf('Stage 1 Round: %d, Accuracy: %.4f\n', ii, acc_s1);
    
    
    [stage2_predict_ts, stage2_predict_reject] = Stage2_Classification(total_data_s2, total_label_s2, tr_idx, ts_idx, reject_index);
    acc_s2 = length(find(stage2_predict_ts == total_label(ts_idx)))/length(total_label(ts_idx));
    fprintf('Stage 2 Round: %d, Accuracy: %.4f\n', ii, acc_s2);
    
    two_stage_predict_ts = stage1_predict_ts;
    for i = 1:length(reject_index)
        two_stage_predict_ts(reject_index(i)) = stage2_predict_ts(reject_index(i));
    end 
    two_stage_acc = length(find(two_stage_predict_ts == total_label(ts_idx)))/length(total_label(ts_idx));
    fprintf('2 stage Round: %d, Accuracy: %.4f\n', ii, two_stage_acc);
    
    all_acc_s1 = [all_acc_s1; acc_s1];
    all_acc_s2 = [all_acc_s2; acc_s2];
    all_acc_2_stage = [all_acc_2_stage; two_stage_acc]; % Record each round of the results of Stage 1 on all the test samples

end
avg_acc_s1 = mean(all_acc_s1);
std_acc_s1 = std(all_acc_s1);

avg_acc_s2 = mean(all_acc_s2);
std_acc_s2 = std(all_acc_s2);

avg_acc_2_stage = mean(all_acc_2_stage);
std_acc_2_stage = std(all_acc_2_stage);

fprintf('Stage 1 Avg, std acc %.4f %.4f \n',avg_acc_s1, std_acc_s1);
fprintf('Stage 2 Avg, std acc %.4f %.4f \n',avg_acc_s2, std_acc_s2);
fprintf('2 stage Avg, std acc %.4f %.4f \n',avg_acc_2_stage, std_acc_2_stage);



