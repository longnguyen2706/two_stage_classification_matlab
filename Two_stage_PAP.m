% SIFT + BoW + RO-SVM Script

clear all;close all;clc;
nRounds = 3;    % # of experiments
validation_ratio = 0.25;
tr_ratio = 0.8;                     % Training image ratio, e.g., 80%
rejectionRates= 0.05:0.2:0.5;
genFeature = true;                % True: Build BoW model from SIFT features; False: Load saved BoW model;
genData = true;                 % True: Need to fetch saved BoW features from folders. False: use the saved .mat file to read BoW features.

%*******SVM?Package? Liblinear *********************

addpath('liblinear-2.1\liblinear-2.1\matlab');

%*******Directories*******************

img_dir = 'image/Hep';       % directory for the image database
data_dir = 'data/Hep';       % directory for saving SIFT descriptors
bow_fea_dir = 'gen_feature_matlab/bow/Hep';    % directory for saving BoW features
cnn_fea_dir = 'gen_feature_matlab/cnn/Hep';
Bpath = ['dictionary/dictionary_1000_Hep.mat'];
% retrieve the directory of the database and load the codebook
database = retr_database_dir(data_dir);

if isempty(database)
    error('Data directory error!');
end

%****************************************

[total_data_s1, total_label_s1, nclass_s1, clabel_s1, fdatabase_s1] = load_bow_features(database, bow_fea_dir, genFeature, genData, Bpath);
[total_data_s2, total_label_s2, nclass_s2, clabel_s2, fdatabase_s2] = load_cnn_features(database, cnn_fea_dir);

fprintf ('nclass1, nclass2 %d %d \n', nclass_s1, nclass_s2);
%fprintf ('is total label 2 stage equal %s \n', isequal(total_label_s1, total_label_s2));

nclass = nclass_s1;
clabel = clabel_s1;
fdatabase = fdatabase_s1;

all_acc_ts_s1 = [];
all_acc_ts_s2 = [];
all_acc_ts_2_stage = [];
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
    
    total_label = total_label_s1;
    %***************************
    
    all_acc_v_s1 = [];
    all_acc_v_s2 = [];
    all_acc_v_2_stage = [];
    all_t_opt = [];
    for iter = 1: length(rejectionRates)
        rejectionRate_thr = rejectionRates(iter);
        
        [t_opt] = Build_RO_SVM(total_data_s1, total_label_s1, tr_without_v_idx, v_idx, nclass,rejectionRate_thr);
        
        [reject_index_v, stage1_predict_v] = Stage1_Classification(t_opt, total_data_s1, total_label_s1, tr_without_v_idx, v_idx, nclass);
        
        acc_v_s1 = length(find(stage1_predict_v == total_label(v_idx)))/length(total_label(v_idx));
        fprintf('Stage 1 Round: %d, Val Accuracy: %.4f, Rejection rate: %.4f, T_opt: %.4f\n', ii, acc_v_s1, rejectionRate_thr, t_opt);
        
        [stage2_predict_v, stage2_predict_reject_v] = Stage2_Classification(total_data_s2, total_label_s2, tr_without_v_idx, v_idx, reject_index_v);
        acc_v_s2 = length(find(stage2_predict_v == total_label(v_idx)))/length(total_label(v_idx));
        fprintf('Stage 2 Round: %d, Val Accuracy: %.4f, Rejection rate: %.4f, T_opt: %.4f\n', ii, acc_v_s2, rejectionRate_thr, t_opt);
        
        two_stage_predict_v = stage1_predict_v;
        for i = 1:length(reject_index_v)
            two_stage_predict_v(reject_index_v(i)) = stage2_predict_v(reject_index_v(i));
        end
        two_stage_acc_v = length(find(two_stage_predict_v == total_label(v_idx)))/length(total_label(v_idx));
        fprintf('2 stage Round: %d, Val Accuracy: %.4f, Rejection rate: %.4f, T_opt: %.4f\n', ii, two_stage_acc_v, rejectionRate_thr, t_opt);
        
        all_acc_v_s1 = [all_acc_v_s1; acc_v_s1];
        all_acc_v_s2 = [all_acc_v_s2; acc_v_s2];
        all_acc_v_2_stage = [all_acc_v_2_stage; two_stage_acc_v];
        all_t_opt = [all_t_opt; t_opt];
    end
    
    [best_acc_v_2_stage, best_acc_v_index] = max(all_acc_v_2_stage);
    best_t_opt = all_t_opt(best_acc_v_index);
    fprintf('Best 2_stage_val_acc: %.4f, best_t_opt:  %.4f, best_rejection_rate: %.4f \n', best_acc_v_2_stage, best_t_opt, rejectionRates(best_acc_v_index));
    
    fprintf('Now retrain 2 stage with best t_opt \n');
    [reject_index_ts, stage1_predict_ts] = Stage1_Classification(t_opt, total_data_s1, total_label_s1, tr_idx, ts_idx, nclass);
    acc_ts_s1 =  length(find(stage1_predict_ts == total_label(ts_idx)))/length(total_label(ts_idx));
    fprintf('Stage 1 Round: %d, Test Accuracy: %.4f\n', ii, acc_ts_s1);
    
    [stage2_predict_ts, stage2_predict_reject_ts] = Stage2_Classification(total_data_s2, total_label_s2, tr_idx, ts_idx, reject_index_ts);
    acc_ts_s2 = length(find(stage2_predict_ts == total_label(ts_idx)))/length(total_label(ts_idx));
    fprintf('Stage 2 Round: %d, Test Accuracy: %.4f\n', ii, acc_ts_s2);
    
    two_stage_predict_ts = stage1_predict_ts;
    for i = 1:length(reject_index_ts)
        two_stage_predict_ts(reject_index_ts(i)) = stage2_predict_ts(reject_index_ts(i));
    end
    two_stage_acc_ts = length(find(two_stage_predict_ts == total_label(ts_idx)))/length(total_label(ts_idx));
    fprintf('2 stage Round: %d, Test Accuracy: %.4f\n', ii, two_stage_acc_ts);
    
    all_acc_ts_s1 = [all_acc_ts_s1; acc_ts_s1];
    all_acc_ts_s2 = [all_acc_ts_s2; acc_ts_s2];
    all_acc_ts_2_stage = [all_acc_ts_2_stage; two_stage_acc_ts];
end
avg_acc_ts_s1 = mean(all_acc_ts_s1);
std_acc_ts_s1 = std(all_acc_ts_s1);

avg_acc_ts_s2 = mean(all_acc_ts_s2);
std_acc_ts_s2 = std(all_acc_ts_s2);

avg_acc_ts_2_stage = mean(all_acc_ts_2_stage);
std_acc_ts_2_stage = std(all_acc_ts_2_stage);

fprintf('Stage 1 Avg, std acc %.4f %.4f \n',avg_acc_ts_s1, std_acc_ts_s1);
fprintf('Stage 2 Avg, std acc %.4f %.4f \n',avg_acc_ts_s2, std_acc_ts_s2);
fprintf('2 stage Avg, std acc %.4f %.4f \n',avg_acc_ts_2_stage, std_acc_ts_2_stage);