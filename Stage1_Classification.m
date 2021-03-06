function [reject_index,stage1_predict_label] = Stage1_Classification(t_opt,SVM_model,total_data,total_label,tr_idx,ts_idx)

%********Package********************
addpath('liblinear-2.1\liblinear-2.1\matlab'); 


%*******Parameters********************
nclass = length(unique(total_label));

%**********************************
    ts_data = total_data(ts_idx,:);
    tr_data = total_data(tr_idx,:);
    tr_label = total_label(tr_idx,:);
    
    
%
  score_acc = [];
    for iter = 1:size(SVM_model.w,1)

        w = SVM_model.w(iter,:);

        score = ts_data*transpose(w)/norm(w);
    %     score = tr_fea*transpose(w)/norm(w);

        score_acc = [score_acc score];

    end

    score_svm = sigmf(score_acc,[15 0]);

    if nclass>2

        for iter = 1:size(score_svm,1)

            score_svm(iter,:) = score_svm(iter,:)/sum(score_svm(iter,:));
        end
    else
        score_svm = [score_svm 1-score_svm];
    end
    
    ts_score_dis = score_svm;
    
%***********************Prior score calculation*********************

centroid_all = [];
for jj = 1:nclass
    class_data = tr_data(tr_label == jj,:);
    centroid_all = [centroid_all; mean(class_data,1)];
end

ts_score_prior = zeros(size(ts_data,1),nclass);

for jj = 1:size(ts_data,1)
   
    n2 = sp_dist2(ts_data(jj,:), centroid_all);
    
    for kk = 1:nclass
        ts_score_prior(jj,kk) = (sum(n2)-n2(kk))/sum(n2);
    end
    ts_score_prior(jj,:) = ts_score_prior(jj,:)./sum(ts_score_prior(jj,:));
    
end

%***********************Calculate combined confidence score**********

ts_score_CS = 0.5*(ts_score_dis+ts_score_prior);

[predict_score_svm_ts,predict_label_svm_ts] =max(ts_score_CS,[],2);

[reject_index,~] = find(predict_score_svm_ts<(1-t_opt));

stage1_predict_label = predict_label_svm_ts;

    

end