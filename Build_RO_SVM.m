function [t_opt model] = Build_RO_SVM(tr_data, tr_label, tr_without_v_data, tr_without_v_label, v_data, v_label, nclass, rejectionRate_thr)

%*******Package****************
addpath('liblinear-2.1\liblinear-2.1\matlab'); 


%*******************discriminative score calculation**********************

    options = ['-C -s 2'];
    model = train(double(tr_without_v_label), sparse(tr_without_v_data), options);
    options = ['-c ',num2str(model(1)),' -s 2'];
    model = train(double(tr_without_v_label), sparse(tr_without_v_data), options);
    [C] = predict(v_label, sparse(v_data), model);
    
    
    score_acc = [];
    for iter = 1:size(model.w,1)

        w = model.w(iter,:);

        score = v_data*transpose(w)/norm(w);
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
    
    v_score_dis = score_svm;
    
%***********************Prior score calculation*********************

centroid_all = [];
for jj = 1:nclass
    class_data = tr_data(tr_label == jj,:);
    centroid_all = [centroid_all; mean(class_data,1)];
end

v_score_prior = zeros(size(v_data,1),nclass);

for jj = 1:size(v_data,1)
   
    n2 = sp_dist2(v_data(jj,:), centroid_all);
    
    for kk = 1:nclass
        v_score_prior(jj,kk) = (sum(n2)-n2(kk))/sum(n2);
    end
    v_score_prior(jj,:) = v_score_prior(jj,:)./sum(v_score_prior(jj,:));
    
end

%***********************Calculate combined confidence score**********

v_score_CS = 0.5*(v_score_dis+v_score_prior);
% v_score_CS = v_score_dis;

%***********************Calculate RO-SVM****************************

[predict_score_svm_v,predict_label_svm_v] =max(v_score_CS,[],2);
acc_svm = length(find(predict_label_svm_v==v_label))/length(v_label);

acc_svm_t_accumulate_v = [];
reject_svm_t_accumulate_v = [];
error_svm_t_accumulate_v = [];
gndacc_svm_t_accumulate_v = [];

T = 0:0.05:1;
T = T';
parfor iterpar = 1:length(T) 
    
    t = T(iterpar);

 
[reject_index,~] = find(predict_score_svm_v<(1-t));
predict_label_svm_temp = predict_label_svm_v;
ts_label_temp = v_label;
%  tr_label_temp = tr_label;

predict_label_svm_temp(reject_index) =[];
ts_label_temp(reject_index) =[];
%  tr_label_temp(index) =[];

if length(ts_label_temp)~=0
% if length(tr_label_temp)~-0

acc_svm_t = length(find(predict_label_svm_temp==ts_label_temp))/length(v_label);
error_svm_t = length(find(predict_label_svm_temp~=ts_label_temp))/length(v_label);

reject_svm_t = length(reject_index)/length(v_label);

% acc_svm_t = length(find(predict_label_svm_temp==tr_label_temp))/length(tr_label_temp);
% error_svm_t = length(find(predict_label_svm_temp~=tr_label_temp))/length(tr_label_temp);
% 
% reject_svm_t = length(index)/length(tr_label);

acc_svm_t_accumulate_v = [acc_svm_t_accumulate_v;acc_svm_t];
reject_svm_t_accumulate_v = [reject_svm_t_accumulate_v;reject_svm_t];
error_svm_t_accumulate_v = [error_svm_t_accumulate_v;error_svm_t];

else
  acc_svm_t_accumulate_v = [acc_svm_t_accumulate_v;1];
reject_svm_t_accumulate_v = [reject_svm_t_accumulate_v;1];
error_svm_t_accumulate_v = [error_svm_t_accumulate_v;0];  

end
gndacc_svm_t_accumulate_v = [gndacc_svm_t_accumulate_v;acc_svm];
end

% B = [T error_svm_t_accumulate_v reject_svm_t_accumulate_v 1-gndacc_svm_t_accumulate_v];


%*******************************************************************************
% a = find(error_svm_t_accumulate_v==0);
% b = find(error_svm_t_accumulate_v == (1-gndacc_svm_t_accumulate_v(1)));
% 
% y1 = reject_svm_t_accumulate_v(a(end));
% y2 = 0;

%*****************************Determine the threshold*******
a = find(reject_svm_t_accumulate_v>rejectionRate_thr);
b = find(reject_svm_t_accumulate_v<rejectionRate_thr);

y1 = reject_svm_t_accumulate_v(a(end));
x1 = T(a(end));
y2 = reject_svm_t_accumulate_v(b(1));
x2 = T(b(1));

t_opt = x2-(x2-x1)*(y1-rejectionRate_thr)/(y1-y2);
% t_opt = (x1+x2)/2;
options = ['-C -s 2'];
model = train(double(tr_label), sparse(tr_data), options);
options = ['-c ',num2str(model(1)),' -s 2'];
model = train(double(tr_label), sparse(tr_data), options);
end