function [stage2_predict_ts, stage2_predict_reject] = Stage2_Classification(total_data, total_label, tr_idx, ts_idx, reject_index)

%********Package********************
addpath('liblinear-2.1\liblinear-2.1\matlab'); 


%*******Parameters********************
nclass = length(unique(total_label));

%**********************************
ts_data = total_data(ts_idx,:);
ts_label = total_label(ts_idx, :);
tr_data = total_data(tr_idx,:);
tr_label = total_label(tr_idx,:);
    
options = ['-C -s 2'];
model2 = train(double(tr_label), sparse(tr_data), options);
options = ['-c ',num2str(model2(1)),' -s 2'];
model2 = train(double(tr_label), sparse(tr_data), options);
[stage2_predict_ts] = predict(ts_label, sparse(ts_data), model2);

reject_data = [];
reject_label = [];
for idx = 1:numel(reject_index)
    reject_data = [reject_data total_data(reject_index)];
    reject_label = [reject_label total_label(reject_index)];
end 

[stage2_predict_reject] = predict(double(reject_label), sparse(reject_data), model2); 

end