function [total_data, total_label, nclass, clabel, fdatabase_bow] = load_bow_features(database, bow_fea_dir, genFeature, genData, Bpath, dataset)

%******Param settings***************
knn = 5;
pyramid = [1,2,4];

%******Load a BoW dictionary**********************
load(Bpath);
B = dictionary';
nCodebook = size(B, 2);


dFea = sum(nCodebook*pyramid.^2);   %dimensionality of BoW features
nFea = length(database.path);       % Number of images

%*********BoW feature building************************

fdatabase_bow = struct;
fdatabase_bow.path = cell(nFea, 1);         % path for each image feature
fdatabase_bow.label = zeros(nFea, 1);       % class label for each image feature

if(genFeature) % True: Build BoW model from SIFT features; False: Load saved BoW model;
    
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
	    pathSplit =  strsplit(fpath,'/');
	    className = pathSplit(length(pathSplit)-1);
	    feaPath = fullfile(bow_fea_dir, className{1}, [fname '.mat']);
	    
	 
	    fea = LLC_pooling(feaSet, B, pyramid, knn);
	    label = database.label(iter1);
	    if ~isdir(fullfile(bow_fea_dir, className{1})),
		mkdir(fullfile(bow_fea_dir, className{1}));
		
	    end      
	    save(feaPath, 'fea', 'label');
        
        fdatabase_bow.label(iter1) = flabel;
        fdatabase_bow.path{iter1} = feaPath;
    end
    
    save(strcat('FeaInfo_bow_',dataset, '.mat'),'fdatabase_bow','fdatabase_bow');
else
    load(strcat('FeaInfo_bow_',dataset, '.mat'));
end

clabel = unique(fdatabase_bow.label);   % Class labels
nclass = length(clabel);            % # of classes
%**************Start experiments**************************
if (genData)  % True: Need to fetch saved BoW features from folders. False: use the saved .mat file to read BoW features.
    
    total_data = [];
    total_label = [];
    
    for jj = 1:nclass
        idx_label = find(fdatabase_bow.label == clabel(jj));
        
        for kk = 1:length(idx_label)
            fpath = fdatabase_bow.path{idx_label(kk)};
            load(fpath);
            total_data = [total_data; fea'];
            total_label = [total_label;label];
        end
    end
    
    save ('PAP_data.mat','total_data','total_label');
else
    load ('PAP_data.mat');
end

end