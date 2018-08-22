% SIFT + BoW + RO-SVM Script

clear all;close all;clc;

pyramid = [1,2,4];                % spatial block structure for the SPM
knn = 7;                          % # of visual words for each LLC coding 
nRounds = 1;                     % # of experiments

NoFeature = true;                % True: Build BoW model from SIFT features; False: Load saved BoW model;
NoData = true;                   % True: Need to fetch saved BoW features from folders. False: use the saved .mat file to read BoW features. 

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
    pathSplit =  strsplit(fpath,'/');
    className = pathSplit(length(pathSplit)-1);
    feaPath = fullfile(fea_dir, className{1}, [fname '.mat']);
    
 
    fea = LLC_pooling(feaSet, B, pyramid, knn);
    label = database.label(iter1);

    if ~isdir(fullfile(fea_dir, className{1})),
        mkdir(fullfile(fea_dir, className{1}));
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
