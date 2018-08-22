function [total_data, total_label, nclass, fdatabase_cnn] = load_cnn_features(database, cnn_fea_dir)
	nFea = length(database.path);       % Number of images
	fdatabase_cnn = struct;
	fdatabase_cnn.path = cell(nFea, 1);         % path for each image feature
	fdatabase_cnn.label = zeros(nFea, 1);       % class label for each image feature


	for iter1 = 1: nFea
		if ~mod(iter1, 5),
			fprintf('.');
		end
		if ~mod(iter1, 100),
			fprintf(' %d images processed\n', iter1);
		end
		
		fpath = database.path{iter1};
		flabel = database.label(iter1);
		[rtpath, fname] = fileparts(fpath);
		pathSplit =  strsplit(fpath,'/');
	    className = pathSplit(length(pathSplit)-1);
	    feaPath = fullfile(cnn_fea_dir, className{1}, [fname '.mat']);
        
		fdatabase_cnn.label(iter1) = flabel;
		fdatabase_cnn.path{iter1} = feaPath;
	end 
	save('FeaInfo_cnn_PAP.mat','fdatabase_cnn','fdatabase_cnn');

	clabel = unique(fdatabase_cnn.label);   % Class labels
	nclass = length(clabel);            % # of classes
	%**************Start experiments**************************
		total_data = [];
		total_label = [];

	for jj = 1:nclass
		idx_label = find(fdatabase_cnn.label == clabel(jj));

		for kk = 1:length(idx_label)
			fpath = fdatabase_cnn.path{idx_label(kk)};
			label = fdatabase_cnn.label(idx_label(kk));
			load(fpath);
			total_data = [total_data; features];
			total_label = [total_label; label];
		end 
	end
end 