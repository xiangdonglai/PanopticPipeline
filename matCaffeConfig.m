function param = config()
%% set this part

% CPU mode or GPU mode
param.use_gpu = 1;

% GPU device number (doesn't matter for CPU mode)
GPUdeviceNumber = 3;

% Select model (default: 5)
param.modelID = 7;

% Use click mode or not. If yes (1), you will be asked to click on the center
% of person to be pose-estimated (for multiple people image). If not (0),
% the model will simply be applies on the whole image.
param.click = 1;

% Scaling paramter: starting and ending ratio of person height to image
% height, and number of scales per octave
% warning: setting too small starting value on non-click mode will take
% large memory
param.starting_range = 0.8;
param.ending_range = 1.2;
param.octave = 3;


%% don't edit this part

% path of your caffe
caffepath = textread('/media/posefs1b/Users/tsimon/loop/devel/Convolutional-Pose-Machines/caffePathMatlab.cfg', '%s', 'whitespace', '\n\t\b ');
disp(caffepath);
caffepath = [caffepath{1} '/matlab/'];
addpath(caffepath);
caffe.reset_all();

modelBasePath = '/media/posefs1b/Users/tsimon/loop/devel/Convolutional-Pose-Machines/'

param.model(1).caffemodel = fullfile(modelBasePath, '/model/_trained_LEEDS_OC_304/3S2L/pose_iter_220000.caffemodel');
param.model(1).deployFile = fullfile(modelBasePath, '/model/_trained_LEEDS_OC_304/3S2L/pose_deploy.prototxt');
param.model(1).description = 'LEEDS observer centric 3 stage 2 level';
param.model(1).boxsize = 304;
param.model(1).np = 14;
param.model(1).padValue = 0;
param.model(1).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(1).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};

param.model(2).caffemodel = fullfile(modelBasePath, '/model/_trained_LEEDS_PC_304/3S2L/pose_iter_80000.caffemodel');
param.model(2).deployFile = fullfile(modelBasePath, '/model/_trained_LEEDS_PC_304/3S2L/pose_deploy.prototxt');
param.model(2).description = 'LEEDS person centric 3 stage 2 level';
param.model(2).boxsize = 304;
param.model(2).np = 14;
param.model(2).padValue = 0;
param.model(2).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(2).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};

param.model(3).caffemodel = fullfile(modelBasePath, '/model/_trained_FLIC_304/3S2L/pose_iter_65000.caffemodel');
param.model(3).deployFile = fullfile(modelBasePath, '/model/_trained_FLIC_304/3S2L/pose_deploy.prototxt');
param.model(3).description = 'FLI1C observer centric 3 stage 2 level';
param.model(3).boxsize = 304;
param.model(3).np = 9;
param.model(3).padValue = 0;
param.model(3).limbs = [2 3; 3 4; 5 6; 6 7];
param.model(3).part_str = {'head', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Lhip', 'bkg'};

param.model(4).caffemodel = fullfile(modelBasePath, '/model/_trained_MPI/pose_iter_500000.caffemodel');
param.model(4).deployFile = fullfile(modelBasePath, '/model/_trained_MPI/pose_deploy.prototxt');
param.model(4).description = 'MPII 3 stage 2 level';
param.model(4).boxsize = 304;
param.model(4).np = 14;
param.model(4).padValue = 0;
param.model(4).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(4).part_str = {'head', 'neck', 'Rsho', 'Right Elbow', 'Right Wrist', ...
                         'Lsho', 'Left Elbow', 'Left Wrist', ...
                         'Rhip', 'Right Knee', 'Right Ankle', ...
                         'Lhip', 'Left Knee', 'Light Ankle', 'bkg'};

param.model(5).caffemodel = fullfile(modelBasePath, '/model/_trained_MPI/pose_iter_630000.caffemodel');
param.model(5).deployFile = fullfile(modelBasePath, '/model/_trained_MPI/pose_deploy_centerMap.prototxt');
param.model(5).description = 'MPII 6 stage L level';
param.model(5).boxsize = 368;
param.model(5).padValue = 128;
param.model(5).np = 14;
param.model(5).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(5).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
param.model(6).caffemodel = fullfile(modelBasePath, '/model/face/pose_iter_21238.caffemodel');
param.model(6).deployFile = fullfile(modelBasePath, '/model/face/pose_deploy.prototxt');
param.model(6).description = 'face';
param.model(6).boxsize = 368;
param.model(6).padValue = 128;
param.model(6).np = 68;
param.model(6).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(6).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
ind = 6+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face2/pose_iter_24000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face2/pose_deploy.prototxt');
param.model(ind).description = 'face';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 68;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/hands/pose_iter_26000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/hands/pose_deploy.prototxt');
param.model(ind).description = 'hands';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 21;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_eg/pose_iter_50000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_eg/pose_deploy.prototxt');
param.model(ind).description = 'face_eg';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 70;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/hands_ue/pose_iter_43000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/hands_ue/pose_deploy.prototxt');
param.model(ind).description = 'hands';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 21;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_eg2/pose_iter_39194.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_eg2/pose_deploy_imw.prototxt');
param.model(ind).description = 'face_eg2';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 70;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/hands_ue_mask/pose_iter_60000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/hands_ue_mask/pose_deploy.prototxt');
param.model(ind).description = 'hands_ue_mask';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 21;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);

ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/hands_ue2/pose_iter_66000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/hands_ue2/pose_deploy.prototxt');
param.model(ind).description = 'hands_ue2';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 21;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_eg3/pose_iter_30000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_eg3/pose_deploy.prototxt');
param.model(ind).description = 'face_eg3';
param.model(ind).boxsize = 440;
param.model(ind).padValue = 128;
param.model(ind).np = 70;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_eg_243/pose_iter_70000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_eg_243/pose_deploy_imw.prototxt');
param.model(ind).description = 'face_eg_243';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 70;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);

ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/hands_v5/pose_iter_142255.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/hands_v5/pose_deploy.prototxt');
param.model(ind).description = 'hands_v5';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 21;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};

fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);

ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/hands_v5_246/pose_iter_40000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/hands_v5_246/pose_deploy.prototxt');
param.model(ind).description = 'hands_v5';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 21;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};

fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_eg_248/pose_iter_125000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_eg_248/pose_deploy.prototxt');
param.model(ind).description = 'face_eg_248';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 70;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_eg_254/pose_iter_157000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_eg_254/pose_deploy.prototxt');
param.model(ind).description = 'face_eg_254';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 70;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_eg_256/pose_iter_56423.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_eg_256/pose_deploy.prototxt');
param.model(ind).description = 'face_eg_256';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 70;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);
ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_eg_257/pose_iter_44635.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_eg_257/pose_deploy.prototxt');
param.model(ind).description = 'face_eg_257';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 70;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);

ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_eg_258/pose_iter_56739.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_eg_258/pose_deploy.prototxt');
param.model(ind).description = 'face_eg_258';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 70;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);

ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/hands_v5_259/pose_iter_65000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/hands_v5_259/pose_deploy.prototxt');
param.model(ind).description = 'hands_v5_259';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 21;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};

fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);

ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_coco_265/pose_iter_4000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_coco_265/pose_deploy_imw.prototxt');
param.model(ind).description = 'face_coco_265';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 24;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};

fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);


ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/face_coco_bb_268/pose_iter_144223.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/face_coco_bb_268/pose_deploy.prototxt');
param.model(ind).description = 'face_coco_265';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 24+9;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};

fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);

ind = ind+1;
param.model(ind).caffemodel = fullfile(modelBasePath, '/model/hands_v6_271/pose_iter_192000.caffemodel');
param.model(ind).deployFile = fullfile(modelBasePath, '/model/hands_v6_271/pose_deploy.prototxt');
param.model(ind).description = 'hands_v6_271';
param.model(ind).boxsize = 368;
param.model(ind).padValue = 128;
param.model(ind).np = 21;
param.model(ind).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(ind).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};

fprintf('%s -> %d\n', regexprep(param.model(ind).caffemodel, '.*/model/',''), ind);



if(param.modelID >= 5)
    param.click = 1;
end
