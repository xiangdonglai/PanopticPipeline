% Dataset Path
addpath('./jsonlab');

seq_name = '190215_uthand2';
frames = [223:225];
poseDirHD = [seq_name '/hdPose3d_stage1_coco19'];
calibFileName = '/media/posefs1a/Calibration/190211_calib_norm/calib_norm.json';
nodeIdxs = [0:30];
panelIdxs = zeros(1,31);
hand_factor = 1.6;
exp_name = 'exp551b_fv101b';
iter_num = 116000;
method_name = sprintf('%s_%dk', exp_name, ceil(iter_num/1000));
face_caffemodel = { sprintf('/media/posefs3b/Users/tsimon/caffe_model/%s/model/pose_iter_%d.caffemodel', exp_name, iter_num),
	'/media/posefs1b/Users/tsimon/loop/devel/Convolutional-Pose-Machines/model/face_exp550/pose_deploy_imw.prototxt'};
out_path = sprintf('/media/posefs3b/Users/tsimon/outputs/hands/%s/%s/', seq_name, method_name);
mkdir(out_path);
mkdir(fullfile(out_path, 'json'));
mkdir(fullfile(out_path, 'images'));
calibData = loadjson(calibFileName);
cameras = [calibData.cameras{:}];
hand_factor = 1.6;
head_factor_coco = 2;
head_factor_face70 = 1.65;
gpu_device = 0;
CPM_reproject_hands_faces_coco19;
