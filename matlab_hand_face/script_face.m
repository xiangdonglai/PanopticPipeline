% Note that Matlab runs such that current working directory is this folder!
addpath('../jsonlab');
addpath('../cocoapi/MatlabAPI/');  % for gason used in 'PoseLoaderJson19', make sure it is already compiled in Matlab

% using about 1.3G GPU memory

% Input argument that should be set;
% seq_name: to be passed in
% calib_name: to be passed in
% processed-path: to be passed in
% frames_start: to be passed in
% frames_end: to be passed in
% e.g.
% seq_name = '190215_uthand2';
% calib_name = '190211_calib_norm';
% processed_path = '/media/posefs11b/Processed/specialEvents/190215_uthand2/';
% frames_start = 223; frames_end = 225;

disp(['seq_name: ', seq_name]);
disp(['processed_path: ', processed_path]);
disp(['calib_name: ', calib_name]);
disp(['frames: ', num2str(frames_start), ' to ', num2str(frames_end)]);

videoDir = sprintf('/media/domedbweb/develop/webdata/dataset/%s/videos/hd_shared_crf20', seq_name);
frames = [frames_start:frames_end];
pose_frame_offset = 0;
poseDirHD = [processed_path, '/body_mpm/coco19_body3DPSRecon_json_normCoord/hd/'];
calibFileName = sprintf('/media/posefs1a/Calibration/%s/calib_norm.json', calib_name);
nodeIdxs = [0:30];
panelIdxs = zeros(1,31);
hand_factor = 1.6;
exp_name = 'exp551b_fv101b';
iter_num = 116000;
method_name = sprintf('%s_%dk', exp_name, ceil(iter_num/1000));
face_caffemodel = {sprintf('/media/posefs3b/Users/tsimon/caffe_model/%s/model/pose_iter_%d.caffemodel', exp_name, iter_num),
	'/media/posefs1b/Users/tsimon/loop/devel/Convolutional-Pose-Machines/model/face_exp550/pose_deploy_imw.prototxt'};
out_path = sprintf('%s/%s/', processed_path, method_name);
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
Func_Converter_face_json2txt(sprintf('%s/json/', out_path), sprintf('%s/facedetect_pm_org/hd_30/', out_path), frames_start, frames_end);
fclose(fopen(sprintf('%s/done_face_2d.log', processed_path), 'a'));
