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
% pose_folder = 'coco19_body3DPSRecon_json_normCoord' or 'op25_body3DPSRecon_json_normCoord'
% e.g.
% seq_name = '190215_uthand2';
% calib_name = '190211_calib_norm';
% processed_path = '/media/posefs11b/Processed/specialEvents/190215_uthand2/';
% frames_start = 223; frames_end = 225;

disp(['seq_name: ', seq_name]);
disp(['processed_path: ', processed_path]);
disp(['calib_name: ', calib_name]);
disp(['frames: ', num2str(frames_start), ' to ', num2str(frames_end)]);
disp(['pose_folder: ', pose_folder]);

videoDir = sprintf('/media/domedbweb/develop/webdata/dataset/%s/videos/hd_shared_crf20', seq_name);
frames = [frames_start:frames_end];
pose_frame_offset = 0;
poseDirHD = sprintf('%s/body_mpm/%s/hd/', processed_path, pose_folder);
calibFileName = sprintf('/media/posefs1a/Calibration/%s/calib_norm.json', calib_name);
nodeIdxs = [0:30];
panelIdxs = zeros(1,31);
hand_factor = 1.6;
hand_iteration_model = '/model/hands_v143_noft/pose_iter_120000.caffemodel';
out_path = sprintf('%s/hands_v143_120k/', processed_path);
mkdir(out_path);
mkdir(fullfile(out_path, 'json'));
mkdir(fullfile(out_path, 'images'));
gpu_device = 0;   % TODO: change this!
CPM_reproject_hands_coco19; 
Converter_hand_json2txt_single(sprintf('%s/json/', out_path), sprintf('%s/handdetect_pm_org/hd_30/', out_path), frames_start, frames_end, 31);
fclose(fopen(sprintf('%s/done_hand_2d.log', processed_path), 'a'));
