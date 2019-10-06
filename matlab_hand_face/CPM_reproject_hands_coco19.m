%% Setup CPM

% [gpu_device, hand_iteration_model] is set by the the caller script;
param = matCaffeConfig();
param.modelID = 26;
new_caffemodel = regexprep(param.model(param.modelID).caffemodel, ...
    '/model/hands_v6_271/pose_iter_192000.caffemodel', ...
    hand_iteration_model);
param.model(param.modelID).caffemodel = new_caffemodel;
param.model(param.modelID).deployFile = ...
    regexprep(new_caffemodel, 'pose_iter.*', ...
              'pose_deploy.prototxt');

param.warp_sampling = 'linear';
model = param.model(param.modelID);
caffe.reset_all();
caffe.set_device(gpu_device);
caffe.set_mode_gpu();
net = caffe.Net(model.deployFile, model.caffemodel, 'test');
fprintf('Description of selected model: %s \n', param.model(param.modelID).caffemodel);

%% Load calibration data
calibData= loadjson(calibFileName);
cameras = [calibData.cameras{:}];

rand('seed', 1389);

prev_center_r = [];
thresh_hand = 0.05;
indices_j15 = [2 1 10 11 12 4 5 6 13 14 15 7  8  9 ]; % Convert from 3d 15 joints to pm 14
edges_body14 = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
bc = hsv(5);
edge_cols = kron( bc, repmat([0.25, 0.5, 0.75, 1]', 1, 1));
cols=hsv(21);
edges = [1,2;2,3;3,4;4,5;1,6;6,7;7,8;8,9;1,10;10,11;11,12;12,13;
    1,14;14,15;15,16;16,17;1,18;18,19;19,20;20,21];
bc = hsv(5);
edge_cols = kron( bc, repmat([0.25, 0.5, 0.75, 1]', 1, 1));

clear views;
cnt =1;
for idc=1:length(nodeIdxs)
    idx = find( [cameras.panel]==panelIdxs(idc) & [cameras.node]==nodeIdxs(idc));
    if isempty(idx)
        continue;
    end
    views(cnt) = cameras(idx);
    cam = views(cnt);
    mkdir(sprintf('%s/json/%02d_%02d', out_path, cam.panel, cam.node));
    cnt = cnt+1;
end

poseDataBuffer = {};

for idc=1:length(views)
    tic
    cam = views(idc);
    cam.M = [cam.R, cam.t(:); 0 0 0 1];
    videoName = sprintf('%s/hd_%02d_%02d.mp4', videoDir, cam.panel, cam.node);
    if ~isfile(videoName)
        continue;
    end
    vidObj = VideoReader(videoName);
    
    % add 1 here: the video index read by Matlab starts from 1, our image/skeleton index starts from 0
    % imgs = read(vidObj, [frames_start+1, frames_end+1]);

    for idni=1:length(frames)
        idn = frames(idni);

        do_plot = false;
    	do_plot_write = false;

        if idc == 1
            poseData = PoseLoaderJson(poseDirHD,idn+pose_frame_offset,idn+pose_frame_offset);
        	if isempty(poseData)
        		fprintf('Empty poseData! %d\n', idn+pose_frame_offset);
                continue;
        	end
            poseData = poseData{1};
            poseDataBuffer{idni} = poseData;
        else
            if idni <= length(poseDataBuffer) && ~isempty(poseDataBuffer{idni})
                poseData = poseDataBuffer{idni};
            else
                continue;
            end
        end
        
        hdidx = idn;
        fprintf('HD frame: %d\n', hdidx);

        test_imagen = sprintf('%02d_%02d_%08d.jpg', cam.panel, cam.node, hdidx);
        
        out_file = sprintf('%s/json/%02d_%02d/%s_l.json',out_path, cam.panel, cam.node, test_imagen);
        try
            data=loadjson((out_file));
            fprintf('Found %s, skipping\n', out_file);
            continue;
        catch
        end

        lms = [];
        for idp=1:length(poseData.bodies)
            pose2D = PoseProject2D(poseData.bodies{idp}.joints15, cam, true);
            lms = cat(3, lms, pose2D');
        end

        if ~isempty(lms)
            lms = lms(:, indices_j15, :);
        end
        im = read(vidObj, idn + 1);
        
        % imshow(im);
        % hold on;
        % scatter(lms(1, :, 1), lms(2, :, 1), 'b');
        % scatter(lms(1, :, 2), lms(2, :, 2), 'r');
        
        clear joint_entries;
        im_o = im;
        
        if max(im(:))<20
            fprintf('Blank image!\n');
        else
            for idper=1:length(poseData.bodies)
                clear joint_entry;
                lm = lms(:,:,idper);
                
                ind_rwrist = 5;
                ind_relb = 4;
                ind_lelb = 7;
                ind_lwrist = 8;
                ind_head = 1;
                ind_neck = 2;
                
                
                % Find 3D scale by projection. This isn't great.
                j15 = reshape(poseData.bodies{idper}.joints15', 3, []);
                j15(4,:) = poseData.bodies{idper}.scores;
                p3d = j15(1:3,:);
                wp3d = bsxfun(@plus, cam.R*p3d, cam.t(:));
                wp3dpm = wp3d(:, indices_j15);
                head_scale_cm = norm( wp3dpm(1:3, ind_head)-wp3dpm(1:3,ind_neck) );
				head_scale_cm = 25;
                
                head_center3d = mean( wp3dpm(1:3,[ind_head,ind_neck]), 2);
                head_scale = (cam.K(2,2))*head_scale_cm/abs(head_center3d(3))*0.7;
                
                joint_entry.head_scale = head_scale;
                
                % Find hand center in 3d, then project.
                rp = wp3dpm(:, [ind_rwrist,ind_lwrist,ind_relb,ind_lelb]);
                rab = rp(:,1)+(rp(:,1)-rp(:,3))*0.15;
                lab = rp(:,2)+(rp(:,2)-rp(:,4))*0.15;
                rp = [rp, rab, lab];
                rp = bsxfun(@rdivide, rp(1:2,:), rp(3,:));
                rpp = bsxfun(@plus, cam.K(1:2,1:2)*rp, cam.K(1:2,3));
                
                joint_entry.rwrist = rpp(1:2,1);
                joint_entry.lwrist = rpp(1:2,2);
                joint_entry.rhand = rpp(1:2,5);
                joint_entry.lhand = rpp(1:2,6);
                
                joint_entry.rhand_scale = (cam.K(2,2))*head_scale_cm/abs(rab(3))*0.7;
                joint_entry.lhand_scale = (cam.K(2,2))*head_scale_cm/abs(lab(3))*0.7;
                
                joint_entry.face = sum( ...
                    bsxfun(@times, [0.4 0.6], lm(1:2, [ind_head,ind_neck])), 2);
                
                rp = lm(:, [ind_rwrist,ind_lwrist,ind_relb,ind_lelb]);
                rab = (rp(:,1)-rp(:,3))*0.15;
                lab = (rp(:,2)-rp(:,4))*0.15;
                joint_entry.rwrist = rp(1:2,1);
                joint_entry.lwrist = rp(1:2,2);
                joint_entry.rhand = rp(1:2,1)+rab(1:2,1);
                joint_entry.lhand = rp(1:2,2)+lab(1:2,1);
                joint_entry.face = sum( bsxfun(@times, [0.4 0.6], lm(1:2, [ind_head,ind_neck])), 2);
                joint_entry.scale = (head_scale*1.4)/0.15;
                joint_entry.vertices = lm(:,:);
                joint_entry.vertices = joint_entry.vertices(:)';
                joint_entry.id = poseData.bodies{idper}.id;
                leftMode = 0;
                
                % Right hand
                center = joint_entry.rhand(:)';
                if all(center>0) && center(1)<size(im,2) && center(2)<size(im,1)
                    
                    height = joint_entry.rhand_scale*hand_factor;
                    middle_scale = 368/height;
                    
                    [heatMaps, prediction] = applyModel_giveCenterAndScale(net,im, param, center, middle_scale);
                    
                    if do_plot
                        for idp=1:8
                            im_o  = insertShape(im_o, 'FilledCircle', [lm(1,idp), lm(2,idp), 2], ...
                                'Color', [255, 0, 0], 'LineWidth', 1);
                        end
                        pts = lm';
                        for ide=2:5
                            im_o  = insertShape(im_o, 'line', ...
                                [pts(edges_body14(ide,1),1:2), pts(edges_body14(ide,2),1:2)], ...
                                'Color', [255,0,0], 'Opacity', 0.25, 'LineWidth', 2);
                        end
                    end
                    
                    pts = zeros(21,3);
                    for idp=1:21,
                        pts(idp,1:3) = prediction(idp,1:3);
                        m = pts(idp,3);
                        if do_plot && m>thresh_hand
                            im_o  = insertShape(im_o, 'FilledCircle', [pts(idp,1:2), 2], ...
                                'Color', cols(idp,:), 'LineWidth', 1);
                        end
                    end
                    if do_plot
                        for ide=1:size(edges,1)
                            if mean([pts(edges(ide,1),3), pts(edges(ide,2),3)])>thresh_hand
                                im_o  = insertShape(im_o, 'line', ...
                                    [pts(edges(ide,1),1:2), pts(edges(ide,2),1:2)], ...
                                    'Color', edge_cols(ide,:)*255, 'Opacity', 0.5, 'LineWidth', 1);
                            end
                        end
                    end
                    %lout = joint_entry;
                    lout = [];
                    lout.visible = ones(21,1);
                    lout.vertices = [pts]';
                    lout.center = center;
                    lout.height = height;
                    lout.visible = lout.visible(:)';
                    lout.vertices = lout.vertices(:)';
                    lout.annotator = ['CPM_', model.caffemodel];
                else
                    % center outside of image bounds
                    lout = [];
                end
                joint_entry.right_hand = lout;
                
                % Left hand
                center = joint_entry.lhand(:)';
                im_l = im(:, end:-1:1, :);
                center(1) = size(im,2)-center(1);
                
                if all(center>0) && center(1)<size(im,2) && center(2)<size(im,1)
                    height = joint_entry.lhand_scale*hand_factor;
                    middle_scale = 368/height;
                    
                    [heatMaps, prediction] = applyModel_giveCenterAndScale(net,im_l, param, center, middle_scale);
                    prediction(:,1) = size(im,2)-prediction(:,1)+1;
                    heatMaps = heatMaps(:,end:-1:1,:);
                    center(1) = size(im,2)-center(1);
                    
                    pts = lm';
                    
                    
                    pts = zeros(21,3);
                    for idp=1:21,
                        pts(idp,1:3) = prediction(idp,1:3);
                        m = pts(idp,3);
                        if do_plot && m>thresh_hand
                            im_o  = insertShape(im_o, 'FilledCircle', [pts(idp,1:2), 2], ...
                                'Color', cols(idp,:), 'LineWidth', 1);
                        end
                    end
                    if do_plot
                        for ide=1:size(edges,1)
                            if mean([pts(edges(ide,1),3), pts(edges(ide,2),3)])>thresh_hand
                                im_o  = insertShape(im_o, 'line', ...
                                    [pts(edges(ide,1),1:2), pts(edges(ide,2),1:2)], ...
                                    'Color', edge_cols(ide,:)*255, 'Opacity', 0.5, 'LineWidth', 1);
                            end
                        end
                    end
                    %lout = joint_entry;
                    lout = [];
                    lout.visible = ones(21,1);
                    lout.vertices = [pts]';
                    lout.center = center;
                    lout.height = height;
                    lout.visible = lout.visible(:)';
                    lout.vertices = lout.vertices(:)';
                    lout.annotator = ['CPM_', model.caffemodel];
                else
                    % center outside of image bounds
                    lout = [];
                end
                joint_entry.left_hand = lout;
                
                
                joint_entry.test_image = test_imagen;
                joint_entries(idper) = joint_entry;
            end
        end
        
        if ~exist('joint_entries', 'var')
            joint_entries = struct;
        end
        jsonstr=savejson('', joint_entries, ...
            out_file );
    end
    toc
end
