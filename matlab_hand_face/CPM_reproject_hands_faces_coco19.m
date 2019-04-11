%% Setup CPM
edges = [1,2;2,3;3,4;4,5;1,6;6,7;7,8;8,9;1,10;10,11;11,12;,12,13;
    1,14;14,15;15,16;16,17;1,18;18,19;19,20;20,21];
bc = hsv(5);
edge_cols = kron( bc, repmat([0.25, 0.5, 0.75, 1]', 1, 1));

param = matCaffeConfig();
param.octave = 1;
param.modelID = 26;
param.warp_sampling = 'linear';
model = param.model(param.modelID);
param2 = param;
param2.modelID = 11;
param2.octave = 5;
param2.model(param2.modelID).caffemodel = face_caffemodel{1};
param2.model(param2.modelID).deployFile = face_caffemodel{2};
model2 = param2.model(param2.modelID);

param3 = param;
param3.modelID = 24;
param3.octave = 3;
model3 = param3.model(param3.modelID);
%%
caffe.reset_all();

caffe.set_device(gpu_device);
caffe.set_mode_gpu();

net = caffe.Net(model.deployFile, model.caffemodel, 'test');
net2 = caffe.Net(model2.deployFile, model2.caffemodel, 'test');
net3 = caffe.Net(model3.deployFile, model3.caffemodel, 'test');

fprintf('Description of selected model: %s \n', param.model(param.modelID).description);
fprintf('Description of selected model2: %s \n', param2.model(param2.modelID).description);
fprintf('Description of selected model3: %s \n', param3.model(param3.modelID).description);


%% Load calibration data
calibData= loadjson(calibFileName);
cameras = [calibData.cameras{:}];

rand('seed', 1389);


prev_center_r = [];
thresh_hand = 0.05;
thresh_face = 0.15;

indices_j15 = [2 1 10 11 12 4 5 6 13 14 15 7  8  9 ]; % Convert from 3d 15 joints to pm 14
edges_body14 = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
edges70 = [1,2;2,3;3,4;4,5;5,6;6,7;7,8;8,9;9,10;10,11;11,12;12,13;13,14;14,15;15,16;16,17;18,19;19,20;20,21;21,22;23,24;24,25;25,26;26,27;28,29;29,30;30,31;32,33;33,34;34,35;35,36;37,38;38,39;39,40;40,41;41,42;42,37;43,44;44,45;45,46;46,47;47,48;48,43;49,50;50,51;51,52;52,53;53,54;54,55;55,56;56,57;57,58;58,59;59,60;60,49;61,62;62,63;63,64;64,65;65,66;66,67;67,68;68,61];

bc = hsv(5);
edge_cols = kron( bc, repmat([0.25, 0.5, 0.75, 1]', 1, 1));
cols=hsv(21);
fcols=hsv(24);
cam_angle = 0;
clear views;
cnt = 1;
for idc=1:length(nodeIdxs)
    idx = find( [cameras.panel]==panelIdxs(idc) & [cameras.node]==nodeIdxs(idc));
    if isempty(idx)
        continue;
    end
    views(cnt) = cameras(idx);
    cam = views(cnt);
    mkdir(sprintf('%s/json/%02d_%02d', out_path, cam.panel, cam.node));
    cnt = cnt +1;
end

im = zeros(1080,1920,3);
for idc=1:length(views)
    tic
    cam = views(idc);
    cam.M = [cam.R, cam.t(:); 0 0 0 1];
    videoName = sprintf('%s/hd_%d_%d.mp4', videoDir, cam.panel, cam.node);
    vidObj = VideoReader(videoName);

    for idni=1:length(frames)
        idn = frames(idni);
        if idni == 1
            assert(idn > 0);
            vidObj.currentTime = (idn - 1) / 30;  % the HD frame starts from 1, check 'IndexMap25to30_offset.txt'
        end
        do_plot = false;
        do_plot_write = false;
        
        poseData = PoseLoaderJson19(poseDirHD,idn+pose_frame_offset,idn+pose_frame_offset);
    	if isempty(poseData)
    		fprintf('Empty poseData! %d\n', idn+pose_frame_offset);
            continue;
    	end
        poseData = poseData{1};
        
        hdidx = idn;
        fprintf('HD frame: %d', hdidx);

        test_imagen = sprintf('%02d_%02d_%08d.jpg', cam.panel, cam.node, hdidx);
        out_file = sprintf('%s/json/%02d_%02d/%s_l.json',out_path, cam.panel, cam.node, test_imagen);
        try
            data=loadjson((out_file));
            %             data=gason(fileread(out_file));
            fprintf('Found %s, skipping\n', out_file);
            continue;
        catch
        end
        centps = [];
        cam_angles = [];
        lms = [];
        for idp=1:length(poseData.bodies)
            pose2D = PoseProject2D(poseData.bodies{idp}.joints15, cam, true);
            lms = cat(3, lms, pose2D');
        end
        if ~isempty(lms)
            lms = lms(:, [indices_j15, 15, 16:19], :);
            centps = zeros(2, 2, size(lms,3));
            cam_angles = zeros(1, size(lms,3));
        end
        try
            % im = imread(test_image);
            % read a frame from the video
            % im = getFrameFromVideo(videoDir, cam, idn);
            im = readFrame(vidObj);
        catch
            fprintf('Error reading %s\n', test_image);
            im = im*0;
        end
        
        clear joint_entries;
        im_o = im;
        
        if max(im(:))<10
            fprintf('Blank image!\n');
        else
            mi = mean(im(:));
            if mi<120
                im = uint8(im*(128.0/mi));
                im_o = im;
            end
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
                wp3dpm = wp3d(:, [indices_j15, 15,16:19]);
                
                
                head_scale_cm = 1.8*norm( wp3dpm(1:3, ind_head)-wp3dpm(1:3,ind_neck) );
                
                head_center3d = wp3dpm(1:3,ind_head);%mean( wp3dpm(1:3,[ind_head,ind_neck]), 2);
                cam_angle = 0;
                if all(wp3dpm(1,[ind_head,16,18])~=0)
					eyel2r = wp3dpm(:, 18)-wp3dpm(:, 16);
                    cent = mean(wp3dpm(:, [16,18]),2);
                    eyel2r = eyel2r/norm(eyel2r);
                    nosev = -(cent - wp3dpm(1:3, ind_head));
                    nosev = nosev/norm(nosev);
                    forwardup = cross( eyel2r', nosev')';
                    forward = nosev*0.4 + forwardup*0.6;
                    forward = forward/norm(forward);
                    cam_cent = [0;0;0];%-cam.R'*cam.t;
                    cdir = cam_cent - cent;
                    cdir = cdir/norm(cdir);
                    cam_angle = 180/pi*acos(dot(cdir,forward));
                    cam_angles(idper) = cam_angle;
                    camb = cam; camb.R = eye(3); camb.t(:) = 0;
                    centp = PoseProject2D([cent(:), cent(:)+forward*10]', camb, true)';
                    centps(1:2, 1:2, idper) = centp;
				end

				if all(wp3dpm(1,16:19)~=0)
					head_center3d = mean( wp3dpm(1:3, [ind_head, 16:19]), 2);
				end
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
                
                %joint_entry.face = sum( ...
                %    bsxfun(@times, [0.9 0.1], lm(1:2, [ind_head,ind_neck])), 2);
                
                rp = lm(:, [ind_rwrist,ind_lwrist,ind_relb,ind_lelb]);
                rab = (rp(:,1)-rp(:,3))*0.15;
                lab = (rp(:,2)-rp(:,4))*0.15;
                joint_entry.rwrist = rp(1:2,1);
                joint_entry.lwrist = rp(1:2,2);
                joint_entry.rhand = rp(1:2,1)+rab(1:2,1);
                joint_entry.lhand = rp(1:2,2)+lab(1:2,1);
                joint_entry.face = sum( bsxfun(@times, [0.9 0.1], lm(1:2, [ind_head,ind_neck])), 2);
				if all(wp3dpm(1,16:19)~=0)
                	joint_entry.face = sum( bsxfun(@times, ones(1,5)/5, lm(1:2, [ind_head,16:19])), 2);
				end
                joint_entry.scale = (head_scale*1.4)/0.15;
                joint_entry.vertices = lm(:,:);
                joint_entry.vertices = joint_entry.vertices(:)';
                joint_entry.id = poseData.bodies{idper}.id;
                leftMode = 0;
                
                % Right hand
                center = joint_entry.rhand(:)';

                lout = [];
                joint_entry.right_hand = lout;
                
                % Left hand
                center = joint_entry.lhand(:)';
                im_l = im(:, end:-1:1, :);
                center(1) = size(im,2)-center(1);
                
                lout = [];
                joint_entry.left_hand = lout;
                
                height = head_scale*head_factor_face70;
                middle_scale = 368/height;
                
                center = joint_entry.face(:)';
                center(2) = center(2)+20;
                if cam_angle<120 && all(center>0) && center(1)<size(im,2) && center(2)<size(im,1)
                    param2.do_plot_warps = 0;
                    [heatMaps, prediction] = applyModel_giveCenterAndScale(net2,...
                        im, param2, center, middle_scale);
                    pts = zeros(1,3);
                    for idp=1:size(prediction,1),
                        pts(idp,1:3) = prediction(idp,1:3);
                        m = pts(idp,3);
                        if do_plot && m>thresh_face && idp>68
                            im_o  = insertShape(im_o, ...
                                'FilledCircle', [pts(idp,1), pts(idp,2), 2], ...
                                'Color', 'r', 'LineWidth', 1);
                        end
                    end
                    fw = (max(pts(:,1:2))-min(pts(:,1:2)))*2/0.9;
                    if do_plot
                        for ide=1:size(edges70,1)
                            if mean([pts(edges70(ide,1),3), pts(edges70(ide,2),3)])>thresh_face
                                im_o  = insertShape(im_o, ...
                                    'line', [pts(edges70(ide,1),1:2), pts(edges70(ide,2),1:2)], ...
                                    'Color', [0,255,0], 'Opacity', 0.5, 'LineWidth', 1);
                            end
                        end
                    end
                    lout = [];
                    lout.visible = ones(70,1);
                    lout.vertices = [pts]';
                    lout.center = center;
                    lout.height = height;
                    
                    lout.visible = lout.visible(:)';
                    lout.vertices = lout.vertices(:)';
                    lout.annotator = ['CPM_', model2.deployFile];
                else
                    lout = [];
                end
                joint_entry.face70 = lout;
                
                
                
                joint_entry.test_image = test_imagen;
                joint_entries(idper) = joint_entry;
            end
        end
        
        if do_plot
            figure(1);
            hold off;
            imshow(im_o);
            if 1
                hold on;
                if ~isempty(lms)
                    for idper=1:size(lms,3)
                        plot(centps(1,:,idper), centps(2,:,idper), 'm', 'LineWidth', 2);
                        text( centps(1,1,idper), centps(2,1,idper)-15, sprintf('%0.1fº', cam_angles(idper)), 'Color', 'w', 'FontSize', 15);
                    end
                    plot(lms(1,:), lms(2,:), 'w.', 'MarkerSize', 15);
                end
            end
            
            title(sprintf('Frame %d, cam: %02d %02d', hdidx, cam.panel, cam.node));
            drawnow;
        end
        
        if ~exist('joint_entries', 'var')
            joint_entries = [];
        end
        jsonstr=savejson('', joint_entries, ...
            out_file );
        
        if do_plot_write
            imwrite(im_o, sprintf('%s/images/%s_lm.jpg',out_path, test_imagen), 'Quality', 95);
        end
    end
    toc
end
