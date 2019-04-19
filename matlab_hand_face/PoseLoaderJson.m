function [ poseData ] = PoseLoaderJson( folderName,frameStart,frameEnd )

poseData ={};
for f=frameStart:frameEnd
    fileName = sprintf('%s/body3DScene_%08d.json',folderName,f);
    if (exist(fileName,'file') ==0)
        disp(sprintf('##ERROR: Cannot find the file: %s\n',fileName));
        break;
    end    
    data = gason(fileread(fileName));
    if ~iscell(data.bodies)
        data.bodies = num2cell(data.bodies);
    end
    %data = webread(fileName);
    poseData{end+1} = data;
    poseData{end}.frameIdx = f;
    for i=1:length(poseData{end}.bodies);
        if isfield(poseData{end}.bodies{i}, 'joints26')
            num_kp = 26;
            temp = reshape(poseData{end}.bodies{i}.joints26,4,num_kp)';
        else
            assert(isfield(poseData{end}.bodies{i}, 'joints19'));
            num_kp = 19;
            temp = reshape(poseData{end}.bodies{i}.joints19,4,num_kp)';
        end
        poseData{end}.bodies{i}.joints15 = temp(1:num_kp,1:3); %15x3 matrix where each row represents 3D joint location
        poseData{end}.bodies{i}.scores = temp(1:num_kp,4);   %15x1 matrix where each row represents 3D joint score
    end
end
end

