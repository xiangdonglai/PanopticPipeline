function [heatMaps, prediction, box_in_scale, imageToTest] = applyModel_giveCenterAndScale(net, oriImg, param, center, middle_scale)
compute_maps = false;
%% select model and other parameters from variable param
model = param.model;
model = model(param.modelID);
boxsize = model.boxsize;
np = model.np;

octave = param.octave;

starting_scale = middle_scale * 0.8;
ending_scale = middle_scale * 1.2;
if param.modelID==11
    starting_scale = middle_scale*0.9;
end
%multiplier = 2.^(log2(starting_scale):(1/octave):log2(ending_scale));
multiplier = linspace(starting_scale, ending_scale, octave);
if (octave==1)
  multiplier = middle_scale;
end

%multiplier = middle_scale;
%starting_scale = 1;
%ending_scale = 1;

% data container for each scale
score = cell(1,length(multiplier));
preds = cell(1,length(multiplier));
peakValue = zeros(length(multiplier), np+1);
pad = cell(1, length(multiplier));
ori_size = cell(1, length(multiplier));
box_in_scale = zeros(1, length(multiplier));

% tic
input_data = {[],[],[]};
input_data{1} = single(permute(oriImg(:,:,[3,2,1]),[2,1,3]))/256 - 0.5;
% toc
sz = size(oriImg);
net.blobs('data_full').reshape( [sz([2,1,3]) 1] );
for m = 1:length(multiplier)
  scale = multiplier(m);
%   center
  M_1 = [[eye(2) 0.5-reshape(center(1:2)-1, 2, 1)]; 0 0 1];
  M_2 = [[scale*eye(2), [0;0]]; 0 0 1];
  M_3 = eye(3);
  M_3(1:2,3) = [368/2; 368/2]-0.5;
  M = M_3*M_2*M_1;
%   M
  Mi = inv(M);

  M_1 = [[eye(2) [0.5;0.5]]; 0 0 1];
  M_2 = [[8*eye(2), [0;0]]; 0 0 1];
  M_3 = eye(3);
  M_3(1:2,3) = -0.5;
  M2 = M_3*M_2*M_1;
  M2 = inv(M2);
  input_data{2} = single(reshape(Mi(1:2,1:3)',1,3,2));
  input_data{3} = single(reshape(M2(1:2,1:3)',1,3,2));
   % do forward pass to get scores
  % scores are now Width x Height x Channels x Num
  s_vec = net.forward(input_data);
  score{m} = s_vec{1}; % note this score is transposed
    
%   tic
%   RO = imref2d([368 368], 1, 1);
%   RI = imref2d([size(oriImg,1) size(oriImg,2)], 1, 1);
%   tf = (affine2d(M'));
%   imageToTest = imwarp( oriImg, RI, tf, param.warp_sampling, ...
%     'OutputView', RO, 'FillValues', 0);
%   toc
%   tic
%   imageToTest = cv.warpAffine(oriImg, M(1:2,:), 'DSize', [368,368]);

%   imageToTest = preprocess(imageToTest, 0.5, param);
%   score{m} = (single(applyDNN_full(imageToTest, Mi, net)));
  % score{m} = gpuArray(single(applyDNN(imageToTest, net)));

%   pool_time = 368 / size(score{m},1);

%   tic
%   scorer = imresize(score{m}, pool_time);
%   toc
%   tic
%   nCh = size(score{m},3);
%   for idc=0:4:nCh-1
%     scorer2(:,:,idc+1:min(idc+4, nCh)) = cv.resize(score{m}(:,:,idc+1:min(idc+4, nCh)), [368 368], 'Interpolation', 'Cubic');
%   end
%   toc
%   score{m} = scorer;
  score1 = permute(score{m}, [2 1 3]);
  pts = zeros( size(score{m},3)-1, 3, 'double');
  pts368 = pts;
  for idp=1:size(pts,1)
    [c,I]=max(reshape(score1(:,:,idp),[],1));
    [i,j]=ind2sub(size(score1(:,:,idp)), I);
    pts368(idp, 1:3) =[j,i,c];
    pt = Mi(1:2,1:2)*[j;i] + Mi(1:2, 3);
    pts(idp, 1:3) = [pt(1),pt(2),c];
  end
  
  if param.modelID==26 && sum(pts(:,3)>0.2)>=5
      mx = max(pts(pts(:,3)>0.2,1:2));
      mn = min(pts(pts(:,3)>0.2,1:2));
      center = (mx+mn)/2; % Bounding box centering
%       center = mean(pts(pts(:,3)>0.2,1:2));
  end
  if param.modelID==11 && sum(pts(:,3)>0.2)>=5
      center(1) = mean(pts(pts(:,3)>0.2,1));
  end
  % fprintf('Scale: %f, mean %f\n', scale, mean(pts(:,3)));
  preds{m} = pts;
  
  if compute_maps
    RI = imref2d([368 368], 1, 1);
    RO = imref2d([size(oriImg,1) size(oriImg,2)], 1, 1);
    tf = affine2d(Mi');
    score2 = imwarp( score1, RI, tf, 'OutputView', RO, 'FillValues', 0);
    score{m} = score2;
  end
  if isfield(param, 'do_plot_warps') && param.do_plot_warps ==1
  imz = net.blobs('image').get_data()+0.5;
  imz = imz(:,:,[3,2,1]);
  figure(2);hold off;imshow(permute(imz, [2,1,3]));
  hold on;
  plot(pts368(:,1), pts368(:,2), 'r.');
  drawnow;
  title(sprintf('Score: %.2f',mean(pts(:,3))));
  pause;
  end
  %imwrite(max(score{m}(:,:,1:end-1),[],3), '/media/posefs3b/Users/tsimon/outputs/160422_ultimatum1/testscore.jpg');
  %imwrite(max(score2a(:,:,1:end-1),[],3), '/media/posefs3b/Users/tsimon/outputs/160422_ultimatum1/testscore2.jpg');

  box_in_scale(m) = boxsize / scale;

end

%% make heatmaps into the size of scaled image according to pad
heatMaps = [];
if compute_maps
  final_score = zeros(size(score{1,1}), 'single');
end

final_pred = zeros(size(score{1},3)-1,3);
final_weight = zeros(size(score{1},3)-1,1);
maxScore = -10;
maxInd = 1;
for m = 1:size(score,2)
  mscore = mean(preds{m}(:,3));
  if mscore>maxScore
    maxScore = mscore;
    maxInd = m;
  end
  if compute_maps
    final_score = final_score + score{m};
  end
end

% Average predictions if not too far from bestPred
bestPred = preds{maxInd};
mmult = mean(multiplier);

% for m=1%:size(score,2)
%   dists = sqrt(sum((preds{m}(1:2,:) - bestPred(1:2,:)).^2));
%   w = preds{m}(:,3);
%   w(w<0) = 0;
%   w(dists*mmult>12) = 0;
%   final_pred = final_pred + bsxfun(@times, w, preds{m}(:,1:3));
%   final_weight = final_weight+w;
% end
% final_pred = bsxfun(@times, final_pred, 1./(final_weight+eps));
final_pred = bestPred;
if compute_maps
  final_score = final_score / length(multiplier);
  heatMaps = final_score;
end

prediction = final_pred;

function img_out = preprocess(img, mean, param)
  img_out = single(img)/256;
  img_out = single(img_out) - mean;  
  img_out = permute(img_out, [2 1 3]);
  img_out2 = zeros( size(img_out,1), size(img_out,2), size(img_out,3)+1, 'single');
  img_out2(:,:,1:3) = img_out;
  if(param.modelID >= 5)
    img_out2 = img_out2(:,:,[3 2 1]); % BGR for opencv training in caffe !!!!!
    boxsize = param.model(param.modelID).boxsize;
    centerMapCell = produceCenterLabelMap([boxsize boxsize], boxsize/2, boxsize/2);
    img_out(:,:,4) = centerMapCell{1};
  end

function scores = applyDNN(images, net)
  input_data = {single(images)};
  % do forward pass to get scores
  % scores are now Width x Height x Channels x Num
  s_vec = net.forward(input_data);
  scores = s_vec{1}; % note this score is transposed

function scores = applyDNN_full(images, M, net)
  M_1 = [[eye(2) [0.5;0.5]]; 0 0 1];
  M_2 = [[8*eye(2), [0;0]]; 0 0 1];
  M_3 = eye(3);
  M_3(1:2,3) = -0.5;
  M2 = M_3*M_2*M_1;
  M2 = inv(M2);
  
  input_data = {single(images), single(reshape(M(1:2,1:3)',1,3,2)), single(reshape(M2(1:2,1:3)',1,3,2)) };
  % do forward pass to get scores
  % scores are now Width x Height x Channels x Num
  s_vec = net.forward(input_data);
  scores = s_vec{1}; % note this score is transposed

function label = produceCenterLabelMap(im_size, x, y) %this function is only for center map in testing
  sigma = 21;
  %label{1} = zeros(im_size(1), im_size(2));
  [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
  X = X - x;
  Y = Y - y;
  D2 = X.^2 + Y.^2;
  Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
  label{1} = exp(-Exponent);
