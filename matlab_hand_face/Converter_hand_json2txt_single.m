function [ Subject ] = Converter_hand_json2txt_single( folderName,saveFolderName,frameStart,frameEnd,numCam)
%POSELOADERJSON Summary of this function goes here

mkdir(saveFolderName);
Elapsedtime =0;
tic;
actualStartFrame =-1;

for f=frameStart:frameEnd
    saveFileName = sprintf('%s/handDetectMC_hd%08d.txt',saveFolderName,f);

    if (exist(saveFileName,'file'))
      continue;
    end
    if actualStartFrame <0
        actualStartFrame  = f;
    end
    
    if mod(f,10)==0 || f==frameStart
        elapsedTime = toc;
        remainingFrames = frameEnd-f;
        remainingTime = remainingFrames/(f-actualStartFrame) * elapsedTime;
        fprintf(2,sprintf('Save to %s\n',saveFileName));
        fprintf(2,sprintf('RemainingTime: %f min\n',remainingTime/60));        
    end
    
    validCnt =0;
    dataArray=cell(31,1);
    for camIdx=0:(numCam-1)%10%30
        fileName = sprintf('%s/00_%02d/00_%02d_%08d.jpg_l.json',folderName,camIdx,camIdx,f);
        if (exist(fileName,'file') ==0)
            fileName = sprintf('%s/00_%02d_%08d.jpg_l.json',folderName,camIdx,f);
            if (exist(fileName,'file') ==0) %Checking again
                disp(sprintf('##ERROR: Cannot find the file: %s\n',fileName));
                continue;
            end
            %return;
        end
        try
            data = loadjson(fileName);      
        catch
            disp(sprintf('## ERROR in Loading: %s\n',fileName));
            continue;
            %return;
        end
        if(length(data)==1)
            data = {data};
        end
        for i=1:length(data)
            if(isfield(data{i},'left_hand')==1 && length(data{i}.left_hand)>0 && sum(isnan(data{i}.left_hand.vertices)==1)==0)
                data{i}.left_hand.whichside = 'l'; 
                if(isfield(data{i},'id')==1)
                    data{i}.left_hand.id = data{i}.id;
                else
                    data{i}.left_hand.id  =-1;
                end
                
                dataArray{camIdx+1}{end+1} = data{i}.left_hand;
            end
            if(isfield(data{i},'right_hand')==1 && length(data{i}.right_hand)>0 && sum(isnan(data{i}.right_hand.vertices)==1)==0)
                data{i}.right_hand.whichside = 'r';
                if(isfield(data{i},'id')==1)
                    data{i}.right_hand.id = data{i}.id;
                else
                    data{i}.right_hand.id  =-1;
                end
                dataArray{camIdx+1}{end+1} = data{i}.right_hand;
            end
        end
        validCnt = validCnt+1;
   end
   
   fout = fopen(saveFileName,'wt');
   if(fout<0)
       saveFileName
       disp('Failed in saving data');
       return;
   end
   fprintf(fout,'%d\n',numCam);
   
   for c=1:length(dataArray)
       handNum = length(dataArray{c});
       camIdx = c-1;
       if(handNum==0)
           continue;
       end
      fprintf(fout,'%d %d %d %d\n',handNum,camIdx,0,camIdx);  %numOfhands %camIdx %panelIdx %camIdx
      for i=1:handNum
          landNum=length(dataArray{c}{i}.vertices)/3;     %x y score
          fprintf(fout,'%c %d %d\n',dataArray{c}{i}.whichside,dataArray{c}{i}.id,landNum);  %numOfhands %camIdx %panelIdx %camIdx
          fprintf(fout,'%f ',dataArray{c}{i}.vertices);  %numOfhands %camIdx %panelIdx %camIdx
          fprintf(fout,'\n');
      end
   end
   fclose(fout);
end

end