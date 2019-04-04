function Func_Converter_face_json2txt( folderName,saveFolderName,frameStart,frameEnd )

mkdir(saveFolderName);
for f=frameStart:frameEnd
    
    saveFileName = sprintf('%s/faceDetectMC_hd%08d.txt',saveFolderName,f);
      if (exist(saveFileName,'file'))
          continue;
      end
    validCnt =0;
    dataArray=cell(31,1);
    for camIdx=0:30
        
        fileName = sprintf('%s/00_%02d/00_%02d_%08d.jpg_l.json',folderName,camIdx,camIdx,f);
        if (exist(fileName,'file') ==0)
            fileName = sprintf('%s/00_%02d_%08d.jpg_l.json',folderName,camIdx,f);    
            if (exist(fileName,'file') ==0) %Checking again
                disp(sprintf('##ERROR: Cannot find the file: %s\n',fileName));
                continue;
            end
        end

        try
            %data = loadjson(fileName);      
            data = gason(fileread(fileName));
        catch
            disp(sprintf('## ERROR in Loading: %s\n',fileName));
            continue;
        end

        for i=1:length(data)
            if(isfield(data(i),'face70')==1 && length(data(i).face70)>0 && sum(isnan(data(i).face70.vertices)==1)==0)
               
                if(isfield(data(i),'id')==1)
                    data(i).face70.id = data(i).id;
                else
                    data(i).face70.id  =-1;
                end
                
                dataArray{camIdx+1}{end+1} = data(i).face70;
            end
        end
        validCnt = validCnt+1;
    end
    
   
   fout = fopen(saveFileName,'wt');
   %fprintf(fout,'%d\n',validCnt);
   fprintf(fout,'31\n');        %Always 31
   
   for c=1:length(dataArray)
       faceNum = length(dataArray{c});
       camIdx = c-1;
       if(faceNum==0)
           continue;
       end
      fprintf(fout,'%d %d %d %d\n',faceNum,camIdx,0,camIdx);  %numOfhands %camIdx %panelIdx %camIdx
      for i=1:faceNum
          landNum=length(dataArray{c}{i}.vertices)/3;     %x y score
          fprintf(fout,'%d %d\n',dataArray{c}{i}.id,landNum);  %numOfhands %camIdx %panelIdx %camIdx
          fprintf(fout,'%f ',dataArray{c}{i}.vertices);  %numOfhands %camIdx %panelIdx %camIdx
          fprintf(fout,'\n');
      end
   end
   fclose(fout);
end


end