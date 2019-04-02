outputFolder=/media/domedbweb/develop/webdata/dataset
machineIdx=31
diskIdx=2


if [ $# -eq 0 ]; then
    #no parameter is given
    inputNasIdx=3
    intputCategory=""
    inputDataName=""
else
    inputNasIdx=$1
    inputCategory=$2
    inputDataName=$3
fi


if [ $# -eq 5 ]; then
    machineIdx=$4
    diskIdx=$5
elif [ $# -eq 6 ]; then
    machineIdx=$4
    diskIdx=$5
    outputPathRoot=$6
fi


for nasIdx in $1
do
captureFolder=$(printf "/media/posefs%s/Captures" $nasIdx )
#outputFolder_nas=$(printf "$outputFolder/posefs%da" $nasIdx )
outputFolder_nas=$outputFolder
[ -d $outputFolder_nas ] || mkdir -p $outputFolder_nas
  
for categoryFolder in $captureFolder/$2*
do
echo $categoryFolder

for dataPath in $categoryFolder/$3*
do
  
  fileName=$(basename $dataPath)
  echo ""
  echo DataPath: $dataPath;
  echo DatasetName: $fileName;
  targetRawFile=$dataPath/hd/ve$machineIdx-$diskIdx/$fileName.hdraw

  echo TargetRawFile: $targetRawFile 
  if [ ! -f $targetRawFile ]; then
    echo "ERROR: $targetRawFile is not found"
    continue;
  fi

  #HD:: video from vga 31-2
  if [ -d $dataPath/hd ]; then

    outputFolder_sub=$outputFolder_nas/$fileName/videos/hd
    outputFileName=$(printf "$outputFolder_sub/ve%d-%d.mp4" $machineIdx $diskIdx)
    echo "outputFileName: $outputFileName"
    if [ ! -f $outputFileName ]; then

      mkdir -p $outputFolder_sub
      [ -d $outputFolder_sub ] || mkdir -p $outputFolder_sub
      echo "Video Generation: $outputFileName"
      videoFileName=$(printf %s_vga $fileName)
      python2 EncodeVideos/encode_hd_sync_to_vga.py --crf 23 $dataPath $fileName $machineIdx $diskIdx $outputFolder_sub
    else
      echo "HD video gen: File already exist"
    fi

  fi

done

done

done
