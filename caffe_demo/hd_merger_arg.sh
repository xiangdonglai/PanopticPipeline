if [ "$#" -ne 5 ]; then
  echo "Usage: $0 mainPath peakFolder outputFolder frameStart frameEnd" >&2
  echo "Usage: $0 /posefs5b/Processed/SocialGames/160422_ultimatum heatmaps_org/vga_25 poseDetect_pm_org 50 100" >&2
  exit 1
fi
mainPath=$1
inputPath=$2
outputPath=$3
frameStart=$4
frameEnd=$5

interval=`expr $frameEnd - $frameStart`
interval=`expr $interval + 1`
interval=`expr $interval / 4` #interval should 3 for parallel 5
mid1=`expr $frameStart + $interval`
mid2=`expr $mid1 + $interval`
mid3=`expr $mid2 + $interval`

#backup old result
#if [ -d $mainPath/$outputPath ]; then
#	echo "## There is a previous result in : $mainPath/$outputPath"
#	now=`date +"%m-%d-%y-%T"`
 #       mv -v $mainPath/$outputPath $mainPath/${outputPath}-$now
#fi

echo $frameStart $mid1 $mid2 $frameEnd
mkdir -pv $mainPath/$outputPath
#echo ../build/examples/rtcpm/poseResultMerger.bin $mainPath/$inputPath $mainPath/$outputPath $frameStart $frameEnd
build/examples/rtpose/poseResultMerger_hd.bin $mainPath/$inputPath $mainPath/$outputPath $frameStart $frameEnd
#parallel -j4 --xapply ../build/examples/rtcpm/poseResultMerger.bin $mainPath/$inputPath $mainPath/$outputPath {1} {2} ::: $frameStart $mid1 $mid2 $mid3 ::: $mid1 $mid2 $mid3 $frameEnd
echo "" > $mainPath/done_pose_org_hd.log
exit;
