rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_a1
startIdx=730;
startEnd=19900;
./peakdetect_hd_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd
echo ./hd_merger_arg.sh $saveFolder/$datasetName heatmaps_org/hd_30 poseDetect_pm_org/hd_30 $startIdx $startEnd
./hd_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/hd_30 poseDetect_mpm_org/hd_30 $startIdx $startEnd 
echo "" > $saveFolder/$datasetName/done_pose_hd_org.log
exit;

rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_a2
startIdx=989;
startEnd=19869;
#./peakdetect_hd_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd
echo ./hd_merger_arg.sh $saveFolder/$datasetName heatmaps_org/hd_30 poseDetect_pm_org/hd_30 $startIdx $startEnd
#./hd_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/hd_30 poseDetect_mpm_org/hd_30 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_hd_org.log

rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_a3
startIdx=1043;
startEnd=10333;
#./peakdetect_hd_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd
echo ./hd_merger_arg.sh $saveFolder/$datasetName heatmaps_org/hd_30 poseDetect_pm_org/hd_30 $startIdx $startEnd
#./hd_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/hd_30 poseDetect_mpm_org/hd_30 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_hd_org.log


rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_b1
startIdx=1773;
startEnd=25479;
./peakdetect_hd_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd
echo ./hd_merger_arg.sh $saveFolder/$datasetName heatmaps_org/hd_30 poseDetect_pm_org/hd_30 $startIdx $startEnd
./hd_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/hd_30 poseDetect_mpm_org/hd_30 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_hd_org.log


rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_b2
startIdx=997;
startEnd=25033;
./peakdetect_hd_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd
echo ./hd_merger_arg.sh $saveFolder/$datasetName heatmaps_org/hd_30 poseDetect_pm_org/hd_30 $startIdx $startEnd
./hd_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/hd_30 poseDetect_mpm_org/hd_30 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_hd_org.log

rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_b3
startIdx=1019;
startEnd=12887;
./peakdetect_hd_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd
echo ./hd_merger_arg.sh $saveFolder/$datasetName heatmaps_org/hd_30 poseDetect_pm_org/hd_30 $startIdx $startEnd
./hd_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/hd_30 poseDetect_mpm_org/hd_30 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_hd_org.log

exit;


rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_a3
startIdx=750;
startEnd=8200;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd
echo "" > $saveFolder/$datasetName/done_pose_org.log

exit;


rawFolder=/media/posefs6a/Captures/specialEvents
saveFolder=/media/posefs6b/Processed/specialEvents
mkdir -p $saveFolder
datasetName=170214_gaze1
startIdx=10;
startEnd=7500;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd
echo "" > $saveFolder/$datasetName/done_pose_org.log


exit;



rawFolder=/media/posefs5a/Captures/SocialGames
saveFolder=/media/posefs3b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=160317_meeting1
startIdx=3511;
startEnd=3750;
camNum=480;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd
echo "" > $saveFolder/$datasetName/done_pose_org.log

rawFolder=/media/posefs5a/Captures/SocialGames
saveFolder=/media/posefs3b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=160317_meeting1
startIdx=3751;
startEnd=4000;
camNum=480;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd
echo "" > $saveFolder/$datasetName/done_pose_org.log


rawFolder=/media/posefs5a/Captures/SocialGames
saveFolder=/media/posefs3b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=160906_pizza1
startIdx=2000;
startEnd=2250;
camNum=480;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd
echo "" > $saveFolder/$datasetName/done_pose_org.log

awFolder=/media/posefs3a/Captures/cello
saveFolder=/media/posefs3b/Processed/cello
mkdir -p $saveFolder
datasetName=150209_celloCapture2
startIdx=30;
startEnd=4450;
camNum=480;
echo ./peakdetect_vga_raw_arg.sh /media/posefs5a/Captures/SocialGames/$datasetName /media/posefs5b/Processed/SocialGames/$datasetName $startIdx $startEnd $camNum
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/$datasetName $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log

exit;

rawFolder=/media/posefs6a/Captures/specialEvents
saveFolder=/media/posefs5b/Processed/specialEvents
mkdir -p $saveFolder
datasetName=160906_ian5
startIdx=10;
startEnd=3000;
camNum=480;
echo ./peakdetect_vga_raw_arg.sh /media/posefs5a/Captures/SocialGames/$datasetName /media/posefs5b/Processed/SocialGames/$datasetName $startIdx $startEnd $camNum
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/$datasetName $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log
exit;

rawFolder=/media/posefs6a/Captures/specialEvents
saveFolder=/media/posefs5b/Processed/specialEvents
mkdir -p $saveFolder
datasetName=160906_ian1
startIdx=10;
startEnd=3000;
camNum=480;
echo ./peakdetect_vga_raw_arg.sh /media/posefs5a/Captures/SocialGames/$datasetName /media/posefs5b/Processed/SocialGames/$datasetName $startIdx $startEnd $camNum
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/$datasetName $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log


exit;

rawFolder=/media/posefs2a/Captures/poseMachine
saveFolder=/media/posefs2b/Processed/poseMachine
mkdir -p $saveFolder
datasetName=141215_pose6
startIdx=10;
startEnd=6800;
camNum=480;
echo ./peakdetect_vga_raw_arg.sh /media/posefs5a/Captures/SocialGames/$datasetName /media/posefs5b/Processed/SocialGames/$datasetName $startIdx $startEnd $camNum
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/$datasetName $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log


exit;

rawFolder=/media/posefs4a/Captures/SocialGames
saveFolder=/media/posefs4b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=150129_007game
startIdx=10;
startEnd=6000;
camNum=480;
echo ./peakdetect_vga_raw_arg.sh /media/posefs5a/Captures/SocialGames/$datasetName /media/posefs5b/Processed/SocialGames/$datasetName $startIdx $startEnd $camNum
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/$datasetName $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd &
#echo "" > $saveFolder/$datasetName/done_pose_org.log
exit;

rawFolder=/media/posefs2a/Captures/poseMachine
saveFolder=/media/posefs2b/Processed/poseMachine
mkdir -p $saveFolder
datasetName=141217_pose1
startIdx=10;
startEnd=6800;
camNum=480;
echo ./peakdetect_vga_raw_arg.sh /media/posefs5a/Captures/SocialGames/$datasetName /media/posefs5b/Processed/SocialGames/$datasetName $startIdx $startEnd $camNum
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/$datasetName $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd &
#echo "" > $saveFolder/$datasetName/done_pose_org.log


exit;


rawFolder=/media/posefs5a/Captures/SocialGames
saveFolder=/media/posefs5b/Processed/SocialGames
datasetName=160422_mafia1
startIdx=17187
startEnd=19500
camNum=480
echo ./peakdetect_vga_raw_arg.sh /media/posefs5a/Captures/SocialGames/$datasetName /media/posefs5b/Processed/SocialGames/$datasetName $startIdx $startEnd $camNum
#./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/$datasetName $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd &
#echo "" > $saveFolder/$datasetName/done_pose_org.log

exit;



awFolder=/media/posefs5a/Captures/SocialGames
saveFolder=/media/posefs5b/Processed/SocialGames
datasetName=160224_ultimatum2
startIdx=100
startEnd=11900
camNum=480
echo ./peakdetect_vga_raw_arg.sh /media/posefs5a/Captures/SocialGames/$datasetName /media/posefs5b/Processed/SocialGames/$datasetName $startIdx $startEnd $camNum
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/$datasetName $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd &
#echo "" > $saveFolder/$datasetName/done_pose_org.log


