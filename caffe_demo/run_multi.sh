rawFolder=/media/posefs12a/Captures/specialEvents
saveFolder=/media/posefs11b/Processed/specialEvents
mkdir -p $saveFolder
datasetName=190125_pose1
startIdx=100;
startEnd=105;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
# echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
# ./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
# echo "" > $saveFolder/$datasetName/done_pose_org.log

exit;


rawFolder=/media/posefs5a/Captures/SocialGames
saveFolder=/media/posefs3b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=160226_ultimatum1
startIdx=100;
startEnd=20900;
camNum=480;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log

exit;


rawFolder=/media/posefs11a/Captures/Pose
saveFolder=/media/posefs2b/Processed/specialEvents
mkdir -p $saveFolder
datasetName=171204_jiujitsu2
startIdx=10;
startEnd=7450;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log

rawFolder=/media/posefs11a/Captures/Pose
saveFolder=/media/posefs2b/Processed/specialEvents
mkdir -p $saveFolder
datasetName=171204_jiujitsu3
startIdx=10;
startEnd=7450;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log




rawFolder=/media/posefs11a/Captures/Pose
saveFolder=/media/posefs2b/Processed/specialEvents
mkdir -p $saveFolder
datasetName=171204_jiujitsu4
startIdx=10;
startEnd=7450;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log

rawFolder=/media/posefs11a/Captures/Pose
saveFolder=/media/posefs2b/Processed/specialEvents
mkdir -p $saveFolder
datasetName=171204_jiujitsu5
startIdx=10;
startEnd=8950;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log



exit;









rawFolder=/media/posefs11a/Captures/Pose
saveFolder=/media/posefs2b/Processed/poseMachine
mkdir -p $saveFolder
datasetName=171026_pose3
startIdx=10;
startEnd=6000;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log



rawFolder=/media/posefs11a/Captures/Cello
saveFolder=/media/posefs3b/Processed/cello
mkdir -p $saveFolder
datasetName=171026_cello1
startIdx=10;
startEnd=7500;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log





rawFolder=/media/posefs11a/Captures/Pose
saveFolder=/media/posefs2b/Processed/poseMachine
mkdir -p $saveFolder
datasetName=171026_pose2
startIdx=10;
startEnd=12500;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log

rawFolder=/media/posefs11a/Captures/Pose
saveFolder=/media/posefs2b/Processed/poseMachine
mkdir -p $saveFolder
datasetName=171026_pose1
startIdx=10;
startEnd=18750;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log


rawFolder=/media/posefs11a/Captures/Cello
saveFolder=/media/posefs3b/Processed/cello
mkdir -p $saveFolder
datasetName=171026_cello2
startIdx=10;
startEnd=7450;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log




exit;



rawFolder=/media/posefs11a/Captures/Cello
saveFolder=/media/posefs3b/Processed/cello
mkdir -p $saveFolder
datasetName=171026_cello3
startIdx=10;
startEnd=1450;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log


exit;

rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_a3
startIdx=8500;
startEnd=15300;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log


rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_b3
startIdx=10650;
startEnd=18400;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
#echo "" > $saveFolder/$datasetName/done_pose_org.log



exit;


rawFolder=/media/posefs9a/Captures/dance
saveFolder=/media/posefs6b/Processed/dance
mkdir -p $saveFolder
datasetName=170307_dance2
startIdx=10;
startEnd=13500;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log

rawFolder=/media/posefs9a/Captures/dance
saveFolder=/media/posefs6b/Processed/dance
mkdir -p $saveFolder
datasetName=170307_dance3
startIdx=10;
startEnd=7125;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log



rawFolder=/media/posefs9a/Captures/dance
saveFolder=/media/posefs6b/Processed/dance
mkdir -p $saveFolder
datasetName=170307_dance4
startIdx=10;
startEnd=6750;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log



rawFolder=/media/posefs9a/Captures/dance
saveFolder=/media/posefs6b/Processed/dance
mkdir -p $saveFolder
datasetName=170307_dance5
startIdx=10;
startEnd=9700;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log


rawFolder=/media/posefs9a/Captures/dance
saveFolder=/media/posefs6b/Processed/dance
mkdir -p $saveFolder
datasetName=170307_dance6
startIdx=10;
startEnd=4625;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log


exit;


rawFolder=/media/posefs9a/Captures/dance
saveFolder=/media/posefs6b/Processed/dance
mkdir -p $saveFolder
datasetName=170307_dance2
startIdx=10;
startEnd=13500;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log







rawFolder=/media/posefs8a/Captures/hands
saveFolder=/media/posefs6b/Processed/hands
mkdir -p $saveFolder
datasetName=161029_sports1
startIdx=10;
startEnd=7400;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log



rawFolder=/media/posefs8a/Captures/hands
saveFolder=/media/posefs6b/Processed/hands
mkdir -p $saveFolder
datasetName=161029_tools1
startIdx=10;
startEnd=7400;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log

exit;


rawFolder=/media/posefs9b/Captures/SocialGames
saveFolder=/media/posefs10b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170404_haggling_a2
startIdx=1130;
startEnd=9200;
camNum=140;
#./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
#./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log

rawFolder=/media/posefs9b/Captures/SocialGames
saveFolder=/media/posefs10b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170404_haggling_b1
startIdx=1100;
startEnd=17000;
camNum=140;
#./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
#./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log

rawFolder=/media/posefs9b/Captures/SocialGames
saveFolder=/media/posefs10b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170404_haggling_b3
startIdx=320;
startEnd=15800;
camNum=140;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/body_mpm/done_pose_org.log

exit;




rawFolder=/media/posefs6a/Captures/specialEvents
saveFolder=/media/posefs5b/Processed/specialEvents
mkdir -p $saveFolder
datasetName=160906_ian1
startIdx=10;
startEnd=3000;
camNum=480;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log


rawFolder=/media/posefs3a/Captures/specialEvents
saveFolder=/media/posefs4b/Processed/specialEvents
mkdir -p $saveFolder
datasetName=160401_ian2
startIdx=50;
startEnd=7499;
camNum=480;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
echo "" > $saveFolder/$datasetName/done_pose_org.log

exit;
rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_b1
startIdx=1375;
startEnd=21150;
camNum=480;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
#./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
#echo "" > $saveFolder/$datasetName/done_pose_org.log


rawFolder=/media/posefs8b/Captures/SocialGames
saveFolder=/media/posefs6b/Processed/SocialGames
mkdir -p $saveFolder
datasetName=170228_haggling_b2
startIdx=750;
startEnd=20800;
camNum=480;
./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
#./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd &
#echo "" > $saveFolder/$datasetName/done_pose_org.log
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



rawFolder=/media/posefs5a/Captures/SocialGames
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


