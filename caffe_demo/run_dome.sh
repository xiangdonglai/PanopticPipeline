CAPTURES_NAS=$1
PROCESSED_NAS=$2
datasetName=$3
startIdx=$4
startEnd=$5  # end index
camNum=$6
numgpu=$7
category=$8

rawFolder=/media/posefs${CAPTURES_NAS}/Captures/$category
saveFolder=/media/posefs${PROCESSED_NAS}/Processed/$category
mkdir -p $saveFolder

./peakdetect_vga_raw_arg.sh $rawFolder/$datasetName $saveFolder/${datasetName}/body_mpm $startIdx $startEnd $camNum $numgpu
echo ./vga_merger_arg.sh $saveFolder/$datasetName heatmaps_org/vga_25 poseDetect_pm_org/vga_25 $startIdx $startEnd
./vga_merger_arg.sh $saveFolder/${datasetName}/body_mpm heatmaps_org/vga_25 poseDetect_mpm_org/vga_25 $startIdx $startEnd
echo "" > $saveFolder/$datasetName/done_pose_org.log
