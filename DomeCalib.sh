#!/bin/bash

cd domeCalibrator/DomeCalibScript/

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 (CalibFolderPath) {KinectCalibFilePath}";
  echo "(CalibFolderPath) is the root folder containing scene image folders, named the dates"
  
  echo "For example, let us assumes that you have a folder hiarachy as follows,"
  echo "   D:/User/160111/scene1"
  echo "   D:/User/160111/scene2"
  echo "   D:/User/160111/scene3"
  echo "   Then, (CalibFolderPath) should be D:/User/160111_calib"
  echo "   Note that subfolder names in (CalibFolderPath) can be arbitrary"
  echo ""
  echo "{KinectCalibFilePath} is the .mat file generated from Kinoptic side (e.g.,kinect_calibration_XX_XXXX.mat)"
  echo "{KinectCalibFilePath} is optional and used to compute mk2nvm matrix to transform kinect data to this calibration space."
  exit 1
fi
path=$1; # D:/calibrationTest/160111_calib/p3
kinectCalibPath=$2; # d:/calibration/kinopticCalib/kinect_calibration_10_150721
nViewThresh=3 # points with visibility equal or larger than this number are used for bundle adjustment
echo "CalibFolder: $path"
echo "kinectCalibPath: $kinectCalibPath"
echo "nViewThresh: $nViewThresh"

date=$(basename $path)
echo $date
if ! [[ $date =~ ^[0-9]+$ ]] ; then
   echo "error: CalibFolder not a date" >&2; exit 1
fi

# Goal: Camera calibration using multiple scenes (multiple visualSfM files)
# Assume that all the scenes (captured by static cameras) are located in "path" folder
# e.g., $path/xx, $path/yy, $path/zz and so on. The folder names xx, yy are not important.
# (But note that you should not use $path/full folder, since it is used for the final output)
#
# This script runs the following
# 1. Run visualSfM in each folder by generating "nvm" files
# 2. Generate a consolidated scene (containing all the 3D points from all scenes) by the generating a unified matching file and SIFT files. 
#    As output, $path/full folder has been generated
# 2.5 Copy jpg files to $path/full. This images are dummy files, and are only required to run visualSfM, (The scene can be arbitrary, but should have images from all sensors)
# 3. Run visualSfM on $path/full. 
#    As output, $path/full/result.nvm has been generated. The output contains all 3D points from all scenes.  You can check the result by loading in VisualSfM.
# 4. Run Bundle Adjustment from  $path/full/result.nvm. 
#    As output, $path/full/output folder has been generated, where all the final outputs are located. 
#    You can visualize the final calibration output by loading $path/full/output/BA_caliball.nvm in VisualSfM. 
#    For the details on the calibration output in $path/full/output, see Gaku's original documentation:
#    https://github.com/CMU-Panoptic-Studio/ProjectorBasedCalibration/blob/master/docs/DomeCalibration.pdf
# 5. Compute mk2nvm whicih is a 4x4 matrix from multiple kinect space (define wrt the depth cam of kinectNode1) to nvm's coord.
#	 Ignored, if kinectCalibPath is not valid. 

# Note: This script doesn't care the number of scenes (subfolders) in $path file. 
# You may put many scenes by capturing with changing projected patterns  or our "white tent" location
# Note 2: Each scene can have images from any sensors (VGA, HD, Kinect color). VisualSfM doesn't care it. 

# Naming rules
# 00_00 - 00_31 mean  HD cameras
# 01_01 - 20_24 mean VGA cameras
# 50_1 - 50_10 mean Kinect color cameras

cnt=0;
for n in $path/*
do
	if [[ $n == *"full" ]]; then
		continue;
	fi
	cnt=`expr $cnt + 1`
done

echo "Number of folder: $cnt"
if [ $cnt == 1	]
then
	echo ""
	echo There is only one folder. 
fi

#Run Each Sfm
for n in $path/*
do
if [[ $n == *"full" ]]; then
	continue;
fi
echo $n
if [ $cnt == 1	]
then
	finalFolder=$n
fi
if [ ! -f $n/result.nvm ]
then
	vsfm/bin/VisualSFM sfm $n $n/result.nvm
else
	echo "Already exists: $n/result.nvm"
fi
done

if [ $cnt == 1	]
then
	echo ""
	echo There is only one folder. 
	echo "Just run bundle adjust on the folder" 
	#Final Bundle Adjustment (added 1 at the end of the command to avoid duplication check)
	DomeBundle/DomeBundle.exe  $finalFolder  $finalFolder/output $nViewThresh 1

	#Compute mk2nvm.txt
	myCalibPath=$finalFolder/output/calibFiles
	mk2nvmPath=$finalFolder/output/mk2nvm.txt
	matlab -nojvm -nodisplay -r "Compute_MK2NVM('$myCalibPath','$kinectCalibPath','$mk2nvmPath'); exit"
	exit;
fi

#copy jpg file for dummy
for n in $path/*
do
	if [[ $n == *"full" ]]; then
		continue;
	fi
echo copy $n to full
`mkdir $path/full`
	for img in $n/*.jpg
	do
		cp -v $img $path/full
	done
break;
done

#Merge Matchin Info
echo ""
echo "Merge Matchings"
if [ ! -f $path/full/FeatureMatches.txt ]
then
	../../Social-Capture-Ubuntu/DomeCorres/build/DomeCorres $path
fi
 
#SfM with all the merged matchngs
vsfm/bin/VisualSFM sfm+import $path/full $path/full/result.nvm $path/full/FeatureMatches.txt

#Final Bundle Adjustment
../../Social-Capture-Ubuntu/DomeBundle/build/DomeBundle $path/full $path/full/output $nViewThresh

#Generate normalized calibration files
../../DomeCalib_norm_check.sh $path/full/output

#Generate json version for panoptic dataset
calibNormPath=$path/full/output_norm;
matlab -nojvm -nodisplay -r "addpath('../../jsonlab/');ConvCalibTxtToJson('$calibNormPath'); exit"

#Compute mk2nvm.txt
myCalibPath=$path/full/output/calibFiles
mk2nvmPath=$path/full/output/mk2nvm.txt
matlab -nojvm -nodisplay -r "Compute_MK2NVM('$myCalibPath','$kinectCalibPath','$mk2nvmPath'); exit"

cp -R $path/full/output_norm /media/posefs1a/Calibration/${date}_calib_norm/
mv $path /media/posefs1a/Calibration/${date}_calib_rawData
