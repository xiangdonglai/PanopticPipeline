calibPath=$1
newpath=${calibPath}_norm
echo $newpath
mkdir -p $newpath

#Generate Normalized Calibration Params
cp -vr $calibPath/calibFiles $newpath/
cp -vr $calibPath/calibFiles_withoutDistortion $newpath/
../../Social-Capture-Ubuntu/SFMProject/build/SFMProject calibNorm $newpath/calibFiles/
mv $newpath/calibFiles/normalized $newpath/
cp -f $newpath/normalized/* $newpath/calibFiles/
cp -f $newpath/normalized/* $newpath/calibFiles_withoutDistortion/

#Detect checkerboard
#checkerPath='M:/Users/hanbyulj/160502_checkerboard3/originalImgs/hd_30'
#matlab -nojvm -nodisplay -r "detectCheckerFunc('$checkerPath',[2000],'.',false); exit"

#Leave-One-Out Test using checker board
#SMC_calib/SFMProject.exe calibCheckbyChecker d:/calibration/160223_calib_tent13_norm/calibFiles d:/codes/Checkerboard/160502_checkerboard3 (outputFolder)

