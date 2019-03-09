#include "BodyPoseReconDM.h"
#include "DomeImageManager.h"
#include "Utility.h"

namespace SkeletonGeneratorDlg
{
void Script_Util_Undistort_PoseMachine2DResult_mpm_19joints(bool bHD)
{
	char poseDetectFolder[512];
	if(bHD)
		sprintf(poseDetectFolder,"%s/poseDetect_mpm_org/hd_30",g_dataMainFolder);
	else
		sprintf(poseDetectFolder,"%s/poseDetect_mpm_org/vga_25",g_dataMainFolder);
	char newPoseDetectFolder[512];
	//sprintf(newPoseDetectFolder,"%s/poseDetect_mpm_15",g_dataMainFolder);
	sprintf(newPoseDetectFolder,"%s/coco19_poseDetect_pm",g_dataMainFolder);
	CreateFolder(newPoseDetectFolder);
	//sprintf(newPoseDetectFolder,"%s/poseDetect_mpm_15/vga_25",g_dataMainFolder);
	if(bHD)
		sprintf(newPoseDetectFolder,"%s/coco19_poseDetect_pm/hd_30",g_dataMainFolder);
	else
		sprintf(newPoseDetectFolder,"%s/coco19_poseDetect_pm/vga_25",g_dataMainFolder);

	CreateFolder(newPoseDetectFolder);
	CDomeImageManager domeImgMan;
	domeImgMan.SetCalibFolderPath(g_calibrationFolder);
	if(bHD)
		domeImgMan.InitDomeCamVgaHdKinect(0, CDomeImageManager::LOAD_SENSORS_HD);
	else
		domeImgMan.InitDomeCamVgaHdKinect();
	int frameEnd = g_dataFrameStartIdx + g_dataFrameNum;
	#pragma omp parallel for
	for(int f = g_dataFrameStartIdx; f <= frameEnd; ++f)
	{
		printf("## Performing frame %d\n",f);
		Module_BodyPose::Load_Undist_PoseDetectMultipleCamResult_MultiPoseMachine_19jointFull(poseDetectFolder, newPoseDetectFolder, f, domeImgMan, bHD);
	}
}
}
