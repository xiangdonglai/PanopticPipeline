#include "HandReconDlg.h"

namespace Module_Hand
{
void UndistortDetectionResult()
{
	char poseDetectFolder[512];
	sprintf(poseDetectFolder,"%s/handdetect_pm_org/hd_30",g_dataMainFolder);
	char newPoseDetectFolder[512];
	sprintf(newPoseDetectFolder,"%s/handDetect_pm",g_dataMainFolder);
	CreateFolder(newPoseDetectFolder);
	sprintf(newPoseDetectFolder,"%s/handDetect_pm/hd_30",g_dataMainFolder);
	CreateFolder(newPoseDetectFolder);
	CDomeImageManager domeImgMan;
	domeImgMan.SetCalibFolderPath(g_calibrationFolder);
	domeImgMan.InitDomeCamVgaHdKinect(480,CDomeImageManager::LOAD_SENSORS_VGA_HD);
	if(domeImgMan.m_domeViews.front()->m_distortionParams.size()==0 || domeImgMan.m_domeViews.front()->m_distortionParams[0]==0)
	{
		printf("ERROR: No distortion paramters in this calibration data\n");
		return;
	}
	int frameEnd = g_dataFrameStartIdx+ g_dataFrameNum;
	#pragma omp parallel for
	for(int f=g_dataFrameStartIdx;f<frameEnd;++f)
	{
		printf("## Performing frame %d\n",f);
		Load_Undist_HandDetectMultipleCamResult_PoseMachine(poseDetectFolder,newPoseDetectFolder,f,domeImgMan,true);
	}
}

void ReconFingers()
{
	//Parameter Settings
	int startFrameIdx = g_dataFrameStartIdx;
	int frameNum = g_dataFrameNum;
	CDomeImageManager domeImageManager;
	domeImageManager.SetCalibFolderPath(g_calibrationFolder);
	domeImageManager.SetImageFolderPath(g_dataImageFolder);
	domeImageManager.InitDomeCamVgaHdKinect(0,CDomeImageManager::LOAD_SENSORS_HD);
	char faceDetectFilePath[512];
	sprintf(faceDetectFilePath,"%s/handDetect_pm",g_dataMainFolder);		//if there is no precomputed detection files, it compute detection first.
	
	//Make a save folder
	char reconFolderPath[512];
	sprintf(reconFolderPath,"%s/handRecon",g_dataMainFolder);
	CreateFolder(reconFolderPath);
	sprintf(reconFolderPath,"%s/handRecon/hd_30",g_dataMainFolder);
	CreateFolder(reconFolderPath);

	//Process
	for(int f=0;f<frameNum;++f)
	{
		char fullPath[512];
		sprintf(fullPath,"%s/handRecon3D_hd%08d.txt",reconFolderPath,startFrameIdx + f);//m_domeImageManager.m_currentFrame);
		if(IsFileExist(fullPath))
			continue;
		domeImageManager.SetFrameIdx(startFrameIdx + f);
		vector<SHand3D> handReconResult;
		g_handReconManager.Finger_Landmark_Reconstruction_Voting_hd(faceDetectFilePath,domeImageManager,handReconResult);
		g_handReconManager.SaveHandReconResult(reconFolderPath,handReconResult,domeImageManager.GetFrameIdx(),true);
	}
}

void Script_ExportToJson()
{
	//Make a save folder
	char reconFolderPath[512];
	sprintf(reconFolderPath,"%s/handRecon_json",g_dataMainFolder);
	CreateFolder(reconFolderPath);
	sprintf(reconFolderPath,"%s/handRecon_json/hd_30",g_dataMainFolder);
	CreateFolder(reconFolderPath);

	for(int f=0;f<g_handReconManager.m_handReconMem.size();++f)
	{
		g_handReconManager.SaveHandReconResult_json(reconFolderPath,g_handReconManager.m_handReconMem[f].handReconVect,g_handReconManager.m_handReconMem[f].frameIdx,true);
	}
}

}