#include "Constants.h"
#include "Utility.h"
#include "FaceReconByPMDM.h"
#include "FaceReconByPMDlg.h"

namespace Module_Face_pm
{
void UndistortDetectionResult()
{
	char poseDetectFolder[512];
	sprintf(poseDetectFolder,"%s/facedetect_pm_org/hd_30",g_dataMainFolder);
	char newPoseDetectFolder[512];
	sprintf(newPoseDetectFolder,"%s/faceDetect_pm",g_dataMainFolder);
	CreateFolder(newPoseDetectFolder);
	sprintf(newPoseDetectFolder,"%s/faceDetect_pm/hd_30",g_dataMainFolder);
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
		Load_Undist_FaceDetectMultipleCamResult_face70_PoseMachine(poseDetectFolder,newPoseDetectFolder,f,domeImgMan,true);
	}
}

void ReconFacePM70()
{
	//Parameter Settings
	int startFrameIdx = g_dataFrameStartIdx;
	int frameNum = g_dataFrameNum;
	CDomeImageManager domeImageManager;
	domeImageManager.SetCalibFolderPath(g_calibrationFolder);
	printf(g_calibrationFolder);
	printf(g_dataImageFolder);
	domeImageManager.SetImageFolderPath(g_dataImageFolder);
	domeImageManager.InitDomeCamVgaHdKinect(0,CDomeImageManager::LOAD_SENSORS_HD);
	char faceDetectFilePath[512];
	sprintf(faceDetectFilePath,"%s/faceDetect_pm/",g_dataMainFolder);		//if there is no precomputed detection files, it compute detection first.

	//Make a save folder
	char reconFolderPath[512];
	sprintf(reconFolderPath,"%s/faceRecon_pm",g_dataMainFolder);
	CreateFolder(reconFolderPath);
	sprintf(reconFolderPath,"%s/faceRecon_pm/hd_30",g_dataMainFolder);
	CreateFolder(reconFolderPath);

	//Process
	for(int f=0;f<frameNum;++f)
	{
		//	int f =1650;
		//g_handReconManager.m_handReconMem.resize(g_handReconManager.m_handReconMem.size()+1);
		//g_handReconManager.m_handReconMem.back().frameIdx = f;
		//domeImageManager.SetFrameIdx(f);
		//g_handReconManager.Finger_Landmark_Reconstruction_Voting_hd(faceDetectFilePath,domeImageManager,g_handReconManager.m_handReconMem.back().handReconVect);

		char fullPath[512];
		sprintf(fullPath,"%s/faceRecon3Dpm_hd%08d.txt",reconFolderPath,startFrameIdx + f);//m_domeImageManager.m_currentFrame);
		if(IsFileExist(fullPath))
		{
			printf("Already exists: %s\n",fullPath);
			continue;
		}
		domeImageManager.SetFrameIdx(startFrameIdx + f);
		//domeImageManager.LoadDomeImagesCurrentFrame(CDomeImageManager::LOAD_MODE_GRAY);
		vector<SFace3D_pm> handReconResult;
		g_faceReconManager_pm.Face_Landmark_Reconstruction_hd(faceDetectFilePath,domeImageManager,handReconResult);
		g_faceReconManager_pm.SaveFaceReconResult(reconFolderPath,handReconResult,domeImageManager.GetFrameIdx(),true);
	}
}

void Load3DFace()
{
	//Assume that face is reconstructed by HD_30 mode
	int startIdx;
	int loadingNum;
	if(g_fpsType == FPS_VGA_25)
	{
		startIdx = g_syncMan.ClosestHDfromVGA(g_poseEstLoadingDataFirstFrameIdx);
		loadingNum = int(g_poseEstLoadingDataNum *30.0/25.0);
	}
	else  //FPS_HD_30
	{
		startIdx = g_poseEstLoadingDataFirstFrameIdx;
		loadingNum = g_poseEstLoadingDataNum;
	}

	vector<string> folderPathCand;
	char fullPath[512];
	sprintf(fullPath,"%s/faceRecon_pm/hd_30",g_dataMainFolder);//m_domeImageManager.m_currentFrame);

	printf("Loading startIdx: %d, loadingNum %d\n",startIdx,loadingNum);
	g_faceReconManager_pm.LoadFace3DByFrame(fullPath,startIdx,loadingNum,true);
}

void ExportToJson()
{
	//Make a save folder
	char reconFolderPath[512];
	sprintf(reconFolderPath,"%s/faceRecon_pm_json",g_dataMainFolder);
	CreateFolder(reconFolderPath);
	sprintf(reconFolderPath,"%s/faceRecon_pm_json/hd_30",g_dataMainFolder);
	CreateFolder(reconFolderPath);

	for(int f=0;f<g_faceReconManager_pm.m_faceReconMem.size();++f)
	{
		g_faceReconManager_pm.SaveFaceReconResult_json(reconFolderPath
			,g_faceReconManager_pm.m_faceReconMem[f].faceReconVect
			,g_faceReconManager_pm.m_faceReconMem[f].frameIdx,true);
	}
}
}