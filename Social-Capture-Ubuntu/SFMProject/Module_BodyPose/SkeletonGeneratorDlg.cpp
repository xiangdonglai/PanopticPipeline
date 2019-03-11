#include "BodyPoseReconDM.h"
#include "BodyPoseReconDT.h"
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

//Using Pose Machine COCO 19
void Script_NodePartProposalRecon_fromPoseMachine_coco19()
{
	for(int i = 0; i < g_dataFrameNum; i += g_dataFrameInterval)
	{
		int frameIdx = g_dataFrameStartIdx + i;
		printf("\n## NodePartProposalRecon:: Frame %d\n",frameIdx);

		//File exist check
		char outputFileName[512];
		sprintf(outputFileName, "%s/coco19_bodyNodeProposal/%04d/nodePartProposals_%08d.txt", g_dataMainFolder, g_askedVGACamNum, frameIdx);
		ifstream fin(outputFileName);
		if(IsFileExist(outputFileName)==true)
			continue;
		Module_BodyPose::CBodyPoseRecon tempPoseManager;
		tempPoseManager.ProbVolumeRecoe_nodePartProposals_fromPoseMachine_coco19(g_dataMainFolder,g_calibrationFolder,g_askedVGACamNum,frameIdx,false,false,true);
		tempPoseManager.ClearData();
	}
	// ((MainControlDlg*)GetParent())->ComputeSliderRange();
	// g_sfm.InitDomeCamIntExtVHK(g_calibrationFolder,g_visData);
	// ((MainControlDlg*)GetParent())->VisualizeEverything();
}

void Script_3DPS_Reconstruct_PoseMachine_coco19()
{
	using Module_BodyPose::g_bodyPoseManager;
	using Module_BodyPose::MODEL_JOINT_NUM_COCO_19;
	using Module_BodyPose::SBody3DScene;
	g_bodyPoseManager.ClearDetectionHull();
	g_bodyPoseManager.m_nodePropScoreVector.reserve(g_dataFrameNum);
	g_bodyPoseManager.ConstructJointHierarchy(MODEL_JOINT_NUM_COCO_19);
	g_bodyPoseManager.SetfpsType(FPS_VGA_25);

	char partPropFolderPath[512];
	sprintf(partPropFolderPath,"%s/coco19_bodyNodeProposal/%04d",g_dataMainFolder,g_askedVGACamNum);

	char saveFolderPath[512];
	sprintf(saveFolderPath,"%s/coco19_body3DPSRecon",g_dataMainFolder);
	CreateFolder(saveFolderPath);
	sprintf(saveFolderPath,"%s/coco19_body3DPSRecon/%04d",g_dataMainFolder,g_askedVGACamNum);
	CreateFolder(saveFolderPath);

	for(int f = 0; f < g_dataFrameNum; f += g_dataFrameInterval)
	{
		int frameIdx = g_dataFrameStartIdx + f;
		char fullPath[512];
		sprintf(fullPath,"%s/nodePartProposals_%08d.txt",partPropFolderPath,frameIdx);//m_domeImageManager.m_currentFrame);
		if(IsFileExist(fullPath)==false)
		{
			printf("## WARNING: cannot find file %s\n",fullPath);
			break;
		}

		//Skip if file already exists
		char outputPath[512];
		sprintf(outputPath,"%s/body3DScene_%08d.txt",saveFolderPath,frameIdx);//m_domeImageManager.m_currentFrame);
		printf("Check existence from %s\n",outputPath);
		if(IsFileExist(outputPath))
		{
			printf("## Skip: file already exists in %s\n",outputPath);
			continue;
		}

		g_bodyPoseManager.LoadNodePartProposals(fullPath,frameIdx,false);
		SBody3DScene newPoseReconMem;
		InferenceJoint15_OnePass_Multiple_DM(g_bodyPoseManager.m_nodePropScoreVector.back(), g_bodyPoseManager.m_skeletonHierarchy, newPoseReconMem);	
		SaveBodyReconResult(saveFolderPath,newPoseReconMem,newPoseReconMem.m_imgFramdIdx);
		g_bodyPoseManager.ClearDetectionHull();
	}
}

void Script_Load_body3DPS_byFrame(bool bCoco19) 
{
	using Module_BodyPose::g_bodyPoseManager;
	g_bodyPoseManager.m_3DPSPoseMemVector.clear();
	g_bodyPoseManager.SetfpsType(FPS_VGA_25);

	//////////////////////////////////////////////////////////////////////////
	/// Try to load from (g_dataMainFolder)/ or (g_dataMainFolder)/poseRecon/
	for(int f=0;f<g_poseEstLoadingDataNum;f+=g_poseEstLoadingDataInterval)
	{
		int targetFrameNum = g_poseEstLoadingDataFirstFrameIdx + f;
		char fullPath[512];
		if(bCoco19)
			sprintf(fullPath,"%s/coco19_body3DPSRecon/%04d/body3DScene_%08d.txt",g_dataMainFolder,g_askedVGACamNum,targetFrameNum);//m_domeImageManager.m_currentFrame);
		else
			sprintf(fullPath,"%s/body3DPSRecon/%04d/body3DScene_%08d.txt",g_dataMainFolder,g_askedVGACamNum,targetFrameNum);//m_domeImageManager.m_currentFrame);
		bool bSuccess = g_bodyPoseManager.LoadBodyReconResult(fullPath,targetFrameNum);
		if(bSuccess ==false)
			printf("Load failure from %s\n",fullPath);
	}
	g_bodyPoseManager.AssignHumanIdentityColorFor3DPSResult();
}

void Script_Load_body3DPS_byFrame_folderSpecify() 
{
	using Module_BodyPose::g_bodyPoseManager;
	g_bodyPoseManager.m_3DPSPoseMemVector.clear();
	g_bodyPoseManager.SetfpsType(FPS_VGA_25);

	//////////////////////////////////////////////////////////////////////////
	/// Try to load from (g_dataMainFolder)/ or (g_dataMainFolder)/poseRecon/
	for(int f=0;f<g_poseEstLoadingDataNum;f+=g_poseEstLoadingDataInterval)
	{
		//int targetFrameNum = g_poseEstLoadingDataFirstFrameIdx +f*5;
		int targetFrameNum = g_poseEstLoadingDataFirstFrameIdx + f;
		char fullPath[512];
		sprintf(fullPath,"%s/body3DScene_%08d.txt",g_dataSpecificFolder,targetFrameNum);//m_domeImageManager.m_currentFrame);
		
		bool bSuccess = g_bodyPoseManager.LoadBodyReconResult(fullPath,targetFrameNum);
		if(bSuccess ==false)
			printf("Load failure from %s\n",fullPath);
		else if(f%500==0)
			printf("Loaded:: %d/%d ::%s\n",f,g_poseEstLoadingDataNum,fullPath);

	}
	g_bodyPoseManager.AssignHumanIdentityColorFor3DPSResult();
}

void Script_3DPS_Optimization_usingDetectionPeaks(bool bCoco19)
{
	using Module_BodyPose::g_bodyPoseManager;
	g_bodyPoseManager.Optimization3DPS_fromDetection_oneToOne_coco19(g_dataMainFolder,g_calibrationFolder,g_askedVGACamNum,false,bCoco19); //this version can handle both 15joints, 19 joints
	g_bodyPoseManager.SetfpsType(FPS_VGA_25);

	//Resave result
	char saveFolderPath[512];
	if(bCoco19)
	{
		sprintf(saveFolderPath,"%s/coco19_body3DPSRecon_updated",g_dataMainFolder,g_askedVGACamNum);
		CreateFolder(saveFolderPath);
		sprintf(saveFolderPath,"%s/coco19_body3DPSRecon_updated/%04d",g_dataMainFolder,g_askedVGACamNum);
		CreateFolder(saveFolderPath);
	}
	else
	{
		sprintf(saveFolderPath,"%s/body3DPSRecon_updated",g_dataMainFolder,g_askedVGACamNum);
		CreateFolder(saveFolderPath);
		sprintf(saveFolderPath,"%s/body3DPSRecon_updated/%04d",g_dataMainFolder,g_askedVGACamNum);
		CreateFolder(saveFolderPath);
	}
	
	for(int i=0;i<g_bodyPoseManager.m_3DPSPoseMemVector.size();++i)
	{
		int frameIdx = g_bodyPoseManager.m_3DPSPoseMemVector[i].m_imgFramdIdx;
		SaveBodyReconResult(saveFolderPath,g_bodyPoseManager.m_3DPSPoseMemVector[i],frameIdx);
	}
}

void Script_Export_3DPS_Json(bool bNormCoord)
{
	using Module_BodyPose::g_bodyPoseManager;
	using Module_BodyPose::MODEL_JOINT_NUM_COCO_19;
	using Module_BodyPose::MODEL_JOINT_NUM_OP_25;
	bNormCoord=false;
	char saveFolderPath[512];
	if(g_bodyPoseManager.m_skeletonHierarchy.size()==MODEL_JOINT_NUM_COCO_19)
	{
		sprintf(saveFolderPath,"%s/coco19_body3DPSRecon_json_normCoord",g_dataMainFolder);
	}
	else if (g_bodyPoseManager.m_skeletonHierarchy.size()==MODEL_JOINT_NUM_OP_25)
	{
		sprintf(saveFolderPath,"%s/op25_body3DPSRecon_json_normCoord",g_dataMainFolder);	
	}
	else
	{
		sprintf(saveFolderPath,"%s/body3DPSRecon_json_normCoord",g_dataMainFolder);	
	}
	
	CreateFolder(saveFolderPath);

	if(g_fpsType==FPS_HD_30)
		sprintf(saveFolderPath,"%s/hd",saveFolderPath);
	else
		sprintf(saveFolderPath,"%s/%04d",saveFolderPath,g_askedVGACamNum);
	CreateFolder(saveFolderPath);
	g_bodyPoseManager.Save3DPSBodyReconResult_Json(saveFolderPath,bNormCoord);
}

}
