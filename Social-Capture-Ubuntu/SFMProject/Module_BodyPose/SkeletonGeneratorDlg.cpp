#include "BodyPoseReconDM.h"
#include "BodyPoseReconDT.h"
#include "DomeImageManager.h"
#include "Utility.h"
#include "SyncMan.h"

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

void Script_Util_Undistort_PoseMachine2DResult_mpm_25joints(bool bHD)
{
	char poseDetectFolder[512];
	if(bHD)
		sprintf(poseDetectFolder,"%s/op25_poseDetect_mpm_org/hd_30",g_dataMainFolder);
	else
		sprintf(poseDetectFolder,"%s/op25_poseDetect_mpm_org/vga_25",g_dataMainFolder);
	char newPoseDetectFolder[512];
	//sprintf(newPoseDetectFolder,"%s/poseDetect_mpm_15",g_dataMainFolder);
	sprintf(newPoseDetectFolder,"%s/op25_poseDetect_pm",g_dataMainFolder);
	CreateFolder(newPoseDetectFolder);
	//sprintf(newPoseDetectFolder,"%s/poseDetect_mpm_15/vga_25",g_dataMainFolder);
	if(bHD)
		sprintf(newPoseDetectFolder,"%s/op25_poseDetect_pm/hd_30",g_dataMainFolder);
	else
		sprintf(newPoseDetectFolder,"%s/op25_poseDetect_pm/vga_25",g_dataMainFolder);

	CreateFolder(newPoseDetectFolder);
	CDomeImageManager domeImgMan;
	domeImgMan.SetCalibFolderPath(g_calibrationFolder);
	if(bHD)
		domeImgMan.InitDomeCamVgaHdKinect(0,CDomeImageManager::LOAD_SENSORS_HD);
	else
		domeImgMan.InitDomeCamVgaHdKinect();
	int frameEnd = g_dataFrameStartIdx+ g_dataFrameNum;
	#pragma omp parallel for
	for(int f=g_dataFrameStartIdx;f<=frameEnd;++f)
	{
		printf("## Performing frame %d\n",f);
		// This function handles both 19 & 25 kp cases.
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

//Using OpenPose 25 keypoints
void Script_NodePartProposalRecon_op25()
{
	//#pragma omp parallel for num_threads(4)
	//for(int i=0;i<g_dataFrameNum;++i)
	for(int i=0;i<g_dataFrameNum;i+=g_dataFrameInterval)
	{
		int frameIdx = g_dataFrameStartIdx + i;
		printf("\n## NodePartProposalRecon:: Frame %d\n",frameIdx);

		//File exist check 
		char outputFileName[512];
		sprintf(outputFileName,"%s/op25_bodyNodeProposal/%04d/nodePartProposals_%08d.txt",g_dataMainFolder,g_askedVGACamNum,frameIdx);
		ifstream fin(outputFileName);
		if(IsFileExist(outputFileName)==true)
			continue;
		//printf("Processing frame %d\n",frameIdx);
		Module_BodyPose::CBodyPoseRecon tempPoseManager;
		tempPoseManager.ProbVolumeRecoe_nodePartProposals_op25(g_dataMainFolder,g_calibrationFolder,g_askedVGACamNum,frameIdx,false,false);
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

void Script_3DPS_Reconstruct_op25()
{
	using Module_BodyPose::g_bodyPoseManager;
	using Module_BodyPose::MODEL_JOINT_NUM_OP_25;
	using Module_BodyPose::SBody3DScene;
	g_bodyPoseManager.ClearDetectionHull();
	g_bodyPoseManager.m_nodePropScoreVector.reserve(g_dataFrameNum);
	g_bodyPoseManager.ConstructJointHierarchy(MODEL_JOINT_NUM_OP_25);
	g_bodyPoseManager.SetfpsType(FPS_VGA_25);

	char partPropFolderPath[512];
	sprintf(partPropFolderPath,"%s/op25_bodyNodeProposal/%04d",g_dataMainFolder,g_askedVGACamNum);

	char saveFolderPath[512];
	sprintf(saveFolderPath,"%s/op25_body3DPSRecon",g_dataMainFolder);
	CreateFolder(saveFolderPath);
	sprintf(saveFolderPath,"%s/op25_body3DPSRecon/%04d",g_dataMainFolder,g_askedVGACamNum);
	CreateFolder(saveFolderPath);

	//for(int f=0;f<g_dataFrameNum;++f)
	for(int f=0;f<g_dataFrameNum;f+=g_dataFrameInterval)
	{
		int frameIdx= g_dataFrameStartIdx +f;
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

		g_bodyPoseManager.LoadNodePartProposals(fullPath,frameIdx,false);				//Loaded g_bodyPoseManager.m_nodePropScoreVector, and partProposal in m_skeletonHierarchy (which is a copy of m_edgeCostVector)
		SBody3DScene newPoseReconMem;
		InferenceJoint15_OnePass_Multiple_DM(g_bodyPoseManager.m_nodePropScoreVector.back(),g_bodyPoseManager.m_skeletonHierarchy,newPoseReconMem);	

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

void Script_Load_body3DPS_byFrame_op25() 
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

		sprintf(fullPath,"%s/op25_body3DPSRecon/%04d/body3DScene_%08d.txt",g_dataMainFolder,g_askedVGACamNum,targetFrameNum);//m_domeImageManager.m_currentFrame);
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

void Script_3DPS_Optimization_usingDetectionPeaks_op25()
{
	using Module_BodyPose::g_bodyPoseManager;
	g_bodyPoseManager.Optimization3DPS_fromDetection_oneToOne_op25(g_dataMainFolder,g_calibrationFolder,g_askedVGACamNum,false); //this version can handle both 15joints, 19 joints
	g_bodyPoseManager.SetfpsType(FPS_VGA_25);

	//Resave result
	char saveFolderPath[512];
	sprintf(saveFolderPath,"%s/op25_body3DPSRecon_updated",g_dataMainFolder,g_askedVGACamNum);
	CreateFolder(saveFolderPath);
	sprintf(saveFolderPath,"%s/op25_body3DPSRecon_updated/%04d",g_dataMainFolder,g_askedVGACamNum);
	CreateFolder(saveFolderPath);
	
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

void Script_VGA_SaveAsHDFrameIdxViaInterpolation()
{
	using Module_BodyPose::g_bodyPoseManager;
	using Module_BodyPose::MODEL_JOINT_NUM_COCO_19;
	using Module_BodyPose::MODEL_JOINT_NUM_OP_25;
	using Module_BodyPose::SBody3DScene;
	using Module_BodyPose::CBody3D;
	if(g_syncMan.IsLoaded()==false)
	{
		printf("## ERROR:: You should have valid sync index table\n");
		return;
	}

	//int offset=1;;
	char saveFolderPath[512];
	if(g_bodyPoseManager.m_skeletonHierarchy.size()==MODEL_JOINT_NUM_COCO_19)
		sprintf(saveFolderPath,"%s/coco19_body3DPSRecon_updated_vga_hdidx",g_dataMainFolder);
	else
	{
		assert(g_bodyPoseManager.m_skeletonHierarchy.size()==MODEL_JOINT_NUM_OP_25);
		sprintf(saveFolderPath,"%s/op25_body3DPSRecon_updated_vga_hdidx",g_dataMainFolder);
	}
	CreateFolder(saveFolderPath);

	//Find closest HD from VGA
	int vga_first_frame = g_bodyPoseManager.m_3DPSPoseMemVector.front().m_imgFramdIdx;
	int hd_first_frame = g_syncMan.ClosestHDfromVGA(vga_first_frame);
	int vga_last_frame = g_bodyPoseManager.m_3DPSPoseMemVector.back().m_imgFramdIdx;
	int hd_last_frame = g_syncMan.ClosestHDfromVGA(vga_last_frame);

	for(int hdIdx= hd_first_frame;hdIdx<=hd_last_frame ;++hdIdx)
	{
		vector< pair<int,double> > vgaIdxWithOff;
		bool success = g_syncMan.ClosestVGAsWithOffInfofromHD(hdIdx,vgaIdxWithOff);		//note the +1 offset
		if(success==false && vgaIdxWithOff.size()<=1)
		{
			int vgaIdx = g_syncMan.ClosestVGAfromHD(hdIdx);
			int idx = vgaIdx - vga_first_frame;
			if(idx<0 || idx>g_bodyPoseManager.m_3DPSPoseMemVector.size())
			{
				printf("ERROR: idx<0 || idx>g_bodyPoseManager.m_3DPSPoseMemVector.size()\n");
				continue;
			}
			SaveBodyReconResult(saveFolderPath,g_bodyPoseManager.m_3DPSPoseMemVector[idx],hdIdx,true);
		}
		else   //hdIdxWithOff.size() ==2
		{
			int vgaIdx_before = vgaIdxWithOff.front().first ;//- offset ;		//note the +1 offset
			float offset_before = abs(vgaIdxWithOff.front().second);

			int vgaIdx_after = vgaIdxWithOff.back().first;// - offset ;		//note the +1 offset
			float offset_after = abs(vgaIdxWithOff.back().second);

			SBody3DScene newBodyScene;

			float offset_sum  = offset_before+offset_after;
			for(int h=0;h<g_bodyPoseManager.m_skeletalTrajProposals.size();++h)
			{
				CBody3D* pBody_before=NULL;
				CBody3D* pBody_after=NULL;
				if(g_bodyPoseManager.m_skeletalTrajProposals[h].GetBody3DFromFrameIdx(vgaIdx_before,&pBody_before)==false)
					continue;

				if(g_bodyPoseManager.m_skeletalTrajProposals[h].GetBody3DFromFrameIdx(vgaIdx_after,&pBody_after)==false)
					continue;

				newBodyScene.m_articBodyAtEachTime.push_back(*pBody_before);  
				CBody3D& targetBody = newBodyScene.m_articBodyAtEachTime.back();
				for(int j=0;j<targetBody.m_jointPosVect.size();++j)
				{
					// targetBody.m_jointPosVect[j] = (offset_after*pBody_before->m_jointPosVect[j] + offset_before*pBody_after->m_jointPosVect[j])/offset_sum;
					const auto sum = offset_after*pBody_before->m_jointPosVect[j] + offset_before*pBody_after->m_jointPosVect[j];
					targetBody.m_jointPosVect[j].x = sum.x / offset_sum;
					targetBody.m_jointPosVect[j].y = sum.y / offset_sum;
					targetBody.m_jointPosVect[j].z = sum.z / offset_sum;
				}
			}
			SaveBodyReconResult(saveFolderPath,newBodyScene,hdIdx ,true);			//note the offset (because current sync table has 1 frame offset to real index)
		}
	}
}

void Script_HD_Load_body3DPS_byFrame(bool bFromSpecificFolder,bool bCoco19) 
{
	using Module_BodyPose::g_bodyPoseManager;
	g_bodyPoseManager.m_3DPSPoseMemVector.clear();
	g_bodyPoseManager.SetfpsType(FPS_HD_30);

	//////////////////////////////////////////////////////////////////////////
	/// Try to load from the following orders:
	//	(g_dataMainFolder)/body_mpm/coco19_body3DPSRecon_updated_vga_hdidx
	//	(g_dataMainFolder)/body3DPSRecon_updated_vga_hdidx/
	//	(g_dataMainFolder)/body3DPSRecon_updated/hd

	vector<string> folderPathCand;
	char fullPath[512];
	if(bFromSpecificFolder)
		folderPathCand.push_back(string(g_dataSpecificFolder));
	sprintf(fullPath,"%s/body_mpm/coco19_body3DPSRecon_updated_vga_hdidx",g_dataMainFolder);//m_domeImageManager.m_currentFrame);
	folderPathCand.push_back(string(fullPath));
	sprintf(fullPath,"%s/body_mpm/coco19_body3DPSRecon_updated_vga_hdidx_140",g_dataMainFolder);//m_domeImageManager.m_currentFrame);
	folderPathCand.push_back(string(fullPath));
	sprintf(fullPath,"%s/body3DPSRecon_updated_vga_hdidx",g_dataMainFolder);//m_domeImageManager.m_currentFrame);
	folderPathCand.push_back(string(fullPath));
	sprintf(fullPath,"%s/body3DPSRecon_updated/hd",g_dataMainFolder);//m_domeImageManager.m_currentFrame);
	folderPathCand.push_back(string(fullPath));
	if(bFromSpecificFolder==false)
		folderPathCand.push_back(string(g_dataSpecificFolder));

	char validFolderPath[512];
	bool bFound=false;
	for(int i=0;i<folderPathCand.size();++i)
	{
		if(IsFileExist(folderPathCand[i].c_str()))
		{
			bFound = true;
			sprintf(validFolderPath,"%s",folderPathCand[i].c_str());
			printf("## Found a valid folder: %s",validFolderPath);
			break;
		}
	}
	if(bFound==false)
	{
		printf("Load failure from all possible folders:\n");
		for(int i=0;i<folderPathCand.size();++i)
		{
			printf("\t Failed from: %s\n",folderPathCand[i].c_str());
		}
		return;
	}

	for(int f=0;f<g_poseEstLoadingDataNum;f++)
	{
		if(f%500==0)
		{
			printf("## Loading: %d/%d\n",f,g_poseEstLoadingDataNum);
		}
		//int targetFrameNum = g_poseEstLoadingDataFirstFrameIdx +f*5;
		int targetFrameNum = g_poseEstLoadingDataFirstFrameIdx + f;
		
		char fullPath[512];
		sprintf(fullPath,"%s/body3DScene_hd%08d.txt",validFolderPath,targetFrameNum);//m_domeImageManager.m_currentFrame);
		bool bSuccess = g_bodyPoseManager.LoadBodyReconResult(fullPath,targetFrameNum);
		if(bSuccess ==false)
			printf("Load failure from %s\n",fullPath);
	}
	g_bodyPoseManager.AssignHumanIdentityColorFor3DPSResult();
}

}
