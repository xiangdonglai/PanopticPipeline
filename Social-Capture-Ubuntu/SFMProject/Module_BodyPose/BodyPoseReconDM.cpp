#include "BodyPoseReconDM.h"
#include <iostream>
#include <fstream>
#include <list>
#include <opencv2/core/core.hpp>
#include <cstdio>
#include "Utility.h"
#include "UtilityGPU.h"
#include "PatchOptimization.h"

static CClock g_clock;
#define VOXEL_SIZE_CM 4
#define POSEMACHINE_NODESCORE_THRESHOLD_AVG 0.05
#define DATASCORE_RATIO 0

using namespace std;
using namespace cv;


namespace Module_BodyPose
{

CBodyPoseRecon g_bodyPoseManager;

//domeViews is required to load only the corresponding view's results
bool Load_Undist_PoseDetectMultipleCamResult_MultiPoseMachine_19jointFull(
	const char* poseDetectFolder,const char* poseDetectSaveFolder,
	const int currentFrame, CDomeImageManager& domeImMan,bool isHD)
{
	//Export Face Detect Results
	char fileName[512];
	char savefileName[512];
	if(isHD==false)
	{
		sprintf(fileName,"%s/poseDetectMC_%08d.txt",poseDetectFolder,currentFrame);
		sprintf(savefileName,"%s/poseDetectMC_%08d.txt",poseDetectSaveFolder,currentFrame);
	}
	else
	{
		sprintf(fileName,"%s/poseDetectMC_hd%08d.txt",poseDetectFolder,currentFrame);
		sprintf(savefileName,"%s/poseDetectMC_hd%08d.txt",poseDetectSaveFolder,currentFrame);
	}
	ifstream fin(fileName);
	if(fin.is_open()==false)
	{
		printf("LoadPoseDetectMultipleCamResult_PoseMachine:: Failed from %s\n\n",fileName);
		return false;
	}
	CreateFolder(poseDetectSaveFolder);
	ofstream fout(savefileName);

	printf("%s\n",fileName);
	char buf[512];
	fin >> buf;
	//fout << buf;
	fout << "ver_mpm_19";
	float version;	
	fin >> version;	
	fout << " " <<version <<"\n";

	if(version<0.1)
	{
		printf("## ERROR: version information is wrong: %f\n",version);
		fin.close();
		fout.close();
		return false;
	}

	int processedViews;	
	if(version>0.49)
	{
		fin >> buf;	//processedViews
		fin >> processedViews;	
		fout << buf << " " << processedViews <<"\n";
	}

	//for(int i=0;i<domeViews.size();++i)
	for(int i=0;   ;++i)
	{ 
		int frameIdx,panelIdx,camIdx,peopleNum,jointNum;
		fin >> frameIdx >> panelIdx >>camIdx >> peopleNum >> jointNum;
		if(fin.eof())
			break;

		fout << frameIdx <<" "<< panelIdx <<" " <<camIdx <<" "<< peopleNum <<" "<< jointNum <<"\n";

		CamViewDT* pCamDT = domeImMan.GetViewDTFromPanelCamIdx(panelIdx,camIdx);
		for(int p=0;p<peopleNum;++p)
		{
			for(int j=0;j<jointNum;++j)
			{
				Point2f tempPt;
				double score;
				fin >>tempPt.x >> tempPt.y >> score;  
				//printf("%f %f %f\n",tempPt.x,tempPt.y,score);
				// if(j>=18) // need to handle 25 pts from Op now
				//	continue;
				if(panelIdx ==14 && camIdx==18 || pCamDT==NULL)		//Not a valid parameters
				{
					fout << -1.0 << " " << -1.0 << " " << -1.0 <<" ";  
					continue;
				}
				Point2f idealPt = pCamDT->ApplyUndistort(tempPt);
				fout << idealPt.x << " " << idealPt.y << " " << score <<" ";  
			}
			fout <<"\n";
			if(version<0.31)
			{
				double dummy;
				fin >>dummy >> dummy >> dummy>> dummy;		//scale information. Not useful
			}
		}
	}
	fin.close();
	fout.close();
	return true;
}

CBodyPoseRecon::CBodyPoseRecon(void)
{
	m_curImgFrameIdx =-1;
	m_detectionCostVolumeDispThresh =-1;
	m_detectionEdgeCostDispThresh =-1;

	m_targetPoseReconUnitIdx= -1;
	m_targetPoseReconUnitIdx_kinect= -1;
	m_targetdetectHullManIdx = -1;

	m_nodePropMaxScore = m_nodePropMinScore = m_partPropMaxScore = m_partPropMinScore = 0;
	m_voxMarginalMaxScore = m_voxMarginalMinScore =0;

	m_loadedKinectNum =0;
	//////////////////////////////////////////////////////////////////////////
	// Kinect edge setting
	/* Joint order
	'AnkleLeft'0
	'AnkleRight'1
	'ElbowLeft'2
	'ElbowRight'3
	'FootLeft'4
	'FootRight'5
	'HandTipLeft6'		-->HandLeft
	'HandTipRight7'		-->HandRight
	'Head'8
	'HipLeft'9
	'HipRight'10
	'KneeLeft'11
	'KneeRight'12
	'Neck'13
	'ShoulderLeft'14
	'ShoulderRight'15
	'SpineBase'16
	'SpineMid'17
	'SpineShoulder'18
	'ThumbLeft'19
	'ThumbRight'20
	'WristLeft'21
	'WristRight'22
	*/

	//Body
	m_ourNodeIdxToKinect.push_back(18);
	m_ourNodeIdxToKinect.push_back(8);
	m_ourNodeIdxToKinect.push_back(16);
	//Left Limbs
	m_ourNodeIdxToKinect.push_back(14);
	m_ourNodeIdxToKinect.push_back(2);
	m_ourNodeIdxToKinect.push_back(21);

	m_ourNodeIdxToKinect.push_back(9);
	m_ourNodeIdxToKinect.push_back(11);
	m_ourNodeIdxToKinect.push_back(0);

	m_ourNodeIdxToKinect.push_back(15);
	m_ourNodeIdxToKinect.push_back(3);
	m_ourNodeIdxToKinect.push_back(22);

	m_ourNodeIdxToKinect.push_back(10);
	m_ourNodeIdxToKinect.push_back(12);
	m_ourNodeIdxToKinect.push_back(1);


	m_kinectEdges.push_back(make_pair(8,18));
	m_kinectEdges.push_back(make_pair(18,14));
	m_kinectEdges.push_back(make_pair(14,2));
	m_kinectEdges.push_back(make_pair(2,21));

	m_kinectEdges.push_back(make_pair(18,15));
	m_kinectEdges.push_back(make_pair(15,3));
	m_kinectEdges.push_back(make_pair(3,22));

	m_kinectEdges.push_back(make_pair(18,16));
	m_kinectEdges.push_back(make_pair(16,9));
	m_kinectEdges.push_back(make_pair(9,11));
	m_kinectEdges.push_back(make_pair(11,0));

	m_kinectEdges.push_back(make_pair(16,10));
	m_kinectEdges.push_back(make_pair(10,12));
	m_kinectEdges.push_back(make_pair(12,1));

	GenerateHumanColors();

	m_fpsType = FPS_VGA_25;
}

CBodyPoseRecon::~CBodyPoseRecon(void)
{
	for(int i=0;i<m_nodePropScoreVector.size();++i)
		delete m_nodePropScoreVector[i];
	m_nodePropScoreVector.clear();

	for(int i=0;i<m_skeletonHierarchy.size();++i)
		delete m_skeletonHierarchy[i];
	m_skeletonHierarchy.clear();

	for(int i=0;i<m_edgeCostVector.size();++i)
		delete m_edgeCostVector[i];
	m_edgeCostVector.clear();
}

void CBodyPoseRecon::GenerateHumanColors()
{
	if(m_colorSet.size()>0)
		return;
	
	m_colorSet.push_back(Point3f(127,255,0));
	m_colorSet.push_back(Point3f(0,206,209));
	
	m_colorSet.push_back(Point3f(128,0,0));
	m_colorSet.push_back(Point3f(153,50,204));


	m_colorSet.push_back(Point3f(220,20,60));
	m_colorSet.push_back(Point3f(0,128,0));
	m_colorSet.push_back(Point3f(70,130,180));
	m_colorSet.push_back(Point3f(255,20,147));


	m_colorSet.push_back(Point3f(240,128,128));
	m_colorSet.push_back(Point3f(0,250,154));
	m_colorSet.push_back(Point3f(0,0,128));
	m_colorSet.push_back(Point3f(210,105,30));

	m_colorSet.push_back(Point3f(255,165,0));
	m_colorSet.push_back(Point3f(32,178,170));
	m_colorSet.push_back(Point3f(123,104,238));
	
	for(int i=0;i<m_colorSet.size();++i)
	{
		m_colorSet[i].x =m_colorSet[i].x/255.0;
		m_colorSet[i].y =m_colorSet[i].y/255.0;
		m_colorSet[i].z =m_colorSet[i].z/255.0;
	}
}

// Donglai's comment: Load 2D detector output in which multiple views have been merged into a single file.
bool LoadPoseDetectMultipleCamResult_PoseMachine(const char* poseDetectFolder,const int currentFrame, const vector<CamViewDT*>& domeViews,vector< vector<SPose2D> >& detectedPoseVect,vector<int>& validCamIdxVect,bool isHD)
{
	detectedPoseVect.resize(domeViews.size());
	validCamIdxVect.reserve(domeViews.size());

	//Generation mapping between <panel,cam> -> idx
	map< pair<int,int> ,int> camNameToOrderIdx;
	for(int i=0;i<domeViews.size();++i)
	{
		pair<int,int> tempPair = make_pair(domeViews[i]->m_actualPanelIdx,domeViews[i]->m_actualCamIdx);
		camNameToOrderIdx[tempPair] = i;
	}

	//Export Face Detect Results
	char fileName[512];
	if(isHD==false)
		sprintf(fileName,"%s/poseDetectMC_%08d.txt",poseDetectFolder,currentFrame);
	else
		sprintf(fileName,"%s/poseDetectMC_hd%08d.txt",poseDetectFolder,currentFrame);
	ifstream fin(fileName);
	if(fin.is_open()==false)
	{
		printf("LoadPoseDetectMultipleCamResult_PoseMachine:: Failed from %s\n\n",fileName);
		return false;
	}
	//printf("%s\n",fileName);
	char buf[512];
	fin >> buf;
	float version;	
	fin >> version;	

	if(version<0.1)
	{
		printf("## ERROR: version information is wrong: %f\n",version);
		fin.close();
		return false;
	}

	int processedViews;	
	if(version>0.49)
	{
		fin >> buf;	//processedViews
		fin >> processedViews;	
	}

	if(processedViews!=detectedPoseVect.size())
	{
		printf("Warning: Missing cameras in the pose machine results: (asked) %d vs (pm) %d \n",detectedPoseVect.size(),processedViews);
	}

	//for(int i=0;i<domeViews.size();++i)
	for(int i=0;   ;++i)
	{ 
		int frameIdx,panelIdx,camIdx,peopleNum,jointNum;
		fin >> frameIdx >> panelIdx >>camIdx >> peopleNum >> jointNum;
		if(fin.eof())
			break;
		/*
		if(version<0.35)		 //there was a bug  in ver0.3
		{
			if(camIdx ==0 && peopleNum ==0) //if no people are detected, it mistakenly didn't write frameIdx...So camIdx becomes peopleNum
				continue;
		}*/

		int camOrderIdx; //= camNameToOrderIdx[ make_pair(panelIdx,camIdx)];
		map< pair<int,int> ,int>::iterator it = camNameToOrderIdx.find(make_pair(panelIdx,camIdx));
		/*printf("%d: %d,%d\n ",i,panelIdx,camIdx);
		if(panelIdx==6 && camIdx==3)
			printf("here\n");*/
		if(it == camNameToOrderIdx.end())		//check existence for sampled camera case
		{
			//Meaningless. To read data.  
			for(int p=0;p<peopleNum;++p)
			{
				//fin >> detectionScore;
				for(int j=0;j<jointNum;++j)
				{
					Point2f tempPt;
					double score;
					fin >>tempPt.x >> tempPt.y >> score;
				}
				if(version<0.31)
				{
					double dummy;
					fin >>dummy >> dummy >> dummy>> dummy;		//scale information. Not useful
				}
			}
			continue;
		}
		else
			camOrderIdx = it->second;


		if(domeViews[camOrderIdx]->m_actualPanelIdx != panelIdx || domeViews[camOrderIdx]->m_actualCamIdx!=camIdx)
		{
			printf("ERROR!!: %d_%d vs %d %d\n",domeViews[camOrderIdx]->m_actualPanelIdx,domeViews[camOrderIdx]->m_actualCamIdx,panelIdx,camIdx);
			fin.close();
			return false;
		}

		vector<SPose2D> &poseVector = detectedPoseVect[camOrderIdx];
		for(int p=0;p<peopleNum;++p)
		{
			poseVector.push_back(SPose2D());
			SPose2D& tempPose = poseVector.back();
			//fin >> tempPose.detectionScore;
			tempPose.detectionScore = 0;
			//tempPose.detectionScore +=1;		//so that 0<detectionScore<2
			for(int j=0;j<jointNum;++j)
			{
				Point2f tempPt;
				double score;
				fin >>tempPt.x >> tempPt.y >>score;  

				//score +=1;
				tempPose.bodyJoints.push_back(tempPt);
				tempPose.bodyJointScores.push_back(score);
			}
			if(version<0.31)
			{
				double dummy;
				fin >>dummy >> dummy >> dummy>> dummy;		//scale information. Not useful
			}
		}

		validCamIdxVect.push_back(camOrderIdx);
	}
	fin.close();
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// This code is for coco 19. Should be backward compatible (meaning it should work with 15 joint case)
////////////////////////////////////////////////////////////////////////////////////////////////
// Node Proposal and Part Proposal Generation
// From PoseMachine
// Don't consider mirroring (which is the only difference at this moment)
// Can be applied for both HD and VGA cases
// askedCamNum is only vaild if isHD==false
// if isHD==true, just use all HD cameras
void CBodyPoseRecon::ProbVolumeRecoe_nodePartProposals_fromPoseMachine_coco19(const char* dataMainFolder,const char* calibFolder,const int askedCamNum, const int frameIdx, bool bSaveCostMap, bool isHD, bool bCoco19)
{
	int modelJointNum;
	char poseDetectFolderName[512];
	if(bCoco19)
	{
		modelJointNum = MODEL_JOINT_NUM_COCO_19;
		sprintf(poseDetectFolderName,"coco19_poseDetect_pm");
	}
	else
	{
		modelJointNum = MODEL_JOINT_NUM_15;
		sprintf(poseDetectFolderName,"poseDetect_pm");
	}

	double gaussianBandwidth = 20;
	if(isHD)
		gaussianBandwidth =30;

	g_clock.tic();
	//To compute working volume, we should load full dome
	CDomeImageManager fullDome;
	fullDome.InitDomeCamOnlyByExtrinsic(calibFolder);
	fullDome.CalcWorldScaleParameters();
	fullDome.DeleteMemory();

	//Parameter setting
	CDomeImageManager domeImageManager;
	domeImageManager.SetCalibFolderPath(calibFolder);
	if(isHD==false)
		domeImageManager.InitDomeCamVgaHd(askedCamNum,false);
	else
		domeImageManager.InitDomeCamVgaHdKinect(0,CDomeImageManager::LOAD_SENSORS_HD);

	domeImageManager.SetFrameIdx(frameIdx);		//Should set this, because domeImageManager.GetFrameIdx() is used 

	//////////////////////////////////////////////////////////////////////////
	/// Load detected 2d pose. Save outer vector is for each image, and inner vector is for each instance (person)
	vector< vector<SPose2D> > detectedPoseVect;				//this order should be the same as enum PoseMachine2DJointEnum 
	char poseDetectFolder[128];
	if(isHD==false)
		sprintf(poseDetectFolder,"%s/%s/vga_25",dataMainFolder,poseDetectFolderName);
	else
		sprintf(poseDetectFolder,"%s/%s/hd_30",dataMainFolder,poseDetectFolderName);
	vector<int> validCamIdxVect;
	bool isLoadDetect = LoadPoseDetectMultipleCamResult_PoseMachine(poseDetectFolder,domeImageManager.GetFrameIdx(),domeImageManager.m_domeViews,detectedPoseVect,validCamIdxVect,isHD);
	if(isLoadDetect==false)
	{
		printf("## Error: Cannot load pose detection results. \n");
		return;
	}
	else
		printf("## PoseDetect result has been successfully loaded \n");

	if(validCamIdxVect.size()!=domeImageManager.GetCameraNum())
	{
		printf("## Warning: askedCamNum (%d) is larger than actually valid camera number (%d)in detection\n", domeImageManager.m_domeViews.size(),validCamIdxVect.size());
		vector<int> newOrders;
		domeImageManager.DeleteSelectedCamerasByValidVect(validCamIdxVect,newOrders);
		vector< vector<SPose2D> > detectedPoseVect_updated;				
		detectedPoseVect_updated.reserve(validCamIdxVect.size());
		for(int i=0;i<newOrders.size();++i)
		{
			detectedPoseVect_updated.push_back(detectedPoseVect[newOrders[i]]);
		}
		detectedPoseVect =detectedPoseVect_updated;
		printf("## After Filtering: askedCamNum (%d) == valid camera number (%d) in detection\n", domeImageManager.m_domeViews.size(),validCamIdxVect.size());
	}

	int cameraNum  = domeImageManager.m_domeViews.size();
	//make voxelVect (can be substituted by Octree)
	//Visual Hull
	float xMin = DOME_VOLUMECUT_CENTER_PT.x - DOME_VOLUMECUT_RADIOUS;
	float xMax = DOME_VOLUMECUT_CENTER_PT.x + DOME_VOLUMECUT_RADIOUS;
	float yMin = DOME_VOLUMECUT_CENTER_PT.y - DOME_VOLUMECUT_RADIOUS*0.5;
	float yMax = DOME_VOLUMECUT_CENTER_PT.y + DOME_VOLUMECUT_RADIOUS*0.7;
	float zMin = DOME_VOLUMECUT_CENTER_PT.z - DOME_VOLUMECUT_RADIOUS;
	float zMax = DOME_VOLUMECUT_CENTER_PT.z + DOME_VOLUMECUT_RADIOUS;
	printfLog("DomeCenter: %f %f %f\n",DOME_VOLUMECUT_CENTER_PT.x,DOME_VOLUMECUT_CENTER_PT.y,DOME_VOLUMECUT_CENTER_PT.z);
	printfLog("VolumeSize: %f %f %f %f %f %f\n",xMin,xMax,yMin,yMax,zMin,zMax);
	float voxelSize_inCM = VOXEL_SIZE_CM;
	float voxelSize = voxelSize_inCM/WORLD_TO_CM_RATIO;		//CM to WorldSize
	printfLog("voxelSize %f cm --> %f(WU) \n",voxelSize_inCM,voxelSize);
	g_clock.toc("Loading Detection Data");

	//////////////////////////////////////////////////////////////////////////
	//Select target joints
	ConstructJointHierarchy(modelJointNum);		//jointNum 14: with the head top
	vector<BodyJointEnum> targetJointIdxVect;
	for(int i=0;i<m_skeletonHierarchy.size();++i)
	{
		targetJointIdxVect.push_back(m_skeletonHierarchy[i]->jointName);	
	}	
	g_clock.tic();
	//////////////////////////////////////////////////////////////////////////
	/// For each target joint,
	// 1. Probability volume Reconstruction (saved to m_nodePropScoreVector.back())
	// 2. Joint candidate after NMS and thresholding (saved to JointReconVector)
	CVisualHullManager* detectionHullManager =  new CVisualHullManager();
	m_nodePropScoreVector.push_back(detectionHullManager);		//push a CVisualHullManager which is describing current frames prob volumes
	detectionHullManager->m_imgFramdIdx =  domeImageManager.GetFrameIdx(); 
	detectionHullManager->m_visualHullRecon.resize(targetJointIdxVect.size()); //contains each joint's visual hull recon
	int jointNum = targetJointIdxVect.size();
	bool bCudaSuccessAll=true;
	//#pragma omp parallel for num_threads(10)
	#pragma omp parallel for
	for(int jj=0;jj<targetJointIdxVect.size();++jj)
	{
		BodyJointEnum targetJointIdx = targetJointIdxVect[jj];
		int targetJointFliipped = -1;

		vector<Mat_<float> > costMapVect;		//changed to float
		vector<Mat_<float> > projectMatVect;
		vector<Point3f> camCenterVect;
		projectMatVect.resize(detectedPoseVect.size());
		assert(cameraNum==detectedPoseVect.size());
		//////////////////////////////////////////////////////////////////////////
		/// Generate Cost Map for each detected 2D pose
		for(int camIdx=0;camIdx<detectedPoseVect.size();++camIdx)
		{
			pair< int, int> imageSize = domeImageManager.GetImageSize(camIdx);		//x, y
			Mat_<float> costMap = Mat_<float>::zeros(imageSize.second,imageSize.first);	//row, column
			for(int j=0;j<detectedPoseVect[camIdx].size();++j)
			{
				//Body Center is a special joint I have created. Center of JOINT_lHip and JOINT_rHip
				if(targetJointIdx==Joint_bodyCenter)
				{
					Point2f bodyCenter = detectedPoseVect[camIdx][j].bodyJoints[PM_JOINT_lHip] + detectedPoseVect[camIdx][j].bodyJoints[PM_JOINT_rHip];
					bodyCenter = bodyCenter*0.5;

					double score = detectedPoseVect[camIdx][j].bodyJointScores[PM_JOINT_lHip] + detectedPoseVect[camIdx][j].bodyJointScores[PM_JOINT_rHip];
					score =  score*0.5;
					PutGaussianKernel( costMap,bodyCenter,gaussianBandwidth,score);//detectedFaceVect[camIdx][j].detectionScore);
				}
				else 
				{
					PoseMachine2DJointEnum targetJointIdx_PoseMachine = m_map_devaToPoseMachineIdx[targetJointIdx];
					if(targetJointIdx_PoseMachine != PM_JOINT_Unknown)
					{
						double score = detectedPoseVect[camIdx][j].bodyJointScores[targetJointIdx_PoseMachine];
						PutGaussianKernel( costMap,detectedPoseVect[camIdx][j].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,score);//detectedFaceVect[camIdx][j].detectionScore);
					}
				}
			}
			
			costMapVect.push_back(costMap);
			domeImageManager.m_domeViews[camIdx]->m_P.convertTo(projectMatVect[camIdx],CV_32F);
			Point3f c = MatToPoint3d(domeImageManager.m_domeViews[camIdx]->m_CamCenter);
			camCenterVect.push_back(c);
		}

		if(bSaveCostMap)
		{
			for(int camIdx=0;camIdx<detectedPoseVect.size();++camIdx)
			{
				//	Save Cost map to images
				if(domeImageManager.m_domeViews[camIdx]->m_actualPanelIdx==14 && domeImageManager.m_domeViews[camIdx]->m_actualCamIdx==7)
				//if(domeImageManager.m_domeViews[camIdx]->m_actualCamIdx==4)
				{
					char fileName[128];
					sprintf(fileName,"c:/tempResult/poseCostMap/poseCostMap_p%d_c%d_f%d_j%d.txt",domeImageManager.m_domeViews[camIdx]->m_actualPanelIdx
						,domeImageManager.m_domeViews[camIdx]->m_actualCamIdx,domeImageManager.GetFrameIdx(),jj);
					ofstream fout(fileName,ios_base::trunc);
					for(int c=0;c<costMapVect[camIdx].cols;++c)
						for(int r=0;r<costMapVect[camIdx].rows;++r)
						{
							if(costMapVect[camIdx](r,c)>0)
							{
								fout << r <<" " <<c <<" "<<costMapVect[camIdx](r,c)<<"\n";
							}
						}
				
					fout.close();
					//imwrite(fileName,costMap);
				}
				/*if(domeImageManager.m_domeViews[camIdx]->m_actualPanelIdx==10 && domeImageManager.m_domeViews[camIdx]->m_actualCamIdx==4)
				{
					imshow("test",costMap);
					cvWaitKey(0);
				}*/
			}
		}

		//////////////////////////////////////////////////////////////////////////
		/// Cost Volume Generation
		CVisualHull& tempVisualHull = detectionHullManager->m_visualHullRecon[jj];
		tempVisualHull.SetParameters(xMin,xMax,yMin,yMax,zMin,zMax,voxelSize);
		tempVisualHull.AllocMemory();		//needed to transform between vIdx to pos and vice verca (we don't need mem allocation but..I was too lazy)
		tempVisualHull.m_actualFrameIdx = domeImageManager.GetFrameIdx();			//same as detectionHullManager.m_imageFrameIdx
		//printfLog("Voxel Num %d\n",tempVisualHull.GetVoxelNum());
		//g_clock.tic();
		int voxelNum = tempVisualHull.GetVoxelNum();
		//printfLog("Raw Voxel Number : %d\n",voxelNum);
		float* voxelCoordVect = new float[voxelNum*3];
		//#pragma omp parallel for
		for(int voxelIdx=0;voxelIdx<voxelNum;++voxelIdx)
		{
			Point3f tempCoord = tempVisualHull.GetVoxelPos(voxelIdx) ;
			voxelCoordVect[3*voxelIdx ] = tempCoord.x;
			voxelCoordVect[3*voxelIdx +1] = tempCoord.y;
			voxelCoordVect[3*voxelIdx +2] = tempCoord.z;
		}
		float* occupancyOutput = new float[voxelNum];
		memset(occupancyOutput,0,voxelNum*sizeof(float));
		bool bCudaSuccess;

		if(isHD ==false)
			bCudaSuccess = DetectionCostVolumeGeneration_float_GPU_average_vga(voxelCoordVect, voxelNum, costMapVect,projectMatVect,occupancyOutput);
		else
			bCudaSuccess = DetectionCostVolumeGeneration_float_GPU_average_hd(voxelCoordVect, voxelNum, costMapVect,projectMatVect,occupancyOutput);
		if(bCudaSuccess==false)
			bCudaSuccessAll = false;

		//////////////////////////////////////////////////////////////////////////
		//Non-Maximum suppression
		vector<int> nonMaxVoxIdx;		//keep the index of local max voxels
		int bandwidth =3;
		for(int voxelIdx=0;voxelIdx<voxelNum;++voxelIdx)
		{
			if(occupancyOutput[voxelIdx]>0)
			{
				float currentProb = occupancyOutput[voxelIdx];
				vector<int> neighVoxIdx;
				tempVisualHull.GetNeighborVoxIdx(voxelIdx,neighVoxIdx,1);
				bool bLocalMax = true;
				for(int i=0;i<neighVoxIdx.size();++i)
				{
					if(currentProb<  occupancyOutput[neighVoxIdx[i]])
					{
						bLocalMax = false;
						break;
					}
				}
				if(bLocalMax)
					nonMaxVoxIdx.push_back(voxelIdx);
			}
		}

		//Non Max Suppression and face pose estimation
		float sphereSize_CM = 30.0;
		float MAGIC_OFFSET = 1000;		//added to flag valid voxels (subtracted again after all others are eliminated)
		int sphereSize = (sphereSize_CM/VOXEL_SIZE_CM);
		for(int i=0;i<nonMaxVoxIdx.size();++i)
		{
			int voxelIdx = nonMaxVoxIdx[i];
			vector<int> neighVoxIdx;
			if(occupancyOutput[voxelIdx] <MAGIC_OFFSET)
				occupancyOutput[voxelIdx] +=MAGIC_OFFSET;
		}
		//Eliminate outliers
		for(int voxelIdx=0;voxelIdx<voxelNum;++voxelIdx)
		{
			if(occupancyOutput[voxelIdx]<MAGIC_OFFSET)
				occupancyOutput[voxelIdx] =0;
			else
				occupancyOutput[voxelIdx] -=MAGIC_OFFSET;
		}

		//////////////////////////////////////////////////////////////////////////
		//Save volumeRecon as surfaceVoxels for visualization
		tempVisualHull.m_surfaceVoxelOriginal.clear();
		double minCost=1e5;
		double maxCost=-1e5;
		for(int voxelIdx=0;voxelIdx<voxelNum;++voxelIdx)
		{
			if(occupancyOutput[voxelIdx] >POSEMACHINE_NODESCORE_THRESHOLD_AVG)
			{
				Point3f voxPos = tempVisualHull.GetVoxelPos(voxelIdx);
				int colorIdx = occupancyOutput[voxelIdx]*255;
				Point3f voxColor(1.0f, 1.0f, 1.0f);  // Donglai's comment: no need for visualization, put random color here.!
				tempVisualHull.m_surfaceVoxelOriginal.push_back(SurfVoxelUnit(voxPos,voxColor));
				tempVisualHull.m_surfaceVoxelOriginal.back().voxelIdx = voxelIdx;
				tempVisualHull.m_surfaceVoxelOriginal.back().prob = occupancyOutput[voxelIdx];

				if(minCost >occupancyOutput[voxelIdx])
					minCost =occupancyOutput[voxelIdx];
				if(maxCost <occupancyOutput[voxelIdx])
					maxCost =occupancyOutput[voxelIdx];
			}
		}
		printfLog("jj: %d: Final Voxel Number : %d:: minCost %f, maxCost %f\n",jj,tempVisualHull.m_surfaceVoxelOriginal.size(),minCost,maxCost);

		tempVisualHull.deleteMemory();	//Volume is no longer needed
		delete[] voxelCoordVect;
		delete[] occupancyOutput;
	}//end of for(int j=0;j<targetJointIdx.size();++j)
	g_clock.toc("Node Proposal Generation");
	if(bCudaSuccessAll==false)
		return;

	//////////////////////////////////////////////////////////////////////////
	/// Part Proposal generation
	g_clock.tic();
	gaussianBandwidth = gaussianBandwidth*0.5;
	//////////////////////////////////////////////////////////////////////////
	/// Set 2D group Map: Left (100) - right (200) 
	vector< vector< Mat_<Vec3b> > > costMap;   //costMap[camIdx][jointIdx]
	costMap.resize(detectedPoseVect.size());
	for(int c=0;c<costMap.size();++c)
	{
		costMap[c].resize(modelJointNum);
		Mat& inputImage = domeImageManager.m_domeViews[c]->m_inputImage;
		for(int j=0;j<costMap[c].size();++j)
		{
			costMap[c][j] = Mat_<Vec3b>::zeros(inputImage.rows,inputImage.cols);
		}
	}

	int LEFT = 100;
	int RIGHT = 200;
	int MAGIC_NUM = 100;		//I need this to distinguish jointRegions from background
	#pragma omp parallel for //num_threads(10)
	for(int camIdx=0;camIdx<detectedPoseVect.size();++camIdx)
	{
		//Mat& inputImage = domeImageManager.m_domeViews[camIdx]->m_inputImage;
		//costMap[camIdx] = Mat_<Vec3b>::zeros(inputImage.rows,inputImage.cols);

		for(int h=0;h<detectedPoseVect[camIdx].size();++h)	//for h is the index for each person in camera of the camIdx 
		{
			for(int jj=0;jj<targetJointIdxVect.size();++jj)		//jj is my joint index
			{
				BodyJointEnum targetJointIdx = targetJointIdxVect[jj];
				PoseMachine2DJointEnum targetJointIdx_PoseMachine = m_map_devaToPoseMachineIdx[targetJointIdx];
					
				if(targetJointIdx>=JOINT_lShoulder && targetJointIdx<=JOINT_lFoot)
				{
					Scalar tempColor(jj + MAGIC_NUM, h,LEFT);				//(jointIDx, humanIdx, LEFT RIGHT flag)
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				else if(targetJointIdx>=JOINT_rShoulder && targetJointIdx<=JOINT_rFoot)
				{
					//int mirroredIdx = jj - (jointNum-3)/2;			//I forgot this.. what a "huge" bug....
					//Scalar tempColor(mirroredIdx + MAGIC_NUM, h,RIGHT);
					Scalar tempColor(jj + MAGIC_NUM, h,RIGHT);
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,200+mirroredIdx,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				else if(targetJointIdx==JOINT_headTop || targetJointIdx==JOINT_neck)
				{
					Scalar tempColor(jj + MAGIC_NUM, h,LEFT);
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				//Coco additional
				else if(targetJointIdx==JOINT_lEye || targetJointIdx==JOINT_lEar)
				{
					Scalar tempColor(jj + MAGIC_NUM, h,LEFT);
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				else if(targetJointIdx==JOINT_rEye || targetJointIdx==JOINT_rEar)
				{
					Scalar tempColor(jj + MAGIC_NUM, h,RIGHT);
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
			}
			Point2f bodyCenter = detectedPoseVect[camIdx][h].bodyJoints[PM_JOINT_lHip] + detectedPoseVect[camIdx][h].bodyJoints[PM_JOINT_rHip];
			bodyCenter = bodyCenter*0.5;
			int bodyCenterJointidxInVector = 2;			//index in targetJointIdxVect
			Scalar tempColor(bodyCenterJointidxInVector+ MAGIC_NUM, h,LEFT);
			circle( costMap[camIdx][SMC_BodyJoint_bodyCenter],bodyCenter,gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
			//circle( costMap[camIdx],bodyCenter,gaussianBandwidth,100+bodyCenterJointidxInVector,-1);//detectedFaceVect[camIdx][j].detectionScore);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// Part Proposal generation
	//0 1 2 // 3 4 5   6 7 8 // 9 10 11   12 13 14 
	int iterEnd = m_skeletonHierarchy.size();
	int iterStart = 1;		//start from head top. This value was 3 before, where I didn't consider head and body
	//if(bDoJointMirroring)
		//iterEnd = 3+(m_skeletonHierarchy.size()-3)/2;
	float increasingpStepUnit = 1.0/detectedPoseVect.size(); 
	#pragma omp parallel for
	for(int jointIdx=iterStart;jointIdx<iterEnd;++jointIdx)	
	{
		if(m_skeletonHierarchy[jointIdx]->parents.size()==0)
			continue;

		STreeElement* childTreeNode = m_skeletonHierarchy[jointIdx];
		STreeElement* parentTreeNode = m_skeletonHierarchy[jointIdx]->parents.front();		//WARNING!!! I am assuming single child mode
		int parentJointidx = parentTreeNode->idxInTree;
		vector< SurfVoxelUnit >& surfVoxChild = detectionHullManager->m_visualHullRecon[jointIdx].m_surfaceVoxelOriginal;
		vector< SurfVoxelUnit >& surfVoxParent  = detectionHullManager->m_visualHullRecon[parentTreeNode->idxInTree].m_surfaceVoxelOriginal;

		//printf("Edge generation: childJoint %d, parentJoint %d\n",jointIdx,parentJointidx);

		//initialize edge strength vectors
		childTreeNode->childParent_partScore.clear();			//should throw away previous data
		childTreeNode->childParent_counter.clear();			//should throw away previous data

		childTreeNode->childParent_partScore.resize(surfVoxChild.size());
		childTreeNode->childParent_counter.resize(surfVoxChild.size());
		for(int c=0;c<surfVoxChild.size();++c)
		{
			childTreeNode->childParent_partScore[c].resize(surfVoxParent.size(),0);
			childTreeNode->childParent_counter[c].resize(surfVoxParent.size(),0);
		}
		//For every possible pair between current joint's node and parent's node
		for(int c=0;c<surfVoxChild.size();++c)
		{
			for(int camIdx=0;camIdx<detectedPoseVect.size();++camIdx)
			{
				Point2d childImgPt = Project3DPt(surfVoxChild[c].pos, domeImageManager.m_domeViews[camIdx]->m_P);
				if(IsOutofBoundary(costMap[camIdx][jointIdx],childImgPt.x,childImgPt.y))
					continue;

				Vec3b childLabel = costMap[camIdx][jointIdx].at<Vec3b>(childImgPt.y,childImgPt.x);
				if(childLabel[0]-MAGIC_NUM != jointIdx)		//not for this joint
					continue;
				int childWhichSideLabel =childLabel[2];		//either 100 or 200
				
				for(int p=0;p<surfVoxParent.size();++p)
				{
					float edgeDist = Distance(surfVoxChild[c].pos,surfVoxParent[p].pos);
					float expected = childTreeNode->parent_jointLength.front();
					if(abs(expected-edgeDist)>expected)		//reject if the length is too long or short
						continue;
					double weightByNodes = (surfVoxChild[c].prob + surfVoxParent[p].prob)/2.0;
					Point2d parentImgPt = Project3DPt(surfVoxParent[p].pos, domeImageManager.m_domeViews[camIdx]->m_P);
					if(IsOutofBoundary(costMap[camIdx][parentJointidx],parentImgPt.x,parentImgPt.y))
						continue;

					Vec3b parentLabel = costMap[camIdx][parentJointidx].at<Vec3b>(parentImgPt.y,parentImgPt.x);
					//if(parentLabel<100)
					if(parentLabel[0]-MAGIC_NUM != parentJointidx)		//not for this joint
						continue;
					int parentWhichSideLabel = parentLabel[2];		//either 100 or 200

					childTreeNode->childParent_counter[c][p] +=1;		//for taking average.. consider the view whenever the part is inside of image. I put this here so that the "non-detected" view provided kind of "negative" voting

					//if the parent is either neck or bodyCenter, side label doesn't matter. Only human label is important
					//For ms coco, if parent if head top (actually nose), side doesn't matter (left/right eye) 
					if(jointIdx==SMC_BodyJoint_lShoulder || jointIdx==SMC_BodyJoint_lHip || jointIdx==SMC_BodyJoint_rShoulder || jointIdx==SMC_BodyJoint_rHip 
									|| jointIdx==SMC_BodyJoint_lEye || jointIdx==SMC_BodyJoint_rEye)			//bug fixed Neck and Bodycenter has label 100 which was never connected with 200 (right Shoulder and Hip)
					{
						if(childLabel[1] == parentLabel[1])		//human label check
						{
							//childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes*increasingpStepUnit;		//max value become 1
							childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes;		//sum
						}
					}
					else if(childWhichSideLabel == parentWhichSideLabel && childLabel[1] == parentLabel[1]) // childLabel[1]  and parentLabel[1] means human index
					{
						//childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes*increasingpStepUnit;		//max value become 1
						childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes;	//sum
					//	childTreeNode->childParent_counter[c][p] +=1;	//for taking average
					}
				}
			}

			//Taking average
			for(int p=0;p<surfVoxParent.size();++p)
			{
				if(childTreeNode->childParent_counter[c][p]>0)
					childTreeNode->childParent_partScore[c][p] /=childTreeNode->childParent_counter[c][p];
			}
		}
	}

	g_clock.toc("Part Proposal Generation\n");

	//Save the edge result to m_edgeCostVector for visualization
	//Todo: only use m_edgeCostVector to avoid copy. (Not the one in the  m_skeleton
	SEdgeCostVector* pEdgeCostUnit = new SEdgeCostVector();
	m_edgeCostVector.push_back(pEdgeCostUnit);
	pEdgeCostUnit->parent_connectivity_vis.resize(m_skeletonHierarchy.size());
	for(int jointIdx=0;jointIdx<m_skeletonHierarchy.size();++jointIdx)	
	{
		pEdgeCostUnit->parent_connectivity_vis[jointIdx] = m_skeletonHierarchy[jointIdx]->childParent_partScore;
	}

	printf("done\n");

	g_clock.tic();
	SaveNodePartProposals(dataMainFolder,g_askedVGACamNum,domeImageManager.GetFrameIdx(),detectionHullManager,m_skeletonHierarchy,validCamIdxVect.size(),isHD);
	g_clock.toc("SaveNodePartProposals");
}


void CBodyPoseRecon::ProbVolumeRecoe_nodePartProposals_op25(const char* dataMainFolder,const char* calibFolder,const int askedCamNum, const int frameIdx,bool bSaveCostMap,bool isHD)
{
	int modelJointNum;
	char poseDetectFolderName[512];

	modelJointNum= MODEL_JOINT_NUM_OP_25;
	sprintf(poseDetectFolderName,"op25_poseDetect_pm");

	double gaussianBandwidth = 20;
	if(isHD)
		gaussianBandwidth =30;

	g_clock.tic();
	//To compute working volume, we should load full dome
	CDomeImageManager fullDome;
	fullDome.InitDomeCamOnlyByExtrinsic(calibFolder);
	fullDome.CalcWorldScaleParameters();
	fullDome.DeleteMemory();

	//Parameter setting
	CDomeImageManager domeImageManager;
	domeImageManager.SetCalibFolderPath(calibFolder);
	if(isHD==false)
		domeImageManager.InitDomeCamVgaHd(askedCamNum,false);
	else
		domeImageManager.InitDomeCamVgaHdKinect(0,CDomeImageManager::LOAD_SENSORS_HD);

	domeImageManager.SetFrameIdx(frameIdx);		//Should set this, because domeImageManager.GetFrameIdx() is used 

	//////////////////////////////////////////////////////////////////////////
	/// Load detected 2d pose. Save outer vector is for each image, and inner vector is for each instance (person)
	vector< vector<SPose2D> > detectedPoseVect;				//this order should be the same as enum PoseMachine2DJointEnum 
	char poseDetectFolder[128];
	if(isHD==false)
		sprintf(poseDetectFolder,"%s/%s/vga_25",dataMainFolder,poseDetectFolderName);
	else
		sprintf(poseDetectFolder,"%s/%s/hd_30",dataMainFolder,poseDetectFolderName);
	vector<int> validCamIdxVect;
	bool isLoadDetect = Module_BodyPose::LoadPoseDetectMultipleCamResult_PoseMachine(poseDetectFolder,domeImageManager.GetFrameIdx(),domeImageManager.m_domeViews,detectedPoseVect,validCamIdxVect,isHD);
	if(isLoadDetect==false)
	{
		printf("## Error: Cannot load pose detection results. \n");
		return;
	}
	else
		printf("## PoseDetect result has been successfully loaded \n");

	if(validCamIdxVect.size()!=domeImageManager.GetCameraNum())
	{
		printf("## Warning: askedCamNum (%d) is larger than actually valid camera number (%d)in detection\n", domeImageManager.m_domeViews.size(),validCamIdxVect.size());
		vector<int> newOrders;
		domeImageManager.DeleteSelectedCamerasByValidVect(validCamIdxVect,newOrders);
		vector< vector<SPose2D> > detectedPoseVect_updated;				
		detectedPoseVect_updated.reserve(validCamIdxVect.size());
		for(int i=0;i<newOrders.size();++i)
		{
			detectedPoseVect_updated.push_back(detectedPoseVect[newOrders[i]]);
		}
		detectedPoseVect =detectedPoseVect_updated;
		printf("## After Filtering: askedCamNum (%d) == valid camera number (%d) in detection\n", domeImageManager.m_domeViews.size(),validCamIdxVect.size());
	}

	int cameraNum  = domeImageManager.m_domeViews.size();
	//make voxelVect (can be substituted by Octree)
	//Visual Hull
	float xMin = DOME_VOLUMECUT_CENTER_PT.x - DOME_VOLUMECUT_RADIOUS;
	float xMax = DOME_VOLUMECUT_CENTER_PT.x + DOME_VOLUMECUT_RADIOUS;
	float yMin = DOME_VOLUMECUT_CENTER_PT.y - DOME_VOLUMECUT_RADIOUS*0.5;
	float yMax = DOME_VOLUMECUT_CENTER_PT.y + DOME_VOLUMECUT_RADIOUS*0.7;
	float zMin = DOME_VOLUMECUT_CENTER_PT.z - DOME_VOLUMECUT_RADIOUS;
	float zMax = DOME_VOLUMECUT_CENTER_PT.z + DOME_VOLUMECUT_RADIOUS;
	printfLog("DomeCenter: %f %f %f\n",DOME_VOLUMECUT_CENTER_PT.x,DOME_VOLUMECUT_CENTER_PT.y,DOME_VOLUMECUT_CENTER_PT.z);
	printfLog("VolumeSize: %f %f %f %f %f %f\n",xMin,xMax,yMin,yMax,zMin,zMax);
	float voxelSize_inCM = VOXEL_SIZE_CM;
	float voxelSize = voxelSize_inCM/WORLD_TO_CM_RATIO;		//CM to WorldSize
	printfLog("voxelSize %f cm --> %f(WU) \n",voxelSize_inCM,voxelSize);
	g_clock.toc("Loading Detection Data");

	//////////////////////////////////////////////////////////////////////////
	//Select target joints
	ConstructJointHierarchy(modelJointNum);		//jointNum 14: with the head top
	vector<BodyJointEnum> targetJointIdxVect;
	for(int i=0;i<m_skeletonHierarchy.size();++i)
	{
		targetJointIdxVect.push_back(m_skeletonHierarchy[i]->jointName);	
	}	
	g_clock.tic();
	//////////////////////////////////////////////////////////////////////////
	/// For each target joint,
	// 1. Probability volume Reconstruction (saved to m_nodePropScoreVector.back())
	// 2. Joint candidate after NMS and thresholding (saved to JointReconVector)
	CVisualHullManager* detectionHullManager =  new CVisualHullManager();
	m_nodePropScoreVector.push_back(detectionHullManager);		//push a CVisualHullManager which is describing current frames prob volumes
	detectionHullManager->m_imgFramdIdx =  domeImageManager.GetFrameIdx(); 
	detectionHullManager->m_visualHullRecon.resize(targetJointIdxVect.size()); //contains each joint's visual hull recon
	int jointNum = targetJointIdxVect.size();
	bool bCudaSuccessAll=true;
	//#pragma omp parallel for num_threads(10)
	#pragma omp parallel for
	for(int jj=0;jj<targetJointIdxVect.size();++jj)
	{
		BodyJointEnum targetJointIdx = targetJointIdxVect[jj];
		int targetJointFliipped = -1;

		vector<Mat_<float> > costMapVect;		//changed to float
		vector<Mat_<float> > projectMatVect;
		vector<Point3f> camCenterVect;
		projectMatVect.resize(detectedPoseVect.size());
		assert(cameraNum==detectedPoseVect.size());
		//////////////////////////////////////////////////////////////////////////
		/// Generate Cost Map for each detected 2D pose
		for(int camIdx=0;camIdx<detectedPoseVect.size();++camIdx)
		{
			pair< int, int> imageSize = domeImageManager.GetImageSize(camIdx);		//x, y
			Mat_<float> costMap = Mat_<float>::zeros(imageSize.second,imageSize.first);	//row, column
			for(int j=0;j<detectedPoseVect[camIdx].size();++j)
			{
				if(targetJointIdx==Joint_bodyCenter)
				{
					Point2f bodyCenter = detectedPoseVect[camIdx][j].bodyJoints[OP_JOINT_LHip] + detectedPoseVect[camIdx][j].bodyJoints[OP_JOINT_RHip];
					bodyCenter = bodyCenter*0.5;

					double score = detectedPoseVect[camIdx][j].bodyJointScores[OP_JOINT_LHip] + detectedPoseVect[camIdx][j].bodyJointScores[OP_JOINT_RHip];
					score =  score*0.5;
					PutGaussianKernel( costMap,bodyCenter,gaussianBandwidth,score);//detectedFaceVect[camIdx][j].detectionScore);
				}
				else
				{
					OpenPose25JointEnum targetJointIdx_OpenPose = m_map_devaToOpenPoseIdx[targetJointIdx];
					if(targetJointIdx_OpenPose != OP_JOINT_Unknown)
					{
						double score = detectedPoseVect[camIdx][j].bodyJointScores[targetJointIdx_OpenPose];
						PutGaussianKernel( costMap,detectedPoseVect[camIdx][j].bodyJoints[targetJointIdx_OpenPose],gaussianBandwidth,score);//detectedFaceVect[camIdx][j].detectionScore);
					}
				}
			}
			
			costMapVect.push_back(costMap);
			domeImageManager.m_domeViews[camIdx]->m_P.convertTo(projectMatVect[camIdx],CV_32F);
			Point3f c = MatToPoint3d(domeImageManager.m_domeViews[camIdx]->m_CamCenter);
			camCenterVect.push_back(c);
		}

		if(bSaveCostMap)
		{
			for(int camIdx=0;camIdx<detectedPoseVect.size();++camIdx)
			{
				//	Save Cost map to images
				if(domeImageManager.m_domeViews[camIdx]->m_actualPanelIdx==14 && domeImageManager.m_domeViews[camIdx]->m_actualCamIdx==7)
				//if(domeImageManager.m_domeViews[camIdx]->m_actualCamIdx==4)
				{
					char fileName[128];
					sprintf(fileName,"c:/tempResult/poseCostMap/poseCostMap_p%d_c%d_f%d_j%d.txt",domeImageManager.m_domeViews[camIdx]->m_actualPanelIdx
						,domeImageManager.m_domeViews[camIdx]->m_actualCamIdx,domeImageManager.GetFrameIdx(),jj);
					ofstream fout(fileName,ios_base::trunc);
					for(int c=0;c<costMapVect[camIdx].cols;++c)
						for(int r=0;r<costMapVect[camIdx].rows;++r)
						{
							if(costMapVect[camIdx](r,c)>0)
							{
								fout << r <<" " <<c <<" "<<costMapVect[camIdx](r,c)<<"\n";
							}
						}
				
					fout.close();
					//imwrite(fileName,costMap);
				}
				/*if(domeImageManager.m_domeViews[camIdx]->m_actualPanelIdx==10 && domeImageManager.m_domeViews[camIdx]->m_actualCamIdx==4)
				{
					imshow("test",costMap);
					cvWaitKey(0);
				}*/
			}
		}

		//////////////////////////////////////////////////////////////////////////
		/// Cost Volume Generation
		CVisualHull& tempVisualHull = detectionHullManager->m_visualHullRecon[jj];
		tempVisualHull.SetParameters(xMin,xMax,yMin,yMax,zMin,zMax,voxelSize);
		tempVisualHull.AllocMemory();		//needed to transform between vIdx to pos and vice verca (we don't need mem allocation but..I was too lazy)
		tempVisualHull.m_actualFrameIdx = domeImageManager.GetFrameIdx();			//same as detectionHullManager.m_imageFrameIdx
		//printfLog("Voxel Num %d\n",tempVisualHull.GetVoxelNum());
		//g_clock.tic();
		int voxelNum = tempVisualHull.GetVoxelNum();
		//printfLog("Raw Voxel Number : %d\n",voxelNum);
		float* voxelCoordVect = new float[voxelNum*3];
		//#pragma omp parallel for
		for(int voxelIdx=0;voxelIdx<voxelNum;++voxelIdx)
		{
			Point3f tempCoord = tempVisualHull.GetVoxelPos(voxelIdx) ;
			voxelCoordVect[3*voxelIdx ] = tempCoord.x;
			voxelCoordVect[3*voxelIdx +1] = tempCoord.y;
			voxelCoordVect[3*voxelIdx +2] = tempCoord.z;
		}
		//g_clock.toc("voxelCoordVect generation");
		float* occupancyOutput = new float[voxelNum];
		memset(occupancyOutput,0,voxelNum*sizeof(float));
		//DetectionCostVolumeGenerationWithOrientation_GPU(voxelCoordVect, voxelNum, costMapVect,projectMatVect,camCenterVect,occupancyOutput);
		//DetectionCostVolumeGeneration_float_GPU_vga(voxelCoordVect, voxelNum, costMapVect,projectMatVect,occupancyOutput);
		bool bCudaSuccess;
		if(isHD ==false)
			bCudaSuccess = DetectionCostVolumeGeneration_float_GPU_average_vga(voxelCoordVect, voxelNum, costMapVect,projectMatVect,occupancyOutput);
		else
			bCudaSuccess = DetectionCostVolumeGeneration_float_GPU_average_hd(voxelCoordVect, voxelNum, costMapVect,projectMatVect,occupancyOutput);
		if(bCudaSuccess==false)
			bCudaSuccessAll = false;

		//////////////////////////////////////////////////////////////////////////
		//Non-Maximum suppression
		vector<int> nonMaxVoxIdx;		//keep the index of local max voxels
		int bandwidth =3;
		for(int voxelIdx=0;voxelIdx<voxelNum;++voxelIdx)
		{
			if(occupancyOutput[voxelIdx]>0)
			{
				float currentProb = occupancyOutput[voxelIdx];
				vector<int> neighVoxIdx;
				tempVisualHull.GetNeighborVoxIdx(voxelIdx,neighVoxIdx,1);
				bool bLocalMax = true;
				for(int i=0;i<neighVoxIdx.size();++i)
				{
					if(currentProb<  occupancyOutput[neighVoxIdx[i]])
					{
						bLocalMax = false;
						break;
					}
				}
				if(bLocalMax)
					nonMaxVoxIdx.push_back(voxelIdx);
			}
		}

		//Non Max Suppression and face pose estimation
		float sphereSize_CM = 30.0;
		float MAGIC_OFFSET = 1000;		//added to flag valid voxels (subtracted again after all others are eliminated)
		int sphereSize = (sphereSize_CM/VOXEL_SIZE_CM);
		for(int i=0;i<nonMaxVoxIdx.size();++i)
		{
			int voxelIdx = nonMaxVoxIdx[i];
			vector<int> neighVoxIdx;
			if(occupancyOutput[voxelIdx] <MAGIC_OFFSET)
				occupancyOutput[voxelIdx] +=MAGIC_OFFSET;
		/*	// Add cube like neighborhoods	
			tempVisualHull.GetNeighborVoxIdx(voxelIdx,neighVoxIdx,sphereSize);
			for(int j=0;j<neighVoxIdx.size();++j)
			{
				if(occupancyOutput[neighVoxIdx[j]]<MAGIC_OFFSET)		//To avoid duplicated addition
					occupancyOutput[neighVoxIdx[j]] +=MAGIC_OFFSET;
			}*/
		}
		//Eliminate outliers
		for(int voxelIdx=0;voxelIdx<voxelNum;++voxelIdx)
		{
			if(occupancyOutput[voxelIdx]<MAGIC_OFFSET)
				occupancyOutput[voxelIdx] =0;
			else
				occupancyOutput[voxelIdx] -=MAGIC_OFFSET;
		}

		//////////////////////////////////////////////////////////////////////////
		//Save volumeRecon as surfaceVoxels for visualization
		tempVisualHull.m_surfaceVoxelOriginal.clear();
		double minCost=1e5;
		double maxCost=-1e5;
		for(int voxelIdx=0;voxelIdx<voxelNum;++voxelIdx)
		{
			if(occupancyOutput[voxelIdx] >POSEMACHINE_NODESCORE_THRESHOLD_AVG)
			{
				//printf("%.2f     ",occupancyOutput[voxelIdx]);
				Point3f voxPos = tempVisualHull.GetVoxelPos(voxelIdx);
				int colorIdx = occupancyOutput[voxelIdx]*255;
				Point3f voxColor(1.0f, 1.0f, 1.0f);  // Donglai's comment: no need for visualization, put random color here.!
				tempVisualHull.m_surfaceVoxelOriginal.push_back( SurfVoxelUnit(voxPos,voxColor));
				tempVisualHull.m_surfaceVoxelOriginal.back().voxelIdx = voxelIdx;
				tempVisualHull.m_surfaceVoxelOriginal.back().prob = occupancyOutput[voxelIdx];

				if(minCost >occupancyOutput[voxelIdx])
					minCost =occupancyOutput[voxelIdx];
				if(maxCost <occupancyOutput[voxelIdx])
					maxCost =occupancyOutput[voxelIdx];
			}
		}
		printfLog("jj: %d: Final Voxel Number : %d:: minCost %f, maxCost %f\n",jj,tempVisualHull.m_surfaceVoxelOriginal.size(),minCost,maxCost);

		//printfLog("Final Voxel Number : %d\n",tempVisualHull.m_surfaceVoxelOriginal.size());
		tempVisualHull.deleteMemory();	//Volume is no longer needed
		delete[] voxelCoordVect;
		delete[] occupancyOutput;
		//printfLog("minCost %f, maxCost %f\n",minCost,maxCost);
	}//end of for(int j=0;j<targetJointIdx.size();++j)
	g_clock.toc("Node Proposal Generation");
	if(bCudaSuccessAll==false)
		return;

	//////////////////////////////////////////////////////////////////////////
	/// Part Proposal generation
	g_clock.tic();
	gaussianBandwidth = gaussianBandwidth*0.5;
	//////////////////////////////////////////////////////////////////////////
	/// Set 2D group Map: Left (100) - right (200) 
	vector< vector< Mat_<Vec3b> > > costMap;   //costMap[camIdx][jointIdx]
	costMap.resize(detectedPoseVect.size());
	for(int c=0;c<costMap.size();++c)
	{
		costMap[c].resize(modelJointNum);
		Mat& inputImage = domeImageManager.m_domeViews[c]->m_inputImage;
		for(int j=0;j<costMap[c].size();++j)
		{
			costMap[c][j] = Mat_<Vec3b>::zeros(inputImage.rows,inputImage.cols);
		}
	}

	int LEFT = 100;
	int RIGHT = 200;
	int MAGIC_NUM = 100;		//I need this to distinguish jointRegions from background
	#pragma omp parallel for //num_threads(10)
	for(int camIdx=0;camIdx<detectedPoseVect.size();++camIdx)
	{
		//Mat& inputImage = domeImageManager.m_domeViews[camIdx]->m_inputImage;
		//costMap[camIdx] = Mat_<Vec3b>::zeros(inputImage.rows,inputImage.cols);

		for(int h=0;h<detectedPoseVect[camIdx].size();++h)	//for h is the index for each person in camera of the camIdx 
		{
			for(int jj=0;jj<targetJointIdxVect.size();++jj)		//jj is my joint index
			{
				BodyJointEnum targetJointIdx = targetJointIdxVect[jj];
				OpenPose25JointEnum targetJointIdx_OpenPose = m_map_devaToOpenPoseIdx[targetJointIdx];
					
				if(targetJointIdx>=JOINT_lShoulder && targetJointIdx<=JOINT_lFoot)
				{
					Scalar tempColor(jj + MAGIC_NUM, h,LEFT);				//(jointIDx, humanIdx, LEFT RIGHT flag)
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_OpenPose],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				else if(targetJointIdx>=JOINT_rShoulder && targetJointIdx<=JOINT_rFoot)
				{
					//int mirroredIdx = jj - (jointNum-3)/2;			//I forgot this.. what a "huge" bug....
					//Scalar tempColor(mirroredIdx + MAGIC_NUM, h,RIGHT);
					Scalar tempColor(jj + MAGIC_NUM, h,RIGHT);
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_OpenPose],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,200+mirroredIdx,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				else if(targetJointIdx==JOINT_headTop || targetJointIdx==JOINT_neck 
					|| targetJointIdx==Joint_bodyCenter || targetJointIdx==JOINT_realheadtop)  // body center is added by Donglai due to new OpenPose
				{
					Scalar tempColor(jj + MAGIC_NUM, h,LEFT);
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_OpenPose],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				//Coco additional
				else if(targetJointIdx==JOINT_lEye || targetJointIdx==JOINT_lEar)
				{
					Scalar tempColor(jj + MAGIC_NUM, h,LEFT);
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_OpenPose],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				else if(targetJointIdx==JOINT_rEye || targetJointIdx==JOINT_rEar)
				{
					Scalar tempColor(jj + MAGIC_NUM, h,RIGHT);
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_OpenPose],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				// foot
				else if(targetJointIdx >= JOINT_lBigToe && targetJointIdx <= JOINT_lHeel)
				{
					Scalar tempColor(jj + MAGIC_NUM, h,LEFT);
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_OpenPose],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				else if(targetJointIdx >= JOINT_rBigToe && targetJointIdx <= JOINT_rHeel)
				{
					Scalar tempColor(jj + MAGIC_NUM, h,RIGHT);
					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_OpenPose],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
				}
				//if(domeImageManager.m_domeViews[camIdx]->m_actualPanelIdx==0 && domeImageManager.m_domeViews[camIdx]->m_actualCamIdx==21)
			/*	{
					imshow("test",costMap[camIdx][jj]);
					cvWaitKey();
				}*/
			}
			Point2f bodyCenter = detectedPoseVect[camIdx][h].bodyJoints[OP_JOINT_LHip] + detectedPoseVect[camIdx][h].bodyJoints[OP_JOINT_RHip];
			bodyCenter = bodyCenter*0.5;
			int bodyCenterJointidxInVector = 2;			//index in targetJointIdxVect
			Scalar tempColor(bodyCenterJointidxInVector+ MAGIC_NUM, h,LEFT);
			circle( costMap[camIdx][SMC_BodyJoint_bodyCenter],bodyCenter,gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
		}
		/*//Debug
		if(true)//domeImageManager.m_domeViews[camIdx]->m_actualPanelIdx==0 && domeImageManager.m_domeViews[camIdx]->m_actualCamIdx==1)
		{
			imshow("test",costMap[camIdx]);
			cvWaitKey();
		}*/
	}

	//////////////////////////////////////////////////////////////////////////
	/// Part Proposal generation
	//0 1 2 // 3 4 5   6 7 8 // 9 10 11   12 13 14 
	int iterEnd = m_skeletonHierarchy.size();
	int iterStart = 1;		//start from head top. This value was 3 before, where I didn't consider head and body
	//if(bDoJointMirroring)
		//iterEnd = 3+(m_skeletonHierarchy.size()-3)/2;
	float increasingpStepUnit = 1.0/detectedPoseVect.size(); 
	#pragma omp parallel for
	for(int jointIdx=iterStart;jointIdx<iterEnd;++jointIdx)	
	{
		if(m_skeletonHierarchy[jointIdx]->parents.size()==0)
			continue;
		//if(m_skeletonHierarchy[jointIdx]->jointName==JOINT_headTop || m_skeletonHierarchy[jointIdx]->jointName==Joint_bodyCenter)
//			continue;
		STreeElement* childTreeNode = m_skeletonHierarchy[jointIdx];
		assert(m_skeletonHierarchy[jointIdx]->parents.size()==1);
		STreeElement* parentTreeNode = m_skeletonHierarchy[jointIdx]->parents.front();		//WARNING!!! I am assuming single child mode
		int parentJointidx = parentTreeNode->idxInTree;
		vector< SurfVoxelUnit >& surfVoxChild = detectionHullManager->m_visualHullRecon[jointIdx].m_surfaceVoxelOriginal;
		vector< SurfVoxelUnit >& surfVoxParent  = detectionHullManager->m_visualHullRecon[parentTreeNode->idxInTree].m_surfaceVoxelOriginal;

		//printf("Edge generation: childJoint %d, parentJoint %d\n",jointIdx,parentJointidx);

		//initialize edge strength vectors
		childTreeNode->childParent_partScore.clear();			//should throw away previous data
		childTreeNode->childParent_counter.clear();			//should throw away previous data

		childTreeNode->childParent_partScore.resize(surfVoxChild.size());
		childTreeNode->childParent_counter.resize(surfVoxChild.size());
		for(int c=0;c<surfVoxChild.size();++c)
		{
			childTreeNode->childParent_partScore[c].resize(surfVoxParent.size(),0);
			childTreeNode->childParent_counter[c].resize(surfVoxParent.size(),0);
		}

		//For every possible pair between current joint's node and parent's node
		for(int c=0;c<surfVoxChild.size();++c)
		{
			for(int camIdx=0;camIdx<detectedPoseVect.size();++camIdx)
			{
				Point2d childImgPt = Project3DPt(surfVoxChild[c].pos, domeImageManager.m_domeViews[camIdx]->m_P);
				if(IsOutofBoundary(costMap[camIdx][jointIdx],childImgPt.x,childImgPt.y))
					continue;

				Vec3b childLabel = costMap[camIdx][jointIdx].at<Vec3b>(childImgPt.y,childImgPt.x);
				if(childLabel[0]-MAGIC_NUM != jointIdx)		//not for this joint
					continue;
				int childWhichSideLabel =childLabel[2];		//either 100 or 200
				
				for(int p=0;p<surfVoxParent.size();++p)
				{
					float edgeDist = Distance(surfVoxChild[c].pos,surfVoxParent[p].pos);
					float expected = childTreeNode->parent_jointLength.front();
					if(abs(expected-edgeDist)>expected)		//reject if the length is too long or short
						continue;

					double weightByNodes = (surfVoxChild[c].prob + surfVoxParent[p].prob)/2.0;
					Point2d parentImgPt = Project3DPt(surfVoxParent[p].pos, domeImageManager.m_domeViews[camIdx]->m_P);
					if(IsOutofBoundary(costMap[camIdx][parentJointidx],parentImgPt.x,parentImgPt.y))
						continue;

					Vec3b parentLabel = costMap[camIdx][parentJointidx].at<Vec3b>(parentImgPt.y,parentImgPt.x);
					//if(parentLabel<100)
					if(parentLabel[0]-MAGIC_NUM != parentJointidx)		//not for this joint
						continue;
					int parentWhichSideLabel = parentLabel[2];		//either 100 or 200

					childTreeNode->childParent_counter[c][p] +=1;		//for taking average.. consider the view whenever the part is inside of image. I put this here so that the "non-detected" view provided kind of "negative" voting

					//if the parent is either neck or bodyCenter, side label doesn't matter. Only human label is important
					//For ms coco, if parent if head top (actually nose), side doesn't matter (left/right eye) 
					// (According to Han's comment, should not matter in modern pose detectors. --Donglai)
					if(jointIdx==SMC_BodyJoint_lShoulder || jointIdx==SMC_BodyJoint_lHip || jointIdx==SMC_BodyJoint_rShoulder || jointIdx==SMC_BodyJoint_rHip 
									|| jointIdx==SMC_BodyJoint_lEye || jointIdx==SMC_BodyJoint_rEye)			//bug fixed Neck and Bodycenter has label 100 which was never connected with 200 (right Shoulder and Hip)
					{
						if(childLabel[1] == parentLabel[1])		//human label check
						{
							//childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes*increasingpStepUnit;		//max value become 1
							childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes;		//sum
						}
					}
					else if(childWhichSideLabel == parentWhichSideLabel && childLabel[1] == parentLabel[1]) // childLabel[1]  and parentLabel[1] means human index
					{
						//childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes*increasingpStepUnit;		//max value become 1
						childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes;	//sum
					//	childTreeNode->childParent_counter[c][p] +=1;	//for taking average
					}
				}
			}

			//Taking average
			for(int p=0;p<surfVoxParent.size();++p)
			{
				if(childTreeNode->childParent_counter[c][p]>0)
					childTreeNode->childParent_partScore[c][p] /=childTreeNode->childParent_counter[c][p];
			}
		}
	}

	/*if(bDoJointMirroring)
	{
		for(int jointIdx=iterEnd;jointIdx<m_skeletonHierarchy.size();++jointIdx)
		{
			int mirrorJoint = jointIdx - (jointNum-3)/2;
			printf("Edge generation: joint %d - mirrored by %d\n",jointIdx,mirrorJoint);

			STreeElement* childTreeNode = m_skeletonHierarchy[jointIdx];
			STreeElement* mirrorNode = m_skeletonHierarchy[mirrorJoint];

			childTreeNode->childParent_partScore.resize(mirrorNode->childParent_partScore.size());
			for(int c=0;c<childTreeNode->childParent_partScore.size();++c)
				childTreeNode->childParent_partScore[c].reserve(mirrorNode->childParent_partScore[c].size());

			for(int c=0;c<mirrorNode->childParent_partScore.size();++c)
			{
				childTreeNode->childParent_partScore[c] = mirrorNode->childParent_partScore[c];
			}
		}
	}*/
	g_clock.toc("Part Proposal Generation\n");

	//Save the edge result to m_edgeCostVector for visualization
	//Todo: only use m_edgeCostVector to avoid copy. (Not the one in the  m_skeleton
	SEdgeCostVector* pEdgeCostUnit = new SEdgeCostVector();
	m_edgeCostVector.push_back(pEdgeCostUnit);
	pEdgeCostUnit->parent_connectivity_vis.resize(m_skeletonHierarchy.size());
	for(int jointIdx=0;jointIdx<m_skeletonHierarchy.size();++jointIdx)	
	{
		pEdgeCostUnit->parent_connectivity_vis[jointIdx] = m_skeletonHierarchy[jointIdx]->childParent_partScore;
	}

	printf("done\n");

	g_clock.tic();
	SaveNodePartProposalsOP25(dataMainFolder,g_askedVGACamNum,domeImageManager.GetFrameIdx(),detectionHullManager,m_skeletonHierarchy,validCamIdxVect.size(),isHD);
	g_clock.toc("SaveNodePartProposals");
}

//Node proposal: saved in pDetectHull
//Part proposal: saved in skeletonHierarchy
void CBodyPoseRecon::SaveNodePartProposals(const char* dataMainFolder,const int cameraNum,int frameIdx,const CVisualHullManager* pDetectHull,const vector<STreeElement*>& skeletonHierarchy,int actualValidCamNum,bool isHD)
{
	char outputFileFolder[512];
	if(m_skeletonHierarchy.size()==MODEL_JOINT_NUM_COCO_19)//MODEL_JOINT_NUM_COCO_19)
	{
		if(isHD==false)
			sprintf(outputFileFolder,"%s/coco19_bodyNodeProposal",dataMainFolder);
		else
			sprintf(outputFileFolder,"%s/coco19_bodyNodeProposal_hd",dataMainFolder);
	}
	else
	{
		if(isHD==false)
			sprintf(outputFileFolder,"%s/bodyNodeProposal",dataMainFolder);
		else
			sprintf(outputFileFolder,"%s/bodyNodeProposal_hd",dataMainFolder);
	}
	
	CreateFolder(outputFileFolder);
	sprintf(outputFileFolder,"%s/%04d",outputFileFolder,cameraNum);
	CreateFolder(outputFileFolder);

	char outputFileName[512];
	if(isHD==false)
		sprintf(outputFileName,"%s/nodePartProposals_%08d.txt",outputFileFolder,frameIdx);
	else
		sprintf(outputFileName,"%s/nodePartProposals_hd%08d.txt",outputFileFolder,frameIdx);
	printfLog("## Save NodePartProposals: %s\n",outputFileName);

	ofstream fout(outputFileName, ios::out | ios::binary);
	
	int versionMagic = -98;   //added this to distinguish weighted part version	//-99:weightedversion (/480), -98: weightedversion2 (/cnt)
	fout.write((char*)&versionMagic,sizeof(int));
	//float version = 0.1;
	float version = 0.5;		//added actually valid number of cameras
	fout.write((char*)&version,sizeof(float));

	int dataInt;
	if(actualValidCamNum<0)
	{
		dataInt =cameraNum;
		fout.write((char*)&dataInt,sizeof(int));
	}
	else
	{
		dataInt = actualValidCamNum;
		fout.write((char*)&dataInt,sizeof(int));
	}

	dataInt = pDetectHull->m_visualHullRecon.size();
	fout.write((char*)&dataInt,sizeof(int));

	for(int v=0;v<pDetectHull->m_visualHullRecon.size();++v)
	{
		//if (v==(3+(pDetectHull->m_visualHullRecon.size()-3)/2))
			//break;		//left-right are the same 

		const CVisualHull& tempVisualHull = pDetectHull->m_visualHullRecon[v]; 

		//float voxelSize;
		//float xMin ,xMax,yMin,yMax,zMin,zMax;
		//tempVisualHull.GetParameters(xMin,xMax,yMin,yMax,zMin,zMax,voxelSize);
		//fout << xMin << " " << xMax << " " <<yMin << " " <<yMax << " " <<zMin <<" " <<zMax << " " <<voxelSize << "\n";  //Voxel Size

		float dataFloat[7];
		tempVisualHull.GetParameters(dataFloat[0],dataFloat[1],dataFloat[2],dataFloat[3],dataFloat[4],dataFloat[5],dataFloat[6]);
		fout.write((char*)dataFloat,sizeof(float)*7);

		int dataInt =tempVisualHull.m_surfaceVoxelOriginal.size();
		//printf("size %d\n",dataInt);
		fout.write((char*)&dataInt,sizeof(int));

		for(int i=0;i<tempVisualHull.m_surfaceVoxelOriginal.size();++i)
		{
			//fout << tempVisualHull.m_surfaceVoxelOriginal[i].pos.x <<" "<< tempVisualHull.m_surfaceVoxelOriginal[i].pos.y <<" "<< tempVisualHull.m_surfaceVoxelOriginal[i].pos.z <<" ";
			//fout << tempVisualHull.m_surfaceVoxelOriginal[i].color.x <<" "<< tempVisualHull.m_surfaceVoxelOriginal[i].color.y <<" "<< tempVisualHull.m_surfaceVoxelOriginal[i].color.z <<" ";
			//fout << tempVisualHull.m_surfaceVoxelOriginal[i].prob << tempVisualHull.m_surfaceVoxelOriginal[i].voxelIdx <<"\n";		//this is an additoonal value compared SaveVisualHullSurface
			
			fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].pos.x,sizeof(float));
			fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].pos.y,sizeof(float));
			fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].pos.z,sizeof(float));

			//fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].color.x,sizeof(float));
			//fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].color.y,sizeof(float));
			//fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].color.z,sizeof(float));

			fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].prob,sizeof(float));
			fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].voxelIdx,sizeof(int));
		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// Save Part Proposal Information
	char magicCharacter='e';		//there exist edge information
	fout.write(&magicCharacter,sizeof(char));
	int jointNum = skeletonHierarchy.size();
	fout.write((char*)&jointNum,sizeof(int));
	for(int j=0;j<skeletonHierarchy.size();++j)
	{
		int childNum=0,parentNum=0;
		if(skeletonHierarchy[j]->childParent_partScore.size()==0)
		{
			fout.write((char*)&childNum,sizeof(int));
			fout.write((char*)&parentNum,sizeof(int));
			continue;
		}
		else
		{
			childNum = skeletonHierarchy[j]->childParent_partScore.size();
			parentNum = skeletonHierarchy[j]->childParent_partScore.front().size();
			fout.write((char*)&childNum,sizeof(int));
			fout.write((char*)&parentNum,sizeof(int));
		}
		vector<int> cVector;	//to save these at once to the disk
		vector<int> fVector;
		vector<float> edgeVector;
		cVector.reserve(childNum*parentNum);
		fVector.reserve(childNum*parentNum);
		edgeVector.reserve(childNum*parentNum);
		for(int c=0;c<skeletonHierarchy[j]->childParent_partScore.size();++c)
		{
			for(int f=0;f<skeletonHierarchy[j]->childParent_partScore[c].size();++f)
			{
				if(skeletonHierarchy[j]->childParent_partScore[c][f]>0)
				{
					cVector.push_back(c);
					fVector.push_back(f);
					edgeVector.push_back(skeletonHierarchy[j]->childParent_partScore[c][f]);
				}
//				printf("Save: %f\n",skeletonHierarchy[j]->childParent_partScore[c][f]);
			}
		}
		int cnt= cVector.size();
		fout.write((char*)&cnt,sizeof(int));	//number of will-be saved value

		fout.write((char*)cVector.data(),sizeof(int)*cnt);		//index 1
		fout.write((char*)fVector.data(),sizeof(int)*cnt);		//index 2
		fout.write((char*)edgeVector.data(),sizeof(float)*cnt);		//value
	}
	fout.close();
}

void CBodyPoseRecon::SaveNodePartProposalsOP25(const char* dataMainFolder,const int cameraNum,int frameIdx,const CVisualHullManager* pDetectHull,const vector<STreeElement*>& skeletonHierarchy,int actualValidCamNum,bool isHD)
{
	char outputFileFolder[512];
	if(isHD==false)
		sprintf(outputFileFolder,"%s/op25_bodyNodeProposal",dataMainFolder);
	else
		sprintf(outputFileFolder,"%s/op25_bodyNodeProposal_hd",dataMainFolder);
	
	CreateFolder(outputFileFolder);
	sprintf(outputFileFolder,"%s/%04d",outputFileFolder,cameraNum);
	CreateFolder(outputFileFolder);

	char outputFileName[512];
	if(isHD==false)
		sprintf(outputFileName,"%s/nodePartProposals_%08d.txt",outputFileFolder,frameIdx);
	else
		sprintf(outputFileName,"%s/nodePartProposals_hd%08d.txt",outputFileFolder,frameIdx);
	printfLog("## Save NodePartProposals: %s\n",outputFileName);

	ofstream fout(outputFileName, ios::out | ios::binary);
	
	int versionMagic = -98;   //added this to distinguish weighted part version	//-99:weightedversion (/480), -98: weightedversion2 (/cnt)
	fout.write((char*)&versionMagic,sizeof(int));
	//float version = 0.1;
	float version = 0.5;		//added actually valid number of cameras
	fout.write((char*)&version,sizeof(float));

	int dataInt;
	if(actualValidCamNum<0)
	{
		dataInt =cameraNum;
		fout.write((char*)&dataInt,sizeof(int));
	}
	else
	{
		dataInt = actualValidCamNum;
		fout.write((char*)&dataInt,sizeof(int));
	}

	dataInt = pDetectHull->m_visualHullRecon.size();
	fout.write((char*)&dataInt,sizeof(int));

	for(int v=0;v<pDetectHull->m_visualHullRecon.size();++v)
	{
		const CVisualHull& tempVisualHull = pDetectHull->m_visualHullRecon[v]; 

		float dataFloat[7];
		tempVisualHull.GetParameters(dataFloat[0],dataFloat[1],dataFloat[2],dataFloat[3],dataFloat[4],dataFloat[5],dataFloat[6]);
		fout.write((char*)dataFloat,sizeof(float)*7);

		int dataInt =tempVisualHull.m_surfaceVoxelOriginal.size();
		//printf("size %d\n",dataInt);
		fout.write((char*)&dataInt,sizeof(int));

		for(int i=0;i<tempVisualHull.m_surfaceVoxelOriginal.size();++i)
		{
	
			fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].pos.x,sizeof(float));
			fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].pos.y,sizeof(float));
			fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].pos.z,sizeof(float));

			fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].prob,sizeof(float));
			fout.write((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].voxelIdx,sizeof(int));
		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// Save Part Proposal Information
	char magicCharacter='e';		//there exist edge information
	fout.write(&magicCharacter,sizeof(char));
	int jointNum = skeletonHierarchy.size();
	fout.write((char*)&jointNum,sizeof(int));
	for(int j=0;j<skeletonHierarchy.size();++j)
	{
		int childNum=0,parentNum=0;
		if(skeletonHierarchy[j]->childParent_partScore.size()==0)
		{
			fout.write((char*)&childNum,sizeof(int));
			fout.write((char*)&parentNum,sizeof(int));
			continue;
		}
		else
		{
			childNum = skeletonHierarchy[j]->childParent_partScore.size();
			parentNum = skeletonHierarchy[j]->childParent_partScore.front().size();
			fout.write((char*)&childNum,sizeof(int));
			fout.write((char*)&parentNum,sizeof(int));
		}
		vector<int> cVector;	//to save these at once to the disk
		vector<int> fVector;
		vector<float> edgeVector;
		cVector.reserve(childNum*parentNum);
		fVector.reserve(childNum*parentNum);
		edgeVector.reserve(childNum*parentNum);
		for(int c=0;c<skeletonHierarchy[j]->childParent_partScore.size();++c)
		{
			for(int f=0;f<skeletonHierarchy[j]->childParent_partScore[c].size();++f)
			{
				if(skeletonHierarchy[j]->childParent_partScore[c][f]>0)
				{
					cVector.push_back(c);
					fVector.push_back(f);
					edgeVector.push_back(skeletonHierarchy[j]->childParent_partScore[c][f]);
				}
//				printf("Save: %f\n",skeletonHierarchy[j]->childParent_partScore[c][f]);
			}
		}
		int cnt= cVector.size();
		fout.write((char*)&cnt,sizeof(int));	//number of will-be saved value

		fout.write((char*)cVector.data(),sizeof(int)*cnt);		//index 1
		fout.write((char*)fVector.data(),sizeof(int)*cnt);		//index 2
		fout.write((char*)edgeVector.data(),sizeof(float)*cnt);		//value
	}
	fout.close();
}


void CBodyPoseRecon::LoadNodePartProposals(const char* fullPath,const int imgFrameIdx,bool bRevisualize,bool bLoadPartProposal)
{
	m_curImgFrameIdx =-100;
	printf("BodyVolumeRecon::Loading from %s\n",fullPath);
	ifstream fin(fullPath, ios::in | ios::binary);
	if(fin)
	{
		CVisualHullManager* detectionHullManager =  new CVisualHullManager();
		m_nodePropScoreVector.push_back(detectionHullManager);		//push a CVisualHullManager which is describing current frames prob volumes
		detectionHullManager->m_imgFramdIdx =  imgFrameIdx; 

		int jointNum;
		//newly added version
		int magicNum;
		fin.read((char*)&magicNum,sizeof(int));
		if(magicNum==-99 || magicNum==-98)
		{
			float version;
			fin.read((char*)&version,sizeof(float));			//0.1

			if(version>0.2)  //from 0.2
			{
				int actualValidCamNum;
				fin.read((char*)&actualValidCamNum,sizeof(int));	//Num of cameras used to make this voting results. Maybe diff from g_askedCamNum
			}

			int dataInt;
			fin.read((char*)&dataInt,sizeof(int));
			jointNum = dataInt;
			if(magicNum==-99)
			{
				g_partPropScoreMode = PART_TRAJ_SCORE_WEIGHTED;			//important to determine threshold
				printf("## PartScoreMode: LoadNodePartProposals: Weighted version (div by 480)\n");	
			}
			else //-98
				g_partPropScoreMode = PART_TRAJ_SCORE_WEIGHTED_DIV_CNT;			//important to determine threshold
				printf("## PartScoreMode: LoadNodePartProposals: Weighted version (div by counter)\n");	
		}
		else
		{
			printf("## PartScoreMode: LoadNodePartProposals: Non-weighted version\n");
			jointNum = magicNum;		//this is an old version (non-weighted part score)
			g_partPropScoreMode = PART_TRAJ_SCORE_NON_WEIGHTED;
		}
		detectionHullManager->m_visualHullRecon.resize(jointNum); //contains each joint's visual hull recon

		if(m_skeletonHierarchy.size()==0)
			ConstructJointHierarchy(jointNum);

		for(int v=0;v<jointNum;++v)
		{
			CVisualHull& tempVisualHull = detectionHullManager->m_visualHullRecon[v]; 
			tempVisualHull.m_actualFrameIdx =imgFrameIdx;

			float dataFloat[7];
			fin.read((char*)dataFloat,sizeof(float)*7);	//fout << xMin << " " << xMax << " " <<yMin << " " <<yMax << " " <<zMin <<" " <<zMax << " " <<voxelSize << "\n";  //Voxel Size
			tempVisualHull.SetParameters(dataFloat[0],dataFloat[1],dataFloat[2],dataFloat[3],dataFloat[4],dataFloat[5],dataFloat[6]);

			int surfVoxNum; //tempVisualHull.m_surfaceVoxelOriginal.size();
			fin.read((char*)&surfVoxNum,sizeof(int));
			tempVisualHull.m_surfaceVoxelOriginal.resize(surfVoxNum);
			for(int i=0;i<surfVoxNum;++i)
			{
				fin.read((char*)dataFloat,sizeof(float)*4);

				tempVisualHull.m_surfaceVoxelOriginal[i].pos.x = dataFloat[0];
				tempVisualHull.m_surfaceVoxelOriginal[i].pos.y = dataFloat[1];
				tempVisualHull.m_surfaceVoxelOriginal[i].pos.z = dataFloat[2];

				tempVisualHull.m_surfaceVoxelOriginal[i].prob = dataFloat[3];

				fin.read((char*)&tempVisualHull.m_surfaceVoxelOriginal[i].voxelIdx,sizeof(int));
			}
		}
		//////////////////////////////////////////////////////////////////////////
		/// Lode Edge Information
 		char magicCharacter;		//there exist edge information
		fin.read(&magicCharacter,1);
		if(magicCharacter!='e' || bLoadPartProposal==false)		//there exists valid edge information
		{
			fin.read((char*)&jointNum,sizeof(int));
			printf("Done\n");
			return;
		}

		fin.read((char*)&jointNum,sizeof(int));
		if(jointNum!=m_skeletonHierarchy.size())
		{
			printfLog("## ERROR: joint number is not same as current setting (%d vs %d)",jointNum,m_skeletonHierarchy.size());
			printf("Done\n");
			return;
		}

		SEdgeCostVector* pEdgeCostUnit = new SEdgeCostVector();
		m_edgeCostVector.push_back(pEdgeCostUnit);
		pEdgeCostUnit->parent_connectivity_vis.resize(m_skeletonHierarchy.size());
		for(int jointIdx=0;jointIdx<m_skeletonHierarchy.size();++jointIdx)	
		{
			int childNum=0,parentNum=0;
			fin.read((char*)&childNum,sizeof(int));
			fin.read((char*)&parentNum,sizeof(int));

			if(childNum==0)
				continue;
			
			pEdgeCostUnit->parent_connectivity_vis[jointIdx].resize(childNum);
			for(int c=0;c<childNum;++c)
				pEdgeCostUnit->parent_connectivity_vis[jointIdx][c].resize(parentNum);

			int cnt=0;
			fin.read((char*)&cnt,sizeof(int));	//number of saved value

			int* cVector = new int[cnt];	//to save these at once to the disk
			int* fVector = new int[cnt];
			float* edgeVector = new float[cnt];
			fin.read((char*)cVector,sizeof(int)*cnt);
			fin.read((char*)fVector,sizeof(int)*cnt);
			fin.read((char*)edgeVector,sizeof(float)*cnt);

			for(int i=0;i<cnt;++i)
			{
				pEdgeCostUnit->parent_connectivity_vis[jointIdx][cVector[i]][fVector[i]]  =  edgeVector[i];
				//printf("load-:: %f\n",edgeVector[i]);
				//m_skeletonHierarchy[j]->childParent_partScore[cVector[i]][fVector[i]] = edgeVector[i];
			}
			delete[] cVector;
			delete[] fVector;
			delete[] edgeVector;
		}
		
		//Save the loaded one to m_skeletonHierarchy, which is used for inference
		//Todo: only use m_edgeCostVector to avoid copy. (Not the one in the  m_skeleton
		for(int jointIdx=0;jointIdx<m_skeletonHierarchy.size();++jointIdx)	
		{
			m_skeletonHierarchy[jointIdx]->childParent_partScore = pEdgeCostUnit->parent_connectivity_vis[jointIdx];
		}
		fin.close();
	}
	else
	{
		printf("LoadProbVolumeReconResult:: Failed from %s\n\n",fullPath);
	}
	printf("Done\n");
}

void InferenceJoint15_OnePass_Multiple_DM(CVisualHullManager* pTempSceneProb,vector<STreeElement*>& m_skeletonHierarchy,SBody3DScene& newPoseReconMem)
{
	int prevNumPeople = newPoseReconMem.m_articBodyAtEachTime.size();
	while(true)
	{
		std::cout << "PrevNumPeople: " << prevNumPeople << "---------------------\n" << std::endl;
		InferenceJointCoco19_OnePass_singleSkeleton(pTempSceneProb,m_skeletonHierarchy,newPoseReconMem);		//This version can handle both 15, 19 joints

		if(prevNumPeople == newPoseReconMem.m_articBodyAtEachTime.size())
			break;
		prevNumPeople = newPoseReconMem.m_articBodyAtEachTime.size();
	}
}

//Only generate the best single skeleton
//Made this to avoid double counting between multiple subjects
void InferenceJointCoco19_OnePass_singleSkeleton(CVisualHullManager* pTempSceneProb,vector<STreeElement*>& m_skeletonHierarchy,SBody3DScene& newPoseReconMem)
{
	//double connectivity_score_thresh = 2.0/g_askedVGACamNum;			//non-weighted part cost
	double connectivity_score_thresh;
	double headThreshold =-1e5;
	if(g_partPropScoreMode == PART_TRAJ_SCORE_WEIGHTED)
		connectivity_score_thresh = 0.2/g_askedVGACamNum;		
	else if(g_partPropScoreMode == PART_TRAJ_SCORE_WEIGHTED_DIV_CNT)
	{
		//Found this empirically
		//480: 0.05
		//10: 0.3
		//connectivity_score_thresh = (0.3* (480-g_askedVGACamNum) + 0.05* (g_askedVGACamNum-10) ) / (480.0-10.0);		//linear interpolation
		connectivity_score_thresh = (0.2* (480-g_askedVGACamNum) + 0.05* (g_askedVGACamNum-10) ) / (480.0-10.0);		//linear interpolation
		headThreshold = connectivity_score_thresh;
		//connectivity_score_thresh = 0.05;				//fixed and independent from number of cameras
		//headThreshold = 0.05;
	}
	else
		connectivity_score_thresh = 2.0/g_askedVGACamNum;		

	//connectivity_score_thresh/=5.0;
	printf("Inference: Param: connectivity_score_thresh: %f\n",connectivity_score_thresh);
	if(pTempSceneProb->m_visualHullRecon.size()==0)
	{
		printf("There is no prob volume data here\n");
		return;
	}

	int jointNum = pTempSceneProb->m_visualHullRecon.size();

	if(m_skeletonHierarchy.size()==0)
	{
		printf("## ERROR: Construct Skeleton Hierarchy first\n");
		return;
	}

	vector<SJointCandGroup> jointCandGroupVector;
	jointCandGroupVector.resize(m_skeletonHierarchy.size());
	std::cout << "m_skeletonHierarchy.size(): " << m_skeletonHierarchy.size() << std::endl;
	//////////////////////////////////////////////////////////////////////////
	/// Generate Nodes
	for(int i=0;i<m_skeletonHierarchy.size();++i)
	{
		SJointCandGroup& jointCandGroup = jointCandGroupVector[i];
		jointCandGroup.pAssociatedJoint=m_skeletonHierarchy[i];

		//initialize infer node
		vector< SurfVoxelUnit >& surfVoxVector = pTempSceneProb->m_visualHullRecon[i].m_surfaceVoxelOriginal;
		jointCandGroup.nodeCandidates.reserve(surfVoxVector.size());
		for(int s=0;s<surfVoxVector.size();++s)
		{
			if(surfVoxVector[s].label>=0)
			{
			//	printf("3DPS: skip this node. It is already take by skeleton %d\n",surfVoxVector[s].label);
				continue;
			}

			InferNode* pTempNode = new InferNode;
			pTempNode->pos = surfVoxVector[s].pos;
			pTempNode->associatedJointIdx = i;
			pTempNode->originalPropIdx = s;		//Bug fixed....
			pTempNode->pTargetSurfVox = &surfVoxVector[s];
			pTempNode->dataScore = surfVoxVector[s].prob;		
			pTempNode->scoreSum =0;
			pTempNode->dataScoreSum =0;
			jointCandGroup.nodeCandidates.push_back(pTempNode);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// Generate Edges
	for(int j=0;j<m_skeletonHierarchy.size();++j)
	{
		STreeElement* pChildTreeNode= m_skeletonHierarchy[j];
		int childJointIdx = j;
		for(int p=0;p<pChildTreeNode->parents.size();++p)
		{
			STreeElement* pParentTreeNode = pChildTreeNode->parents[p];
			int parentJointIdx = pParentTreeNode->idxInTree;

			SJointCandGroup& childJointCandGroup = jointCandGroupVector[childJointIdx ];
			SJointCandGroup& parentJointCandGroup = jointCandGroupVector[parentJointIdx];

			vector< SurfVoxelUnit >& childNodePropVector = pTempSceneProb->m_visualHullRecon[childJointIdx].m_surfaceVoxelOriginal;
			vector< SurfVoxelUnit >& parentNodePropVector = pTempSceneProb->m_visualHullRecon[parentJointIdx].m_surfaceVoxelOriginal;

			childJointCandGroup.parentCandGroupVector.push_back(&parentJointCandGroup);
			parentJointCandGroup.childCandGroupVector.push_back(&childJointCandGroup);
			
			float expectedDist = pChildTreeNode->parent_jointLength[p];
			
			vector<InferEdge> dummy;
			for(int cc=0;cc<childJointCandGroup.nodeCandidates.size();++cc)
				childJointCandGroup.nodeCandidates[cc]->m_edgesToParent.push_back(dummy);

			for(int pp=0;pp<parentJointCandGroup.nodeCandidates.size();++pp)
				parentJointCandGroup.nodeCandidates[pp]->m_edgesToChild.push_back(dummy);

			for(int cc=0;cc<childJointCandGroup.nodeCandidates.size();++cc)
			{
				for(int pp=0;pp<parentJointCandGroup.nodeCandidates.size();++pp)
				{
					InferEdge edge;
					float tempDist = Distance(childJointCandGroup.nodeCandidates[cc]->pos,parentJointCandGroup.nodeCandidates[pp]->pos);
					bool bReject =false; 

					if(pChildTreeNode->jointName==Joint_bodyCenter )				//Tougher constraint, avoid to jump to the proximal next person
					{
						if(tempDist>expectedDist + cm2world(20))		//10cm, distError>0, if tempDist> expectedDist
							bReject =true;
					}
					else
					{
						//if(tempDist>expectedDist + cm2world(10))		//20cm, distError>0, if tempDist> expectedDist
							//bReject =true;
					}

					float distError = abs(expectedDist-tempDist) / expectedDist;		//0 ~
					//float magicNumber =10;
					//float springScore = exp(- magicNumber* distError*distError);			//penalty
					float springScore = 1 - abs(expectedDist-tempDist) / expectedDist;			//penalty
					springScore *= 0;//SPRING_RATIO;
					//if(springScore<0)
						//springScore = -1e3;		//infinite

					//////////////////////////////////////////////////////////////////////////
					/// Connectivity Score
					float connectivityScore=0;
					int childOriginalIdx = childJointCandGroup.nodeCandidates[cc]->originalPropIdx;	//Bug fixed....
					int parentOriginalIdx = parentJointCandGroup.nodeCandidates[pp]->originalPropIdx;
					if(pChildTreeNode->childParent_partScore.size()>0)
						if(!(pChildTreeNode->jointName==JOINT_neck))// || pChildTreeNode->jointName==JOINT_lHip || pChildTreeNode->jointName==JOINT_rHip))
						{
							connectivityScore  = pChildTreeNode->childParent_partScore[childOriginalIdx][parentOriginalIdx];
							/*if(connectivityScore<0.01)
								continue;*/
						}
					//if(connectivityScore<0.2/g_askedVGACamNum)
					//if(connectivityScore<2/g_askedVGACamNum)
					double connectivity_score_thresh_temp = connectivity_score_thresh;
					if(connectivityScore<connectivity_score_thresh_temp)
						continue;

					if(bReject)
					{
						continue;
					}
					else
					{
						//Compute Edge Score
						edge.edgeScroe =springScore + connectivityScore;			//temporary, need to compute.
						edge.debugSpringScore = springScore;
						edge.debugConnectScore = connectivityScore;
					}
					//Add this edge to both nodes
					edge.otherNode = parentJointCandGroup.nodeCandidates[pp];
					childJointCandGroup.nodeCandidates[cc]->m_edgesToParent.back().push_back(edge);

					edge.otherNode = childJointCandGroup.nodeCandidates[cc];
					parentJointCandGroup.nodeCandidates[pp]->m_edgesToChild.back().push_back(edge);
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//Initialize: Find leaf of the tree hierarchy, and add active nodes
	for(int j=0;j<jointCandGroupVector.size();++j)
	{
		if(jointCandGroupVector[j].childCandGroupVector.size()==0)
		{
			for(int i=0;i<jointCandGroupVector[j].nodeCandidates.size();++i)
			{
				jointCandGroupVector[j].nodeCandidates[i]->scoreSum = DATASCORE_RATIO*jointCandGroupVector[j].nodeCandidates[i]->dataScore;
				jointCandGroupVector[j].nodeCandidates[i]->dataScoreSum = DATASCORE_RATIO*jointCandGroupVector[j].nodeCandidates[i]->dataScore;
			}

			//Send ready message to the parents (or neighbors)
			for(int p=0;p<jointCandGroupVector[j].parentCandGroupVector.size();++p)
			{
				jointCandGroupVector[j].parentCandGroupVector[p]->groupsReadyToSendMessageToMe.push_back(&jointCandGroupVector[j]);
			//	jointCandGroupVector[j].groupsISentReadyMessage.push_back(jointCandGroupVector[j].parentCandGroupVector[p]);
				jointCandGroupVector[j].bDoneAllProcess = true;			//this is One-pass case
			}
		}
	}

	while(true)
	{
		vector<SJointCandGroup*> activeJoints;//contains currently active nodes. 

		//Find active joint
		//one-pass case: if groupsReadyToSendMessage.size() is same as childCandGroupVector, get messages and do process, and send ready message to the parents
		//two-pass case: if groupsReadyToSendMessage.size() is same as (neighborNum-1), get messages and do process, and send ready message to the other reaming neighbor
		//				 if groupsReadyToSendMessage.size() is same as (neighborNum), do above for all (neighborNum-1) combinations.
		//				 if this number is same as (neighborNum), compute max marginal for current node. 
		for(int j=0;j<jointCandGroupVector.size();++j)
		{
			if(jointCandGroupVector[j].bDoneAllProcess)
				continue;

			//one pass case
			if(jointCandGroupVector[j].groupsReadyToSendMessageToMe.size() == jointCandGroupVector[j].childCandGroupVector.size())
			{
				activeJoints.push_back(&jointCandGroupVector[j]);
			}
		}

		printf("## INFERENCE:: Active nodes Num %d\n",activeJoints.size());

		if(activeJoints.size()==0)
			break;

		//In current implementation, get message from all children (already verified that all child are ready to send message to me, once it is activated)
		for(int j=0;j<activeJoints.size();++j)
		{
			for(int a=0;a<activeJoints[j]->nodeCandidates.size();++a)
			{
				InferNode* parentNode = activeJoints[j]->nodeCandidates[a];

				//printf("sizeChild %d\n",parentNode->m_edgesToChild.size());
				//#pragma omp parallel for		
				for(int c=0;c<parentNode->m_edgesToChild.size();++c)		//This loop is for multiple children
				{
					float maxScore=-1e10;			//score might be very small inverse number
					float maxCorresSpringScore=-1;
					float maxCorresConnectScore=-1;
					//SurfVoxelUnit* pBestPrevSurfVox = NULL;
					InferNode* pBestPrevInferNode = NULL; 

					for(int cc=0;cc<parentNode->m_edgesToChild[c].size();++cc)
					{
						InferNode* childNode = parentNode->m_edgesToChild[c][cc].otherNode;

						float tempScore = childNode->scoreSum + parentNode->m_edgesToChild[c][cc].edgeScroe; //I didn't add current data term yet. 
						if(pBestPrevInferNode==NULL || maxScore< tempScore)
						{
							maxScore = tempScore;
							pBestPrevInferNode = childNode;

							//For debugging
							maxCorresSpringScore =   childNode->springScoreSum+  parentNode->m_edgesToChild[c][cc].debugSpringScore;
							maxCorresConnectScore =   childNode->connectivityScoreSum +parentNode->m_edgesToChild[c][cc].debugConnectScore;
						}
					}

					//Set best path
					if(pBestPrevInferNode!=NULL)
					{
						parentNode->pPrevInferNodeVect.push_back(pBestPrevInferNode);
						parentNode->scoreSum += maxScore;
						parentNode->springScoreSum += maxCorresSpringScore;
						parentNode->connectivityScoreSum += maxCorresConnectScore;

						if(c+1 == parentNode->m_edgesToChild.size())  //Bug fixed. It should be added once. At the end of the iteration
						{
							parentNode->scoreSum += DATASCORE_RATIO*parentNode->dataScore;		//Bug fixed. It should be "+=" for multiple children cases
							parentNode->dataScoreSum += DATASCORE_RATIO*parentNode->dataScore;
						}
					}
					else
						parentNode->scoreSum =0;// += -1e3;		//- infinit
				}
			}	//end of for(int c=0;activeJoints[j]->nodeCandidates[c].size();++c)
		} //end of for(int j=0;j<activeJoints.size();++j)

		//One-pass case: Send ready messages to parents
		for(int j=0;j<activeJoints.size();++j)
		{
			//actually there is only one parent
			for(int p=0;p<activeJoints[j]->parentCandGroupVector.size();++p)
			{
				activeJoints[j]->parentCandGroupVector[p]->groupsReadyToSendMessageToMe.push_back(activeJoints[j]);
//				activeJoints[j]->groupsISentReadyMessage.push_back(activeJoints[j]->parentCandGroupVector[p]);
			}

			//this is always true in the one-pass case
			//if(activeJoints[j]->groupsISentReadyMessage.size() == activeJoints[j]->parentCandGroupVector.size())
				activeJoints[j]->bDoneAllProcess = true;	
		}
	}
	//////////////////////////////////////////////////////////////////////////
	/// Save marginal(probSum) for visualization
	for(int j=0;j<jointCandGroupVector.size();++j)
	{
		for(int n=0;n<jointCandGroupVector[j].nodeCandidates.size();++n)
		{
			jointCandGroupVector[j].nodeCandidates[n]->pTargetSurfVox->marginalProb = jointCandGroupVector[j].nodeCandidates[n]->scoreSum;
		}
	}

	//Save inference results
	newPoseReconMem.m_imgFramdIdx = pTempSceneProb->m_imgFramdIdx;

	//////////////////////////////////////////////////////////////////////////
	//Select Root node candidates based on NMS
	vector<InferNode*>& rootNodeCandOriginal = jointCandGroupVector[0].nodeCandidates;
	vector<InferNode*> rootNodeCand;
	float bandwidth = cm2world(5);
	vector<bool> isLocalMax;
	isLocalMax.resize(rootNodeCandOriginal.size(),true);
	#pragma omp parallel for
	for(int i=0;i<rootNodeCandOriginal.size();++i)
	{
		for(int j=0;j<rootNodeCandOriginal.size();++j)
		{
			if(i==j)
				continue;
			float dist = Distance(rootNodeCandOriginal[i]->pos,rootNodeCandOriginal[j]->pos);
			if(dist<bandwidth)
			{
				if(rootNodeCandOriginal[i]->scoreSum > rootNodeCandOriginal[j]->scoreSum  )
					isLocalMax[j] = false;
			}
		}
	}
	for(int i=0;i<rootNodeCandOriginal.size();++i)
	{
		if(isLocalMax[i]==true)
			rootNodeCand.push_back(rootNodeCandOriginal[i]);
	}
	printf("rootNodeCand: %d\n",rootNodeCand.size());

	//////////////////////////////////////////////////////////////////////////
	/// Save all valid skeleton candidates
	std::list<InferNode*> rootNodeCandList(rootNodeCand.begin(),rootNodeCand.end());
	while(rootNodeCandList.size()>0)
	{
		printf("Number of candidate: %d\n",rootNodeCandList.size());
		//////////////////////////////////////////////////////////////////////////
		// Find Best One
		InferNode* pBestNode=NULL;
		float maxProb=-1e10;

		list<InferNode*>::iterator iter = rootNodeCandList.begin();
		list<InferNode*>::iterator bestIter;// = rootNodeCandList.end();
		while(iter!=rootNodeCandList.end())
		{
			if(iter==rootNodeCandList.begin() ||  (*iter)->scoreSum > maxProb)
			{
				maxProb = (*iter)->scoreSum;
				bestIter = iter;
			}
			iter++;
		}
		pBestNode = *bestIter;
		rootNodeCandList.erase(bestIter);

		//if(maxProb <-1 || pBestNode ==NULL)//maxProb==-1e10)
		double bestDataCost = pBestNode->scoreSum-pBestNode->springScoreSum-pBestNode->connectivityScoreSum;
		printf("bestDataCost  %f\n",bestDataCost);

		if(pBestNode->dataScore<headThreshold)
		{
			printf("## 3DPS: rejected by headtop datascore threshold: %f\n",pBestNode->dataScore);
			continue;
		}
		//////////////////////////////////////////////////////////////////////////
		//Back tracking
		double dataTermSum = 0;
		vector<InferNode*> allRelatedInferNode; 
		allRelatedInferNode.push_back(pBestNode);
		dataTermSum += DATASCORE_RATIO*pBestNode->dataScore;			//debug

		int iterCnt =0;
		bool bValid = true;
		int bodyPartCnt = 0;
		int shoulderCnt = 0;
		while(iterCnt<allRelatedInferNode.size())
		{
			InferNode* pTempInferNode= allRelatedInferNode[iterCnt];
			 
			//if(pTempInferNode==NULL || pTempInferNode->bTaken==true)
			
			if(pTempInferNode!=NULL && pTempInferNode->associatedJointIdx<=2)
				bodyPartCnt++;
			if(pTempInferNode!=NULL && (pTempInferNode->associatedJointIdx==3 || pTempInferNode->associatedJointIdx==9) )
				shoulderCnt++;
			for(int i=0;i<pTempInferNode->pPrevInferNodeVect.size();++i)
			{
				InferNode* pPrevInferNode = pTempInferNode->pPrevInferNodeVect[i];
				if(pPrevInferNode->bTaken==true )		//reject if any torso parts are overlapped
				{
					if(pPrevInferNode->associatedJointIdx<=3)
					{
						bValid = false;
						break;
					}
					else  //Then, just ignore this part
						continue;
				}

				allRelatedInferNode.push_back(pPrevInferNode);			//I am sure no infinite loop by doing this. 
				dataTermSum += DATASCORE_RATIO*pPrevInferNode->dataScore;			//debug
			}
			if(bValid==false)
				break;

			iterCnt++;
		}
		if(bodyPartCnt<3)		//reject if it doesn't have full body (head, neck, body)
		{
			printf("## 3DPS: rejected by body-head validity constraint: bodyPartCnt: %d\n",bodyPartCnt);
			bValid=false;
		}
		if(iterCnt<=6 || shoulderCnt<2)
			bValid=false;
		if(bValid==false)
		{
			printf("Not a valid candidate\n");
			continue;
		}

		printfLog("Best Prob %f (data: %f, spring: %f, connect: %f): validjointNum %d\n",maxProb,pBestNode->dataScoreSum,pBestNode->springScoreSum,pBestNode->connectivityScoreSum,allRelatedInferNode.size());

		//////////////////////////////////////////////////////////////////////////
		// Add Skeleton	
		CBody3D newHuman;
		newHuman.m_jointPosVect.resize(m_skeletonHierarchy.size());
		newHuman.m_jointConfidenceVect.resize(m_skeletonHierarchy.size(),-1);
		for(int i=0;i<allRelatedInferNode.size();++i)
		{
			int jointIdx = allRelatedInferNode[i]->associatedJointIdx;
			newHuman.m_jointPosVect[jointIdx] = allRelatedInferNode[i]->pos;
			//if(DATASCORE_RATIO>0)
				newHuman.m_jointConfidenceVect[jointIdx] = allRelatedInferNode[i]->dataScore;
			/*else
				newHuman.m_jointConfidenceVect[jointIdx] = 0;*/
			allRelatedInferNode[i]->bTaken = true;
			allRelatedInferNode[i]->pTargetSurfVox->label = newPoseReconMem.m_articBodyAtEachTime.size();		//exclude this node. It is already taken.
		}
		//if(newHuman.m_jointPosVect.size() == m_skeletonHierarchy.size())
		{
			newPoseReconMem.m_articBodyAtEachTime.push_back(newHuman);
		}

		printf("Found a valid skeleton\n");
		break;
	}

	//////////////////////////////////////////////////////////////////////////
	/// Deallocate memories

	for(int i=0;i<jointCandGroupVector.size();++i)
	{
		for(int j=0;j<jointCandGroupVector[i].nodeCandidates.size();++j)
			delete jointCandGroupVector[i].nodeCandidates[j];
		jointCandGroupVector[i].nodeCandidates.clear();
	}
}

//SBody3DScene contains the pose of all subjects at a time instance.
//Version Log
//ver 0.6: Added Joint Scores
//ver 0.8: Added Outlier information
void SaveBodyReconResult(const char* folderPath,const SBody3DScene& poseReconMem,const int frameIdx,bool isHD)
{
	//Save Face Reconstruction Result
	//char folderPath[512];
	//sprintf(folderPath,"%s/poseRecon_c%d",g_dataMainFolder,g_askedVGACamNum);
	CreateFolder(folderPath);

	char fullPath[512];
	if(isHD==false)
		sprintf(fullPath,"%s/body3DScene_%08d.txt",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
	else
		sprintf(fullPath,"%s/body3DScene_hd%08d.txt",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
	if(frameIdx%500==0)
		printf("Save to %s\n",fullPath);
	ofstream fout(fullPath,std::ios_base::trunc);
	//fout << "ver 0.6\n";		//0.6: added joint score compared to 0.6
	fout << "ver 0.8\n";		//0.8: added joint outlier information
	fout << poseReconMem.m_articBodyAtEachTime.size() <<" ";

	if(poseReconMem.m_articBodyAtEachTime.size()==0)
		fout <<"0\n";
	else
		fout << poseReconMem.m_articBodyAtEachTime.front().m_jointPosVect.size() << "\n";

	for(unsigned int s=0;s<poseReconMem.m_articBodyAtEachTime.size();++s)
	{
		const CBody3D& skeleton = poseReconMem.m_articBodyAtEachTime[s];

		for(int i=0;i<skeleton.m_jointPosVect.size();++i)
		{
			fout << skeleton.m_jointPosVect[i].x << " " << skeleton.m_jointPosVect[i].y << " " << skeleton.m_jointPosVect[i].z << " ";
			
			if(skeleton.m_jointConfidenceVect.size()>0)
				fout << skeleton.m_jointConfidenceVect[i] <<" ";
			else
				fout << -1 <<" ";

			if(skeleton.m_isOutlierJoint.size()==skeleton.m_jointPosVect.size())
			{
				if(skeleton.m_isOutlierJoint[i])
					fout << 1 << " ";
				else
					fout << 0 << " ";
			}
			else
					fout << 0 << " ";
		}
		fout <<"\n";
	}
	fout.close();
}

//SBody3DScene contains the pose of all subjects at a time instance.
void SaveBodyReconResult_json(const char* folderPath,const SBody3DScene& poseReconMem,const int frameIdx,bool bNormCoord,bool bAnnotated)
{
	CreateFolder(folderPath);

	char fullPath[512];
	sprintf(fullPath,"%s/body3DScene_%08d.json",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
	printf("Save to %s\n",fullPath);
	ofstream fout(fullPath,std::ios_base::trunc);
	fout << "{ \"version\": 0.7, \n";		//ver 0.6: added confidence. ver 0.7: added valid humanLabel
	char buf[512];
	if(g_fpsType == FPS_VGA_25)
	{
		sprintf(buf,"\"univTime\" :%0.3f,\n",g_syncMan.GetUnivTimeVGA(frameIdx) );
		fout <<buf;
		sprintf(buf,"\"fpsType\" :\"vga_25\",\n");
		fout <<buf;
		printf("%f vs %0.3f vs %s\n",g_syncMan.GetUnivTimeVGA(frameIdx),g_syncMan.GetUnivTimeVGA(frameIdx),buf);
		//fout <<buf;
		sprintf(buf,"\"vgaVideoTime\" :%0.3f,\n",g_syncMan.GetUnivTimeVGA(frameIdx)-g_syncMan.GetUnivTimeVGA(0));
		fout <<buf;
	}
	else
	{
		sprintf(buf,"\"univTime\" :%0.3f,\n",g_syncMan.GetUnivTimeHD(frameIdx) );
		fout <<buf;
		sprintf(buf,"\"fpsType\" :\"hd_29_97\",\n");
		fout <<buf;
		printf("%f vs %0.3f vs %s\n",g_syncMan.GetUnivTimeHD(frameIdx),g_syncMan.GetUnivTimeHD(frameIdx),buf);
		//fout <<buf;
	}

	fout << "\"bodies\" :\n";
	fout << "[";
	Mat_<double> R,t;
	double scale;
	if(bNormCoord)
	{
		CDomeImageManager fullDome;
		fullDome.InitDomeCamOnlyByExtrinsic(g_calibrationFolder);
		fullDome.CalcWorldScaleParameters();
		fullDome.ComputeRtForNormCoord(R,t,scale);
	}
	bool isFirst=true;
	for(unsigned int s=0;s<poseReconMem.m_articBodyAtEachTime.size();++s)
	{
		const CBody3D& skeleton = poseReconMem.m_articBodyAtEachTime[s];

		if(skeleton.m_humanIdentLabel<0)
			continue;		//false positive
		if(isFirst)
		{
			fout <<"\n{ \"id\": "<<skeleton.m_humanIdentLabel	 <<",\n";
			isFirst = false;
		}
		else
			fout <<",\n{ \"id\": "<<skeleton.m_humanIdentLabel <<",\n";

		//joints15
		if(skeleton.m_jointPosVect.size()==19)
			fout <<"\"joints19\": [";
		else if(skeleton.m_jointPosVect.size() == 26)
			fout << "\"joints26\": [";
		else
			fout <<"\"joints15\": [";
		vector<Mat_<double>> normCoord;
		if(bNormCoord)
		{
			normCoord.resize(skeleton.m_jointPosVect.size());
			#pragma omp parallel for num_threads(8)	
			for(int i=0;i<skeleton.m_jointPosVect.size();++i)
			{
				Mat_<double> x_world;
				Point3fToMat(skeleton.m_jointPosVect[i],x_world);
				x_world = (R*x_world +t)*scale;
				normCoord[i] = x_world;
			}
		}
		

		for(int i=0;i<skeleton.m_jointPosVect.size();++i)
		{
			if(bNormCoord==false)
				fout << skeleton.m_jointPosVect[i].x << ", " << skeleton.m_jointPosVect[i].y << ", " << skeleton.m_jointPosVect[i].z;
			else
			{
				/*Mat_<double> x_world;
				Point3fToMat(skeleton.m_jointPosVect[i],x_world);
				x_world = R*x_world +t;*/
				fout << normCoord[i](0,0) << ", " << normCoord[i](1,0) << ", " << normCoord[i](2,0);
			}
			
			if(skeleton.m_jointConfidenceVect.size()>0)
				fout << ", " << skeleton.m_jointConfidenceVect[i];
			else
				fout << ", " << 0.0;
			if(i+1<skeleton.m_jointPosVect.size())
				fout<<", ";
		}
		fout <<"]\n}";
	}

	fout <<"\n] }";
	fout.close();
}

//SBody3DScene contains the pose of all subjects at a time instance.
//Version Log
//ver 0.6: Added Joint Scores
//ver 0.8: Added Outlier information
bool CBodyPoseRecon::LoadBodyReconResult(char* fullPath,int imgFrameIdx)
{
	//printf("BodyPoseRecon::Loading from %s\n",fullPath);
	//Save Face Reconstruction Result
	ifstream fin(fullPath, ios::in);

	if(fin.is_open()==false)
	{
		printf("# Failed:: load from %s\n",fullPath);
		return false;
	}
	
	//printf("Load from %s\n",fullPath);

	char dummy[512];
	if(fin)
	{
		m_3DPSPoseMemVector.push_back(SBody3DScene());
		SBody3DScene& newPoseReconMem = m_3DPSPoseMemVector.back();
		newPoseReconMem.m_imgFramdIdx = imgFrameIdx;

		float ver;
		fin >> dummy >> ver; //version
		if(ver<0.5)
		{
			printf("version mismatch\n");
			m_3DPSPoseMemVector.pop_back();
			return false;
		}
		int subjectNum;
		int jointNum;
		fin >>subjectNum >>jointNum;
		if(jointNum>0 && jointNum!=m_skeletonHierarchy.size())
		{
			ClearSKeletonHierarchy();
			ConstructJointHierarchy(jointNum);
		}

		//printf("Human Pose Num: %d, JointNum %d\n",subjectNum,jointNum);
		newPoseReconMem.m_articBodyAtEachTime.reserve(subjectNum);
		for(int i=0;i<subjectNum;++i)
		{
			newPoseReconMem.m_articBodyAtEachTime.push_back(CBody3D());
			CBody3D& newSkeleton = newPoseReconMem.m_articBodyAtEachTime.back();
			//if(ver>0.69) //0.7: human identity
				//fin >> newSkeleton.m_humanIdentLabel;
			newSkeleton.m_jointPosVect.reserve(jointNum);
			newSkeleton.m_jointConfidenceVect.reserve(jointNum);
			for(int j=0;j<jointNum;++j)
			{
				Point3f temp;
				fin >>temp.x >> temp.y >> temp.z;
				newSkeleton.m_jointPosVect.push_back(temp);

				if(ver>0.55)		//ver 0.6.. added joint confidence
				{
					double confidence;
					fin >> confidence;
					newSkeleton.m_jointConfidenceVect.push_back(confidence);
				}

				if(ver>0.75)		//ver 0.8.. added (manually specified) outlier information
				{
					int bIsOulier;
					fin >> bIsOulier;			//1 (manually specified), 0: valid, -1: automatically obtained non valid 
					if(bIsOulier)				
						newSkeleton.m_isOutlierJoint.push_back(true);
					else
						newSkeleton.m_isOutlierJoint.push_back(false);

				}
			}

			float tempHeight = abs(newSkeleton.m_jointPosVect.front().y- newSkeleton.m_jointPosVect.back().y);

			//Compute body normal
			Point3d neckToRoot= newSkeleton.m_jointPosVect[SMC_BodyJoint_bodyCenter] - newSkeleton.m_jointPosVect[SMC_BodyJoint_neck];
			Point3d rightToLeft = newSkeleton.m_jointPosVect[SMC_BodyJoint_lShoulder]  -newSkeleton.m_jointPosVect[SMC_BodyJoint_rShoulder];
			Normalize(neckToRoot);
			Normalize(rightToLeft);
			Point3d normalDirect = neckToRoot.cross(rightToLeft);
			Normalize(normalDirect);
			Point3dToMat(normalDirect,newSkeleton.m_bodyNormal);

			//Compute Rotation0Translation
			Point3d yaxis = normalDirect.cross(rightToLeft);
			Mat_<double> R = Mat_<double>::eye(3,3);
			R(0,0) = rightToLeft.x;
			R(1,0) = rightToLeft.y;
			R(2,0) = rightToLeft.z;

			R(0,1) = yaxis.x;
			R(1,1) = yaxis.y;
			R(2,1) = yaxis.z;

			R(0,2) = normalDirect.x;
			R(1,2) = normalDirect.y;
			R(2,2) = normalDirect.z;
			newSkeleton.m_ext_R = R.inv();

			Mat pt;
			Point3fToMat4by1(newSkeleton.m_jointPosVect[SMC_BodyJoint_neck],pt);
			newSkeleton.m_ext_t  = -R*pt.rowRange(0,3);
		}
	}
	fin.close();
	GenerateHumanColors();
	return true;
}

void CBodyPoseRecon::ConstructJointHierarchy(int jointNum)
{
	if(m_skeletonHierarchy.size()>0)
		return;

	if(jointNum!=19 && jointNum != 26)
	{
		printf("Joint number mismatched (should be 19 || 26): input %d \n",jointNum);
		return;
	}

	if (jointNum==19)
	{
		//Construct tree structure
		STreeElement* pNeck = new STreeElement(JOINT_neck);
		pNeck->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pNeck);	//0

		STreeElement* pHeadTop = new STreeElement(JOINT_headTop);
		pHeadTop->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pHeadTop);	//1
		pNeck->children.push_back(pHeadTop);
		pHeadTop->parents.push_back(pNeck);

		STreeElement* pBodyCenter =  new STreeElement(Joint_bodyCenter);
		pBodyCenter->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pBodyCenter);	//2
		pNeck->children.push_back(pBodyCenter);
		pBodyCenter->parents.push_back(pNeck);

		//////////////////////////////////////////////////////////////////////////
		// left part
		STreeElement* pLShoulder =  new STreeElement(JOINT_lShoulder);
		pLShoulder->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLShoulder);	//3
		pNeck->children.push_back(pLShoulder);
		pLShoulder->parents.push_back(pNeck);

	
		STreeElement* pLElbow =  new STreeElement(JOINT_lElbow);
		pLElbow->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLElbow);	//4
		pLShoulder->children.push_back(pLElbow);
		pLElbow->parents.push_back(pLShoulder);

		STreeElement* pLHand =  new STreeElement(JOINT_lHand);
		pLHand->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLHand);	//5
		pLElbow->children.push_back(pLHand);
		pLHand->parents.push_back(pLElbow);

		STreeElement* pLHip =  new STreeElement(JOINT_lHip);
		pLHip->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLHip);	//6
		pBodyCenter->children.push_back(pLHip);
		pLHip->parents.push_back(pBodyCenter);

		STreeElement* pLKnee =  new STreeElement(JOINT_lKnee);
		pLKnee->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLKnee);	//7
		pLHip->children.push_back(pLKnee);
		pLKnee->parents.push_back(pLHip);

		STreeElement* pLFoot =  new STreeElement(JOINT_lFoot);
		pLFoot->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLFoot);	//8
		pLKnee->children.push_back(pLFoot);
		pLFoot->parents.push_back(pLKnee);


		//////////////////////////////////////////////////////////////////////////
		// Right part
		STreeElement* pRShoulder =  new STreeElement(JOINT_rShoulder);
		pRShoulder->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRShoulder);	//9
		pNeck->children.push_back(pRShoulder);
		pRShoulder->parents.push_back(pNeck);

		STreeElement* pRElbow =  new STreeElement(JOINT_rElbow);
		pRElbow->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRElbow);	//10
		pRShoulder->children.push_back(pRElbow);
		pRElbow->parents.push_back(pRShoulder);

		STreeElement* pRHand =  new STreeElement(JOINT_rHand);
		pRHand->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRHand);	//11
		pRElbow->children.push_back(pRHand);
		pRHand->parents.push_back(pRElbow);

		STreeElement* pRHip =  new STreeElement(JOINT_rHip);
		pRHip->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRHip);	//12
		pBodyCenter->children.push_back(pRHip);
		pRHip->parents.push_back(pBodyCenter);

		STreeElement* pRKnee =  new STreeElement(JOINT_rKnee);
		pRKnee->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRKnee);	//13
		pRHip->children.push_back(pRKnee);
		pRKnee->parents.push_back(pRHip);

		STreeElement* pRFoot =  new STreeElement(JOINT_rFoot);
		pRFoot->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRFoot);	//14
		pRKnee->children.push_back(pRFoot);
		pRFoot->parents.push_back(pRKnee);

		//Coco additional landmarks
		STreeElement* pLEye =  new STreeElement(JOINT_lEye);
		pLEye->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLEye);	//16
		pHeadTop->children.push_back(pLEye);
		pLEye->parents.push_back(pHeadTop);

		STreeElement* pLEar =  new STreeElement(JOINT_lEar);
		pLEar->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLEar);	//16
		pLEye->children.push_back(pLEar);
		pLEar->parents.push_back(pLEye);

		STreeElement* pREye =  new STreeElement(JOINT_rEye);
		pREye->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pREye);	//15
		pHeadTop->children.push_back(pREye);
		pREye->parents.push_back(pHeadTop);
		
		STreeElement* pREar =  new STreeElement(JOINT_rEar);
		pREar->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pREar);	//15
		pREye->children.push_back(pREar);
		pREar->parents.push_back(pREye);


		//////////////////////////////////////////////////////////////////////////
		//Set joint length: order is exactly same as above structure (Since I just printed out)
		/*
		Joint 0-1: 45.543312 
		Joint 0-2: 38.904293 
		Joint 0-8: 30.871185 
		Joint 2-3: 59.428207 
		Joint 2-5: 95.444031 
		Joint 3-4: 53.469955 
		Joint 5-6: 65.699959 
		Joint 6-7: 77.267548 
		Joint 8-9: 70.435951 
		Joint 8-11: 87.636108 
		Joint 9-10: 53.206226 
		Joint 11-12: 80.116852 
		Joint 12-13: 72.784691 */
	
		CDomeImageManager dummyImageset;	//To calculate working volume
		dummyImageset.InitDomeCamOnlyByExtrinsic(g_calibrationFolder);
		dummyImageset.CalcWorldScaleParameters();
		dummyImageset.DeleteMemory();

		//cm
		float jointLength[]= {26, 60.4,
				23,  28,  25,
				11,  47,  55,
				23,  28,  25,
				11,  47,  55,
				15, 10, 15, 10};	 //Eyes, Ears

	//	float jointLength[]= {46.368018,  34.742435,  54.772530,  45.276409,  90.691743,  62.649436,  82.916327,  34.742435,  54.772530,  45.276409,  90.691743,  62.649436,  82.916327, 0};
		int cnt =0;
		for(int i=0;i<m_skeletonHierarchy.size();++i)		//to verify this hierarchy
		{
			for(int j=0;j<m_skeletonHierarchy[i]->parents.size();++j)
			{
				//int childJointIdx = m_skeletonHierarchy[i]->children[j]->idxInTree;
				m_skeletonHierarchy[i]->parent_jointLength.push_back(cm2world(jointLength[cnt++]));
			}
		}
		
		//////////////////////////////////////////////////////////////////////////
		/// Colors to be used to draw bones or relavant information
		m_boneColorVector.clear();
		m_boneColorVector.push_back(Point3f(173,167,107));	//neck
		m_boneColorVector.push_back(Point3f(255,230,154));	//head
		m_boneColorVector.push_back(Point3f(194,216,224));  //body

		m_boneColorVector.push_back(Point3f(242,182,224)); //LShoulder magenta
		m_boneColorVector.push_back(Point3f(213,89,176)); 
		m_boneColorVector.push_back(Point3f(132,8,95)); 

		m_boneColorVector.push_back(Point3f(196,246,185)); //LHip green
		m_boneColorVector.push_back(Point3f(118,227,95)); 
		m_boneColorVector.push_back(Point3f(33,186,9)); 

		m_boneColorVector.push_back(Point3f(253,191,197));  //RShoulder red
		m_boneColorVector.push_back(Point3f(249,104,118)); 
		m_boneColorVector.push_back(Point3f(164,55,65)); 

		m_boneColorVector.push_back(Point3f(176,230,233)); //RHip blue
		m_boneColorVector.push_back(Point3f(79,182,187)); 
		m_boneColorVector.push_back(Point3f(7,102,106)); 

		m_boneColorVector.push_back(Point3f(255,230,154));	//lEye  //same color as head 
		m_boneColorVector.push_back(Point3f(255,230,154));	//lEar	//same color as head
		m_boneColorVector.push_back(Point3f(255,230,154));	//rEye	//same color as head
		m_boneColorVector.push_back(Point3f(255,230,154));	//rEar	//same color as head

		for(int i=0;i<m_boneColorVector.size();++i)
			m_boneColorVector[i] = m_boneColorVector[i]*(1/255.0);

		//////////////////////////////////////////////////////////////////////////
		m_skeletonHierarchy[0]->mirrorJointIdx =0;
		m_skeletonHierarchy[1]->mirrorJointIdx =1;
		m_skeletonHierarchy[2]->mirrorJointIdx =2;

		m_skeletonHierarchy[3]->mirrorJointIdx =3;
		m_skeletonHierarchy[4]->mirrorJointIdx =4;
		m_skeletonHierarchy[5]->mirrorJointIdx =5;

		m_skeletonHierarchy[6]->mirrorJointIdx =6;
		m_skeletonHierarchy[7]->mirrorJointIdx =7;
		m_skeletonHierarchy[8]->mirrorJointIdx =8;

		m_skeletonHierarchy[9]->mirrorJointIdx =3;
		m_skeletonHierarchy[10]->mirrorJointIdx =4;
		m_skeletonHierarchy[11]->mirrorJointIdx =5;

		m_skeletonHierarchy[12]->mirrorJointIdx =6;
		m_skeletonHierarchy[13]->mirrorJointIdx =7;
		m_skeletonHierarchy[14]->mirrorJointIdx =8;

		m_skeletonHierarchy[15]->mirrorJointIdx =17;
		m_skeletonHierarchy[16]->mirrorJointIdx =18;
		m_skeletonHierarchy[17]->mirrorJointIdx =15;
		m_skeletonHierarchy[18]->mirrorJointIdx =16;


		//////////////////////////////////////////////////////////////////////////
		m_skeletonHierarchy[0]->levelToRootInHiearchy =0;
		m_skeletonHierarchy[1]->levelToRootInHiearchy =1;
		m_skeletonHierarchy[2]->levelToRootInHiearchy =1;

		m_skeletonHierarchy[3]->levelToRootInHiearchy =1;
		m_skeletonHierarchy[4]->levelToRootInHiearchy =3;
		m_skeletonHierarchy[5]->levelToRootInHiearchy =4;

		m_skeletonHierarchy[6]->levelToRootInHiearchy =2;
		m_skeletonHierarchy[7]->levelToRootInHiearchy =5;
		m_skeletonHierarchy[8]->levelToRootInHiearchy =6;
		
		m_skeletonHierarchy[9]->levelToRootInHiearchy =1;
		m_skeletonHierarchy[10]->levelToRootInHiearchy =3;
		m_skeletonHierarchy[11]->levelToRootInHiearchy =4;

		m_skeletonHierarchy[12]->levelToRootInHiearchy =2;
		m_skeletonHierarchy[13]->levelToRootInHiearchy =5;
		m_skeletonHierarchy[14]->levelToRootInHiearchy =6;

		m_skeletonHierarchy[15]->levelToRootInHiearchy =2;
		m_skeletonHierarchy[16]->levelToRootInHiearchy =3;
		m_skeletonHierarchy[17]->levelToRootInHiearchy =2;
		m_skeletonHierarchy[18]->levelToRootInHiearchy =3;
	}
	else if(jointNum==26)
	{
		//Construct tree structure
		STreeElement* pNeck = new STreeElement(JOINT_neck);
		pNeck->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pNeck);	//0

		STreeElement* pHeadTop = new STreeElement(JOINT_headTop);
		pHeadTop->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pHeadTop);	//1
		pNeck->children.push_back(pHeadTop);
		pHeadTop->parents.push_back(pNeck);

		STreeElement* pBodyCenter =  new STreeElement(Joint_bodyCenter);
		pBodyCenter->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pBodyCenter);	//2
		pNeck->children.push_back(pBodyCenter);
		pBodyCenter->parents.push_back(pNeck);

		//////////////////////////////////////////////////////////////////////////
		// left part
		STreeElement* pLShoulder =  new STreeElement(JOINT_lShoulder);
		pLShoulder->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLShoulder);	//3
		pNeck->children.push_back(pLShoulder);
		pLShoulder->parents.push_back(pNeck);

	
		STreeElement* pLElbow =  new STreeElement(JOINT_lElbow);
		pLElbow->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLElbow);	//4
		pLShoulder->children.push_back(pLElbow);
		pLElbow->parents.push_back(pLShoulder);

		STreeElement* pLHand =  new STreeElement(JOINT_lHand);
		pLHand->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLHand);	//5
		pLElbow->children.push_back(pLHand);
		pLHand->parents.push_back(pLElbow);

		STreeElement* pLHip =  new STreeElement(JOINT_lHip);
		pLHip->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLHip);	//6
		pBodyCenter->children.push_back(pLHip);
		pLHip->parents.push_back(pBodyCenter);

		STreeElement* pLKnee =  new STreeElement(JOINT_lKnee);
		pLKnee->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLKnee);	//7
		pLHip->children.push_back(pLKnee);
		pLKnee->parents.push_back(pLHip);

		STreeElement* pLFoot =  new STreeElement(JOINT_lFoot);
		pLFoot->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLFoot);	//8
		pLKnee->children.push_back(pLFoot);
		pLFoot->parents.push_back(pLKnee);


		//////////////////////////////////////////////////////////////////////////
		// Right part
		STreeElement* pRShoulder =  new STreeElement(JOINT_rShoulder);
		pRShoulder->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRShoulder);	//9
		pNeck->children.push_back(pRShoulder);
		pRShoulder->parents.push_back(pNeck);

		STreeElement* pRElbow =  new STreeElement(JOINT_rElbow);
		pRElbow->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRElbow);	//10
		pRShoulder->children.push_back(pRElbow);
		pRElbow->parents.push_back(pRShoulder);

		STreeElement* pRHand =  new STreeElement(JOINT_rHand);
		pRHand->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRHand);	//11
		pRElbow->children.push_back(pRHand);
		pRHand->parents.push_back(pRElbow);

		STreeElement* pRHip =  new STreeElement(JOINT_rHip);
		pRHip->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRHip);	//12
		pBodyCenter->children.push_back(pRHip);
		pRHip->parents.push_back(pBodyCenter);

		STreeElement* pRKnee =  new STreeElement(JOINT_rKnee);
		pRKnee->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRKnee);	//13
		pRHip->children.push_back(pRKnee);
		pRKnee->parents.push_back(pRHip);

		STreeElement* pRFoot =  new STreeElement(JOINT_rFoot);
		pRFoot->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pRFoot);	//14
		pRKnee->children.push_back(pRFoot);
		pRFoot->parents.push_back(pRKnee);

		//Coco additional landmarks
		STreeElement* pLEye =  new STreeElement(JOINT_lEye);
		pLEye->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLEye);	//15
		pHeadTop->children.push_back(pLEye);
		pLEye->parents.push_back(pHeadTop);

		STreeElement* pLEar =  new STreeElement(JOINT_lEar);
		pLEar->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pLEar);	//16
		pLEye->children.push_back(pLEar);
		pLEar->parents.push_back(pLEye);

		STreeElement* pREye =  new STreeElement(JOINT_rEye);
		pREye->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pREye);	//17
		pHeadTop->children.push_back(pREye);
		pREye->parents.push_back(pHeadTop);
		
		STreeElement* pREar =  new STreeElement(JOINT_rEar);
		pREar->idxInTree = m_skeletonHierarchy.size();
		m_skeletonHierarchy.push_back(pREar);	//18
		pREye->children.push_back(pREar);
		pREar->parents.push_back(pREye);

		STreeElement* pLBigToe = new STreeElement(JOINT_lBigToe);
		pLBigToe->idxInTree = m_skeletonHierarchy.size();  //19
		m_skeletonHierarchy.emplace_back(pLBigToe);
		pLFoot->children.emplace_back(pLBigToe);
		pLBigToe->parents.emplace_back(pLFoot);

		STreeElement* pLSmallToe = new STreeElement(JOINT_lSmallToe);
		pLSmallToe->idxInTree = m_skeletonHierarchy.size();  //20
		m_skeletonHierarchy.emplace_back(pLSmallToe);
		pLFoot->children.emplace_back(pLSmallToe);
		pLSmallToe->parents.emplace_back(pLFoot);

		STreeElement* pLHeel = new STreeElement(JOINT_lHeel);
		pLHeel->idxInTree = m_skeletonHierarchy.size();  //21
		m_skeletonHierarchy.emplace_back(pLHeel);
		pLFoot->children.emplace_back(pLHeel);
		pLHeel->parents.emplace_back(pLFoot);

		STreeElement* pRBigToe = new STreeElement(JOINT_rBigToe);
		pRBigToe->idxInTree = m_skeletonHierarchy.size();  //22
		m_skeletonHierarchy.emplace_back(pRBigToe);
		pRFoot->children.emplace_back(pRBigToe);
		pRBigToe->parents.emplace_back(pRFoot);

		STreeElement* pRSmallToe = new STreeElement(JOINT_rSmallToe);
		pRSmallToe->idxInTree = m_skeletonHierarchy.size();  //23
		m_skeletonHierarchy.emplace_back(pRSmallToe);
		pRFoot->children.emplace_back(pRSmallToe);
		pRSmallToe->parents.emplace_back(pRFoot);

		STreeElement* pRHeel = new STreeElement(JOINT_rHeel);
		pRHeel->idxInTree = m_skeletonHierarchy.size();  //24
		m_skeletonHierarchy.emplace_back(pRHeel);
		pRFoot->children.emplace_back(pRHeel);
		pRHeel->parents.emplace_back(pRFoot);

		STreeElement* pRealHeadTop = new STreeElement(JOINT_realheadtop);
		pRealHeadTop->idxInTree = m_skeletonHierarchy.size();  //25
		m_skeletonHierarchy.emplace_back(pRealHeadTop);
		pNeck->children.emplace_back(pRealHeadTop);
		pRealHeadTop->parents.emplace_back(pNeck);

		//////////////////////////////////////////////////////////////////////////
		//Set joint length: order is exactly same as above structure (Since I just printed out)
		/*
		Joint 0-1: 45.543312 
		Joint 0-2: 38.904293 
		Joint 0-8: 30.871185 
		Joint 2-3: 59.428207 
		Joint 2-5: 95.444031 
		Joint 3-4: 53.469955 
		Joint 5-6: 65.699959 
		Joint 6-7: 77.267548 
		Joint 8-9: 70.435951 
		Joint 8-11: 87.636108 
		Joint 9-10: 53.206226 
		Joint 11-12: 80.116852 
		Joint 12-13: 72.784691 */
	
		CDomeImageManager dummyImageset;	//To calculate working volume
		dummyImageset.InitDomeCamOnlyByExtrinsic(g_calibrationFolder);
		dummyImageset.CalcWorldScaleParameters();
		dummyImageset.DeleteMemory();

		//cm
		// donglai: change middle to LRHip distance from 11cm -> 20cm, due to new OpenPose detector
		float jointLength[]= {26, 60.4,
				23,  28,  25,
				11,  47,  55,
				23,  28,  25,
				11,  47,  55,
				15, 10, 15, 10, //Eyes, Ears
				20, 20, 10, 20, 20, 10, 20};  // toe and heels

	//	float jointLength[]= {46.368018,  34.742435,  54.772530,  45.276409,  90.691743,  62.649436,  82.916327,  34.742435,  54.772530,  45.276409,  90.691743,  62.649436,  82.916327, 0};
		int cnt =0;
		for(int i=0;i<m_skeletonHierarchy.size();++i)		//to verify this hierarchy
		{
			for(int j=0;j<m_skeletonHierarchy[i]->parents.size();++j)
			{
				//int childJointIdx = m_skeletonHierarchy[i]->children[j]->idxInTree;
				m_skeletonHierarchy[i]->parent_jointLength.push_back(cm2world(jointLength[cnt++]));
			}
		}

		//////////////////////////////////////////////////////////////////////////
		/// Colors to be used to draw bones or relavant information
		m_boneColorVector.clear();
		m_boneColorVector.push_back(Point3f(173,167,107));	//neck
		m_boneColorVector.push_back(Point3f(255,230,154));	//head
		m_boneColorVector.push_back(Point3f(194,216,224));  //body

		m_boneColorVector.push_back(Point3f(242,182,224)); //LShoulder magenta
		m_boneColorVector.push_back(Point3f(213,89,176)); 
		m_boneColorVector.push_back(Point3f(132,8,95)); 

		m_boneColorVector.push_back(Point3f(196,246,185)); //LHip green
		m_boneColorVector.push_back(Point3f(118,227,95)); 
		m_boneColorVector.push_back(Point3f(33,186,9)); 

		m_boneColorVector.push_back(Point3f(253,191,197));  //RShoulder red
		m_boneColorVector.push_back(Point3f(249,104,118)); 
		m_boneColorVector.push_back(Point3f(164,55,65)); 

		m_boneColorVector.push_back(Point3f(176,230,233)); //RHip blue
		m_boneColorVector.push_back(Point3f(79,182,187)); 
		m_boneColorVector.push_back(Point3f(7,102,106)); 

		m_boneColorVector.push_back(Point3f(255,230,154));	//lEye  //same color as head 
		m_boneColorVector.push_back(Point3f(255,230,154));	//lEar	//same color as head
		m_boneColorVector.push_back(Point3f(255,230,154));	//rEye	//same color as head
		m_boneColorVector.push_back(Point3f(255,230,154));	//rEar	//same color as head

		m_boneColorVector.push_back(Point3f(33,186,9));  // lBigToe, same as lFoot
		m_boneColorVector.push_back(Point3f(33,186,9)); 
		m_boneColorVector.push_back(Point3f(33,186,9)); 

		m_boneColorVector.push_back(Point3f(7,102,106));  // rBigToe, same as rFoot
		m_boneColorVector.push_back(Point3f(7,102,106));
		m_boneColorVector.push_back(Point3f(7,102,106));

		m_boneColorVector.push_back(Point3f(255,230,154));  // real head top, same as head


		for(int i=0;i<m_boneColorVector.size();++i)
			m_boneColorVector[i] = m_boneColorVector[i]*(1/255.0);

		//////////////////////////////////////////////////////////////////////////
		m_skeletonHierarchy[0]->mirrorJointIdx =0;
		m_skeletonHierarchy[1]->mirrorJointIdx =1;
		m_skeletonHierarchy[2]->mirrorJointIdx =2;

		m_skeletonHierarchy[3]->mirrorJointIdx =3;
		m_skeletonHierarchy[4]->mirrorJointIdx =4;
		m_skeletonHierarchy[5]->mirrorJointIdx =5;

		m_skeletonHierarchy[6]->mirrorJointIdx =6;
		m_skeletonHierarchy[7]->mirrorJointIdx =7;
		m_skeletonHierarchy[8]->mirrorJointIdx =8;

		m_skeletonHierarchy[9]->mirrorJointIdx =3;
		m_skeletonHierarchy[10]->mirrorJointIdx =4;
		m_skeletonHierarchy[11]->mirrorJointIdx =5;

		m_skeletonHierarchy[12]->mirrorJointIdx =6;
		m_skeletonHierarchy[13]->mirrorJointIdx =7;
		m_skeletonHierarchy[14]->mirrorJointIdx =8;

		m_skeletonHierarchy[15]->mirrorJointIdx =17;
		m_skeletonHierarchy[16]->mirrorJointIdx =18;
		m_skeletonHierarchy[17]->mirrorJointIdx =15;
		m_skeletonHierarchy[18]->mirrorJointIdx =16;

		// seemed not used
		m_skeletonHierarchy[19]->mirrorJointIdx =22;
		m_skeletonHierarchy[20]->mirrorJointIdx =23;
		m_skeletonHierarchy[21]->mirrorJointIdx =24;
		m_skeletonHierarchy[22]->mirrorJointIdx =19;
		m_skeletonHierarchy[23]->mirrorJointIdx =20;
		m_skeletonHierarchy[24]->mirrorJointIdx =21;
		m_skeletonHierarchy[25]->mirrorJointIdx =25;

		//////////////////////////////////////////////////////////////////////////
		m_skeletonHierarchy[0]->levelToRootInHiearchy =0;
		m_skeletonHierarchy[1]->levelToRootInHiearchy =1;
		m_skeletonHierarchy[2]->levelToRootInHiearchy =1;

		m_skeletonHierarchy[3]->levelToRootInHiearchy =1;
		m_skeletonHierarchy[4]->levelToRootInHiearchy =3;
		m_skeletonHierarchy[5]->levelToRootInHiearchy =4;

		m_skeletonHierarchy[6]->levelToRootInHiearchy =2;
		m_skeletonHierarchy[7]->levelToRootInHiearchy =5;
		m_skeletonHierarchy[8]->levelToRootInHiearchy =6;
		
		m_skeletonHierarchy[9]->levelToRootInHiearchy =1;
		m_skeletonHierarchy[10]->levelToRootInHiearchy =3;
		m_skeletonHierarchy[11]->levelToRootInHiearchy =4;

		m_skeletonHierarchy[12]->levelToRootInHiearchy =2;
		m_skeletonHierarchy[13]->levelToRootInHiearchy =5;
		m_skeletonHierarchy[14]->levelToRootInHiearchy =6;

		m_skeletonHierarchy[15]->levelToRootInHiearchy =2;
		m_skeletonHierarchy[16]->levelToRootInHiearchy =3;
		m_skeletonHierarchy[17]->levelToRootInHiearchy =2;
		m_skeletonHierarchy[18]->levelToRootInHiearchy =3;

		m_skeletonHierarchy[19]->levelToRootInHiearchy =7;
		m_skeletonHierarchy[20]->levelToRootInHiearchy =7;
		m_skeletonHierarchy[21]->levelToRootInHiearchy =7;
		m_skeletonHierarchy[22]->levelToRootInHiearchy =7;
		m_skeletonHierarchy[23]->levelToRootInHiearchy =7;
		m_skeletonHierarchy[24]->levelToRootInHiearchy =7;

		m_skeletonHierarchy[25]->levelToRootInHiearchy =1;
	}

	if(jointNum == 19)
		m_skeletonHierarchy_nextHalfStart = 9;		//ignore coco face parts
	else if(jointNum == 26)
		m_skeletonHierarchy_nextHalfStart = 9;		//ignore coco face parts

	if (jointNum == 19)
	{
		//Generate DevaToPoseMachineIndex Map
		m_map_devaToPoseMachineIdx.clear();
		m_map_devaToPoseMachineIdx.resize(31, PM_JOINT_Unknown);
		for(int i=0;i<31;++i)
		{
			if(i==JOINT_headTop)
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_HeadTop;
			else if(i==JOINT_neck)  
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_Neck;
			else if(i==JOINT_lShoulder) 
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_lShouder;
			else if(i==JOINT_lElbow)
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_lElbow;
			else if(i==JOINT_lHand)
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_lWrist;
			else if(i==JOINT_lHip) 
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_lHip;
			else if(i==JOINT_lKnee) 
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_lKnee;
			else if(i==JOINT_lFoot)  
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_lAnkle;
			else if(i==JOINT_rShoulder) 
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_rShoulder;
			else if(i==JOINT_rElbow)
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_rElbow;
			else if(i==JOINT_rHand)
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_rWrist;
			else if(i==JOINT_rHip) 
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_rHip;
			else if(i==JOINT_rKnee)  
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_rKnee;
			else if(i==JOINT_rFoot)  
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_rAnkle;
			
			//Coco additional part
			else if(i==JOINT_lEye)  
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_lEye;
			else if(i==JOINT_lEar)  
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_lEar;
			else if(i==JOINT_rEye)  
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_rEye;
			else if(i==JOINT_rEar)  
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_rEar;

			else 
				m_map_devaToPoseMachineIdx[i] = PM_JOINT_Unknown;
		}
		m_map_SMCToPoseMachineIdx.clear();
		for(int i=0;i<m_skeletonHierarchy.size();++i)
		{
			m_map_SMCToPoseMachineIdx.push_back(m_map_devaToPoseMachineIdx[m_skeletonHierarchy[i]->jointName]);		//targetJointIdxVect[ourIdx]==DetectionFile's index. 
		}
		m_map_PoseMachineToSMCIdx.clear();
		m_map_PoseMachineToSMCIdx.resize(18);		//CoCo detection has 18 joints (no body center)
		for(int i=0;i<m_map_SMCToPoseMachineIdx.size();++i)
		{
			int pmIdx = m_map_SMCToPoseMachineIdx[i];
			if(pmIdx==PM_JOINT_Unknown)
				continue;
			m_map_PoseMachineToSMCIdx[pmIdx] = (SMC_BodyJointEnum)i;
		}
	}
	else
	{
		//Generate DevaToOpenPoseIndex Map
		m_map_devaToOpenPoseIdx.clear();
		m_map_devaToOpenPoseIdx.resize(38,OP_JOINT_Unknown);
		for(int i=0;i<38;++i)
		{
			if(i==JOINT_headTop)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_Nose;
			// else if(i==Joint_bodyCenter)
			//	m_map_devaToOpenPoseIdx[i] = OP_JOINT_MidHip;
			else if(i==JOINT_neck)  
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_UpperNeck;
			else if(i==JOINT_lShoulder) 
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LShoulder;
			else if(i==JOINT_lElbow)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LElbow;
			else if(i==JOINT_lHand)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LWrist;
			else if(i==JOINT_lHip) 
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LHip;
			else if(i==JOINT_lKnee) 
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LKnee;
			else if(i==JOINT_lFoot)  
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LAnkle;
			else if(i==JOINT_rShoulder) 
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_RShoulder;
			else if(i==JOINT_rElbow)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_RElbow;
			else if(i==JOINT_rHand)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_RWrist;
			else if(i==JOINT_rHip) 
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_RHip;
			else if(i==JOINT_rKnee)  
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_RKnee;
			else if(i==JOINT_rFoot)  
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_RAnkle;
			
			//Coco additional part
			else if(i==JOINT_lEye)  
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LEye;
			else if(i==JOINT_lEar)  
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LEar;
			else if(i==JOINT_rEye)  
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_REye;
			else if(i==JOINT_rEar)  
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_REar;

			//foot
			else if (i == JOINT_lBigToe)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LBigToe;
			else if (i == JOINT_lSmallToe)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LSmallToe;
			else if (i == JOINT_lHeel)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_LHeel;
			else if (i == JOINT_rBigToe)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_RBigToe;
			else if (i == JOINT_rSmallToe)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_RSmallToe;
			else if (i == JOINT_rHeel)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_RHeel;

			else if (i == JOINT_realheadtop)
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_HeadTop;
			else 
				m_map_devaToOpenPoseIdx[i] = OP_JOINT_Unknown;
		}
		m_map_SMCToOpenPoseIdx.clear();
		for(int i=0;i<m_skeletonHierarchy.size();++i)
		{
			m_map_SMCToOpenPoseIdx.push_back(m_map_devaToOpenPoseIdx[m_skeletonHierarchy[i]->jointName]);		//targetJointIdxVect[ourIdx]==DetectionFile's index. 
		}
	}
}

//Valid only for poseDetectFolder
void CBodyPoseRecon::Optimization3DPS_fromDetection_oneToOne_coco19(const char* dataMainFolder,const char* calibFolder,const int askedCamNum,bool isHD,bool bCoco19)//optimize poseReconMem.m_poseAtEachFrame
{
	int modelJointNum;
	char poseDetectFolderName[512];
	if(bCoco19)
	{
		modelJointNum= MODEL_JOINT_NUM_COCO_19;
		sprintf(poseDetectFolderName,"coco19_poseDetect_pm");
	}
	else
	{
		modelJointNum= MODEL_JOINT_NUM_15;
		sprintf(poseDetectFolderName,"poseDetect_pm");
	}

	printf("Start: Optimization3DPS_fromDetection\n");
	//double gaussianBandwidth = 20;
	double gaussianBandwidth = 10;
	//if(isHD)
	//gaussianBandwidth *=2;

	//To compute working volume, we should load full dome
	CDomeImageManager fullDome;
	fullDome.InitDomeCamOnlyByExtrinsic(calibFolder);
	fullDome.CalcWorldScaleParameters();
	fullDome.DeleteMemory();

	//Parameter setting
	CDomeImageManager domeImageManager;
	domeImageManager.SetCalibFolderPath(calibFolder);
	if(isHD==false)
		domeImageManager.InitDomeCamVgaHd(askedCamNum,false);
	else
		domeImageManager.InitDomeCamVgaHdKinect(0,CDomeImageManager::LOAD_SENSORS_HD);
	//domeImageManager.SetFrameIdx(frameIdx);		//Don't need

	vector<int> jointToDetectIdxMap;
	//vector<int> smc2posemachineIdx;			//This is a bug. 
	//smc2posemachineIdx.resize(MODEL_JOINT_NUM_COCO_19);
	for(int i=0;i<m_skeletonHierarchy.size();++i)
	{
		jointToDetectIdxMap.push_back(m_map_devaToPoseMachineIdx[m_skeletonHierarchy[i]->jointName]);		//targetJointIdxVect[ourIdx]==DetectionFile's index. 
	//	smc2posemachineIdx[jointToDetectIdxMap.back()] = i;   //This is a bug.
	}

	//For each time instance
	for(int t=0;t<m_3DPSPoseMemVector.size();++t)
	{
		int frameIdx = m_3DPSPoseMemVector[t].m_imgFramdIdx;
		vector<CBody3D>& bodySkeletonVect = m_3DPSPoseMemVector[t].m_articBodyAtEachTime;

		printf("## Processing %d\n",frameIdx);

		/// Load detected 2d pose. Outer vector is for each image, and inner vector is for each instance (person)
		vector< vector<SPose2D> > detectedPoseVect;		
		char poseDetectFolder[128];
		if(isHD==false)
			sprintf(poseDetectFolder,"%s/%s/vga_25",dataMainFolder,poseDetectFolderName);
		else
			sprintf(poseDetectFolder,"%s/%s/hd_30",dataMainFolder,poseDetectFolderName);
		printf("Loading posedetection result.....\n");
		vector<int> validCamIdxVect;
		bool isLoadDetect =  Module_BodyPose::LoadPoseDetectMultipleCamResult_PoseMachine(poseDetectFolder,frameIdx,domeImageManager.m_domeViews,detectedPoseVect,validCamIdxVect,isHD);
		if(isLoadDetect==false)
		{
			printf("## Error: Cannot load pose detection results. \n");
			continue;
		}
		else
			printf("## PoseDetect result has been successfully loaded \n");
		if(detectedPoseVect.size() != domeImageManager.m_domeViews.size())
		{
			printf("## ERROR: Optimization3DPS_fromDetection:: detectedPoseVect.size() != domeImageManager.m_domeViews.size()\n");
			return;
		}

		//Find 1-1 correspondence between 3D point and 2D point
		vector< vector< vector< pair<int, float> > > > detectedPoseVect_corrsHuman;		//finding corresponding humna index for each detected 2d joint/
		// each image/ each 2d skeleton / each joint(joint idx is our SMC index)
		//<int,float> ==>(humanIdx, detectionScore)
		detectedPoseVect_corrsHuman.resize(detectedPoseVect.size());
		for(int i=0;i<detectedPoseVect_corrsHuman.size();++i)
		{
			detectedPoseVect_corrsHuman[i].resize(detectedPoseVect[i].size());		//initialize as -1
			for(int h2d=0;h2d<detectedPoseVect_corrsHuman[i].size();++h2d)
			{
				detectedPoseVect_corrsHuman[i][h2d].resize(modelJointNum, make_pair(-1,-1.0f));
			}
		}
		//For each human finding correspondence candidate and claim that they are mine if its value is larter 
		for(int h=0;h<bodySkeletonVect.size();++h)
		{
			for(int j=0; j<bodySkeletonVect[h].m_jointPosVect.size();++j)
			{
				if(j==SMC_BodyJoint_bodyCenter)
					continue;
				if(bodySkeletonVect[h].m_jointConfidenceVect.size()>j && bodySkeletonVect[h].m_jointConfidenceVect[j]<0)
					continue;;
				//printf("Current confidence: %f\n",bodySkeletonVect[h].m_jointConfidenceVect[j]);

				Point3d jointPt3d = bodySkeletonVect[h].m_jointPosVect[j];
				int corresDetectIdx = jointToDetectIdxMap[j];			//find corresponding poseMachine joint idx
				if(corresDetectIdx==PM_JOINT_Unknown)
					continue;

				for(int c=0;c<domeImageManager.m_domeViews.size();++c)
				{
					Point2d jointPt2d = domeImageManager.m_domeViews[c]->ProjectionPt(jointPt3d);
					pair<int,int> imSize = domeImageManager.GetImageSize(c);
					if(IsOutofBoundary(imSize.first,imSize.second,jointPt2d.x,jointPt2d.y))
						continue;
					double bestDist = 1e5;
					double best3DNodeWeight;
					int best2DPosIdx=-1;
					for(int di=0;di<detectedPoseVect[c].size();++di)		//for each human detectedPoseVect[c][di]
					{
						double dist = Distance(jointPt2d, detectedPoseVect[c][di].bodyJoints[corresDetectIdx]);
						if(dist<gaussianBandwidth && dist <bestDist)		//Threshold should consider gaussian variance. Also we can consider all joint to determine correspondence (if two people are close each other)
						{
							bestDist = dist;
							best3DNodeWeight = bodySkeletonVect[h].m_jointConfidenceVect[j];
							best2DPosIdx = di;
						}
					}
					if(bestDist>1e4 || best2DPosIdx<0)
						continue;

					//Claim that this is mine
					if(detectedPoseVect_corrsHuman[c][best2DPosIdx][j].first<0 || detectedPoseVect_corrsHuman[c][best2DPosIdx][j].second<best3DNodeWeight)
						detectedPoseVect_corrsHuman[c][best2DPosIdx][j] = make_pair(h,best3DNodeWeight);
				}
			}
		}

		//Count the number of ownership
		vector<CBody3D> bodySkeletonVect_refined;
		bodySkeletonVect_refined.reserve(bodySkeletonVect.size());
		vector<int> validHumanVector; 
		for(int h=0;h<bodySkeletonVect.size();++h)
		{
			vector<int> ownershipNum(modelJointNum,0);
			for(int j=0; j<bodySkeletonVect[h].m_jointPosVect.size();++j)
			{
				if(j==SMC_BodyJoint_bodyCenter)
					continue;
				//int posemachineJointIdx = smc2posemachineIdx[j];		//This is a bug. smc2posemachineIdx has garbages...
				int posemachineJointIdx = jointToDetectIdxMap[j];
				int cnt=0;
				float aveDetectionScore=0;
				for(int c=0;c<detectedPoseVect_corrsHuman.size();++c)
				{
					for(int h2d=0;h2d<detectedPoseVect_corrsHuman[c].size();++h2d)
					{
						if(detectedPoseVect_corrsHuman[c][h2d][j].first==h)
						{
							cnt++;
							aveDetectionScore += detectedPoseVect[c][h2d].bodyJointScores[posemachineJointIdx];
						}
					}
				}
				if(cnt>0)
					aveDetectionScore /= cnt;
				printf("human %d: joint %d: cnt: %d, avg: %f\n",h,j,cnt,aveDetectionScore);
				ownershipNum[j] = cnt;
			}
			if(ownershipNum[SMC_BodyJoint_headTop] <2 && ownershipNum[SMC_BodyJoint_neck] <2 )//|| ownershipNum[SMC_BodyJoint_neck]<2 )//|| ownershipNum[SMC_BodyJoint_lShoulder]<2 || ownershipNum[SMC_BodyJoint_rShoulder]<2)
			{
				printf("Rejected\n");
				continue;
			}
			
			printf("		Accepted\n");
			bodySkeletonVect_refined.push_back(bodySkeletonVect[h]);
		}
		bodySkeletonVect = bodySkeletonVect_refined;		//save the filtered one
		printf("Filtered human num: %d\n",bodySkeletonVect_refined.size());

		//Optimization based on the corresponding 2D detection
		#pragma omp parallel for
		for(int h=0;h<bodySkeletonVect.size();++h)
		{
			for(int j=0; j<bodySkeletonVect[h].m_jointPosVect.size();++j)
			{
				if(j==SMC_BodyJoint_bodyCenter)
					continue;
				if(bodySkeletonVect[h].m_jointConfidenceVect.size()>j && bodySkeletonVect[h].m_jointConfidenceVect[j]<0)
					continue;;
				//printf("Current confidence: %f\n",bodySkeletonVect[h].m_jointConfidenceVect[j]);

				Point3d jointPt3d = bodySkeletonVect[h].m_jointPosVect[j];
				//int corresDetectIdx = jointToDetectIdxMap[j];
				int corresDetectIdx = jointToDetectIdxMap[j];
				if(corresDetectIdx==PM_JOINT_Unknown)
					continue;

				vector<Mat*> projecMatVect;
				vector<Point2d> imagePtVect;
				vector<double> weightVect;

				for(int c=0;c<domeImageManager.m_domeViews.size();++c)
				{
					Point2d jointPt2d = domeImageManager.m_domeViews[c]->ProjectionPt(jointPt3d);
					pair<int,int> imSize = domeImageManager.GetImageSize(c);
					if(IsOutofBoundary(imSize.first,imSize.second,jointPt2d.x,jointPt2d.y))
						continue;
					double bestDist = 1e5;
					Point2d bestPt2dCorres;
					double bestWeight;
					for(int di=0;di<detectedPoseVect[c].size();++di)		//for each human detectedPoseVect[c][di]
					{
						double dist = Distance(jointPt2d, detectedPoseVect[c][di].bodyJoints[corresDetectIdx]);
						if(dist<gaussianBandwidth && dist <bestDist)		//Threshold should consider gaussian variance. Also we can consider all joint to determine correspondence (if two people are close each other)
						{
							bestDist = dist;
							bestPt2dCorres = detectedPoseVect[c][di].bodyJoints[corresDetectIdx];
							bestWeight = detectedPoseVect[c][di].bodyJointScores[corresDetectIdx];
						}
					}
					if(bestDist>1e4)
						continue;
					projecMatVect.push_back(&domeImageManager.m_domeViews[c]->m_P);
					weightVect.push_back(bestWeight);
					imagePtVect.push_back(bestPt2dCorres);
				}
				if(projecMatVect.size()<3 )
					continue;
				Mat X;
				Point3dToMat4by1(jointPt3d,X);
				//double beforeError = CalcReprojectionError_weighted(projecMatVect,imagePtVect,weightVect,X);
				double beforeError = CalcReprojectionError(projecMatVect,imagePtVect,X);
				int iter = 5;
				while((iter--) > 0)
				{
					//vector<unsigned int> inliers;
					double changesqr = TriangulationOptimization(projecMatVect,imagePtVect,X);
					//double changesqr = triangulateWithRANSAC(projecMatVect,imagePtVect,imagePtVect.size()*10,X,3,inliers);
					//printf("iter %d: humanIdx %d, jointIdx %d, inlierNum %d\n",5-iter,h,j,inliers.size());
					//double changesqr = TriangulationOptimizationWithWeight(projecMatVect,imagePtVect,X);
					//double changesqr = TriangulationOptimizationWithWeight(projecMatVect,imagePtVect,weightVect,X);
					//double afterError = CalcReprojectionError_weighted(projecMatVect,imagePtVect,weightVect,X);
					double afterError = CalcReprojectionError(projecMatVect,imagePtVect,X);
					double changeDiff = abs(afterError - beforeError);
					//printf("changeDiff: %f\n",changeDiff);
					if(changeDiff<0.05)
					{
						//printf("iterNum: %d: finalChangeDiff: %f, finalError: %f\n",10-iter,changeDiff,afterError);
						break;
					}
					beforeError = afterError;
				} 
				bodySkeletonVect[h].m_jointPosVect[j] = MatToPoint3d(X);
				//printf("ReprojectionErrorChange: %f\n",change);
			}
			if(bodySkeletonVect[h].m_jointConfidenceVect[SMC_BodyJoint_lHip] >=0 && bodySkeletonVect[h].m_jointConfidenceVect[SMC_BodyJoint_rHip] >=0)
				bodySkeletonVect[h].m_jointPosVect[SMC_BodyJoint_bodyCenter] = 0.5 * ( bodySkeletonVect[h].m_jointPosVect[SMC_BodyJoint_lHip] + bodySkeletonVect[h].m_jointPosVect[SMC_BodyJoint_rHip] );
		}
	}
}

//Valid only for poseDetectFolder
void CBodyPoseRecon::Optimization3DPS_fromDetection_oneToOne_op25(const char* dataMainFolder,const char* calibFolder,const int askedCamNum,bool isHD)//optimize poseReconMem.m_poseAtEachFrame
{
	int modelJointNum;
	char poseDetectFolderName[512];

	modelJointNum= MODEL_JOINT_NUM_OP_25;
	sprintf(poseDetectFolderName,"op25_poseDetect_pm");

	printf("Start: Optimization3DPS_fromDetection\n");
	//double gaussianBandwidth = 20;
	double gaussianBandwidth = 10;
	//if(isHD)
	//gaussianBandwidth *=2;

	//To compute working volume, we should load full dome
	CDomeImageManager fullDome;
	fullDome.InitDomeCamOnlyByExtrinsic(calibFolder);
	fullDome.CalcWorldScaleParameters();
	fullDome.DeleteMemory();

	//Parameter setting
	CDomeImageManager domeImageManager;
	domeImageManager.SetCalibFolderPath(calibFolder);
	if(isHD==false)
		domeImageManager.InitDomeCamVgaHd(askedCamNum,false);
	else
		domeImageManager.InitDomeCamVgaHdKinect(0,CDomeImageManager::LOAD_SENSORS_HD);
	//domeImageManager.SetFrameIdx(frameIdx);		//Don't need

	vector<int> jointToDetectIdxMap;
	//vector<int> smc2posemachineIdx;			//This is a bug. 
	//smc2posemachineIdx.resize(MODEL_JOINT_NUM_COCO_19);
	for(int i=0;i<m_skeletonHierarchy.size();++i)
	{
		jointToDetectIdxMap.push_back(m_map_devaToOpenPoseIdx[m_skeletonHierarchy[i]->jointName]);		//targetJointIdxVect[ourIdx]==DetectionFile's index. 
	//	smc2posemachineIdx[jointToDetectIdxMap.back()] = i;   //This is a bug.
	}

	//For each time instance
	for(int t=0;t<m_3DPSPoseMemVector.size();++t)
	{
		int frameIdx = m_3DPSPoseMemVector[t].m_imgFramdIdx;
		vector<CBody3D>& bodySkeletonVect = m_3DPSPoseMemVector[t].m_articBodyAtEachTime;

		printf("## Processing %d\n",frameIdx);

		/// Load detected 2d pose. Outer vector is for each image, and inner vector is for each instance (person)
		vector< vector<SPose2D> > detectedPoseVect;		
		char poseDetectFolder[128];
		if(isHD==false)
			sprintf(poseDetectFolder,"%s/%s/vga_25",dataMainFolder,poseDetectFolderName);
		else
			sprintf(poseDetectFolder,"%s/%s/hd_30",dataMainFolder,poseDetectFolderName);
		printf("Loading posedetection result.....\n");
		vector<int> validCamIdxVect;
		bool isLoadDetect =  Module_BodyPose::LoadPoseDetectMultipleCamResult_PoseMachine(poseDetectFolder,frameIdx,domeImageManager.m_domeViews,detectedPoseVect,validCamIdxVect,isHD);
		if(isLoadDetect==false)
		{
			printf("## Error: Cannot load pose detection results. \n");
			continue;
		}
		else
			printf("## PoseDetect result has been successfully loaded \n");
		if(detectedPoseVect.size() != domeImageManager.m_domeViews.size())
		{
			printf("## ERROR: Optimization3DPS_fromDetection:: detectedPoseVect.size() != domeImageManager.m_domeViews.size()\n");
			return;
		}

		//Find 1-1 correspondence between 3D point and 2D point
		vector< vector< vector< pair<int, float> > > > detectedPoseVect_corrsHuman;		//finding corresponding humna index for each detected 2d joint/
		// each image/ each 2d skeleton / each joint(joint idx is our SMC index)
		//<int,float> ==>(humanIdx, detectionScore)
		detectedPoseVect_corrsHuman.resize(detectedPoseVect.size());
		for(int i=0;i<detectedPoseVect_corrsHuman.size();++i)
		{
			detectedPoseVect_corrsHuman[i].resize(detectedPoseVect[i].size());		//initialize as -1
			for(int h2d=0;h2d<detectedPoseVect_corrsHuman[i].size();++h2d)
			{
				detectedPoseVect_corrsHuman[i][h2d].resize(modelJointNum, make_pair(-1,-1.0f));
			}
		}
		//For each human finding correspondence candidate and claim that they are mine if its value is larter 
		for(int h=0;h<bodySkeletonVect.size();++h)
		{
			for(int j=0; j<bodySkeletonVect[h].m_jointPosVect.size();++j)
			{
				if(j==SMC_BodyJoint_bodyCenter)
					continue;
				if(bodySkeletonVect[h].m_jointConfidenceVect.size()>j && bodySkeletonVect[h].m_jointConfidenceVect[j]<0)
					continue;;
				//printf("Current confidence: %f\n",bodySkeletonVect[h].m_jointConfidenceVect[j]);

				Point3d jointPt3d = bodySkeletonVect[h].m_jointPosVect[j];
				int corresDetectIdx = jointToDetectIdxMap[j];			//find corresponding poseMachine joint idx
				if(corresDetectIdx==OP_JOINT_Unknown)
					continue;

				for(int c=0;c<domeImageManager.m_domeViews.size();++c)
				{
					Point2d jointPt2d = domeImageManager.m_domeViews[c]->ProjectionPt(jointPt3d);
					pair<int,int> imSize = domeImageManager.GetImageSize(c);
					if(IsOutofBoundary(imSize.first,imSize.second,jointPt2d.x,jointPt2d.y))
						continue;
					double bestDist = 1e5;
					double best3DNodeWeight;
					int best2DPosIdx=-1;
					for(int di=0;di<detectedPoseVect[c].size();++di)		//for each human detectedPoseVect[c][di]
					{
						double dist = Distance(jointPt2d, detectedPoseVect[c][di].bodyJoints[corresDetectIdx]);
						if(dist<gaussianBandwidth && dist <bestDist)		//Threshold should consider gaussian variance. Also we can consider all joint to determine correspondence (if two people are close each other)
						{
							bestDist = dist;
							best3DNodeWeight = bodySkeletonVect[h].m_jointConfidenceVect[j];
							best2DPosIdx = di;
						}
					}
					if(bestDist>1e4 || best2DPosIdx<0)
						continue;

					//Claim that this is mine
					if(detectedPoseVect_corrsHuman[c][best2DPosIdx][j].first<0 || detectedPoseVect_corrsHuman[c][best2DPosIdx][j].second<best3DNodeWeight)
						detectedPoseVect_corrsHuman[c][best2DPosIdx][j] = make_pair(h,best3DNodeWeight);
				}
			}
		}

		//Count the number of ownership
		vector<CBody3D> bodySkeletonVect_refined;
		bodySkeletonVect_refined.reserve(bodySkeletonVect.size());
		vector<int> validHumanVector; 
		for(int h=0;h<bodySkeletonVect.size();++h)
		{
			vector<int> ownershipNum(modelJointNum,0);
			for(int j=0; j<bodySkeletonVect[h].m_jointPosVect.size();++j)
			{
				if(j==SMC_BodyJoint_bodyCenter)
					continue;
				//int posemachineJointIdx = smc2posemachineIdx[j];		//This is a bug. smc2posemachineIdx has garbages...
				int posemachineJointIdx = jointToDetectIdxMap[j];
				int cnt=0;
				float aveDetectionScore=0;
				for(int c=0;c<detectedPoseVect_corrsHuman.size();++c)
				{
					for(int h2d=0;h2d<detectedPoseVect_corrsHuman[c].size();++h2d)
					{
						if(detectedPoseVect_corrsHuman[c][h2d][j].first==h)
						{
							cnt++;
							aveDetectionScore += detectedPoseVect[c][h2d].bodyJointScores[posemachineJointIdx];
						}
					}
				}
				if(cnt>0)
					aveDetectionScore /= cnt;
				printf("human %d: joint %d: cnt: %d, avg: %f\n",h,j,cnt,aveDetectionScore);
				ownershipNum[j] = cnt;
			}
			if(ownershipNum[SMC_BodyJoint_headTop] <2 && ownershipNum[SMC_BodyJoint_neck] <2 )//|| ownershipNum[SMC_BodyJoint_neck]<2 )//|| ownershipNum[SMC_BodyJoint_lShoulder]<2 || ownershipNum[SMC_BodyJoint_rShoulder]<2)
			{
				printf("Rejected\n");
				continue;
			}
			
			printf("		Accepted\n");
			bodySkeletonVect_refined.push_back(bodySkeletonVect[h]);
		}
		bodySkeletonVect = bodySkeletonVect_refined;		//save the filtered one
		printf("Filtered human num: %d\n",bodySkeletonVect_refined.size());

		//Optimization based on the corresponding 2D detection
		#pragma omp parallel for
		for(int h=0;h<bodySkeletonVect.size();++h)
		{
			for(int j=0; j<bodySkeletonVect[h].m_jointPosVect.size();++j)
			{
				if(j==SMC_BodyJoint_bodyCenter)
					continue;
				if(bodySkeletonVect[h].m_jointConfidenceVect.size()>j && bodySkeletonVect[h].m_jointConfidenceVect[j]<0)
					continue;;
				//printf("Current confidence: %f\n",bodySkeletonVect[h].m_jointConfidenceVect[j]);

				Point3d jointPt3d = bodySkeletonVect[h].m_jointPosVect[j];
				//int corresDetectIdx = jointToDetectIdxMap[j];
				int corresDetectIdx = jointToDetectIdxMap[j];
				if(corresDetectIdx==OP_JOINT_Unknown)
					continue;

				vector<Mat*> projecMatVect;
				vector<Point2d> imagePtVect;
				vector<double> weightVect;

				for(int c=0;c<domeImageManager.m_domeViews.size();++c)
				{
					Point2d jointPt2d = domeImageManager.m_domeViews[c]->ProjectionPt(jointPt3d);
					pair<int,int> imSize = domeImageManager.GetImageSize(c);
					if(IsOutofBoundary(imSize.first,imSize.second,jointPt2d.x,jointPt2d.y))
						continue;
					double bestDist = 1e5;
					Point2d bestPt2dCorres;
					double bestWeight;
					for(int di=0;di<detectedPoseVect[c].size();++di)		//for each human detectedPoseVect[c][di]
					{
						double dist = Distance(jointPt2d, detectedPoseVect[c][di].bodyJoints[corresDetectIdx]);
						if(dist<gaussianBandwidth && dist <bestDist)		//Threshold should consider gaussian variance. Also we can consider all joint to determine correspondence (if two people are close each other)
						{
							bestDist = dist;
							bestPt2dCorres = detectedPoseVect[c][di].bodyJoints[corresDetectIdx];
							bestWeight = detectedPoseVect[c][di].bodyJointScores[corresDetectIdx];
						}
					}
					if(bestDist>1e4)
						continue;
					projecMatVect.push_back(&domeImageManager.m_domeViews[c]->m_P);
					weightVect.push_back(bestWeight);
					imagePtVect.push_back(bestPt2dCorres);
				}
				if(projecMatVect.size()<3 )
					continue;
				Mat X;
				Point3dToMat4by1(jointPt3d,X);
				//double beforeError = CalcReprojectionError_weighted(projecMatVect,imagePtVect,weightVect,X);
				double beforeError = CalcReprojectionError(projecMatVect,imagePtVect,X);
				int iter = 5;
				while((iter--)>0)
				{
					double changesqr = TriangulationOptimization(projecMatVect,imagePtVect,X);
					double afterError = CalcReprojectionError(projecMatVect,imagePtVect,X);
					double changeDiff = abs(afterError - beforeError);
					if(changeDiff<0.05)
					{
						break;
					}
					beforeError = afterError;
				} 
				bodySkeletonVect[h].m_jointPosVect[j] = MatToPoint3d(X);
				//printf("ReprojectionErrorChange: %f\n",change);
			}
			if(bodySkeletonVect[h].m_jointConfidenceVect[SMC_BodyJoint_lHip] >=0 && bodySkeletonVect[h].m_jointConfidenceVect[SMC_BodyJoint_rHip] >=0)
				bodySkeletonVect[h].m_jointPosVect[SMC_BodyJoint_bodyCenter] = 0.5 * ( bodySkeletonVect[h].m_jointPosVect[SMC_BodyJoint_lHip] + bodySkeletonVect[h].m_jointPosVect[SMC_BodyJoint_rHip] );
		}
	}
}

void CBodyPoseRecon::Save3DPSBodyReconResult_Json(char* saveFolderPath,bool bNormCoord,bool bAnnotated)
{
	//Follow the first color set
	for(int t=0;t<m_3DPSPoseMemVector.size();++t)
	{
		SBody3DScene&  currentScene = m_3DPSPoseMemVector[t];
		//SaveBodyReconResult_javaScript(saveFolderPath,currentScene,m_3DPSPoseMemVector[t].m_imgFramdIdx);
		SaveBodyReconResult_json(saveFolderPath,currentScene,m_3DPSPoseMemVector[t].m_imgFramdIdx,bNormCoord,bAnnotated);
	}
}

//Should be double precision
//vector< vector<Point3d,double> >& relatedPts:: outer joint for each time (length should be same or longer than transformVector),
//												inner joint for peak candidates (position, cost)
float ComputJointTrajCostFunction(Point3f startPt,vector< vector<Mat_<double> > >& transformVector,vector< vector<pair<Point3d,double> > >& relatedPts, vector<float>& scoreForEachTime)
{
	double sigma = 30;

	Mat_<double> startPtMat;
	Point3fToMat4by1(startPt,startPtMat);
	double errorSum=0;
	for(int b=0;b<transformVector.size();++b)
	{
		if(b==0)
			scoreForEachTime.reserve(transformVector[b].size());

		for(int f=0;f<transformVector[b].size();++f)
		{
			Mat pt = transformVector[b][f]*startPtMat;
			Point3d tempPt;
			MatToPoint3d(pt,tempPt);
		
			//Compute current cost
			if(relatedPts.size()<=f)
				break;

			double tempMaxScore =-1;
			
			for(int c=0;c<relatedPts[f].size();++c)
			{
				//Compute Gaussian score
				double d = Distance(tempPt,relatedPts[f][c].first);
				double score = relatedPts[f][c].second* exp(- d*d/(2*sigma*sigma));		//range 0 ~ relatedPts[f][c].second
				if(tempMaxScore<score)
					tempMaxScore = score;
			}

			if(tempMaxScore<0)
				tempMaxScore =0;		//just uniform score

			errorSum+=tempMaxScore;

			if(b==0)
				scoreForEachTime.push_back(tempMaxScore);
		}
	}
	return errorSum ;
}

void CBodyPoseRecon::AssignHumanIdentityColorFor3DPSResult()
{
	if(m_3DPSPoseMemVector.size()==0)
	{
		printf("## ERROR: m_3DPSPoseMemVector.size()==0\n");
		return;
	}
	m_skeletalTrajProposals.clear();
	vector< vector<CBody3D*> > humanToCorresSkeleton;		//outer for each humna. need to remember corresponding skeletons, to do final renaming at the end
	int missingThresh=50;			//2 sec

	//Follow the first color set
	for(int t=0;t<m_3DPSPoseMemVector.size();++t)
	{
		//Init
		for(int c=0;c<m_3DPSPoseMemVector[t].m_articBodyAtEachTime.size();++c)
		{
			m_3DPSPoseMemVector[t].m_articBodyAtEachTime[c].m_humanIdentLabel = -1;		//default
		}
		
		//Given already generated association, find their members
		//printf("m_skeletalTrajProposals.size() == %d\n",m_skeletalTrajProposals.size());

		//For each candidate find the best human identity
		//vector< pair<int,double> > correspondingHuman(m_3DPSPoseMemVector[t].m_articBodyAtEachTime.size(),make_pair(-1,1e5));				//humanIdentity, distance
		vector< pair<int,int> > correspondingHuman(m_3DPSPoseMemVector[t].m_articBodyAtEachTime.size(),make_pair(-1,10000));				//humanIdentity, misingHistory
		for(int f=0;f<m_skeletalTrajProposals.size();++f)
		{
			if(m_skeletalTrajProposals[f].m_bStillTracked==false)
				continue;

			Point3d prevPt = m_skeletalTrajProposals[f].m_finalHumanPose.back().m_jointPosVect[SMC_BodyJoint_neck];
			int elementNum = m_skeletalTrajProposals[f].m_finalHumanPose.size();
			int missingHistory = 0;			//check this to avoid the case where a skeleton deappear but mistakenly reassociated for the location after long time frames
			for(int tt=elementNum-1;tt>=0;--tt)
			{
				if(m_skeletalTrajProposals[f].m_finalHumanPose[tt].m_bValid==true)
					break;
				missingHistory++;
			}
			vector<cv::Point3f> prevSkeleton = m_skeletalTrajProposals[f].m_finalHumanPose.back().m_jointPosVect;
			double bestDist = 1e5;
			int bestCandidateIdx =-1;

			for(int c=0;c<m_3DPSPoseMemVector[t].m_articBodyAtEachTime.size();++c)
			{
				if(m_3DPSPoseMemVector[t].m_articBodyAtEachTime[c].m_humanIdentLabel>=0)
					continue;

				double dist = Distance(prevPt,m_3DPSPoseMemVector[t].m_articBodyAtEachTime[c].m_jointPosVect[SMC_BodyJoint_neck]);
				if(bestDist>dist && dist < cm2world(30) )
				{
					bestDist = dist;
					bestCandidateIdx = c;
				}
			}
			if(bestCandidateIdx>=0 && bestDist<cm2world(20))
			{
				/*if( correspondingHuman[bestCandidateIdx].first <0 ||  correspondingHuman[bestCandidateIdx].second >bestDist )
					correspondingHuman[bestCandidateIdx] = make_pair(f,missingHistory);*/
				if( correspondingHuman[bestCandidateIdx].first <0 ||  correspondingHuman[bestCandidateIdx].second >missingHistory )
					correspondingHuman[bestCandidateIdx] = make_pair(f,missingHistory);
			}
		}

		for(int f=0;f<m_skeletalTrajProposals.size();++f)
		{
			if(m_skeletalTrajProposals[f].m_bStillTracked==false)
				continue;

			Point3d prevPt = m_skeletalTrajProposals[f].m_finalHumanPose.back().m_jointPosVect[SMC_BodyJoint_neck];
			vector<cv::Point3f> prevSkeleton = m_skeletalTrajProposals[f].m_finalHumanPose.back().m_jointPosVect;
			//double bestDist = 1e5;
			/*int bestCandidateIdx =-1;
			for(int c=0;c<m_3DPSPoseMemVector[t].m_articBodyAtEachTime.size();++c)
			{
				if(m_3DPSPoseMemVector[t].m_articBodyAtEachTime[c].m_humanIdentLabel>=0)
					continue;

				double dist = Distance(prevPt,m_3DPSPoseMemVector[t].m_articBodyAtEachTime[c].m_jointPosVect[SMC_BodyJoint_neck]);
				if(bestDist>dist && dist < cm2world(30) )
				{
					bestDist = dist;
					bestCandidateIdx = c;
				}
			}*/
			int bestCandidateIdx = -1;
			for(int c=0;c<correspondingHuman.size();++c)
			{
				if(correspondingHuman[c].first == f)
				{
					bestCandidateIdx = c;
					break;
				}
			}
			if(bestCandidateIdx>=0)// && bestDist<cm2world(20))
			{
				//printf("Found %d th (dist %f)\n",t,world2cm(bestDist));
				m_skeletalTrajProposals[f].m_finalHumanPose.push_back(m_3DPSPoseMemVector[t].m_articBodyAtEachTime[bestCandidateIdx]);	//just cloning
				m_skeletalTrajProposals[f].missingCnt =0;
				m_3DPSPoseMemVector[t].m_articBodyAtEachTime[bestCandidateIdx].m_humanIdentLabel = f;

				humanToCorresSkeleton[f].push_back(&m_3DPSPoseMemVector[t].m_articBodyAtEachTime[bestCandidateIdx]);		//remember for renaming
			}
			else
			{
				m_skeletalTrajProposals[f].missingCnt++;
				if(m_skeletalTrajProposals[f].missingCnt>=missingThresh)		//Completely failed
				{
					m_skeletalTrajProposals[f].m_bStillTracked= false;
					while(m_skeletalTrajProposals[f].m_finalHumanPose.size()>=0)
					{
						if(m_skeletalTrajProposals[f].m_finalHumanPose.back().m_bValid ==false)
						{
							m_skeletalTrajProposals[f].m_finalHumanPose.pop_back();
							humanToCorresSkeleton[f].pop_back();
						}
						else
							break;
					}
				}
				else   //Failed to find corresponding human at this moment
				{
					m_skeletalTrajProposals[f].m_finalHumanPose.resize(m_skeletalTrajProposals[f].m_finalHumanPose.size()+1);
					m_skeletalTrajProposals[f].m_finalHumanPose.back().m_bValid = false;
					m_skeletalTrajProposals[f].m_finalHumanPose.back().m_jointPosVect = prevSkeleton;

					humanToCorresSkeleton[f].push_back(NULL);
				}
			}
		}

		//For the non-selected members, generate new groups
		//Init
		for(int c=0;c<m_3DPSPoseMemVector[t].m_articBodyAtEachTime.size();++c)
		{
			if(m_3DPSPoseMemVector[t].m_articBodyAtEachTime[c].m_humanIdentLabel <0)
			{
				m_skeletalTrajProposals.resize(m_skeletalTrajProposals.size()+1);
				CBody4DSubject& newBodySubject = m_skeletalTrajProposals.back();
				newBodySubject.m_initFrameIdx = m_3DPSPoseMemVector[t].m_imgFramdIdx;
				newBodySubject.m_finalHumanPose.push_back(m_3DPSPoseMemVector[t].m_articBodyAtEachTime[c]);		//copy from body3D (points, confidence...etc)

				int idx = (m_skeletalTrajProposals.size()-1)%m_colorSet.size();//original
				/*int idx=0;		//for 2 people's case
				if(m_skeletalTrajProposals.size()>=2)
					idx =1;*/
				newBodySubject.SetBodyColor(m_colorSet[idx]);
				newBodySubject.missingCnt =0;
				newBodySubject.m_bStillTracked = true;

				m_3DPSPoseMemVector[t].m_articBodyAtEachTime[c].m_humanIdentLabel = m_skeletalTrajProposals.size()-1;
				
				humanToCorresSkeleton.resize(humanToCorresSkeleton.size()+1);
				humanToCorresSkeleton.back().push_back(&m_3DPSPoseMemVector[t].m_articBodyAtEachTime[c]);		//remember for renaming
			}
		}
	}

	//Eliminate Too short one
	int validThresh = 100;
/*	int maxLength =0;
	for(int i=0;i<m_skeletalTrajProposals.size();++i)
		maxLength = max(maxLength,(int)m_skeletalTrajProposals[i].m_finalHumanPose.size());
	if(maxLength<=100)		*/

	if(g_poseEstLoadingDataNum<=150)
		validThresh=0;

	printf("## Valid Skeletal Traj Proposal Range Threshold: %d\n",validThresh);
	vector< CBody4DSubject> inliers;
	vector < vector<CBody3D*>> inliers_corresSkel;
	for(int i=0;i<m_skeletalTrajProposals.size();++i)
	{
		//count valid element

		int validCnt=0;
		for(int t=0;t < m_skeletalTrajProposals[i].m_finalHumanPose.size();++t)
		{
			if(m_skeletalTrajProposals[i].m_finalHumanPose[t].m_bValid)
				validCnt++;
		}
		//if(m_skeletalTrajProposals[i].m_finalHumanPose.size()>validThresh)
		if(validCnt>validThresh)
		{
			//printf("Check: %d vs %d\n",m_skeletalTrajProposals[i].m_finalHumanPose.size(),humanToCorresSkeleton[i].size());
			printf("## ValidSkeleton: Id %d, validCnt: %d\n",inliers_corresSkel.size(),validCnt);
			inliers.push_back(m_skeletalTrajProposals[i]);
			inliers_corresSkel.push_back(humanToCorresSkeleton[i]);
		}
	}
	m_skeletalTrajProposals = inliers;

	//Rename human identity labels
	for(int t=0;t<m_3DPSPoseMemVector.size();++t)
	{
		for(int c=0;c<m_3DPSPoseMemVector[t].m_articBodyAtEachTime.size();++c)
			m_3DPSPoseMemVector[t].m_articBodyAtEachTime[c].m_humanIdentLabel = -1;		//default
	}
	for(int h=0;h<inliers_corresSkel.size();++h)
	{
		for(int t=0;t<inliers_corresSkel[h].size();++t)
		{
			if(inliers_corresSkel[h][t]==NULL)
				continue;
			inliers_corresSkel[h][t]->m_humanIdentLabel = h;
		}
	}

	for(int i=0;i<m_skeletalTrajProposals.size();++i)
	{
		for(int t=0;t < m_skeletalTrajProposals[i].m_finalHumanPose.size();++t)
		{
			m_skeletalTrajProposals[i].m_finalHumanPose[t].m_humanIdentLabel = i;
		}
	}

	//special color assignement
	for(int i=0;i<m_skeletalTrajProposals.size();++i)
	{
		Point3f color = m_skeletalTrajProposals[i].GetBodyColor();
		m_skeletalTrajProposals[i].SetBodyColor(color);
	}
	
}

bool CPartTrajProposal::BoneToTrajMinMaxDist_considerNormal(TrajElement3D* refTraj,double& minDist,double& maxDist, double& worstNormalInnerProd
										 ,double childNodeSideOffset_cm, double parentNodeSideOffset_cm)
{
	float partBoneLength = GetBoneLength();
	float minAlongBoneDistThresh = -cm2world(childNodeSideOffset_cm);
	float maxAlongBoneDistThresh = partBoneLength +cm2world(parentNodeSideOffset_cm);

	maxDist = 0;
	minDist = 1e5;
	int compareStartFrameIdx = max(m_startImgFrameIdx,refTraj->GetTrajStartTimeInFrameIdx());
	int compareLastFrameIdx = min(m_startImgFrameIdx+(int)m_articBodyAtEachTime.size()-1,refTraj->GetTrajLastTimeInFrameIdx());
	//if(compareLastFrameIdx-compareStartFrameIdx<10)
	if(compareLastFrameIdx-compareStartFrameIdx<5)
		return false;

	bool bOnceInTheBoundary = false;
	worstNormalInnerProd=1e5;
	for(int f=compareStartFrameIdx;f<=compareLastFrameIdx;++f)
	{
		cv::Point3d tempTrajPt;
		cv::Point3d tempTrajNormal;
		bool bSuccess = refTraj->GetTrajPosNormal_BySelectedFrameIdx(f,tempTrajPt,tempTrajNormal);
		if(bSuccess==false)
			break;

		int t = f-m_startImgFrameIdx;
		if(m_articBodyAtEachTime[t].m_bValid==false)		//ignore outliers
			continue;

		assert(m_articBodyAtEachTime[t].m_jointPosVect.size()==2);
		cv::Point3f& childJointdPos = m_articBodyAtEachTime[t].m_jointPosVect[0];
		cv::Point3f& parentJointdPos = m_articBodyAtEachTime[t].m_jointPosVect[1];
		cv::Point3f boneDirectionalVect = parentJointdPos- childJointdPos;
		Normalize(boneDirectionalVect);

		//////////////////////////////////////////////////////////////////////////
		//Compute orthogonal dist
		cv::Point3f childToPt = cv::Point3f(tempTrajPt.x,tempTrajPt.y,tempTrajPt.z) - childJointdPos;

		float distAlongBoneDirection = childToPt.dot(boneDirectionalVect);

		//added
		if(distAlongBoneDirection> minAlongBoneDistThresh && distAlongBoneDirection  < maxAlongBoneDistThresh)
			bOnceInTheBoundary = true;		//at least once

		cv::Point3f orthoVect = childToPt- boneDirectionalVect*distAlongBoneDirection;
		double orthoDist = norm(orthoVect);

		//Compare normals
		double normalCompare = orthoVect.dot(tempTrajNormal);
		if(worstNormalInnerProd>normalCompare)
			worstNormalInnerProd = normalCompare;
		if( normalCompare<-0.2)
			return false;

		maxDist = max(maxDist,orthoDist);
		minDist = min(minDist,orthoDist);
	}

	if(bOnceInTheBoundary==false)
		return false;
	return true;
}

//Input::
//Start time
//originalTrajVect: to find related peaks among all peaks
//Transformation vectors (from startPt for each joints)
//Related peak points for each frame. 
//thresh is used to filter out meangless peaks
//Return score
double CPartTrajProposal::NodeTrajectoryOptimization(int jointIdx,int startTime,vector<Point3f>& originalTrajVect,vector< vector<Mat_<double> > >& transformVector,vector<CVisualHullManager*>& detectionHullVector,double thresh,vector<Point3f>& optimizedTraj,bool NoOptimization)
{
	if(detectionHullVector.size()==0)
	{
		printf("No Detection Hull result\n");
		return -1e3;
	}

	//Select related pt only using a bandwidth
	/*double bandwidth = 10/WORLD_TO_CM_RATIO;
	if(jointIdx==SMC_BodyJoint_bodyCenter)
		bandwidth = 20/WORLD_TO_CM_RATIO;
	if(jointIdx==SMC_BodyJoint_lShoulder)
		bandwidth = 5/WORLD_TO_CM_RATIO;*/

	double bandwidth = 20/WORLD_TO_CM_RATIO;
	if(jointIdx==SMC_BodyJoint_bodyCenter)
		bandwidth = 25/WORLD_TO_CM_RATIO;
	if(jointIdx==SMC_BodyJoint_lShoulder)
		bandwidth = 15/WORLD_TO_CM_RATIO;

	/*if(jointIdx==8)
		bandwidth = 20/WORLD_TO_CM_RATIO;*/
	int detectionHullStartIdx = startTime - detectionHullVector.front()->m_imgFramdIdx;
	int trackLength =0;

	for(int i=0;i<transformVector.size();++i)
	{
		if(trackLength <transformVector[i].size())
			trackLength  = transformVector[i].size();
	}

	if(detectionHullVector.size()<detectionHullStartIdx +trackLength)
	{
		trackLength = detectionHullVector.size()- detectionHullStartIdx;
			//printf("Detection Hull is shorter than required %d vs %d\n",m_nodePropScoreVector.size(), detectionHullStartIdx +trackLength);
			//return 1e3;
	}

	vector< vector<pair<Point3d,double> > > relatedPts;
	relatedPts.resize(trackLength);
	for(int i=0;i<trackLength;++i)
	{
		int dhIdx = detectionHullStartIdx +i;
		Point3f targetJointPos = originalTrajVect[i];
		CVisualHull& jointPeaks =detectionHullVector[dhIdx]->m_visualHullRecon[jointIdx];
		/*
		for(int k=0;k<jointPeaks.m_surfaceVoxelOriginal.size();k++)
		{
			if(jointPeaks.m_surfaceVoxelOriginal[k].prob <thresh)
				continue;
			if(Distance(targetJointPos,jointPeaks.m_surfaceVoxelOriginal[k].pos) <bandwidth)
			{
				Point3d pos_d;
				Point3fTo3d( jointPeaks.m_surfaceVoxelOriginal[k].pos,pos_d);
				relatedPts[i].push_back( make_pair(pos_d,jointPeaks.m_surfaceVoxelOriginal[k].prob));
			}
		}*/

		//Select best one
		float bestScore =-1e5;
		Point3d bestPos;
		for(int k=0;k<jointPeaks.m_surfaceVoxelOriginal.size();k++)
		{
			
			if(Distance(targetJointPos,jointPeaks.m_surfaceVoxelOriginal[k].pos) <bandwidth)
			{
				if(jointPeaks.m_surfaceVoxelOriginal[k].prob > bestScore)
				{
					Point3d pos_d;
					Point3fTo3d( jointPeaks.m_surfaceVoxelOriginal[k].pos,pos_d);
					bestPos = pos_d;
					bestScore = jointPeaks.m_surfaceVoxelOriginal[k].prob;
				}
			}
		} 
		if(bestScore>0)
			relatedPts[i].push_back( make_pair(bestPos,bestScore));
	}

	Point3f startPt= originalTrajVect.front();
	m_boneGroupScoreForEachTime.clear();
	float score = ComputJointTrajCostFunction(startPt,transformVector,relatedPts,m_boneGroupScoreForEachTime);

	printf("Joint%d: current Score %f ->",jointIdx,score);
	if(NoOptimization)
	{
		printf("\n");
		return score;
	}

	//////////////////////////////////////////////////////////////////////////
	/// Joint optimization
	Point3f optimizedTrajStartPt;
	PoseReconNodeTrajOptimization_ceres(startPt,transformVector,relatedPts,optimizedTrajStartPt);
	m_boneGroupScoreForEachTime.clear();
	score =ComputJointTrajCostFunction(optimizedTrajStartPt,transformVector,relatedPts,m_boneGroupScoreForEachTime);
	printf("afterOptimization -> %f\n",score);

	//Transform
	Mat startPtMat;
	Point3fToMat4by1(optimizedTrajStartPt,startPtMat);
	optimizedTraj.push_back(optimizedTrajStartPt);
	for(int i=1;i<transformVector[0].size();++i)
	{
		Mat pt = transformVector[0][i]*startPtMat;
		optimizedTraj.push_back(MatToPoint3f(pt));
		//cout<<transformVector[0][f-jointStartFrameIdx]<<endl;
	}
	return score ;
}

void CPartTrajProposal::ComputeBoneTraj(vector<CVisualHullManager*>& detectionHullVector,float threshold)
{
	vector< vector<Mat_<double> > > transformVector;		//outer for T from each bone, inner for each frame	.. because there exist multiple transforms from different bones
	transformVector.push_back(m_forwardTransformVect);

	if(m_articBodyAtEachTime.size()==0)
		return;

	int boneEndPtNum = m_jointIdxVect.size();
	double boneGroupScore =0;
	for(int b=0;b<boneEndPtNum;++b)
	{
		vector<Point3f> trajVect;			//generated by just averaging.
		trajVect.reserve(m_articBodyAtEachTime.size());
		for(int t=0;t<m_articBodyAtEachTime.size();++t)
			trajVect.push_back(m_articBodyAtEachTime[t].m_jointPosVect[b]);

		vector<Point3f> optimizedTrajVect;			//final transform is done by transformVector[0] with optimized start pt
		double score = NodeTrajectoryOptimization(m_jointIdxVect[b],m_startImgFrameIdx,trajVect,transformVector,detectionHullVector,threshold,optimizedTrajVect,true);		//transformVector start from jointStartFrameIdx
		boneGroupScore +=score;
	}
	m_boneGroupScore = boneGroupScore/boneEndPtNum;
}

bool CPartTrajProposal::GetBoneDirection(int frameIdx,Point3f& directVect)
{
	int idx = frameIdx - m_startImgFrameIdx;
	if(idx<0 || idx >= m_articBodyAtEachTime.size())
		return false;

	Point3f direct = m_articBodyAtEachTime[idx].m_jointPosVect[0] - m_articBodyAtEachTime[idx].m_jointPosVect[1];
	Normalize(direct);
	directVect = direct;
	
	return true;
}

void CPartTrajProposal::OptimizeBoneTraj(vector<CVisualHullManager*>& detectionHullVector,float threshold)
{
	vector< vector<Mat_<double> > > transformVector;		//outer for T from each bone, inner for each frame	.. because there exist multiple transforms from different bones
	transformVector.push_back(m_forwardTransformVect);

	if(m_articBodyAtEachTime.size()==0)
		return;

	int boneEndPtNum = m_jointIdxVect.size();
	double boneGroupScore =0;
	for(int b=0;b<boneEndPtNum;++b)
	{
		vector<Point3f> trajVect;			//generated by just averaging.
		trajVect.reserve(m_articBodyAtEachTime.size());
		for(int t=0;t<m_articBodyAtEachTime.size();++t)
			trajVect.push_back(m_articBodyAtEachTime[t].m_jointPosVect[b]);

		vector<Point3f> optimizedTrajVect;			//final transform is done by transformVector[0] with optimized start pt
		double score = NodeTrajectoryOptimization(m_jointIdxVect[b],m_startImgFrameIdx,trajVect,transformVector,detectionHullVector,threshold,optimizedTrajVect,false);		//transformVector start from jointStartFrameIdx
		boneGroupScore +=score;

		for(int t=0;t<m_articBodyAtEachTime.size();++t)
			m_articBodyAtEachTime[t].m_jointPosVect[b] = optimizedTrajVect[t];
	}

	m_boneGroupScore = boneGroupScore/boneEndPtNum;
}

bool CBody4DSubject::GetBody3DFromFrameIdx(int frameIdx, CBody3D** pReturnBody)
{
	int idx = frameIdx-m_initFrameIdx;
	if(idx<0 || idx>=m_finalHumanPose.size())
		return false;
	else
	{
		*pReturnBody = &m_finalHumanPose[idx];
		return true;
	}
	return false;
}

}