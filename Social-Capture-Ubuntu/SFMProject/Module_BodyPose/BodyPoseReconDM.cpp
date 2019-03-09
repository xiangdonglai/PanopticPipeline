#include "BodyPoseReconDM.h"
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <cstdio>
#include "Utility.h"

static CClock g_clock;
#define VOXEL_SIZE_CM 4
#define POSEMACHINE_NODESCORE_THRESHOLD_AVG 0.05

using namespace std;
using namespace cv;
//domeViews is required to load only the corresponding view's results
bool Module_BodyPose::Load_Undist_PoseDetectMultipleCamResult_MultiPoseMachine_19jointFull(
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


namespace Module_BodyPose
{

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
	#pragma omp parallel for num_threads(10)
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

		// if(isHD ==false)
		// 	bCudaSuccess = DetectionCostVolumeGeneration_float_GPU_average_vga(voxelCoordVect, voxelNum, costMapVect,projectMatVect,occupancyOutput);
		// else
		// 	bCudaSuccess = DetectionCostVolumeGeneration_float_GPU_average_hd(voxelCoordVect, voxelNum, costMapVect,projectMatVect,occupancyOutput);
		// if(bCudaSuccess==false)
		// 	bCudaSuccessAll = false;

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

		tempVisualHull.deleteMemory();	//Volume is no longer needed
		delete[] voxelCoordVect;
		delete[] occupancyOutput;
	}//end of for(int j=0;j<targetJointIdx.size();++j)
	g_clock.toc("Node Proposal Generation");
	if(bCudaSuccessAll==false)
		return;

	// //////////////////////////////////////////////////////////////////////////
	// /// Part Proposal generation
	// g_clock.tic();
	// gaussianBandwidth = gaussianBandwidth*0.5;
	// //////////////////////////////////////////////////////////////////////////
	// /// Set 2D group Map: Left (100) - right (200) 
	// vector< vector< Mat_<Vec3b> > > costMap;   //costMap[camIdx][jointIdx]
	// costMap.resize(detectedPoseVect.size());
	// for(int c=0;c<costMap.size();++c)
	// {
	// 	costMap[c].resize(modelJointNum);
	// 	Mat& inputImage = domeImageManager.m_domeViews[c]->m_inputImage;
	// 	for(int j=0;j<costMap[c].size();++j)
	// 	{
	// 		costMap[c][j] = Mat_<Vec3b>::zeros(inputImage.rows,inputImage.cols);
	// 	}
	// }

// 	int LEFT = 100;
// 	int RIGHT = 200;
// 	int MAGIC_NUM = 100;		//I need this to distinguish jointRegions from background
// 	#pragma omp parallel for //num_threads(10)
// 	for(int camIdx=0;camIdx<detectedPoseVect.size();++camIdx)
// 	{
// 		//Mat& inputImage = domeImageManager.m_domeViews[camIdx]->m_inputImage;
// 		//costMap[camIdx] = Mat_<Vec3b>::zeros(inputImage.rows,inputImage.cols);

// 		for(int h=0;h<detectedPoseVect[camIdx].size();++h)	//for h is the index for each person in camera of the camIdx 
// 		{
// 			for(int jj=0;jj<targetJointIdxVect.size();++jj)		//jj is my joint index
// 			{
// 				BodyJointEnum targetJointIdx = targetJointIdxVect[jj];
// 				PoseMachine2DJointEnum targetJointIdx_PoseMachine = m_map_devaToPoseMachineIdx[targetJointIdx];
					
// 				if(targetJointIdx>=JOINT_lShoulder && targetJointIdx<=JOINT_lFoot)
// 				{
// 					Scalar tempColor(jj + MAGIC_NUM, h,LEFT);				//(jointIDx, humanIdx, LEFT RIGHT flag)
// 					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 				}
// 				else if(targetJointIdx>=JOINT_rShoulder && targetJointIdx<=JOINT_rFoot)
// 				{
// 					//int mirroredIdx = jj - (jointNum-3)/2;			//I forgot this.. what a "huge" bug....
// 					//Scalar tempColor(mirroredIdx + MAGIC_NUM, h,RIGHT);
// 					Scalar tempColor(jj + MAGIC_NUM, h,RIGHT);
// 					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,200+mirroredIdx,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 				}
// 				else if(targetJointIdx==JOINT_headTop || targetJointIdx==JOINT_neck)
// 				{
// 					Scalar tempColor(jj + MAGIC_NUM, h,LEFT);
// 					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 				}
// 				//Coco additional
// 				else if(targetJointIdx==JOINT_lEye || targetJointIdx==JOINT_lEar)
// 				{
// 					Scalar tempColor(jj + MAGIC_NUM, h,LEFT);
// 					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 				}
// 				else if(targetJointIdx==JOINT_rEye || targetJointIdx==JOINT_rEar)
// 				{
// 					Scalar tempColor(jj + MAGIC_NUM, h,RIGHT);
// 					circle( costMap[camIdx][jj],detectedPoseVect[camIdx][h].bodyJoints[targetJointIdx_PoseMachine],gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 					//circle( costMap[camIdx],detectedPoseVect[camIdx][j].bodyJoints[targetJointIdxByDeva],gaussianBandwidth,100+jj,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 				}
// 				//if(domeImageManager.m_domeViews[camIdx]->m_actualPanelIdx==0 && domeImageManager.m_domeViews[camIdx]->m_actualCamIdx==21)
// 			/*	{
// 					imshow("test",costMap[camIdx][jj]);
// 					cvWaitKey();
// 				}*/
// 			}
// 			Point2f bodyCenter = detectedPoseVect[camIdx][h].bodyJoints[PM_JOINT_lHip] + detectedPoseVect[camIdx][h].bodyJoints[PM_JOINT_rHip];
// 			bodyCenter = bodyCenter*0.5;
// 			int bodyCenterJointidxInVector = 2;			//index in targetJointIdxVect
// 			Scalar tempColor(bodyCenterJointidxInVector+ MAGIC_NUM, h,LEFT);
// 			circle( costMap[camIdx][SMC_BodyJoint_bodyCenter],bodyCenter,gaussianBandwidth,tempColor,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 			//circle( costMap[camIdx],bodyCenter,gaussianBandwidth,100+bodyCenterJointidxInVector,-1);//detectedFaceVect[camIdx][j].detectionScore);
// 		}
// 		/*//Debug
// 		if(true)//domeImageManager.m_domeViews[camIdx]->m_actualPanelIdx==0 && domeImageManager.m_domeViews[camIdx]->m_actualCamIdx==1)
// 		{
// 			imshow("test",costMap[camIdx]);
// 			cvWaitKey();
// 		}*/
// 	}

// 	//////////////////////////////////////////////////////////////////////////
// 	/// Part Proposal generation
// 	//0 1 2 // 3 4 5   6 7 8 // 9 10 11   12 13 14 
// 	int iterEnd = m_skeletonHierarchy.size();
// 	int iterStart = 1;		//start from head top. This value was 3 before, where I didn't consider head and body
// 	//if(bDoJointMirroring)
// 		//iterEnd = 3+(m_skeletonHierarchy.size()-3)/2;
// 	float increasingpStepUnit = 1.0/detectedPoseVect.size(); 
// 	#pragma omp parallel for
// 	for(int jointIdx=iterStart;jointIdx<iterEnd;++jointIdx)	
// 	{
// 		if(m_skeletonHierarchy[jointIdx]->parents.size()==0)
// 			continue;
// 		//if(m_skeletonHierarchy[jointIdx]->jointName==JOINT_headTop || m_skeletonHierarchy[jointIdx]->jointName==Joint_bodyCenter)
// //			continue;
// 		STreeElement* childTreeNode = m_skeletonHierarchy[jointIdx];
// //		assert(m_skeletonHierarchy[jointIdx]->parents->size()==1);
// 		STreeElement* parentTreeNode = m_skeletonHierarchy[jointIdx]->parents.front();		//WARNING!!! I am assuming single child mode
// 		int parentJointidx = parentTreeNode->idxInTree;
// 		vector< SurfVoxelUnit >& surfVoxChild = detectionHullManager->m_visualHullRecon[jointIdx].m_surfaceVoxelOriginal;
// 		vector< SurfVoxelUnit >& surfVoxParent  = detectionHullManager->m_visualHullRecon[parentTreeNode->idxInTree].m_surfaceVoxelOriginal;

// 		//printf("Edge generation: childJoint %d, parentJoint %d\n",jointIdx,parentJointidx);

// 		//initialize edge strength vectors
// 		childTreeNode->childParent_partScore.clear();			//should throw away previous data
// 		childTreeNode->childParent_counter.clear();			//should throw away previous data

// 		childTreeNode->childParent_partScore.resize(surfVoxChild.size());
// 		childTreeNode->childParent_counter.resize(surfVoxChild.size());
// 		for(int c=0;c<surfVoxChild.size();++c)
// 		{
// 			childTreeNode->childParent_partScore[c].resize(surfVoxParent.size(),0);
// 			childTreeNode->childParent_counter[c].resize(surfVoxParent.size(),0);
// 		}
// 		//For every possible pair between current joint's node and parent's node
// 		for(int c=0;c<surfVoxChild.size();++c)
// 		{
// 			for(int camIdx=0;camIdx<detectedPoseVect.size();++camIdx)
// 			{
// 				Point2d childImgPt = Project3DPt(surfVoxChild[c].pos, domeImageManager.m_domeViews[camIdx]->m_P);
// 				if(IsOutofBoundary(costMap[camIdx][jointIdx],childImgPt.x,childImgPt.y))
// 					continue;

// 				Vec3b childLabel = costMap[camIdx][jointIdx].at<Vec3b>(childImgPt.y,childImgPt.x);
// 				if(childLabel[0]-MAGIC_NUM != jointIdx)		//not for this joint
// 					continue;
// 				int childWhichSideLabel =childLabel[2];		//either 100 or 200
				
// 				for(int p=0;p<surfVoxParent.size();++p)
// 				{
// 					float edgeDist = Distance(surfVoxChild[c].pos,surfVoxParent[p].pos);
// 					float expected = childTreeNode->parent_jointLength.front();
// 					if(abs(expected-edgeDist)>expected)		//reject if the length is too long or short
// 						continue;
// 					double weightByNodes = (surfVoxChild[c].prob + surfVoxParent[p].prob)/2.0;
// 					Point2d parentImgPt = Project3DPt(surfVoxParent[p].pos, domeImageManager.m_domeViews[camIdx]->m_P);
// 					if(IsOutofBoundary(costMap[camIdx][parentJointidx],parentImgPt.x,parentImgPt.y))
// 						continue;

// 					Vec3b parentLabel = costMap[camIdx][parentJointidx].at<Vec3b>(parentImgPt.y,parentImgPt.x);
// 					//if(parentLabel<100)
// 					if(parentLabel[0]-MAGIC_NUM != parentJointidx)		//not for this joint
// 						continue;
// 					int parentWhichSideLabel = parentLabel[2];		//either 100 or 200

// 					childTreeNode->childParent_counter[c][p] +=1;		//for taking average.. consider the view whenever the part is inside of image. I put this here so that the "non-detected" view provided kind of "negative" voting

// 					//if the parent is either neck or bodyCenter, side label doesn't matter. Only human label is important
// 					//For ms coco, if parent if head top (actually nose), side doesn't matter (left/right eye) 
// 					if(jointIdx==SMC_BodyJoint_lShoulder || jointIdx==SMC_BodyJoint_lHip || jointIdx==SMC_BodyJoint_rShoulder || jointIdx==SMC_BodyJoint_rHip 
// 									|| jointIdx==SMC_BodyJoint_lEye || jointIdx==SMC_BodyJoint_rEye)			//bug fixed Neck and Bodycenter has label 100 which was never connected with 200 (right Shoulder and Hip)
// 					{
// 						if(childLabel[1] == parentLabel[1])		//human label check
// 						{
// 							//childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes*increasingpStepUnit;		//max value become 1
// 							childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes;		//sum
// 						}
// 					}
// 					else if(childWhichSideLabel == parentWhichSideLabel && childLabel[1] == parentLabel[1]) // childLabel[1]  and parentLabel[1] means human index
// 					{
// 						//childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes*increasingpStepUnit;		//max value become 1
// 						childTreeNode->childParent_partScore[c][p] = childTreeNode->childParent_partScore[c][p] + weightByNodes;	//sum
// 					//	childTreeNode->childParent_counter[c][p] +=1;	//for taking average
// 					}
// 				}
// 			}

// 			//Taking average
// 			for(int p=0;p<surfVoxParent.size();++p)
// 			{
// 				if(childTreeNode->childParent_counter[c][p]>0)
// 					childTreeNode->childParent_partScore[c][p] /=childTreeNode->childParent_counter[c][p];
// 			}
// 		}
// 	}

// 	/*if(bDoJointMirroring)
// 	{
// 		for(int jointIdx=iterEnd;jointIdx<m_skeletonHierarchy.size();++jointIdx)
// 		{
// 			int mirrorJoint = jointIdx - (jointNum-3)/2;
// 			printf("Edge generation: joint %d - mirrored by %d\n",jointIdx,mirrorJoint);

// 			STreeElement* childTreeNode = m_skeletonHierarchy[jointIdx];
// 			STreeElement* mirrorNode = m_skeletonHierarchy[mirrorJoint];

// 			childTreeNode->childParent_partScore.resize(mirrorNode->childParent_partScore.size());
// 			for(int c=0;c<childTreeNode->childParent_partScore.size();++c)
// 				childTreeNode->childParent_partScore[c].reserve(mirrorNode->childParent_partScore[c].size());

// 			for(int c=0;c<mirrorNode->childParent_partScore.size();++c)
// 			{
// 				childTreeNode->childParent_partScore[c] = mirrorNode->childParent_partScore[c];
// 			}
// 		}
// 	}*/
// 	g_clock.toc("Part Proposal Generation\n");

// 	//Save the edge result to m_edgeCostVector for visualization
// 	//Todo: only use m_edgeCostVector to avoid copy. (Not the one in the  m_skeleton
// 	SEdgeCostVector* pEdgeCostUnit = new SEdgeCostVector();
// 	m_edgeCostVector.push_back(pEdgeCostUnit);
// 	pEdgeCostUnit->parent_connectivity_vis.resize(m_skeletonHierarchy.size());
// 	for(int jointIdx=0;jointIdx<m_skeletonHierarchy.size();++jointIdx)	
// 	{
// 		pEdgeCostUnit->parent_connectivity_vis[jointIdx] = m_skeletonHierarchy[jointIdx]->childParent_partScore;
// 	}

// 	//if(g_dataFrameNum==1)
// 		//ReVisualizeNodeProposals(0.1);

// 	printf("done\n");

// 	g_clock.tic();
// 	SaveNodePartProposals(dataMainFolder,g_askedVGACamNum,domeImageManager.GetFrameIdx(),detectionHullManager,m_skeletonHierarchy,validCamIdxVect.size(),isHD);
// 	g_clock.toc("SaveNodePartProposals");
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
}

}