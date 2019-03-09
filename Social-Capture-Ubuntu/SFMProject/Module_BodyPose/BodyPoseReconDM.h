#pragma once
#include "DomeImageManager.h"
#include "BodyPoseReconDT.h"
#include "VisualHullModule.h"

namespace Module_BodyPose
{
bool Load_Undist_PoseDetectMultipleCamResult_MultiPoseMachine_19jointFull(const char* poseDetectFolder,const char* poseDetectSaveFolder,const int currentFrame, CDomeImageManager& domeImMan,bool isHD);
bool LoadPoseDetectMultipleCamResult_PoseMachine(const char* poseDetectFolder, const int currentFrame, const vector<CamViewDT*>& domeViews,vector< vector<SPose2D> >& detectedFaceVect,vector<int>& validCamIdxVect,bool isHD=false);

class CBodyPoseRecon
{
public:
	CBodyPoseRecon(void);
	~CBodyPoseRecon(void);

	void ClearDetectionHull()
	{
		for(int i=0;i<m_nodePropScoreVector.size();++i)
			delete m_nodePropScoreVector[i];
		m_nodePropScoreVector.clear();

		for(int i=0;i<m_edgeCostVector.size();++i)
			delete m_edgeCostVector[i];
		m_edgeCostVector.clear();
	}

	void ClearSKeletonHierarchy()
	{
		for(int i=0;i<m_skeletonHierarchy.size();++i)
			delete m_skeletonHierarchy[i];
		m_skeletonHierarchy.clear();
	}

	// Node Proposal Generation
	void ProbVolumeRecoe_nodePartProposals_fromPoseMachine_coco19(const char* dataMainFolder,const char* calibFolder,const int askedCamNum, const int frameIdx, bool bSaveCostMap, bool isHD=false, bool bCoco19=false);
	void ConstructJointHierarchy(int jointNum);

	int m_targetPoseReconUnitIdx;		//index for m_3DPSPoseMemVector
	int m_targetdetectHullManIdx;		//index for m_nodePropScoreVector
	int m_targetPoseReconUnitIdx_kinect;

	float m_detectionCostVolumeDispThresh;
	float m_detectionEdgeCostDispThresh;

	float m_nodePropMaxScore;
	float m_nodePropMinScore;
	float m_partPropMaxScore;
	float m_partPropMinScore;

	int m_loadedKinectNum;		//increase whenever LoadKinectPoseRecon is called

	vector<STreeElement*> m_skeletonHierarchy;
	vector<CVisualHullManager*> m_nodePropScoreVector;		//Each CVisualHullManager element -> each time instance
															//CVisualHullManager has multiple visual hull, each of which is for each joint  (different from ordinary usage)
															
	vector<SEdgeCostVector*> m_edgeCostVector;		//I made this for visualization. Need to be merged with other costs
													//This number should be same as m_nodePropScoreVector

	vector<PoseMachine2DJointEnum> m_map_devaToPoseMachineIdx;  //m_map_devaToPoseMachineIdx[PoseMachine2DJointEnum] == BodyJointEnum. Why Deva? Superset. 
	vector<PoseMachine2DJointEnum> m_map_SMCToPoseMachineIdx;
	vector<OpenPose25JointEnum> m_map_devaToOpenPoseIdx;
	vector<OpenPose25JointEnum> m_map_SMCToOpenPoseIdx;

	vector<cv::Point3f> m_boneColorVector;		//size is same as bone number
private:
	CDomeImageManager m_domeImageManager;
	
	int m_curImgFrameIdx;

	float m_voxMarginalMaxScore;
	float m_voxMarginalMinScore;

	vector< int > m_ourNodeIdxToKinect;		//m_ourNodeIdxToKinect[ourIdx] == kinectNodeIdx;
	vector<pair<int,int> > m_kinectEdges;
	vector<cv::Point3f> m_colorSet;
	void GenerateHumanColors();
	FPStype m_fpsType;
};

}