#pragma once
#include "DomeImageManager.h"
#include "BodyPoseReconDT.h"
#include "VisualHullModule.h"

namespace Module_BodyPose
{

//Used for the BONE GROUPING ALGORITHM
class CPartTrajProposal			//contains a potential bone candidate
{	
public:

	bool GetParentJointPos(int frameIdx,cv::Point3f& returnPos)		//frameIdx is imageFrameIdx
	{
		int idx = frameIdx - m_startImgFrameIdx;
		if(idx<0 || idx >=m_articBodyAtEachTime.size())
		{
			returnPos =  cv::Point3f(-1,-1,-1);
			return false;
		}
		else
			returnPos = m_articBodyAtEachTime[idx].m_jointPosVect[BoneCandidate::IDX_PARENT];

		return true;
	}

	bool GetChildJointPos(int frameIdx,cv::Point3f& returnPos)		//frameIdx is imageFrameIdx
	{
		int idx = frameIdx - m_startImgFrameIdx;
		if(idx<0 || idx >=m_articBodyAtEachTime.size())
		{
			returnPos =  cv::Point3f(-1,-1,-1);
			return false;
		}
		else
			returnPos = m_articBodyAtEachTime[idx].m_jointPosVect[BoneCandidate::IDX_CHILD];

		return true;
	}

	void ClearData()
	{
		m_jointIdxVect.clear();
		m_articBodyAtEachTime.clear();
		m_boneByDetection.clear();
		m_detectedChildJointGroup.clear();
		m_detectedParentJointGroup.clear();

		m_forwardTransformVect.clear();	
		m_backwardTransformVect.clear();	
		m_refinedHumanSkeletons.clear();	
		m_aggregatedJointAtRefFrame.clear();
		m_relatedTrajSet.clear();
	}

	int GetEndFrameIdx() const
	{
		return m_startImgFrameIdx + m_forwardTransformVect.size() -1;
	}

	//Used to generate new part trajectory proposals, e.g, for post processing
	void Initilaize(int startImgFrameIdx,int childNodeIdx,int parentNodeIdx
					,int initiatedBoneCandImgFrameIdx, BoneCandidate& initiatedBoneCand)
	{
		m_startImgFrameIdx = startImgFrameIdx;
		m_jointIdxVect.push_back(childNodeIdx);
		m_jointIdxVect.push_back(parentNodeIdx);

		m_initiatedBoneCandImgFrameIdx =  initiatedBoneCandImgFrameIdx;
		m_initiatedBoneCand =  initiatedBoneCand;

		m_articBodyAtEachTime.resize(1);
		m_articBodyAtEachTime[0].m_jointPosVect = initiatedBoneCand.jointPts;
	}

	//Assume that we are given m_forwardTransformVect, and init part position at m_articBodyAtEachTime[0]
	void RepropagatePart_GivenForwardTranform_GivenFirstPartPos()
	{
		if(m_articBodyAtEachTime.size()==0)
		{
			printf(" ## Warning: Can't propagate since there is no initial part position]n");
			return;
		}
		CBody3D initPart = m_articBodyAtEachTime[0];
		m_articBodyAtEachTime.clear();
		m_articBodyAtEachTime.push_back(initPart);
		m_articBodyAtEachTime.resize(m_forwardTransformVect.size());
		for(int t=1;t<m_articBodyAtEachTime.size();++t)
			m_articBodyAtEachTime[t].m_jointPosVect.resize(2);		//will be initalized below

		for(int nodeIdx=0;nodeIdx<=1;++nodeIdx)		//i==0 child node, i==1 parent node;
		{
			cv::Mat_<double> startPtMat;
			Point3fToMat4by1(initPart.m_jointPosVect[nodeIdx],startPtMat);
			for(int t=1;t<m_forwardTransformVect.size();++t)
			{
				cv::Mat pt = m_forwardTransformVect[t] * startPtMat;
				m_articBodyAtEachTime[t].m_jointPosVect[nodeIdx] = MatToPoint3f(pt);
			}
		}
	}

	CPartTrajProposal()
	{
		m_startImgFrameIdx =-1;
		m_boneGroupScore=0;
		m_bForwardTracking = true;
	}

	int GetBoneTrajectoryLastFrameIdx()
	{
		return m_startImgFrameIdx + m_articBodyAtEachTime.size()-1;
	}
	//Return NULL if there is no valid skeleton at the selected frame
	//Return skeleton by frameIdx
	CBody3D* GetCurSkeleton(int imgFrameIdx)		
	{
		int idx = imgFrameIdx - m_startImgFrameIdx;
		if(idx>=0 && idx<m_articBodyAtEachTime.size())
			return &m_articBodyAtEachTime[idx];
		else 
			return NULL;
	}
	float GetBoneLength()
	{
		if(m_articBodyAtEachTime.size()==0)
			return -1;
		else
			return Distance(m_articBodyAtEachTime.front().m_jointPosVect[0],m_articBodyAtEachTime.front().m_jointPosVect[1]);
	}

	//return transform matrix from srcFrameIdx -> dstFrameIdx
	//Asumming only m_forwardTransformVect exist.
	bool GetTransform(int srcFrameIdx,int dstFrameIdx,cv::Mat& srcToDstMat)	
	{
		int srcIdx = srcFrameIdx - m_startImgFrameIdx;
		int dstIdx = dstFrameIdx - m_startImgFrameIdx;

		if(srcIdx<0 || srcIdx>=m_forwardTransformVect.size() || dstIdx<0 || dstIdx>=m_forwardTransformVect.size() )
			return false;

		cv::Mat srcMat = m_forwardTransformVect[srcIdx];
		cv::Mat dstMat = m_forwardTransformVect[dstIdx];
		srcToDstMat =  dstMat*srcMat.inv();
		return true;
	}

	template <typename T>
	float DistanceFromBone(int frameIdx, cv::Point3_<T> inputPt)
	{
		int t = frameIdx -m_startImgFrameIdx;
		if(t<0 || t>= m_articBodyAtEachTime.size())
			return 1e5;		//some big number;
			
		cv::Point3_<T> childPt = m_articBodyAtEachTime[t].m_jointPosVect[0];
		cv::Point3_<T> parentPt = m_articBodyAtEachTime[t].m_jointPosVect[1];
		cv::Point3_<T> jointDirectVector = parentPt - childPt ;
		float length = norm(jointDirectVector);
		Normalize(jointDirectVector);
		//Bone range check
		float dist = DistancePtToLine(childPt,parentPt,jointDirectVector,length,inputPt);
		return dist;
	}
	float BoneToTrajLongestDist(TrajElement3D* refTraj)
	{
		float longestDist = 0;
		for(int t=0;t<m_articBodyAtEachTime.size();++t)
		{
			int tempFrameIdx;
			if(m_bForwardTracking)
				tempFrameIdx = m_startImgFrameIdx + t;
			else
				tempFrameIdx = m_startImgFrameIdx - t;

			cv::Point3d tempTrajPt;
			bool bSuccess = refTraj->GetTrajPoseBySelectedFrameIdx(tempFrameIdx,tempTrajPt);
			if(bSuccess==false)
				break;

			assert(m_articBodyAtEachTime[t].m_jointPosVect.size()==2);
			cv::Point3f& childJointdPos = m_articBodyAtEachTime[t].m_jointPosVect[0];
			cv::Point3f& parentJointdPos = m_articBodyAtEachTime[t].m_jointPosVect[1];
			cv::Point3f& boneDirectionalVect = m_articBodyAtEachTime[t].m_boneDirectinalVector[1];

			//////////////////////////////////////////////////////////////////////////
			//Compute orthogonal dist
			cv::Point3f childToPt = cv::Point3f(tempTrajPt.x,tempTrajPt.y,tempTrajPt.z)- childJointdPos;

			float distAlongBoneDirection = childToPt.dot(boneDirectionalVect);

			cv::Point3f orthoVect = childToPt- boneDirectionalVect*distAlongBoneDirection;
			double orthoDist = norm(orthoVect);

			if(orthoDist>longestDist)
				longestDist = orthoDist;
		}
		return longestDist;
	}

	bool BoneToTrajMinMaxDist(TrajElement3D* refTraj,double& minDist,double& maxDist,
								double childNodeSideOffset_cm=0,double parentNodeSideOffset_cm=0)
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
		for(int f=compareStartFrameIdx;f<=compareLastFrameIdx;++f)
		{
			cv::Point3d tempTrajPt;
			bool bSuccess = refTraj->GetTrajPoseBySelectedFrameIdx(f,tempTrajPt);
			if(bSuccess==false)
				break;

			int t = f-m_startImgFrameIdx;
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

			maxDist = max(maxDist,orthoDist);
			minDist = min(minDist,orthoDist);
		}

		if(bOnceInTheBoundary==false)
			return false;
		return true;
	}

	bool BoneToTrajMinMaxDist_considerNormal(TrajElement3D* refTraj,double& minDist,double& maxDist,double& worstNormalInnerProd,double childNodeSideOffset_cm=0,double parentNodeSideOffset_cm=0);

	double NodeTrajectoryOptimization(int jointIdx,int startTime,vector<cv::Point3f>& originalTrajVect,vector< vector<cv::Mat_<double> > >& transformVector,vector<CVisualHullManager*>& detectionHullVector,double thresh,vector<cv::Point3f>& optimizedTraj,bool NoOptimization);
	void ComputeBoneTraj(vector<CVisualHullManager*>& detectionHullVector,float threshold);	//without optimization. Save the result on m_boneGroupScore
	bool GetBoneDirection(int frameIdx,cv::Point3f& directVect);
	void OptimizeBoneTraj(vector<CVisualHullManager*>& detectionHullVector,float threshold);		//with optimization. Save the result on m_boneGroupScore

	///////////////////////////////////////////////////
	//Essential variables
	int m_startImgFrameIdx;		//Very initial frame where this part traj proposal start (usually the end point of the forward tracking). c.f., m_initiatedBoneCandImgFrameIdx is a frame where very initial part bone is initiated (probably frame somewhere in the middle), for backward-forward propagation. 
	int m_initiatedBoneCandImgFrameIdx;		//the time of initial part bone cand is selected for backward-forward tracking
	vector<int> m_jointIdxVect;					//joint index of boneEnds. m_jointIdxVect[0] is the child bone. number should be same as the joint number of each element of m_articBodyAtEachTime
	vector<CBody3D> m_articBodyAtEachTime;		//Skeleton for each time instance. m_articBodyAtEachTime[0] is the pose at m_startImgFrameIdx. For each element, child ==[0] and paranet ==[1] 

	///////////////////////////////////////////////////
	// may not be critically required variables
	set<TrajElement3D*> m_relatedTrajSet;
	vector<float> m_boneGroupScoreForEachTime;	//only child
	cv::Point3f m_boneGroupColor;			//this color is also saved in CBody3D
	bool m_bForwardTracking;		//if false, m_articBodyAtEachTime is save as backward tracking direction

	//utility/debugging (can be deleted if you don't want to keep)
	cv::Point3f m_skeletonBasicColorBackupForPicking;		//if picking happens, original color is saved here, and recovered later.
	BoneCandidate m_initiatedBoneCand;		//[0]: child, [1]: parent: the one of initial bone cand
	vector< vector<cv::Point3f> > m_detectedChildJointGroup;		//outer for time (memIdx)
	vector< vector<cv::Point3f> > m_detectedParentJointGroup;		//outer for time (memIdx)

	// The following used in trajector-part association algorithm
	vector< pair<TrajElement3D*,double> > m_generalPurposeVect;

	//outdated
	float m_boneGroupScore;		//(old version) sum of m_detectedChildJointGroup.size() ... (new version) jointTrajCost
	vector<BoneCandidate> m_boneByDetection;	//(old version) corresponding bones generated by detection.... not meaningful anymore


	//The following used to make final skeleton pose by transforming every joints in this group into the reference frames
	//And propagate the final one at reference frame to each frame again. 
	vector<cv::Mat_<double> > m_forwardTransformVect;		//4x4 matrix [R t ; 0 0 0 1] from m_startImgFrameIdx(or the time of m_articBodyAtEachTime[0])
	// identityMat, m_referenceImgFrameIdx+1, m_referenceImgFrameIdx+2 ...
	vector<cv::Mat_<double> > m_backwardTransformVect;	//This is used only temporary. In the end, we transform this to forwardtracking and only keep m_forwardTransformVect.  4x4 matrix [R t ; 0 0 0 1] // identityMat, m_referenceImgFrameIdx-1, m_referenceImgFrameIdx-2 ...
	vector<CBody3D> m_refinedHumanSkeletons;		//Skeleton for each time instance
	vector< vector<cv::Point3f> > m_aggregatedJointAtRefFrame;		//outer for each joint hierarchy (size()==2 in a bone case, child ==[0] and paranet[1] )
};

//4D (spatio-temporal) body information for a subject for a duration of time starting from a time instance
class CBody4DSubject
{
public:
	CBody4DSubject()
	{
		m_bOwnershipOfInitBoneVect = false;
		m_frameInterval = 1;
		m_avgJointTrajScore =0;

		m_kinectIdx = -1;	//only for kinect
		m_bStillTracked = false;
		missingCnt =0;
	}
	~CBody4DSubject()
	{
		if(m_bOwnershipOfInitBoneVect)
		{
			for(int i=0;i<m_relatedPartTrajPropVector.size();++i)
			{
				delete m_relatedPartTrajPropVector[i];
			}
			m_relatedPartTrajPropVector.clear();
		}
	}
	void SetBodyColor(cv::Point3f c)
	{
		m_bodyColor = c;
		for(int i=0;i<m_finalHumanPose.size();++i)
			m_finalHumanPose[i].skeletonColor = c;
	}
	cv::Point3f GetBodyColor()
	{
		return m_bodyColor;
	}
	float m_avgJointTrajScore;		//average score of all valid joint trajectory score . mainly for debugging at this momoent

	bool GetBody3DFromFrameIdx(int frameIdx,CBody3D** returnBody);
	//int m_startImgFrameIdx;		//start img frame idx
	vector<CPartTrajProposal*> m_relatedPartTrajPropVector;			//DO NOT DELETE this pointers except m_bOwnershipOfInitBoneVect==true !!!: Contains a pointer of m_partTrajProposalVector element.. Each element is for each bone. Order is same as m_skeletonHierarchy
	vector<CBody3D> m_finalHumanPose;			//final result. each element for each time instance. Skeleton's size should be m_skeletonHierarchy.size(). (0,0,0) means not intialized joints, which should be ignored

#ifndef LINUX_COMPILE
	//////////////////////////////////////////////////////////////////////////
	/// Face Detector
	vector<Module_Face::SFace3D*> m_face3DReconInit;			//Very initial association by proximity. starting from m_initFrameIdx. number should be either 0 or same as m_finalHumanPose. Null for missing face
	vector<Module_Face::SFace3D> m_face3DReconOptimized;			//Very initial association by proximity. starting from m_initFrameIdx. number should be either 0 or same as m_finalHumanPose. Null for missing face
#endif

	//////////////////////////////////////////////////////////////////////////
	int m_initFrameIdx;		//frameidx where m_finalHumanPose[0] is generated
	Bbox3D m_bbox;		//made for visibility debug. Initalized in Export_KinectAll_body3DJointPtsFormatBodyPoseVisibility
	bool m_bOwnershipOfInitBoneVect;		//if this true, m_relatedPartTrajPropVector should be deleted (This is the case where data is loaded from file)
	int m_frameInterval;		//only used to Ground truth data, and Kinect (especilly 

	//Only for kinect
	int m_kinectIdx;

	//Used for association
	bool m_bStillTracked;
	int missingCnt;

private:
	cv::Point3f m_bodyColor; 

};


bool Load_Undist_PoseDetectMultipleCamResult_MultiPoseMachine_19jointFull(const char* poseDetectFolder,const char* poseDetectSaveFolder,const int currentFrame, CDomeImageManager& domeImMan,bool isHD);
bool LoadPoseDetectMultipleCamResult_PoseMachine(const char* poseDetectFolder, const int currentFrame, const vector<CamViewDT*>& domeViews,vector< vector<SPose2D> >& detectedFaceVect,vector<int>& validCamIdxVect,bool isHD=false);
void InferenceJoint15_OnePass_Multiple_DM(CVisualHullManager* pTempSceneProb, std::vector<STreeElement*>& m_skeletonHierarchy,SBody3DScene& newPoseReconMem);
void InferenceJointCoco19_OnePass_singleSkeleton(CVisualHullManager* pTempSceneProb,vector<STreeElement*>& m_skeletonHierarchy,SBody3DScene& newPoseReconMem);
void SaveBodyReconResult(const char* folderPath,const SBody3DScene& poseReconMem,const int frameIdx,bool isHD=false);
void SaveBodyReconResult_json(const char* folderPath,const SBody3DScene& poseReconMem,const int frameIdx,bool bNormCoord=false,bool bAnnotated=false);

class CBodyPoseRecon
{
public:
	CBodyPoseRecon(void);
	~CBodyPoseRecon(void);

	void ClearData()
	{
		m_loadedKinectNum =0;
		ClearDetectionHull();
	}
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
	void SaveNodePartProposals(const char* dataMainFolder,const int cameraNum,const int frameIdx,const CVisualHullManager* pDetectHull,const vector<STreeElement*>& skeletonHierarchy,int actualValidCamNum=-1,bool isHD=false);

	// 3DPS pose estimation
	void Optimization3DPS_fromDetection_oneToOne_coco19(const char* dataMainFolder,const char* calibFolder,const int askedCamNum,bool isHD=false,bool bCoco19=false); //optimize poseReconMem.m_poseAtEachFrame
	void Save3DPSBodyReconResult_Json(char* saveFolderPath,bool bNormCoord,bool bAnnotated=false);

	// File IO
	void LoadNodePartProposals(const char* fullPath,const int imgFrameIdx,bool bRevisualize,bool bLoadPartProposal=true);
	bool LoadBodyReconResult(char* fullPath,int imgFrameIdx);

	//Visualization
	void AssignHumanIdentityColorFor3DPSResult();

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
	vector<SMC_BodyJointEnum> m_map_PoseMachineToSMCIdx;
	int m_skeletonHierarchy_nextHalfStart;

	vector<cv::Point3f> m_boneColorVector;		//size is same as bone number
	vector<SBody3DScene> m_3DPSPoseMemVector;		//Each element contains a scene at each time (used for the 3DPS algorithm)
	vector<CBody4DSubject> m_skeletalTrajProposals;

	void SetfpsType(FPStype t)
	{
		m_fpsType = t;
	}
	FPStype GetfpsType()
	{
		return m_fpsType;
	}
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

extern CBodyPoseRecon g_bodyPoseManager;
}