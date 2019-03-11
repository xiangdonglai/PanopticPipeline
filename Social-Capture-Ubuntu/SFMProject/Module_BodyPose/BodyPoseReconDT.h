#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include "DataStructures.h"
#include "TrajElement3D.h"

namespace Module_BodyPose
{

extern const int MODEL_JOINT_NUM_15;
extern const int MODEL_JOINT_NUM_COCO_19;
extern const int MODEL_JOINT_NUM_OP_25;

//////////////////////////////////////////////////////////////////////////
//Originally defined by Deva's code
//We have added serval points (body center, coco points)
//Our code uses this as the "superset" labels for 2D and 3D body skeleton
//////////////////////////////////////////////////////////////////////////
enum BodyJointEnum 
{
	JOINT_headTop=0,
	JOINT_neck,  //1

	//left
	JOINT_lShoulder, //2
	JOINT_lUpperArm,//3
	JOINT_lElbow,//4
	JOINT_lLowerArm,//5
	JOINT_lHand,//6

	JOINT_lUpperBody,//7
	JOINT_lLowerBody, //8

	JOINT_lHip,  //9
	JOINT_lUpperLeg,  //10
	JOINT_lKnee,  //11
	JOINT_lLowerLeg,  //12
	JOINT_lFoot,  //13

	//right
	JOINT_rShoulder, //14
	JOINT_rUpperArm,//15
	JOINT_rElbow,//16
	JOINT_rLowerArm,//17
	JOINT_rHand,//18

	JOINT_rUpperBody,//19
	JOINT_rLowerBody, //20

	JOINT_rHip,  //21
	JOINT_rUpperLeg,  //22
	JOINT_rKnee,  //23
	JOINT_rLowerLeg,  //24
	JOINT_rFoot,  //25

	//Artificial joint
	Joint_bodyCenter, //26

	//from MS COCO dataset
	JOINT_lEye,  //27
	JOINT_lEar,     //28
	JOINT_rEye,			//29
	JOINT_rEar,			//30

	// New OP25
	JOINT_lBigToe,		//31
	JOINT_lSmallToe,	//32
	JOINT_lHeel,		//33
	JOINT_rBigToe,		//34
	JOINT_rSmallToe,	//35
	JOINT_rHeel,		//36

	JOINT_realheadtop, // the previous one is actually nose.
};

//To load Pose Machine.
//Order is same as pose machine
enum PoseMachine2DJointEnum 
{
	PM_JOINT_HeadTop=0,	//0
	PM_JOINT_Neck,	//1
	PM_JOINT_rShoulder,	//2
	PM_JOINT_rElbow,	//3
	PM_JOINT_rWrist,	//4
	PM_JOINT_lShouder,	//5
	PM_JOINT_lElbow,	//6
	PM_JOINT_lWrist,	//7
	PM_JOINT_rHip,	//8
	PM_JOINT_rKnee,	//9
	PM_JOINT_rAnkle,	//10
	PM_JOINT_lHip,	//11
	PM_JOINT_lKnee,	//12
	PM_JOINT_lAnkle,	//13

	//From here, coco
	PM_JOINT_rEye,
	PM_JOINT_lEye,
	PM_JOINT_rEar,
	PM_JOINT_lEar,
	
	PM_JOINT_Unknown
};

enum OpenPose25JointEnum   // copied from OpenPose BODY_25B
{
    OP_JOINT_Nose,
    OP_JOINT_LEye,
    OP_JOINT_REye,
    OP_JOINT_LEar,
    OP_JOINT_REar,
    OP_JOINT_LShoulder,
    OP_JOINT_RShoulder,
    OP_JOINT_LElbow,
    OP_JOINT_RElbow,
    OP_JOINT_LWrist,
    OP_JOINT_RWrist,
    OP_JOINT_LHip,
    OP_JOINT_RHip,
    OP_JOINT_LKnee,
    OP_JOINT_RKnee,
    OP_JOINT_LAnkle,
    OP_JOINT_RAnkle,
    OP_JOINT_UpperNeck,
    OP_JOINT_HeadTop,
    OP_JOINT_LBigToe,
    OP_JOINT_LSmallToe,
    OP_JOINT_LHeel,
    OP_JOINT_RBigToe,
    OP_JOINT_RSmallToe,
    OP_JOINT_RHeel,
    OP_JOINT_Unknown
};


//ICCV 2015 Social Motion Capture Joint Index 
//This is the 3D labels for 15 joint case
enum SMC_BodyJointEnum 
{
	SMC_BodyJoint_neck=0,  //0
	SMC_BodyJoint_headTop =1,
	
	SMC_BodyJoint_bodyCenter,  //2

	//left
	SMC_BodyJoint_lShoulder, //3
	SMC_BodyJoint_lElbow,//4
	SMC_BodyJoint_lHand,//5
	SMC_BodyJoint_lHip,  //6
	SMC_BodyJoint_lKnee,  //7
	SMC_BodyJoint_lFoot,  //8

	//right
	SMC_BodyJoint_rShoulder, //9
	SMC_BodyJoint_rElbow,//10
	SMC_BodyJoint_rHand,//11
	SMC_BodyJoint_rHip,  //12
	SMC_BodyJoint_rKnee,  //13
	SMC_BodyJoint_rFoot,  //14


	//Additional by COCO
	SMC_BodyJoint_lEye,
	SMC_BodyJoint_lEar,
	SMC_BodyJoint_rEye,
	SMC_BodyJoint_rEar,

	SMC_BodyJoint_lBigToe,
	SMC_BodyJoint_lSmallToe,
	SMC_BodyJoint_lHeel,

	SMC_BodyJoint_rBigToe,
	SMC_BodyJoint_rSmallToe,
	SMC_BodyJoint_rHeel
};

class STreeElement
{
public:
	STreeElement(BodyJointEnum j)
	{
		jointName=j;
		bInferActiveJoint =false;

		levelToRootInHiearchy=0;
		mirrorJointIdx =-1;
		//printf("%d %s\n",jointName,GetJointNameStr().c_str());
	}

	std::vector<STreeElement*> parents;			//WARNING!!! parent number should be 1...Due to the edge generation thing
	std::vector<float> parent_jointLength;		//size should be same as "children" vector...For better implementation, move this to each nodes.
											//the only benefit of this codding is that we only need to keep one edge vector, assuming fully conencted graphy

	std::vector< std::vector<float> > childParent_partScore;		//outer: all childNodes, inner: all parentNodes //3D connectity socres for all possible pair
	std::vector< std::vector<int> > childParent_counter;		//outer: all childNodes, inner: all parentNodes //used to compute average score
	std::vector<STreeElement*> children;

	BodyJointEnum jointName; 
	int idxInTree; //index in the m_skeletonHierarchy vector
	int mirrorJointIdx;		//a reference between left-right limbs

	// std::string GetJointNameStr()
	// {
	// 	return BodyJointEnum_str[jointName];
	// }

	bool bInferActiveJoint;
	int levelToRootInHiearchy;		//to adjust skeleton display level;
};

class CBody3D
{
public:
	CBody3D()
	{
		bDraw  =true;
		skeletonColor =g_blue_p3f;
		m_humanIdentLabel = -1;

		m_bValid = true;
	}

	int m_humanIdentLabel;
	std::vector<PickedPtInfo> m_selectedPtsForCurrentJoint;			//used for interpolation
	std::vector<cv::Point3f> m_jointPosVect;			//joint position for the currently selected time instance. Rendering is referring only this data structure
	std::vector<bool>	m_isOutlierJoint;		//same size as m_jointPosVect==15. m_isOutlierJoint.size() == 0 in general.
	std::vector<double>	m_jointConfidenceVect;		//after inference. Save node proposal's score. -1 if not valid
	std::vector<SJointTraj> 	m_jointTrajVect;	//corresponding trajectories for each joint (used for manual selection)
	std::vector<cv::Point3f> m_boneDirectinalVector;		//Each element is a vector from child to parent joint. size is same as m_jointPosVect. Root has zero vector. 
	std::vector<float> m_boneLengthVector;		//Each element is a length from child to parent joint. size is same as m_jointPosVect. Root has zero vector. 
	cv::Point3f skeletonColor;		//to be visualized
	cv::Point3f skeletonColorBackupForPicking;

	////////////////////////////////////
	//Computed from Trajectory Stream
	std::vector<cv::Mat_<double> > m_partTrans_next;		//transformation for t->t+1.. m_partTrans_next[childIdx] means trans. for the parent-child limb. size() is same as joint hierarchy. m_limbTrans_next[0] means nothing. 
	std::vector<cv::Mat_<double> > m_partTrans_prev;		//transformation for t->t-1

	//Debugging tools
	std::vector< TrajElement3D* > m_relatedTraj;			//related trajectories. pair<postiton,color>. Currently only used for display
	std::vector< pair<cv::Point3f,cv::Point3f>  > m_relatedPointCloud;			//related trajectories. pair<postiton,color>. Currently only used for display

	cv::Mat m_bodyNormal;		//Used to check body orientation
	//Transform from world to Human-centric coord
	//ptMat_world = curRefHuman.m_ext_R * ptMat + curRefHuman.m_ext_t;
	cv::Mat m_ext_R;
	cv::Mat m_ext_R_inv;
	cv::Mat m_ext_t;

	bool m_bValid;  //used to denote blank(missing) element in the association process

	bool bDraw;
	bool isBlank()
	{
		if(m_jointPosVect.size()==0) 
			return true;
		else
			return false;
	}
	int GetFirstImgFrameIdx()
	{
		if(m_jointTrajVect.size() ==0)
			return -1;

		int imgIdx = 1e5;
		for(int i=0;i<m_jointTrajVect.size();++i)
		{
			if(imgIdx > m_jointTrajVect[i].m_initFrameIdx)
				imgIdx  = m_jointTrajVect[i].m_initFrameIdx;
		}
		return imgIdx;
	}
	int GetLastImgFrameIdx()
	{
		if(m_jointTrajVect.size() ==0)
			return -1;

		int imgIdx = -1e5;
		for(int i=0;i<m_jointTrajVect.size();++i)
		{
			int lastFrameIdx  = m_jointTrajVect[i].m_initFrameIdx  + (int)m_jointTrajVect[i].m_jointTrajectory.size();
			if(imgIdx <  lastFrameIdx)
				imgIdx  = lastFrameIdx;
		}
		return imgIdx;
	}
	//////////////////////////////////////////////////////////////////////////
	//Set m_jointPosVect from m_jointTrajVect
	//////////////////////////////////////////////////////////////////////////
	void SetCurFrameIdx(int selectedFrameIdx)			//the arguement is actual frame idx, not memVectIdx
	{
		/*if(m_jointPosVect.size() != m_jointTrajVect.size())
		{
			printf("Mismatch: m_jointPosVect.size() != m_jointTrajVect.size()\n");
			return;
		}*/

		int cnt=0;
		for(int i=0;i<m_jointTrajVect.size();++i)
		{
			if(m_jointTrajVect[i].m_jointTrajectory.size()==0)
				continue;		

			int trajTimeIdx = selectedFrameIdx - m_jointTrajVect[i].m_initFrameIdx;
			if(trajTimeIdx<0)
				continue;//trajTimeIdx =0;
			if(trajTimeIdx >=m_jointTrajVect[i].m_jointTrajectory.size())
				continue;
				//trajTimeIdx = int(m_jointTrajVect[i].m_jointTrajectory.size()-1);

			m_jointPosVect[i] = m_jointTrajVect[i].m_jointTrajectory[trajTimeIdx];
			cnt++;
		}

		if(cnt == m_jointTrajVect.size())
			bDraw =true;
		else 
			bDraw =false;
	}

	void ExportJointTrajectory(char* fileName)
	{
		ofstream fout(fileName,std::ios_base::app);
		int maxRange = 1e3;
		int minRange = -1e3;
	/*	for(int i=0;i<m_jointTrajVect.size();++i)
		{
			minRange = max(minRange,m_jointTrajVect[i].m_initFrameIdx);			//in frameIdx
			maxRange = min(maxRange,m_jointTrajVect[i].m_initFrameIdx + (int) m_jointTrajVect[i].m_jointTrajectory.size());	//in frameIdx
		}
		fout << minRange << " " <<maxRange <<"\n";			//in frameIdx*/
		for(int i=0;i<m_jointTrajVect.size();++i)
		{
			fout << m_jointTrajVect[i].m_initFrameIdx <<" " <<m_jointTrajVect[i].m_jointTrajectory.size() <<"\n";
			for(int t=0;t<m_jointTrajVect[i].m_jointTrajectory.size();++t)
			{
				fout << m_jointTrajVect[i].m_jointTrajectory[t].x << " " << m_jointTrajVect[i].m_jointTrajectory[t].y << " " << m_jointTrajVect[i].m_jointTrajectory[t].z << " ";
			}
			fout << "\n";
		}
	}


	void ImportJointTrajectory(ifstream& fin)
	{
		m_jointPosVect.clear();
		m_jointTrajVect.clear();
		//		ifstream fin(fileName);
		int initFrameIdx;
		int trackedFrameNUm;
		for(int i=0;i<CURRENT_JOINT_NUM;++i)		//For each joint
		{
			fin >> initFrameIdx >> trackedFrameNUm ;
			m_jointPosVect.push_back(cv::Point3f(0,0,0));
			m_jointTrajVect.push_back(SJointTraj());
			m_jointTrajVect.back().m_initFrameIdx = initFrameIdx ;
			for(int t=0;t<trackedFrameNUm;++t)
			{
				cv::Point3f temp;
				fin >> temp.x >> temp.y >> temp.z;

				if(temp.x==0 && temp.y==0 &&temp.z==0)
					m_jointTrajVect.back().m_jointTrajectory.push_back(m_jointTrajVect.back().m_jointTrajectory.back());
				else
					m_jointTrajVect.back().m_jointTrajectory.push_back(temp);
			}
			m_jointPosVect.back() = m_jointTrajVect.back().m_jointTrajectory.front();
		}
	}
};

//Body3DScenes by frame. 
struct SBody3DScene			
{
	int m_imgFramdIdx;
	std::vector<CBody3D> m_articBodyAtEachTime;
};

//Each element represents edge cost for each time instance
struct SEdgeCostVector
{
	std::vector< std::vector< std::vector<float> > > parent_connectivity_vis;		//this is only for the visualization . <outer vector>: joints,  <inner vectors> child x parent
	std::vector< std::pair<cv::Point3d,cv::Point3d> > edgeVisualization;	//just for visualization. Initialized by ReVisualizePartProposals()
};

struct SPose2D
{
	float detectionScore;			//used for Deva's detector
	std::vector<cv::Point2d> bodyJointBoxSize;		//landmark pts, detected box size. used in Deva's detector

	std::vector<cv::Point2d> bodyJoints;		//landmark pts	
	std::vector<double> bodyJointScores;		//scores for each joint		//New data for the pose machine

	cv::Mat m_P;
	cv::Point3f m_camCenter;		//to compute baseline
	int m_viewIdx;		//unique index for each cam in the global image array
	int m_panelIdx;	//0~20
	int m_camIdx;		//1~24
};

// Donglai's comment: declare InferNode first such that InferEdge can be defined. InferNode will be defined immediately after InferEdge
struct InferNode;

struct InferEdge
{
	float edgeScroe;
	InferNode* otherNode;

	//debug
	float debugSpringScore;	
	float debugConnectScore;	
};


struct InferNode
{
	InferNode()
	{
		pTargetSurfVox = NULL;
		scoreSum =0;
		springScoreSum =0;
		connectivityScoreSum =0;

		dataScore =0;
		bTaken = false;
	}
	SurfVoxelUnit* pTargetSurfVox;
	
	// Data for this node //////////////////////////////////////////////////////////////////
	cv::Point3f pos;
	float dataScore;
	std::vector< vector<InferEdge> > m_edgesToParent;		//There should be 1 parent. outer vector for each jointGroup, inner vector for edges
	std::vector< vector<InferEdge> > m_edgesToChild;			//There could be multiple children. outer vector for each jointGroup, inner vector for edges

	//Variables for backtracking  //////////////////////////////////////////////////////////////////
	std::vector<InferNode*> pPrevInferNodeVect;	//pointer for backtracking. made this vector for multiple children case
	int associatedJointIdx;		//for easier backtracking
	int originalPropIdx;		//the index when loaded (to look up corresponding part proposals)

	bool bTaken;		//if yes, already used as a joint for a subject

	//Variables for inference //////////////////////////////////////////////////////////////////
	float scoreSum;		//for inference
	float dataScoreSum;
	float springScoreSum;		//for debugging
	float connectivityScoreSum;		//for debugging 

	std::vector< std::pair<float, int> > neighborBestScore;		//used for backtracking. only for root. max value, joint index

	//ETC //////////////////////////////////////////////////////////////////
	Bbox3D bbox;	//for NMS
};

struct SJointCandGroup
{
	SJointCandGroup()
	{
		pAssociatedJoint= NULL;
		bDoneAllProcess= false;
	}
	std::vector<InferNode*> nodeCandidates;		//I use pointer here, which is the different from SJointCandGroup_old

	STreeElement* pAssociatedJoint;

	//The order of the following do not consistent with the order of m_edgesToParent of InferNode
	std::vector<SJointCandGroup*> parentCandGroupVector;
	std::vector<SJointCandGroup*> childCandGroupVector;

	std::vector<SJointCandGroup*> groupsReadyToSendMessageToMe;		//They are ready to send message to me
															//one-pass case: if this number is same as childCandGroupVector, get messages and do process, and send ready message to the parents
															//two-pass case: if this number is same as (neighborNum-1), get messages and do process, and send ready message to the other reaming neighbor
															//				 if this number is same as (neighborNum), do above for all (neighborNum-1) combinations.
															//				 if this number is same as (neighborNum), compute max marginal for current node. 

	//vector<SJointCandGroup*> groupsISentReadyMessage;		//I should send read-message to all parents (one-pass), or all neighbor(two-passes)
	bool bDoneAllProcess;			//Tasks: (1) sending messages to all parents (or neighbors), compute max marginal for the root (or everynodes in two-pass case)
};

//////////////////////////////////////////////////////////////////////////
// Used of find meaningful candidate using Trajecotry Stream
struct BoneCandidate
{
	//For each element of m_articBodyAtEachTime
	enum JOINT_ENUM{
		IDX_CHILD=0,
		IDX_PARENT
	};

	bool isBlank()
	{
		if(jointPts.size()==0) 
			return true;
		else
			return false;
	};

	std::vector<int> idxInNodeProposalVect;			//index in the nodeproposal vectors of child and parent vectors
	//if jointIdx=NULL, use jointPts instead. (For the case sententially generated bone)
	std::vector<cv::Point3f> jointPts;			//child, parent 
	int memIdx;			//generated memIdx
	float score;
};

}