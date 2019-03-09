#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

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
		// inferedData = NULL;

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

	// void ClearInferData()
	// {
	// 	for(int i=0;i<inferedDataCands.size();++i)
	// 		delete inferedDataCands[i];
	// 	inferedDataCands.clear();

	// 	if(inferedData!=NULL)
	// 	{
	// 		delete inferedData;
	// 		inferedData =NULL;
	// 	}
	// }

	// SJointCandGroup_old* inferedData;		//finally generated result by message passing from the leaf-side
	// vector<SJointCandGroup_old*> inferedDataCands;		//Candidate from all children. Made this to consider multiple children nodes case

	bool bInferActiveJoint;
	int levelToRootInHiearchy;		//to adjust skeleton display level;
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

}