#pragma once

#define USE_MODULE_FACERECON_BYPM
#include "Utility.h"

#define FACE_PM_70_LANDMARK_NUM 70
#define FACE_PM_COCO_LANDMARK_NUM 24



namespace Module_Face_pm
{

class SFace2D_pm
{
public:
	SFace2D_pm()
	{
		m_faceIdx =-1;
		m_subjectIdx = -1;
	}

	cv::Mat m_P;
	cv::Point3f m_camCenter;		//to compute baseline
	int m_viewIdx;		//unique index for each cam in the global image array
	int m_panelIdx;	//0~20
	int m_camIdx;		//1~24

	int m_faceIdx;	//unique index in this image detecion
	int m_subjectIdx;		//corresponding human subject index
	//int m_whichSide;

	//float m_detectionScore;
	vector<cv::Point2d> m_faceLandmarkVect;		//landmark pts
	vector<double> m_detectionScore;
	//cv::Point2d centerPt;	//Center found by initial face detector
	//cv::Mat handLandmarkMat;		//to use Intraface tracker.
};

struct SFaceReconInfo_pm
{
	SFaceReconInfo_pm()
	{
		m_averageScore =  0;
		m_averageReproError = 1e5;
	}
	vector< int > m_visibility;		// cameraIdx
	double m_averageScore;
	double m_averageReproError;
};

class SFace3D_pm
{	
public:
	SFace3D_pm()
	{
		m_bValid = true;
		//m_avgDetectScore = m_avgReprError = 0;
		m_identityIdx =-1;
	}

	vector<cv::Point3d> m_landmark3D;
	vector<SFaceReconInfo_pm> m_faceInfo;

	cv::Point3d m_center;
	cv::Point3d m_normal;
	cv::Point3d m_faceUp;
	cv::Point3d m_faceX;
	bool m_bValid;		//this is used for associate for CAssociatedFaceReconMemory where I want to add blank face to
	int m_identityIdx;	//this is the human idendity index determined by CAssociatedFaceReconMemory
};

//These are used for two purpose
//1: same time multiple fingers
//2: across time single subject's face
struct SFaceReconMemory_pm
{
	vector<SFace3D_pm> faceReconVect;
	int frameIdx;			//can be a time for all face, or a tracking start time
};


}	//end of namespace Module_Face
