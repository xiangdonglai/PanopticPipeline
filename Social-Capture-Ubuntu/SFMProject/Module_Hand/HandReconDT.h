#pragma once

#define USE_MODULE_FINGER
#include "../Utility.h"

#define HAND_LEFT 1
#define HAND_RIGHT 2
#define HAND_LANDMARK_NUM 21

namespace Module_Hand
{

class SHand2D
{
public:
	SHand2D()
	{
		m_handIdx =-1;
		m_subjectIdx = -1;
		m_whichSide = HAND_LEFT;		//initialize as left hands
	}
	/*SHand2D(const SHand2D& source)
	{
		P = source.P;		//dont' have to use clone, since they will no be changed in any case.
		viewIdx = source.viewIdx;		//unique index for each cam in the global image array
		panelIdx = source.panelIdx;	//0~20
		camIdx = source.camIdx;		//1~24
		//centerPt = source.centerPt;	//Center found by initial finger detector
		fingerLandmarkVect = source.fingerLandmarkVect;		//landmark pts
		//fingerLandmarkMat = source.faceLandmarkMat.clone();		//to use Intraface tracker. 
		detectionScore = source.detectionScore;
	}*/

	cv::Mat m_P;
	cv::Point3f m_camCenter;		//to compute baseline
	int m_viewIdx;		//unique index for each cam in the global image array
	int m_panelIdx;	//0~20
	int m_camIdx;		//1~24

	int m_handIdx;	//unique index in this image detecion
	int m_subjectIdx;		//corresponding human subject index
	int m_whichSide;

	//float m_detectionScore;
	vector<cv::Point2d> m_fingerLandmarkVect;		//landmark pts
	vector<double> m_detectionScore;
	//cv::Point2d centerPt;	//Center found by initial face detector
	//cv::Mat handLandmarkMat;		//to use Intraface tracker.
};

struct SFingerReconInfo
{
	SFingerReconInfo()
	{
		m_averageScore =  0;
		m_averageReproError = 1e5;
		m_bValid = true;
	}
	vector< int > m_visibility;		// cameraIdx
	double m_averageScore;
	double m_averageReproError;
	double m_bValid;
};

class SHand3D
{	
public:
	SHand3D()
	{
		m_bValid = true;
		//m_avgDetectScore = m_avgReprError = 0;
		m_identityIdx =-1;
		m_normal = cv::Point3d(0,0,0);
	}
	/*float GetFaceWidth()
	{
		if(landmark3D.size()==0)
			return -1;

		return Distance(landmark3D[28],landmark3D[19]);
	}
	void ComputeFaceNormalCenter()
	{
		if(landmark3D.size()<FACE_LANDMARK_NUM)
			return;
		cv::Point3d vect1 = landmark3D[28] - landmark3D[40]; //a left eye pt - a mouth pt
		cv::Point3d vect2 = landmark3D[20] - landmark3D[40];	//a right eye pt - a mouth pt
		faceNormal = vect1.cross(vect2);
		Normalize(faceNormal);

		centerPt3D = landmark3D[FACE_BETWEEN_EYES_PT_INDEX];
	}*/

	vector<cv::Point3d> m_landmark3D;
	vector<SFingerReconInfo> m_fingerInfo;
	bool m_bValid;		//this is used for associate for CAssociatedFaceReconMemory where I want to add blank face to
	int m_identityIdx;	//this is the human idendity index determined by CAssociatedFaceReconMemory
	int m_whichSide;// = HAND_LEFT or HAND_RIGHT;		//initialize as left hands

	cv::Point3d m_normal;
	cv::Point3d m_palmCenter;  //0.5 of landmark[0] -> landmark[9]; 
	cv::Point3d m_palmUp;
	cv::Point3d m_palmX;


	//vector< pair<int,int> > visibility;		// <cameraIdx, faceIdx> where this finger is visible
	//cv::Point3d faceNormal;
	//cv::Point3d centerPt3D;
	//Bbox3D	m_bbox;		//
	//cv::Point3f m_bboxCenter;		//cetner between floor pt and faceCenter. Made this since bbox center may not be accurate
	//double avgReprError;
	//double avgDetectScore;
};

//These are used for two purpose
//1: same time multiple fingers
//2: across time single subject's face
struct SHandReconMemory
{
	vector<SHand3D> handReconVect;
	int frameIdx;			//can be a time for all face, or a tracking start time
};
/*
//////////////////////////////////////////////////////////////////////////
/// Contains face pose of a subject over time
class CAssociatedFaceReconMemory
{
public:
	CAssociatedFaceReconMemory()
	{
		startFrameIdx =-1;
		bStillTracked = true;

		m_bEdited = false;
		missingCnt = 0;
		m_editTrans = cv::Point3d(0,0,0);
	}
	bool GetFace3DFromFrameIdx(int frameIdx, SFace3D** pReturnBody)
	{
		int idx = frameIdx - startFrameIdx;
		if(idx<0 || idx>= faceReconVect.size())
			return false;
		else
		{
			*pReturnBody = &faceReconVect[idx];
			return true;
		}
		return false;
	};

	vector<SFace3D> faceReconVect;		//over time starting from startFrameIdx
	vector<SFace3D*> faceDetectPtVect;		//detection result. The number should be same as faceReconVect
	int startFrameIdx;

	//used for face association step
	bool bStillTracked;
	vector<TrajElement3D*> corresTrajVect;		//contains trajectory 3D belongs to this face
	cv::Point3f m_color;

	//Face editing
	bool m_bEdited;
	int missingCnt;
	cv::Point3d m_editTrans;
};*/


}	//end of namespace Module_Face
