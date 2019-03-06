#pragma once
#include "opencv2/core/core.hpp"
#include <cv.h>
#include <cxcore.h>
#include <ctype.h>
#include <fstream>

#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/legacy/legacy.hpp>
//#include <opencv2/contrib/contrib.hpp>
//#include <opencv2/video/background_segm.hpp>

#include "DataStructures.h"

class TrajElement3D;


class TrajElement3D
{
	//The following is made ade for Moduel_PatchStreamRecon
public:
	bool m_bIsStillTracked;		//Track this point, if m_bIsStillTracked==true
public:

	/* Constructors */
	void Init()  //Called by Constructors. Should be always called
	{
		m_color = cv::Point3d(1,1,1);
		m_scale3D = 0;
		bValid = true;		//used to eliminated some "not perfect" points 

		m_initiatedMemoryIdx = -1;
		m_initiatedFrameIdx = -1;
		m_ptIdxInMemoryVect = -1;
		//m_prevRefImageIdx =-1;
#ifndef LINUX_COMPILE
		//random color generation for display
		m_randomColor.z = int(rand()%256)/255.0;	//blue
		m_randomColor.y = int(rand()%256)/255.0;	//green
		m_randomColor.x = int(rand()%256)/255.0;	//red
		//printf("randColor %f %f %f\n",m_randomColor.x,m_randomColor.y,m_randomColor.z);
#endif
		m_arrow1StepVect4by1 = cv::Mat::zeros(4,1,CV_64F);   m_arrow1StepVect4by1.at<double>(3,0) = 1;
		m_arrow2StepVect4by1 = cv::Mat::zeros(4,1,CV_64F);	 m_arrow2StepVect4by1.at<double>(3,0) = 1; 
		m_normal = cv::Mat::zeros(3,1,CV_64F);  

		m_refCamIdx = -1;

		/*m_linkedPrevTrajectory = NULL;
		m_linkedNextTrajectory = NULL;*/
	}

	TrajElement3D()
	{
		Init();
	}

	TrajElement3D(cv::Mat& X)
	{
		Init();

		if(X.rows==4)
			X.copyTo(m_ptMat4by1);
		else if(X.rows==3)
		{
			m_ptMat4by1 = cv::Mat::ones(4,1,CV_64F);
			X.copyTo(m_ptMat4by1.rowRange(0,3));
		}

		double* coord = X.ptr<double>(0);
		m_pt3D.x = coord[0];
		m_pt3D.y = coord[1];
		m_pt3D.z = coord[2];
	}

	TrajElement3D(cv::Point3d& paraX)
	{
		Init();

		m_pt3D = paraX;
		m_ptMat4by1 = cv::Mat::ones(4,1,CV_64F);
		m_ptMat4by1.at<double>(0,0) = m_pt3D.x;
		m_ptMat4by1.at<double>(1,0) = m_pt3D.y;
		m_ptMat4by1.at<double>(2,0) = m_pt3D.z;
	}	

	/* Patch Initialization */
	void InitializePatchByRefCam(CamViewDT* refCam,double projectedPatchScale);		//normal: parallel to image plane, patch size: defeind by projectedPatchScale
	void InitializePatchByNormal(cv::Point3d normal,double patch3DHalfWidthInCM);
	void InitializeArrowVects();
	void InitializeArrowVectNormal();
	void InitializeArrowsByNormal(cv::Mat& normal,CamViewDT* refImage,double halfPatchSize);	//Initilize ArrowHeads and etc. by normal
	
	/* Set or Get Status */
	//Set status variables (m_ptMat4by1, m_pt3D, m_arrow1stHead, m_arrow2ndHead)
	bool SetTrajStatusByFrameIdx(int frameIdx);	
	//Get Trajectory Unit for a time instance using globalMemIdx
	TrackUnit* GetTrackUnitByMemIdx(int globalMemIdx);	// memIdx is the global memIdx
	TrackUnit* GetTrackUnitByFrameIdx(int frameIdx);	// memIdx is the global memIdx
	template <typename Type>
	bool GetTrajPoseBySelectedFrameIdx(int targetFrameIdx,cv::Point3_<Type>& outputPt)
	{
		int offset = targetFrameIdx-m_initiatedFrameIdx;
		int targetMemIdx = m_initiatedMemoryIdx + offset;
		TrackUnit* targetTrackUnit = GetTrackUnitByMemIdx(targetMemIdx);
		if(targetTrackUnit==NULL)
			return false;
		else
		{
			outputPt = targetTrackUnit->m_pt3D;
			return true;
		}
	}

	template <typename Type>
	bool GetTrajPosNormal_BySelectedFrameIdx(int targetFrameIdx,cv::Point3_<Type>& outputPt,cv::Point3_<Type>& outputNormal)
	{
		int offset = targetFrameIdx-m_initiatedFrameIdx;
		int targetMemIdx = m_initiatedMemoryIdx + offset;
		TrackUnit* targetTrackUnit = GetTrackUnitByMemIdx(targetMemIdx);
		if(targetTrackUnit==NULL)
			return false;
		else
		{
			outputPt = targetTrackUnit->m_pt3D;
			outputNormal = targetTrackUnit->m_normal;
			return true;
		}
	}
	void getArrow1Vect(cv::Mat& arrow);  //return 3x1 vector 
	void getArrow2Vect(cv::Mat& arrow);  //return 3x1 vector 
	void GetPatch3DCoord(int x, int y,cv::Mat& returnMat);
	void GetPatchNormzliaed3DCoord(int x, int y,cv::Mat& returnMat);  //center is origin.
	void GetLocalRotationMatrix(TrajElement3D *pOther,cv::Mat& returnRotMat);
	void SetPos(cv::Point3d& newPos);		
	void SetPosWithoutArrowSet(cv::Point3d& newPos);		
	void SetAverageColorUsingSIFTKey(vector<CamViewDT*>& sequences);
	void SetAverageColorByProjection(vector<CamViewDT*>& camVect,bool bOnlyHD);
	//retutnr 3pts: 1stArrow, 2ndArrow, normal
	//1stArrow & 2ndArrow are not orthogonal
	void GetPointMatrix(vector<cv::Mat>& pointVector); 	//http://www.cs.hunter.cuny.edu/~ioannis/registerpts_allen_notes.pdf
	void GetTransformTo(TrajElement3D* target,cv::Mat& R,cv::Mat& t);
	int GetTrajLength()
	{
		return m_actualPrevTrackUnit.size() + m_actualNextTrackUnit.size() + 1;
	}
	int GetTrajStartTimeInFrameIdx()
	{
		return m_initiatedFrameIdx- m_actualPrevTrackUnit.size(); 
	}
	int GetTrajLastTimeInFrameIdx()
	{
		return m_initiatedFrameIdx+ m_actualNextTrackUnit.size(); 
	}
	int GetTrajStartTimeInMemIdx()
	{
		return m_initiatedMemoryIdx - m_actualPrevTrackUnit.size(); 
	}
	int GetTrajLastTimeInMemIdx()
	{
		return m_initiatedMemoryIdx + m_actualNextTrackUnit.size();
	}
	void GetVisibilityVector(vector<int>& visIdxVect)
	{
		visIdxVect.resize(m_associatedViews.size());
		for(int i=0;i<m_associatedViews.size();++i)
			visIdxVect[i] = m_associatedViews[i].camIdx;
	}

	/* Transforming Patch */
	void OrthgonalizePatchAxis(bool bInitializeArrowVectNormal);			//usually called after optimization. 
	void PatchResize(double arrow1Length,double arrow2Length,bool bInitializeArrowVectNormal);
	void RotateNTransPatch(cv::Mat_<double>& R,cv::Mat_<double>& t);		//global R,t transform
	void InvRotateNTransPatch(cv::Mat_<double>& R,cv::Mat_<double>& t);		//global R,t transform
	void LocallyRotatePatch(const cv::Mat_<double> R);				//R is applied locally such that center is fixed
	void LocallyRotatePatchByYawPitchRoll(double yaw,double pitch,double roll);

	/* Patch Status Backup */
	void BackUpStatus();
	void RollBackPosStatus();
	void CopyTo(TrajElement3D& newOne);

	/* Photometric Consistency related */
	bool Extract3DPatchPixelValue(CamViewDT* pRefImage);
	bool Extract3DPatchPixelValue(CamViewDT* pCamView,cv::Mat_<double>* patchReturn);
	bool Extract3DPatchPixelValue_microStrcuture(CamViewDT* pRefImage);
	static bool Extract3DPatchPixelValue_microStrcuture(CamViewDT* pRefImage,CMicroStructure& microStr,vector< pair< cv::Point2i,cv::Mat > >& patch3DGridVect,cv::Mat_<double>& refPatch);
	static bool Extract3DPatchPixelValue_microStrcuture_rgb(CamViewDT* pRefImage,CMicroStructure& microStr,vector< pair< cv::Point2i,cv::Mat > >& patch3DGridVect,cv::Mat_<cv::Vec3d>& refPatch);
	double ComputePhotoConsistCostForOneView(CamViewDT* pTargetImage,cv::Mat_<double>* patchReturn);

	double TrajectoryDistance_average_max(TrajElement3D* otherT);
	double TrajectoryDistance_minMaxCost(TrajElement3D* otherT);
	void TrajectoryDist_minMax(TrajElement3D* otherT,double& minDist,double& maxDist,int& computeCnt);
	
	/* Micro Structure (advanced patch structure) */
	void GenerateMicroStruct(CMicroStructure& str,double patchArrowSizeCm);		//This function only can initialize CMicroStructure. 

	/* Visualization */
	//void VisualizeTrajectory(vector< pair<cv::Point3d,cv::Point3d> >& trajectoryPts,cv::Point3d color);
	void VisualizeTrajectoryLocalColor(vector< pair<cv::Point3d,cv::Point3d> >& trajectoryPts,cv::Point3d color,bool bShowOnlyForwardTraj);
	void VisualizeTrajectory(vector< pair<cv::Point3d,cv::Point3d> >& trajectoryPts,int curSelectedMemIdx,int curPtMemIdx,cv::Point3d color,bool bShowOnlyForwardTraj);
	void VisualizeTrajectory(vector< pair<cv::Point3d,cv::Point3d> >& trajectoryPts,cv::Point3d color,bool bShowOnlyForwardTraj);
	void VisualizeTrajectory(vector< pair<cv::Point3d,cv::Point3d> >& trajectoryPts,vector< float>& trajectoryPts_alpha,int curSelectedMemIdx
		,vector< cv::Point3d >& colorMapForNext,vector< cv::Point3d >& colorMapForPrev,int colorMode,vector< cv::Point3f >& colorMapGeneral,bool showOnlyFoward,int firstTimeMemIdx,int lastTimeMemIdx,int prevCutLeng,int backCutLeng);
	void VisualizeTrajectoryReturnWithFrame(vector< pair<cv::Point3d,cv::Point3d> >& trajectoryPts,vector<int>& returnFrameIdx,vector< float>& trajectoryPts_alpha,int curSelectedMemIdx
		,vector< cv::Point3d >& colorMapForNext,vector< cv::Point3d >& colorMapForPrev,int colorMode,vector< cv::Point3f >& colorMapGeneral,bool showOnlyFoward,int firstTimeMemIdx,int lastTimeMemIdx,int prevCutLeng,int backCutLeng);
	void VisualizeTripletTrajectory(vector< cv::Point3d >& trajectoryPts);

	/*	Projections */
	cv::Point2d GetProjectedPatchCenter(const cv::Mat& P);
	void GetProjectedPatchBoundary(const cv::Mat& projMat,vector<cv::Point2d>& boundaryPts);
	bool PatchProjectionBoundaryCheck(const cv::Mat& targetImage, const cv::Mat& P);
	void Visualize_projectedMicroStructure(cv::Mat targetImage,const cv::Mat& P,cv::Scalar color);
	void Visualize_projectedPatchBoundary(cv::Mat targetImage,const cv::Mat& P,cv::Scalar color);
	void Visualize_projectedPatchCenter(cv::Mat targetImage,const cv::Mat& P,cv::Scalar color, int crossSize =10);

	/*	Body Reconstruction */
	//propagatedPtsUptoNow is propgated history data 
	//frameIdxforLastElmt is the time for propagatedPtsUptoNow.back()
	//bForwardTracking==false means backward tracking 
	double BoneToTrajLongestDist(vector< vector<cv::Point3f> >& propagatedPtsUptoNow,int frameIdxforLastElmt,bool bForwardTracking);

////////////////////////////////////////////////////////
//// Member Variables

	bool bValid;  //if it is false, then it should be deleted
	//Patch information
	cv::Mat_<double> m_ptMat4by1; // save as m_pt3D, but Mat type
	cv::Point3d m_pt3D;
	CMicroStructure m_microStructure;		//Initialized by GenerateMicroStruct
	cv::Point3d m_arrow1stHead;		//arrow1 and arrow2 are not necessarily right angle...they are just the results of triangluation from correspoding 2D points
	cv::Point3d m_arrow2ndHead;
	double m_scale3D;  //definered by the distance between center and 1stArrowHead
	vector<VisibleElement> m_associatedViews;
	cv::Point3d m_color;  //for point cloud. //0-1 double value
	cv::Point3d m_colorOriginal;  //sometimes, m_color is used for special purpose,  and need to be returned for the original color
	
	//the following is initialized by InitializeArrowVectNormal()
	cv::Mat m_arrow1StepVect4by1;   //4x1 double matrix  the last element is zero//need to be initalized  //length is not 1, see the Initialize() code
	cv::Mat m_arrow2StepVect4by1;	  //4x1 double matrix 
	cv::Mat m_normal;  //3x1 matrix  double type			defined as arrow1.cross(arrow2)

	int m_initiatedMemoryIdx;
	int m_initiatedFrameIdx;
	int m_ptIdxInMemoryVect;		//may not be intialized for some old Loader

	//for Status backup and roll-back
	//Can be used by BackUpStatus(), RollBackPosStatus()
	cv::Mat m_origianl_normal;
	cv::Point3d m_origianl_pt;
	cv::Mat m_origianl_ptMat4by1; // save as m_pt3D, but Mat type
	cv::Point3d m_origianl_arrow1stHead;		//arrow1 and arrow2 are not necessarily right angle...they are just the results of triangluation from correspoding 2D points
	cv::Point3d m_origianl_arrow2ndHead;
	cv::Point3d m_randomColor;
	
	//For tracking
	TrackUnit m_curTrackUnit;
	vector<TrackUnit> m_actualPrevTrackUnit;  //for previous frames
	vector<TrackUnit> m_actualNextTrackUnit;  //for next frames
	vector<TrackUnit>* m_pTrackUnitVect;  //this is made to simplyfy doing both backward and forward 3D tracking.
	
	//For Appearance Score
	vector< pair< cv::Point2i,cv::Mat > > m_patch3DGridVect;		//have the (patch grid position in 2d , corresponding 3D point)
	cv::Mat_<double> m_refPatch;		//used for appearance cost
	int m_refCamIdx;				//this index is valid only on the corresponding camera
	//int m_prevRefImageIdx;		//save the selected reference camera index fron whith m_refPatch is extracted. used only for debug purpose

	vector<int>* m_pCamVisibilityForVis;		//made only for visualization. save the pointer of TrackUnit.m_visibleCamIdxVector

	//The following was made for clssification kind of thing.. very old code around 2012
	//CDetector m_detector;

	//The following is used for tarjectory-bone association optimization
	vector<void*> m_generalPurposePtrVect;	
	vector<double> m_generalPurposePtrVectValue;	

	//The following also used for generating super trajectory
	vector<TrajElement3D*> m_linkedPrevTrajectory;   //for matching between patches by features. old code
	vector<TrajElement3D*> m_linkedNextTrajectory;  //for matching between patches by features. old code
	int m_generalPurposeIntValue;		//initially made for trajectory segmentation
	int m_generalPurposeFloatValue;

	/*
	//The following is for 2D-3D matching (used for PnP for SfM)
	//Mat m_descriptor;
	//void SetAverageDescriptor(int desciptorSize,vector<CamViewDT*>& sequences);
	//void SetColorByReferenceView(vector<CamViewDT*>& sequences);
	//void SetDescriptorByReferenceView(int desciptorSize,vector<CamViewDT*>& sequences);
	*/
};


struct SMachingUnit
{
	SMachingUnit()
	{
		m_3DPtAddr = NULL;
	}
	
	vector<VisibleElement> m_associatedKeys;    //could be meaningless
	TrajElement3D* m_3DPtAddr;   //pointer for TrajElement3D if it has corresponding 3D point, otherwise NULL
};


class MemoryUnit
{
public:

	MemoryUnit()
	{
	}
	~MemoryUnit()
	{
		for(unsigned int frameIdx=0;frameIdx<m_camViewVect.size();++frameIdx)
		{
			if(m_camViewVect[frameIdx])
				delete m_camViewVect[frameIdx];
		}
		m_camViewVect.clear();

		for(unsigned int i=0;i<m_trajStream3D.size();++i)
		{
			delete m_trajStream3D[i];
		}
		m_trajStream3D.clear();
	}

	
	vector<CamViewDT*> m_camViewVect;
	vector<cv::Mat_<float> > m_depthMapVect;		//size should be the same as m_camViewVect
	vector<TrajElement3D*> m_trajStream3D;

	cv::Mat m_Pts3D_descriptor;		//it should be CV_32F to use OpenCV matching functions



	int m_memroyIdx;  //idx in whole memory vector
	int m_frameIdx;   //frame idx from which this memory reconstructed

	char m_fullPath[512];
};

//Utility Function
//int SelectRefImageByNormalOnlyFromHD(TrajElement3D* pPts3D,vector<CamViewDT*>& camVect,vector<int>& visibleCamIdxVector,bool bBoundaryCheck=true);
