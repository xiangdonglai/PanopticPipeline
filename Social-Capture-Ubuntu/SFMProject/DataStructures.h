#pragma once
#include <cv.h>
#include <cxcore.h>
#include <ctype.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//Opengl is included only here
// #ifndef LINUX_COMPILE
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>
// #endif

#include "Utility.h"
#include "Constants.h"
// #include "Detector.h"

#include "SyncMan.h"

using namespace std;

class CamViewDT;

class TrackingLog
{
public:
	TrackingLog()
	{
		for(int i=0;i<3;++i)
		{
			trackDist[i] = -1;
			track_BacktrackDist[i] =-1;
			reproject_TrackDist[i] =-1;
		}
		m_homoEstNCCError = -1;

//#ifdef DEBUG_TRACKING_VISIBILITY_TERM_DRAWING
		visDataTerm_normalCost = 0;
		visDataTerm_motionCost = 0;
		visDataTerm_appearanceCost = 0;
	//	visDataTerm_totalDataCost = 0;
//#endif
	}

	TrackingLog(float tDist,float tbDist,float repDist)
	{
		trackDist[0]  = tDist;
		track_BacktrackDist[0] = tbDist;
		reproject_TrackDist[0] = repDist;
		
//#ifdef DEBUG_TRACKING_VISIBILITY_TERM_DRAWING
		visDataTerm_normalCost = 0;
		visDataTerm_motionCost = 0;
		visDataTerm_appearanceCost = 0;
		//visDataTerm_totalDataCost = 0;
		
//#endif
	}

	float trackDist[3];
	float track_BacktrackDist[3];
	float reproject_TrackDist[3];
	float m_homoEstNCCError;

//#ifdef DEBUG_TRACKING_VISIBILITY_TERM_DRAWING
	//visibility data term
	float visDataTerm_normalCost;
	float visDataTerm_motionCost;
	float visDataTerm_appearanceCost;

	float visDataTerm_ratioMotion;
	float visDataTerm_thresh;
	//float visDataTerm_totalDataCost;
//#endif

};

//Element describing info of cam i at time t of TrackUnit,
class CamInfoForTrackUnit
{
public:

	CamInfoForTrackUnit()
	{
		m_seqIdx = -1;
		m_inlierChecker = false;
		m_lambdaMotionLog =  -1;

		for(int i=0;i<3;++i)
		{
			m_triplet.push_back(cv::Point2f(0,0));
			m_trackedTriplet.push_back(cv::Point2f(0,0));
		}
	}

	int m_seqIdx;
	bool m_inlierChecker;
	cv::Point2f m_projectedPt2D;  //Actually, it is just same as m_triplet[0]

	//for tracking
	cv::Point2f m_trackedPt2D;  //Optical flow result. used for the initialization of homography estimation
	vector<cv::Point2f> m_trackedTriplet;  //tracking result after Homography estimation
	vector<cv::Point2f> m_triplet; //Final triplet after projecting 3D triplet, using vector to make it esay to copy, for new framework //13.0812
	vector<cv::Point2f> m_tripletReconstructed;	//Reconstructed Patch, before optimization. Need for DEBUG

	//for detector
	//DetectedRegion m_finalObjRegion;  //x_region  //temporarily, it is fixed window size
	//vector<DetectedRegion> m_detectedPtVector; //d

#ifdef DEBUG_TRACKING_VISIBILITY_TERM_DRAWING
	//Misc
	TrackingLog m_log;
#endif

	//GraphCut
	float visDataTerm_totalDataCost;
	float m_lambdaMotionLog;
};


class CMicroStructure
{
public:
	CMicroStructure()
	{
		bValid = false;
	}

	void GenerateBlankStructure(int vertexNum)
	{
		m_vertexVect.clear();
		m_vertexIdxInPatch.clear();
		m_gridSize = sqrt(vertexNum);

		int halfSize = m_gridSize/2;
		//double scaleFactor = cm2world(patchArrowSizeCm)/Distance(m_arrow1stHead,m_pt3D);
		m_vertexVect.resize(vertexNum);
		m_vertexIdxInPatch.reserve(vertexNum);
		for(int y=-halfSize;y<=halfSize;y++)
		{
			for(int x=-halfSize;x<=halfSize;x++)
			{
				m_vertexIdxInPatch.push_back(cv::Point2i(x+halfSize,y+halfSize));
			}
		}
		ParamSetting();
		bValid  =true;
	}

	void SetValid(bool b)
	{
		bValid = b;
	}
	bool IsValid()
	{
		return bValid;
	}

	int GetGridSize()
	{
		return m_gridSize;
	}
	const vector<cv::Point3d>&  GetVertexVectReadOnly()
	{
		//vec = m_vertexVect;
		return m_vertexVect;
	}
	const vector<cv::Point2i>&  GetIndexVectReadOnly()
	{
		//vec = m_vertexVect;
		return m_vertexIdxInPatch;
	}
	/* The following code should be verified
	void GetVertexVectReadOnly(vector<cv::Point3f>** vec,vector<cv::Point2i>** idxVect)
	{
		(*vec) = &m_vertexVect;
		(*idxVect) = &m_vertexIdxInPatch;
	}*/
	vector<cv::Point3d>& GetVertexVectForModify()
	{
		return m_vertexVect;
	}
/*	void GetVertexVectForModify(vector<cv::Point3f>** vec,vector<cv::Point2i>** idxVect)
	{
		(*vec) = &m_vertexVect;
		(*idxVect) = &m_vertexIdxInPatch;
	}*/
	vector<cv::Point2i>& GetIndexVectForModify()
	{
		return m_vertexIdxInPatch;
	}
	void SetCenterPos(cv::Point3d& newPt);
	void ParamSetting()		//should be called after init or resize 
	{
		m_gridSize = sqrt((float)m_vertexVect.size());
	}
	//friend void TrajElement3D::GenerateMicroStruct(CMicroStructure& str);

private:
	vector<cv::Point3d> m_vertexVect;		//should be initialized only by GenerateMicroStruct
	vector<cv::Point2i> m_vertexIdxInPatch;		//GenerateMicroStruct initializes this
	bool bValid;
	int m_gridSize;
};

class TrackUnit
{
public:
	TrackUnit()
	{
		m_pt3D = cv::Point3d(0,0,0);
		m_memIdx = -1;
		m_distFromPrevPt = 0;
		m_lambdaLog = -1;
		m_refCamIdx = -1;
	}

	cv::Point3d m_pt3D;  //X
	CMicroStructure m_microStructure;
	cv::Point3d m_arrowHead3D[2];  //for new framework //13.0812
	cv::Point3d m_normal;
	vector<int> m_visibleCamIdxVector;  //visible cam info estimated by previous frame's patch shape (by proposed visibility estimation method)
	int m_refCamIdx;		//index of reference camera

	//May not be important for PatchStreamRecon version
	double m_distFromPrevPt;    //For cost function
	double m_lambdaLog;
	int m_memIdx;   //used only in MotionFlow generation process. After that, it's meaningless
	vector<CamInfoForTrackUnit> m_cameraInfos;

	vector<bool> m_isVisibleInCamOrder;  //orederd as m_pSequence order

	//Experiment
	void AddRandomNoise()
	{
		printf("original %f %f %f\n,",m_arrowHead3D[1].x,m_arrowHead3D[1].y,m_arrowHead3D[1].z);
		cv::Point3d arrow2 = m_arrowHead3D[1] -m_pt3D;
		double dist = norm(arrow2);
		arrow2.x *=1.2;
		//arrow2.y *=1.3;
		arrow2.z *=1.5;
		Normalize(arrow2);


		m_arrowHead3D[1] = m_pt3D+arrow2*dist;
		printf("noisy %f %f %f\n,",m_arrowHead3D[1].x,m_arrowHead3D[1].y,m_arrowHead3D[1].z);
	}

		
	void SetIsVisibleInCamOrder_ByVisibleCamVect(size_t seqNum)
	{
		m_isVisibleInCamOrder.clear();
		for(unsigned int i=0;i<seqNum;++i)
		{
			m_isVisibleInCamOrder.push_back(false);
		}
		for(unsigned int i=0;i<m_visibleCamIdxVector.size();++i)
		{
			int seqIdx = m_visibleCamIdxVector[i];
			m_isVisibleInCamOrder[seqIdx] = true;
		}
	}

	//To save Memeory space
	void ClearNotImportantHistory()
	{
		m_cameraInfos.clear();
		m_isVisibleInCamOrder.clear();
		//m_visibleCamIdxVector.clear();
	}

};

class VisibleElement
{
public:
	VisibleElement()
	{
		camIdx= keyPtIdx = -1;
	}
	VisibleElement(int c,int k)
	{
		camIdx= c;
		keyPtIdx = k;
	}
	bool operator<(const VisibleElement& t)
	{
		return camIdx<t.camIdx;
	}
	int camIdx;		//means camera index....need to change the name (originated from SfM)
	int keyPtIdx;		
	
	cv::Point2f originalImagePt;  //mainly used for bundle adjustment
};

//This had patch representation at a time instance (not patch's trajectory)
//and only contains essentional data for that.
//Note that TrajElement3D is super set of this.
class SimplePatchElement3D
{
public:
	SimplePatchElement3D()
	{
		m_color = cv::Point3d(1,1,1);
		m_normal = cv::Point3d(0,0,0);
		bValid = true;		//used to eliminated some "not perfect" points 
		m_associatedViewNum =0;

#ifndef LINUX_COMPILE
		//random color generation for display
		m_randomColor.z = int(rand()%256)/255.0;	//blue
		m_randomColor.y = int(rand()%256)/255.0;	//green
		m_randomColor.x = int(rand()%256)/255.0;	//red
#endif
	}

	SimplePatchElement3D(cv::Mat& X)
	{
		if(X.rows==4)
			X.copyTo(m_ptMat4by1);
		else if(X.rows==3)
		{
			m_ptMat4by1 = cv::Mat::ones(4,1,CV_64F);
			X.copyTo(m_ptMat4by1.rowRange(0,3));
		}
		double* coord = X.ptr<double>(0);
		m_pt.x = coord[0];
		m_pt.y = coord[1];
		m_pt.z = coord[2];

		SimplePatchElement3D();
	}

	SimplePatchElement3D(cv::Point3d& paraX)
	{
		m_pt = paraX;
		m_ptMat4by1 = cv::Mat::ones(4,1,CV_64F);
		m_ptMat4by1.at<double>(0,0) = m_pt.x;
		m_ptMat4by1.at<double>(1,0) = m_pt.y;
		m_ptMat4by1.at<double>(2,0) = m_pt.z;

		SimplePatchElement3D();
	}	

	bool bValid;  //if it is false, then it should be deleted
	//Patch information
	cv::Mat m_ptMat4by1; // save as pt, but Mat type
	cv::Point3d m_pt;
	cv::Point3d m_normal;
	cv::Point3d m_color;  //for point cloud
	cv::Point3d m_randomColor;
	cv::Point3d m_arrow1stHead;		//arrow1 and arrow2 are not necessarily right angle...they are just the results of triangluation from correspoding 2D points
	cv::Point3d m_arrow2ndHead;
	cv::Mat m_patch;		//patch average Image
	cv::Mat_<double> m_patchGray;   //this is used for patch optimization. Just a gray image, but double type element for interpolation
	int m_associatedViewNum;
};


class PatchCloud
{
public:
	~PatchCloud()
	{
		for(int i=0;i<m_patchVector.size();++i)
			delete m_patchVector[i];
		m_patchVector.clear();
	}

	vector<SimplePatchElement3D*> m_patchVector;
	int m_frameIdx;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//Represent a camera view containing image, calibration data, and labels (camera and panel index).
class CamViewDT
{
public:
	CamViewDT(){
		m_alreadyUsed = false;
		m_bRegistered = false;

		m_actualImageFrameIdx = m_actualPanelIdx = m_actualCamIdx = -1;
		m_heightExpected = m_widthExpected =0;
		m_sensorType = SENSOR_TYPE_UNDEFINED;

		m_imSizeX = m_imSizeY =-1;

	};

	CamViewDT(int idx)
	{
		m_camIdx = idx;
		m_alreadyUsed = false;
		m_bRegistered = false;

		m_actualImageFrameIdx = m_actualPanelIdx = m_actualCamIdx = -1;
		m_actualImageFrameIdx = m_actualPanelIdx = m_actualCamIdx = -1;
		m_heightExpected = m_widthExpected =0;
		m_sensorType = SENSOR_TYPE_UNDEFINED;
	}
	
	//return rgb value for OpenGL display
	cv::Point3d getRGBValue(int x,int y)
	{
		cv::Point3d rgb;
		rgb.z = m_rgbInputImage.at<cv::Vec3b>(y,x)[0]/255.0; //blue
		rgb.y = m_rgbInputImage.at<cv::Vec3b>(y,x)[1]/255.0; //green
		rgb.x = m_rgbInputImage.at<cv::Vec3b>(y,x)[2]/255.0; //red
		return rgb;
	}
	void ShowSubregion(cv::Point2d pt,double scale)
	{
		int left = pt.x-scale;
		int right = pt.x+scale;
		int top = pt.y+scale;
		int bottom = pt.y-scale;
		if(bottom<0 || left <0 || right>=m_rgbInputImage.cols || top >=m_rgbInputImage.rows)
		{
			printf("ShowSubregion::out of region\n");
			return;
		}

		cv::Mat subRegion(m_rgbInputImage, cv::Range(bottom,top), cv::Range(left, right));
		cv::imshow("ShowSubregion",subRegion);
		cv::waitKey();
	}

	void UndistortImage(cv::Mat& originalImg,cv::Mat& idealImg);  //undistortion using parameter of this camera
	cv::Point_<double> ApplyUndistort(const cv::Point_<double>& pt_d); //input: image coord, output: ideal_image coord
	cv::Point_<double> ApplyDistort(const cv::Point_<double>& pt_d);   //input: ideal_image coord, output: image coord


	//Required: m_actualPanelIdx should be defined
	void InitBasicInfo()		//Initialize camera size (based on panelName)
	{
		if(m_actualPanelIdx ==-1)
		{
			printf("## ERROR:: InitBasicInfo:: m_actualPanelIdx should be defined first!!!\n");
			return;
		}
		if(m_actualPanelIdx==PANEL_HD)
		{
			m_heightExpected = 1080;
			m_widthExpected = 1920;
			m_sensorType = SENSOR_TYPE_HD;
		}
		else if(m_actualPanelIdx==PANEL_KINECT)
		{
			m_heightExpected = 1080;
			m_widthExpected = 1920;
			m_sensorType = SENSOR_TYPE_KINECT;
		}
		else if( m_actualPanelIdx  >=1 && m_actualPanelIdx <= 20)
		{
			m_heightExpected = 480;
			m_widthExpected = 640;
			m_sensorType = SENSOR_TYPE_VGA;
		}
		else
		{
			printf("## WARNING:: InitBasicInfo:: %d is undefined panelIdx\n",m_actualPanelIdx);
			return;
		}

		SettingRMatrixGL();		//for camera position visualization as pyramids
		SettingModelViewMatrixGL();		//for depth rendering and shader
	}
	void SettingRMatrixGL();		//to visualize in opengl 
	void SettingModelViewMatrixGL(); //to visualize in opengl 

	template <typename Type>
	cv::Point2d ProjectionPt(const cv::Point3_<Type>& pt,bool applyDistort=false)
	{
		cv::Point2d returnPt;
		cv::Mat_<double> ptMat;
		Point3xToMat4by1(pt,ptMat);
		ptMat = m_P * ptMat;
		ptMat = ptMat/ptMat.at<double>(2,0);

		returnPt.x = ptMat.at<double>(0,0);
		returnPt.y = ptMat.at<double>(1,0);

		if(applyDistort ==false)
		{
			return returnPt;
		}
		else
		{
			returnPt = ApplyDistort(returnPt);
			return returnPt;
		}

		//cannot be reached
		return returnPt;
	}
	cv::Point2d ProjectionPt(const cv::Mat_<double>& ptMat4by1,bool applyDistort=false)
	{
		cv::Point2d returnPt;
		cv::Mat_<double> ptMat = m_P * ptMat4by1;
		ptMat = ptMat/ptMat.at<double>(2,0);

		returnPt.x = ptMat.at<double>(0,0);
		returnPt.y = ptMat.at<double>(1,0);

		if(applyDistort ==false)
		{
			return returnPt;
		}
		else
		{
			returnPt = ApplyDistort(returnPt);
			return returnPt;
		}

		//cannot be reached
		return returnPt;
	}
	template <typename Type>
	bool GetProjectedPtRGBColor(const cv::Point3_<Type>& pt,cv::Point3f& returnColor,bool applyDistort=false)
	{
		if(m_rgbInputImage.rows==0)
			return false;

		cv::Point2d pt2D =  ProjectionPt(pt,applyDistort);
		if(::IsOutofBoundary(m_rgbInputImage,pt2D.x,pt2D.y))
			return false;
		cv::Mat_<cv::Vec3b> m(m_rgbInputImage);
		cv::Vec3d color = BilinearInterpolation(m,pt2D.x,pt2D.y);
		returnColor = cv::Point3f(color(2)/255.0,color(1)/255.0,color(0)/255.0);

		return true;
	}

	cv::Point3d GetPt3DFromZDepth(double imageX,double imageY,double zDepth)
	{
		cv::Mat_<double> pt = cv::Mat_<double>::ones(3,1);
		pt(0,0) = imageX;
		pt(1,0) = imageY;
		cv::Mat pt3DMat = m_invK*pt;
		pt3DMat = pt3DMat*zDepth;
		pt3DMat = m_invR * (pt3DMat - m_t );
		cv::Point3d pt3D  = MatToPoint3d(pt3DMat);
		return pt3D;
	}

	template <typename T>
	cv::Point3d GetPt3DCamCoord(cv::Point3_<T>& pt_w)
	{
		cv::Mat_<double> ptMat;
		Point3ToMatDouble(pt_w,ptMat);		//3x3 matrix
		ptMat = m_R * ptMat + m_t;

		cv::Point3d pt_c;
		pt_c.x = ptMat.at<double>(0,0);
		pt_c.y = ptMat.at<double>(1,0);
		pt_c.z = ptMat.at<double>(2,0);

		return pt_c;
	}

	cv::Point3d GetCamCenter()
	{
		cv::Point3d camCenter;
		MatToPoint3d(m_CamCenter,camCenter);
		return camCenter;
	}

	bool LoadIntrisicParameter(const char* calibFolderPath,int panelIdx,int camIdx);
	bool LoadPhysicalImage(EnumLoadMode loadingMode,int frameIdx,bool bIsHD,bool bUseSavedPath=true);		//if bUseSavedPath==false, make name using InputParameter.txt's info
	bool LoadPhysicalImage(EnumLoadMode loadingMode,bool bIsHD,bool bUseSavedPath=true);


	pair<int,int> GetImageSize()
	{
		if(m_imSizeX>0 && m_imSizeY>0)
		{
			return make_pair(m_imSizeX,m_imSizeY);
		}
		else
		{
			if(m_sensorType==SENSOR_TYPE_VGA)
			{
				m_imSizeX = 640;
				m_imSizeY = 480;
				return make_pair(640,480);
			}
			else if(m_sensorType==SENSOR_TYPE_HD)
			{
				m_imSizeX = 1920;
				m_imSizeY = 1080;
				return make_pair(1920,1080);
			}
			else if(m_sensorType==SENSOR_TYPE_KINECT)
			{
				m_imSizeX = 1920;
				m_imSizeY = 1080;
				return make_pair(1920,1080);
			}
			else 
			{
				printf("# WARNING: GetImageSize:: return make_pair(-1,-1) \n");
				return make_pair(-1,-1);		
			}
		}
		return make_pair(-1,-1);		
	}
	bool IsOutofBoundary(double x,double y)
	{
		pair<int,int> imSize = GetImageSize();
		if(x<0 || y<0 || x>=imSize.first ||y>=imSize.second)
			return true;
		else
			return false;
	}

	CamViewDT(CamViewDT const& other)
	{
		m_idealPts = other.m_idealPts;
		m_keypoints = other.m_keypoints;
		
		m_keypoint1stArrow = other.m_keypoint1stArrow; //arrowHead
		m_keypoint2ndArrow = other.m_keypoint2ndArrow; //2nd principal arrow head
		m_descriptors = other.m_descriptors;
		m_usedKeyFor3DFlag = other.m_usedKeyFor3DFlag;
		m_matchingUnitIdx = other.m_matchingUnitIdx;

		m_inputImage = other.m_inputImage;  //gray image
		m_rgbInputImage = other.m_rgbInputImage;	//rgb image
		m_debugImage = other.m_debugImage;  //for debug
		m_smallIRgbImage = other.m_smallIRgbImage;

		m_P = other.m_P;  //camera matrix
		m_K = other.m_K;
		m_invK = other.m_invK;
		m_R = other.m_R;
		m_invR =other.m_invR;
		m_t = other.m_t;
		m_opticalAxis = other.m_opticalAxis;  //initialized in CamParamSetting()

		m_R_Quat =other.m_R_Quat;
		m_CamCenter =other.m_CamCenter; //-invR*T		//3x1 matrix

		memcpy(m_RMatrixGL,other.m_RMatrixGL,sizeof(float)*16);		//m_RMatrixGL[16];
		memcpy(m_modelViewMatGL,other.m_modelViewMatGL,sizeof(float)*16);		//m_RMatrixGL[16];
		memcpy(m_projMatGL,other.m_projMatGL,sizeof(int)*4);		//m_RMatrixGL[16];
		memcpy(m_mvpMatGL,other.m_mvpMatGL,sizeof(float)*16);		//m_RMatrixGL[16];


		m_alreadyUsed = other.m_alreadyUsed ;
		m_camIdx = other.m_camIdx;	//usedSequence idx
		m_sequenceIdx =other.m_sequenceIdx; //idx for m_sequence[], only used for pereforming Sfm, not for loaded data
		m_associatedPtIdx =other.m_associatedPtIdx;
		
		m_actualImageFrameIdx =other.m_actualImageFrameIdx;   //actual image file Idx
		m_actualPanelIdx =other.m_actualPanelIdx;   //Only for dome system
		m_actualCamIdx =other.m_actualCamIdx;   //Only for dome system
		m_fullPath = other.m_fullPath;  //image full path
		m_fileName =other.m_fileName;  //image full path

		memcpy(m_camParamForBundle,other.m_camParamForBundle, sizeof(double)*9);//m_camParamForBundle[9];  //used for bundler. meaningless outside of bundler function
		m_distortionParams =other.m_distortionParams;			//if m_distortionParams.size()==0, there is no lense distortions
		m_bRegistered =other.m_bRegistered;  //false, if failed in camera pose registering

		m_heightExpected = other.m_heightExpected;
		m_widthExpected = other.m_widthExpected;
		m_sensorType = other.m_sensorType;
	};

	//SIFT-related information
	std::vector<cv::Point2d> m_idealPts;		//Ideal key point position, after undistortion.
	std::vector<cv::KeyPoint> m_keypoints;		//Ideal key point position, after undistortion.
	//std::vector<Point2f> m_originalKeyPt; //before undistorted. Use this to extract to extract pixel value or bundling adjustments
	std::vector<cv::Point2d> m_keypoint1stArrow; //arrowHead
	std::vector<cv::Point2d> m_keypoint2ndArrow; //2nd principal arrow head
	cv::Mat m_descriptors;
	vector<bool> m_usedKeyFor3DFlag;
	vector<int> m_matchingUnitIdx;


	cv::Mat m_inputImage;  //gray image
	cv::Mat m_rgbInputImage;	//rgb image
	cv::Mat m_debugImage;  //for debug
	cv::Mat m_smallIRgbImage;

	//If you load from files, some of below might not be initialized.
	cv::Mat_<double> m_P;  //camera matrix
	cv::Mat m_K;
	cv::Mat m_invK;
	cv::Mat m_R;
	cv::Mat m_invR;
	cv::Mat m_t;
	cv::Mat m_opticalAxis;  //initialized in CamParamSetting()

	cv::Mat m_R_Quat;
	cv::Mat m_CamCenter; //-invR*T		//3x1 matrix
	GLfloat m_RMatrixGL[16];


	//The following is used to render the scene into camera view
	GLfloat m_modelViewMatGL[16];		//World to Camera coordinate. To render cam view
	GLfloat m_projMatGL[16];		//World to Camera coordinate. To render cam view
	GLfloat m_perspectGL[16];		//World to Camera coordinate. To render cam view
	GLfloat m_mvpMatGL[16];		//World to Camera coordinate. To render cam view

	bool m_alreadyUsed;
	int m_camIdx;	//usedSequence idx
	int m_sequenceIdx; //idx for m_sequence[], only used for pereforming Sfm, not for loaded data
	vector<int> m_associatedPtIdx;
		
	int m_actualImageFrameIdx;   //actual image file Idx
	int m_actualPanelIdx;   //Only for dome system
	int m_actualCamIdx;   //Only for dome system
	string m_fullPath;  //image full path
	string m_fileName;  //image full path

	//for bundleAdjustment
	//important!!
	//m_camParamForBundle[7]and [8]: it is the undistortion parameter, which is for x_d -> x_u, it should be positive number;
	double m_camParamForBundle[9];  //used for bundler. meaningless outside of bundler function
	//double m_camParamForBundle[12];  //used for bundler. meaningless outside of bundler function
	vector<double> m_distortionParams;			//if m_distortionParams.size()==0, there is no lense distortions
	
	bool m_bRegistered;  //false, if failed in camera pose registering

	//The following is basic info set by InitBasicInfo()
	int m_heightExpected;			//Expected means the size defined by panel index (HD, Kinect should be 1920x1080, VGA should be 640 480)
	int m_widthExpected;
	int m_sensorType;		//SENSOR_TYPE_VGA, SENSOR_TYPE_HD, SENSOR_TYPE_KINECT

private:
	int m_imSizeX;		//always should be called by function to initialize if this value is not available
	int m_imSizeY;

	cv::Point_<double> ApplyUndistort_visSfM(const cv::Point_<double>& pt_d); //input: image coord, output: ideal_image coord
	cv::Point_<double> ApplyDistort_visSfM(const cv::Point_<double>& pt_d);   //input: ideal_image coord, output: image coord
	cv::Point_<double> ApplyUndistort_openCV(const cv::Point_<double>& pt_d); //input: image coord, output: ideal_image coord
	cv::Point_<double> ApplyDistort_openCV(const cv::Point_<double>& pt_d);   //input: ideal_image coord, output: image coord
};

//if bUseSavedPath==false, make name using camera information (frameIdx, panelIdx, camIdx) and the g_dataImageFolder in InputParameter.txt
//DEPRECIATED, it cannot handle HD and VGA automatically
bool LoadPhysicalImageForAllCams(vector<CamViewDT*>& camVector,EnumLoadMode loadingMode,bool bIsHD=false,bool bUseSavedPath=true);	
bool LoadPhysicalImageForAllCams(vector<CamViewDT*>& camVector,EnumLoadMode loadingMode,int frameIdx,bool bIsHD=false,bool bUseSavedPath=true);	



class Coord3D
{
public:
	Coord3D()
	{
		x= 0;
		y= 0;
		z= 0;
	}
	Coord3D(double xx,double yy,double zz)
	{
		x = xx;
		y = yy;
		z = zz;
	}
	double x,y,z;
};

struct CamVisInfo
{
	CamVisInfo()
	{
		//m_isHDCamera = false;
		m_cameraType = SENSOR_TYPE_VGA;
	}

	//m_RMatrixGL is used for camera visualization
	void SettingRMatrixGL(cv::Mat R)
	{
		cv::Mat invR = R.inv();
		//colwise
		m_RMatrixGL[0] = invR.at<double>(0,0);
		m_RMatrixGL[1] = invR.at<double>(1,0);
		m_RMatrixGL[2] = invR.at<double>(2,0);
		m_RMatrixGL[3] = 0;
		m_RMatrixGL[4] = invR.at<double>(0,1);
		m_RMatrixGL[5] = invR.at<double>(1,1);
		m_RMatrixGL[6] = invR.at<double>(2,1);
		m_RMatrixGL[7] = 0;
		m_RMatrixGL[8] = invR.at<double>(0,2);
		m_RMatrixGL[9] = invR.at<double>(1,2);
		m_RMatrixGL[10] = invR.at<double>(2,2);
		m_RMatrixGL[11] = 0;
		m_RMatrixGL[12] = 0; //4th col
		m_RMatrixGL[13] = 0;
		m_RMatrixGL[14] = 0;
		m_RMatrixGL[15] = 1;
	}

	double m_CamCenter[3]; 
	GLfloat m_RMatrixGL[16];		//To visualize cam cetners

	GLfloat m_modelViewMat[16];		//To render in camera view
	GLfloat m_modelViewMat_inv[16];		//To render in camera view
	GLfloat m_projMat[16];		//To render in camera view
	GLfloat m_mvpMat[16];		//All projection 


	char m_camNameString[64];
	//bool m_isHDCamera;
	int m_cameraType;

	//lookAtVector
	double m_eyeVec[3];
	double m_atVec[3];
	double m_upVec[3];
	cv::Mat m_K;
};


//save every matching results between each pair
class MatchingManager
{
public:
	MatchingManager()
	{
		//m_matching= NULL;
	}

	~MatchingManager()
	{
		for(int i=0;i<m_matching.size();++i)
		{
			m_matching.clear();
		}
		m_totalMatchingPairNum =0;
	}

	void Clear()
	{
		for(int i=0;i<m_matching.size();++i)
		{
			m_matching.clear();
		}
		m_totalMatchingPairNum =0;
	}


	void ClearOneMatching(int im1_idx,int im2_idx)
	{
		m_matching[im1_idx][im2_idx].clear();
		m_matching[im2_idx][im1_idx].clear();
	}

	void Initilize(int machingPair)
	{
		for(int i=0;i<m_matching.size();++i)
		{
			m_matching.clear();
		}
		m_matching.clear();

		m_totalMatchingPairNum = machingPair;
		m_matching.resize(m_totalMatchingPairNum);
		for(int i=0;i<m_totalMatchingPairNum;++i)
		{
			m_matching[i].resize(m_totalMatchingPairNum);
		}
	}

	void PushMatching(int im1_idx,int im1_feauterIdx,int im2_idx,int im2_feauterIdx)
	{
		m_matching[im1_idx][im2_idx].push_back(make_pair(im1_feauterIdx,im2_feauterIdx));
		m_matching[im2_idx][im1_idx].push_back(make_pair(im2_feauterIdx,im1_feauterIdx));  //looks redundant, but easy to use regardless im1, im2 order
	}

	pair<int,int> GetMatching(int im1_idx,int im2_idx,int matching_idx)
	{
		return m_matching[im1_idx][im2_idx][matching_idx];  //same as m_matching[im2_idx][im1_idx][matching_idx];
	}

	unsigned int GetMatchingNum(int im1_idx,int im2_idx)
	{
		return (unsigned int) m_matching[im1_idx][im2_idx].size();
	}
	vector< pair<int,int> >& GetMatchingVector(int im1_idx,int im2_idx)
	{
		return m_matching[im1_idx][im2_idx];
	};


	//return false if there is no previous matching
	//dome with HD camera (can be appliable for general puporse. by using 0 as the panel index for general cameras
	//newCams contains the cameras which have to be matched again. 
	bool LoadPossibleMatchingResultDome(vector<CamViewDT*>& sequence,const char* dataMainFolder,int frameNum,vector<int>& newCams)
	{
		char fullPath[256];
		//sprintf(fullPath,"%s/../matching/matching_%d.txt",dirPath,frameNum);
		sprintf(fullPath,"%s/matching/matching_c%d/matching_%d.txt",dataMainFolder,sequence.size(),frameNum);
		ifstream fin(fullPath);
		if(!fin)
		{
			return false;
		}
		printfLog("Loading matching file start\n");
		int imageNum;
		fin >>imageNum;

		if(imageNum < sequence.size())	//make a backup
		{
			//make a backup
			char backupPath[256];
			sprintf(backupPath,"%s/matching/matching_%d_camNum%d.txt",dataMainFolder,frameNum,imageNum);
			std::ifstream srce( fullPath, std::ios::binary ) ;
			std::ofstream dest( backupPath, std::ios::binary ) ;
			dest << srce.rdbuf() ;
			srce.close();
			srce.close();
		}

		//Initilize(imageNum);
		Initilize((int)sequence.size());  //have to be same as sequence.size()
		vector<bool> loadedCams;
		loadedCams.resize(sequence.size(),false);

		while(fin.eof() == false)
		{
			int panel_1,cam_1,panel_2,cam_2;
			fin >> panel_1 >> cam_1 >>panel_2>>cam_2;

			//search target image index
			int camIdx_1=-1;
			int camIdx_2=-1;
			for(unsigned int camIdx=0;camIdx<sequence.size();++camIdx)
			{
				if(sequence[camIdx]->m_actualPanelIdx == panel_1 && sequence[camIdx]->m_actualCamIdx == cam_1)
				{
					camIdx_1 = camIdx;
				}
				else if(sequence[camIdx]->m_actualPanelIdx == panel_2 && sequence[camIdx]->m_actualCamIdx == cam_2)
				{
					camIdx_2 = camIdx;
				}

				if(camIdx_1>=0 && camIdx_2>=0)		//find bothIndex
					break;
			}
			
			int num;
			fin >> num;
			for(int k=0;k<num;++k)
			{
				pair<int,int> tempPair;
				fin >> tempPair.first >> tempPair.second;
				if(camIdx_1>=0 && camIdx_2>=0)
				{
					m_matching[camIdx_1][camIdx_2].push_back(tempPair);
					m_matching[camIdx_2][camIdx_1].push_back(make_pair(tempPair.second,tempPair.first));  //could be meaningless
				}
			}

			//check
			if(camIdx_1>=0 && camIdx_2>=0)
			{
				loadedCams[camIdx_1] = true;
				loadedCams[camIdx_2] = true;
			}
		}
		fin.close();

		//if there exist missing matings
		if(sequence.size() > imageNum)
		{
			for(unsigned int camIdx=0;camIdx<sequence.size();++camIdx)
			{
				if(loadedCams[camIdx]==false)
					newCams.push_back(camIdx);
			}
			
		}
		printfLog("Loading matching file has been finished\n");
		
		return true;		//loaded at least one matching
	}
	
	//dome with HD camera (can be appliable for general puporse. by using 0 as the panel index for general cameras
	//newCams contains the cameras which have to be matched again. 
	bool LoadMatchingResultDome(vector<CamViewDT*>& sequence,char* dirPath,int frameNum)
	{
		char fullPath[256];
		sprintf(fullPath,"%s/../matching/matching_%d.txt",dirPath,frameNum);
		ifstream fin(fullPath);
		if(!fin)
		{
			return false;
		}
		printfLog("Loading matching file start\n");
		int imageNum;
		fin >>imageNum;

		if(imageNum < sequence.size())	//make a backup
		{
			fin.close();
			//make a backup
			char backupPath[256];
			sprintf(backupPath,"%s/../matching/matching_%d_camNum%d.txt",dirPath,frameNum,imageNum);
			std::ifstream srce( fullPath, std::ios::binary ) ;
			std::ofstream dest( backupPath, std::ios::binary ) ;
			dest << srce.rdbuf() ;
			srce.close();
			srce.close();
			return false;
		}

		//Initilize(imageNum);
		Initilize((int)sequence.size());  //have to be same as sequence.size()
		vector<bool> loadedCams;
		loadedCams.resize(sequence.size(),false);

		while(fin.eof() == false)
		{
			int panel_1,cam_1,panel_2,cam_2;
			fin >> panel_1 >> cam_1 >>panel_2>>cam_2;


			//search target image index
			int camIdx_1=-1;
			int camIdx_2=-1;
			for(unsigned int camIdx=0;camIdx<sequence.size();++camIdx)
			{
				if(sequence[camIdx]->m_actualPanelIdx == panel_1 && sequence[camIdx]->m_actualCamIdx == cam_1)
				{
					camIdx_1 = camIdx;
				}
				else if(sequence[camIdx]->m_actualPanelIdx == panel_2 && sequence[camIdx]->m_actualCamIdx == cam_2)
				{
					camIdx_2 = camIdx;
				}

				if(camIdx_1>=0 && camIdx_2>=0)		//find bothIndex
					break;
			}

			int num;
			fin >> num;
			for(int k=0;k<num;++k)
			{
				pair<int,int> tempPair;
				fin >> tempPair.first >> tempPair.second;
				if(camIdx_1>=0 && camIdx_2>=0)
				{
					m_matching[camIdx_1][camIdx_2].push_back(tempPair);
					m_matching[camIdx_2][camIdx_1].push_back(make_pair(tempPair.second,tempPair.first));  //could be meaningless
				}
			}
		}
		fin.close();

		printfLog("Loading matching file has been finished\n");
		return true;
	}


	void SaveMatchingResultDome(std::vector<CamViewDT*>& sequence,const char* saveDirPath,int frameNum)
	{
		char matchingFolderPath[512];
		sprintf(matchingFolderPath,"%s/matching",saveDirPath);
		CreateFolder(matchingFolderPath);
		sprintf(matchingFolderPath,"%s/matching_c%d",matchingFolderPath,sequence.size());
		CreateFolder(matchingFolderPath);
		char fullPath[256];
		sprintf(fullPath,"%s/matching_%d.txt",matchingFolderPath,frameNum);
		ofstream fout(fullPath,std::ios_base::trunc);
		fout <<m_totalMatchingPairNum <<"\n";

		for(int i=0;i<m_totalMatchingPairNum;++i)
		{
			for(int j=i+1;j<m_totalMatchingPairNum;++j)
			{
				fout << sequence[i]->m_actualPanelIdx << " " << sequence[i]->m_actualCamIdx <<" ";
				fout << sequence[j]->m_actualPanelIdx << " " << sequence[j]->m_actualCamIdx <<" ";  //print cameralabels for this matching
				fout << m_matching[i][j].size() <<"\n";
				for(unsigned int k=0;k<m_matching[i][j].size();++k)
				{
					fout << m_matching[i][j][k].first <<" "<<m_matching[i][j][k].second<<" ";
				}
				fout <<"\n";
			}
		}
		fout.close();
	}
	


	//only for dome
	bool LoadMatchingResultMonoCam(vector<CamViewDT*>& sequence,char* dirPath)
	{
		int firstFrameNum = sequence[0]->m_actualImageFrameIdx;

		char fullPath[256];
		sprintf(fullPath,"%s/matching_c%d.txt",dirPath,firstFrameNum);
		ifstream fin(fullPath);
		if(!fin)
			return false;
		char buf[256];
		fin >> buf; //stepSize

		int stepSize;
		fin >>stepSize;
		if(stepSize!=INPUT_IMAGE_STEP)
		{
			printfLog("stepSizeDismatch: stepSize in the file is %d, compared to current setting %d\n",stepSize,INPUT_IMAGE_STEP);
#ifndef LINUX_COMPILE
			AfxMessageBox("stepSizeDismatch \n");
#endif
			return false;
		}

		int imageNum;
		fin >>imageNum;
		//Initilize(imageNum);
		Initilize((int)sequence.size());  //have to be same as sequence.size()


		//Load
		for(int i=0;i<m_totalMatchingPairNum;++i)
		{
			for(int j=i+1;j<m_totalMatchingPairNum;++j)
			{
				int frame_1,frame_2;
				fin >> frame_1 >> frame_2;

				int num;
				fin >> num;
				for(int k=0;k<num;++k)
				{
					pair<int,int> tempPair;
					fin >> tempPair.first >> tempPair.second;
					m_matching[i][j].push_back(tempPair);
					m_matching[j][i].push_back(make_pair(tempPair.second,tempPair.first));  //could be meaningless
				}
			}
		}
		return true;
	}

	//monocularCam SFM
	void SaveMatchingResultMonoCam(std::vector<CamViewDT*>& sequence,char* dirPath)
	{
		char fullPath[256];
		sprintf(fullPath,"%s/matching_%d.txt",dirPath,sequence[0]->m_actualImageFrameIdx);  //startFrameIdx
		ofstream fout(fullPath,std::ios_base::trunc);
		fout << "stepSize " << INPUT_IMAGE_STEP <<"\n";
		fout <<m_totalMatchingPairNum <<"\n";

		for(int i=0;i<m_totalMatchingPairNum;++i)
		{
			for(int j=i+1;j<m_totalMatchingPairNum;++j)
			{
				fout << sequence[i]->m_actualImageFrameIdx <<" ";
				fout << sequence[j]->m_actualImageFrameIdx <<" ";  //print cameralabels for this matching
				fout << m_matching[i][j].size() <<"\n";
				for(unsigned int k=0;k<m_matching[i][j].size();++k)
				{
					fout << m_matching[i][j][k].first <<" "<<m_matching[i][j][k].second<<" ";
				}
				fout <<"\n";
			}
		}
		fout.close();
	}

private:
	int m_totalMatchingPairNum;   //totalMatchingPair would be greater than actual ImageNum. 
	//int m_usedImageNum;  //top K image which is used, used to calculate the matchingMerge data structure
	//vector< pair<int,int> > **m_matching;
	vector< vector< vector< pair<int,int> > > > m_matching;  //  matching between image_i,image_j ==> m_matchingStructure[i][j][N] = m_matchingStructure[j][i][N] 
};



class CEdgeElement
{
public:
	CEdgeElement()
	{
		cam1 = NULL;
		cam2 = NULL;
		
		centerDistance = 1e10;
	}
	CamViewDT* cam1;
	CamViewDT* cam2;

	double centerDistance;
	double nccScore;
	double howSimilar;
};

bool operator<(const CEdgeElement& a, const CEdgeElement& b);

class DataTermRatio
{
public:
	DataTermRatio(float m,float a,float n)
	{
		motion = m;
		normal =n;
		appearance = a;
		threshold = -1;
	}

	DataTermRatio(float m,float a,float n,float t)
	{
		motion = m;
		normal =n;
		appearance = a;
		threshold = t;
	}

	float motion;
	float normal;
	float appearance;
	float threshold;
};

struct SurfVoxelUnit
{
	SurfVoxelUnit()
	{
		color = cv::Point3f(1,1,1);
		prob=0.0f;
		marginalProb =0.0f;
		label =-1;
	}

	SurfVoxelUnit(cv::Point3f& p)
	{
		pos = p;
		color = cv::Point3f(1,1,1);

		prob=0.0f;
		marginalProb =0.0f;
		label =-1;
	}
	SurfVoxelUnit(const cv::Point3f& p,const cv::Point3f& c)
	{
		pos = p;
		color = c;

		prob=0.0f;
		marginalProb =0.0f;
		label =-1;
	}
	cv::Point3f pos;
	cv::Point3f color;
	vector< pair<int, double> > visibleCams;
	//vector<double> visibleCamDistance;
	int voxelIdx;
	cv::Point3f normal;

	int label;	//segmentation			//to check already wheter taken in 3dPS. If take, label>=0, and the label means skeleton index
	float prob;	//general purpose probablity
	float marginalProb;	//after inference, assuming this as a root
};

class PickedPtInfo
{
public:
	PickedPtInfo()
	{
		m_bUseAssignedColor = false;
		m_color = g_cyan_p3f;		//default color
	}
	PickedPtInfo(int m,int i)
		:memIdx(m),ptIdx(i)
	{
		m_bUseAssignedColor = false;
		m_color = g_cyan_p3f;		//default color
	}

	 bool operator<( const PickedPtInfo& Ref) const { 
		 if(memIdx !=Ref.memIdx)
    		return memIdx < Ref.memIdx; 
		 else
    		return ptIdx < Ref.ptIdx; 

    }

	bool operator == (const PickedPtInfo &Ref) const 
	{
		return (memIdx== Ref.memIdx && ptIdx==Ref.ptIdx);
	}

	int memIdx;
	int ptIdx;


	bool m_bUseAssignedColor;	//if true, use the assigned color for visualization  (used for visualizing cost fot the selected trajectory);
	cv::Point3f m_color;
};


//Scalelton Generation
//#define CURRENT_JOINT_NUM 13
//#define CURRENT_JOINT_NUM 7		//only upper body
extern int CURRENT_JOINT_NUM;
extern pair<int,int> g_selectedJoint;
struct SJointTraj
{
	vector<cv::Point3f> m_jointTrajectory;
	//int m_initMemIdx;
	int m_initFrameIdx;		//This is the actual frame idx for the sequence, not memVectIdx
};

class Bbox3D
{
public:
	Bbox3D()
	{
		minXyz = cv::Point3f(1e10,1e10,1e10);
		maxXyz = cv::Point3f(-1e10,-1e10,-1e10);
	}
	void IncludeThis(cv::Point3f pt)
	{
		minXyz.x = (minXyz.x > pt.x) ? pt.x:minXyz.x;
		minXyz.y = (minXyz.y > pt.y) ? pt.y:minXyz.y;
		minXyz.z = (minXyz.z > pt.z) ? pt.z:minXyz.z;

		maxXyz.x = (maxXyz.x < pt.x) ? pt.x:maxXyz.x;
		maxXyz.y = (maxXyz.y < pt.y) ? pt.y:maxXyz.y;
		maxXyz.z = (maxXyz.z < pt.z) ? pt.z:maxXyz.z;
	}
	bool IsIntersect(Bbox3D& otherBox, float offset)
	{
		for(int i=0;i<otherBox.corners.size();++i)
		{
			if( minXyz.x -offset<= otherBox.corners[i].x && otherBox.corners[i].x <= maxXyz.x+offset
				&& minXyz.y-offset <= otherBox.corners[i].y && otherBox.corners[i].y <= maxXyz.y+offset
				&& minXyz.z-offset <= otherBox.corners[i].z && otherBox.corners[i].z <= maxXyz.z+offset)
				return true;
		}
		if( minXyz.x -offset<= otherBox.centerPt.x && otherBox.centerPt.x <= maxXyz.x+offset
			&& minXyz.y-offset <= otherBox.centerPt.y && otherBox.centerPt.y <= maxXyz.y+offset
			&& minXyz.z -offset<= otherBox.centerPt.z && otherBox.centerPt.z <= maxXyz.z+offset)
			return true;

		for(int i=0;i<corners.size();++i)
		{
			if( otherBox.minXyz.x-offset <= corners[i].x && corners[i].x <= otherBox.maxXyz.x+offset
				&& otherBox.minXyz.y -offset<= corners[i].y && corners[i].y <= otherBox.maxXyz.y+offset
				&& otherBox.minXyz.z -offset<= corners[i].z && corners[i].z <= otherBox.maxXyz.z+offset)
				return true;
		}

		if( otherBox.minXyz.x-offset <= centerPt.x && centerPt.x <= otherBox.maxXyz.x+offset
			&& otherBox.minXyz.y -offset<= centerPt.y && centerPt.y <= otherBox.maxXyz.y+offset
			&& otherBox.minXyz.z -offset<= centerPt.z && centerPt.z <= otherBox.maxXyz.z+offset)
			return true;

		return false;

	}
	void Finalize()
	{
		corners.push_back(cv::Point3f(minXyz.x,minXyz.y,minXyz.z));
		corners.push_back(cv::Point3f(minXyz.x,maxXyz.y,minXyz.z));
		corners.push_back(cv::Point3f(maxXyz.x,maxXyz.y,minXyz.z));
		corners.push_back(cv::Point3f(maxXyz.x,minXyz.y,minXyz.z));

		corners.push_back(cv::Point3f(minXyz.x,minXyz.y,maxXyz.z));
		corners.push_back(cv::Point3f(minXyz.x,maxXyz.y,maxXyz.z));
		corners.push_back(cv::Point3f(maxXyz.x,maxXyz.y,maxXyz.z));
		corners.push_back(cv::Point3f(maxXyz.x,minXyz.y,maxXyz.z));

		centerPt = (minXyz + maxXyz)*0.5;
		//Draw order
		//0-1 1-2 2-3 3-0 4-5 5-6 6-7 7-4
		//0-4 1-5 2-6 3-7 

		maxDistFromCenterToCorner = Distance(maxXyz,centerPt);
	}

	void GetDrawOrder(vector<int>& order)
	{
		order.push_back(0);
		order.push_back(1);
		order.push_back(1);
		order.push_back(2);
		order.push_back(2);
		order.push_back(3);
		order.push_back(3);
		order.push_back(0);

		order.push_back(4);
		order.push_back(5);
		order.push_back(5);
		order.push_back(6);
		order.push_back(6);
		order.push_back(7);
		order.push_back(7);
		order.push_back(4);

		order.push_back(0);
		order.push_back(4);
		order.push_back(1);
		order.push_back(5);
		order.push_back(2);
		order.push_back(6);
		order.push_back(3);
		order.push_back(7);
	}

	vector<cv::Point3d> corners;
	cv::Point3f centerPt;
	float maxDistFromCenterToCorner;

private:

	cv::Point3f minXyz;		//left up top
	cv::Point3f maxXyz;		//right down bottom
};



//For visualization with alpha
struct SColorWithAlpha
{
	SColorWithAlpha()
	{
		alpha =1;
	}
	SColorWithAlpha(cv::Point3f & c)
	{
		color = c;
		alpha =1;
	}
	SColorWithAlpha(cv::Point3f & c,float a)
	{
		color = c;
		alpha =a;
	}

	cv::Point3f color;
	float alpha;
};

//To draw related trajecoty. 
///There is similar version for vector in SFM.cpp
//Why here, not SFM.cpp? to be used at BodyPoseRecon
//void GetPointPosFromTrajSet(int currrentFrameIdx, int veryFirstImgIdxOfTrajMemory, bool bShowOnlyForwardTraj,std::set<TrajElement3D*>& trajectorySet, vector<Point3f>& outputPtCloud, vector<TrajElement3D*>& outputTrajPointers,bool bEnforceAllOutput);


struct SFrameIdx
{
	SFrameIdx()
	{
		fpstype = FPS_VGA_25;
	}

	SFrameIdx(FPStype t,int f)
	{
		fpstype = t;
		frameIdx = f;
	}
	FPStype fpstype;
	int frameIdx;
};

//The following is the space to communicate between application and opengl rendere
struct SRenderPacket
{
	SRenderPacket()
	{
		m_bFlag = false;
		m_in_camVectPtr = NULL;
		
	}
	bool m_bFlag;
	vector<cv::Point3f> m_in_ptCloud;
	vector<CamViewDT*>* m_in_camVectPtr;

	//output
	vector<cv::Mat_<float> > m_output_depthMapVect;
};


//Used for simple TXT load and pt recon
struct SCorres2D
{
	vector<cv::Point2d> ptVect;
	vector<int> camIdxVect;
};
extern SRenderPacket g_depthRenderPacket;
