#pragma once

//#include "DataStructures.h"
#include "TrajElement3D.h"
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <ceres/ceres.h>
#include <ceres/rotation.h>




/////////////////////////////////////////////////////////////////////////////////////////////////////////
//The follwing is for 3D point cloud reconstruction and SfM
double BundleAdjustment_ceres(vector<CamViewDT*>& sequences,vector<TrajElement3D*>& Pts3D);
extern double TriangulationOptimizationF(vector<cv::Mat*>& M,vector<cv::Point2f>& p,cv::Mat& X);			//why inlines???
extern double TriangulationOptimization(vector<cv::Mat*>& M,vector<cv::Point2d>& p,cv::Mat& X);
extern double TriangulationOptimizationWithWeight(vector<cv::Mat*>& M,vector<cv::Point2d>& p,vector<double>& weights,cv::Mat& X);

//Nonlinear Optimization for 3D Triangulation
struct ReprojectionErrorForTriangulation
{
	ReprojectionErrorForTriangulation(double x,double y,double* param)
	{
		observed_x = x;
		observed_y = y;
		memcpy(camParam,param,sizeof(double)*12);
	}
	
	/*template <typename T>
	bool operator()(const T* const pt,
					const T* const dummy,
					T* residuals) const ;*/
	
	template <typename T>
	bool operator()(const T* const pt,
					T* residuals) const ;

	inline virtual bool Evaluate(double const* const* pt,
		double* residuals,
		double** jacobians) const;
					
	double observed_x;
	double observed_y;
	double camParam[12];
};

//Nonlinear Optimization for 3D Triangulation
struct ReprojectionErrorForTriangulation_weighted
{
	ReprojectionErrorForTriangulation_weighted(double x,double y,double w,double* param)
	{
		observed_x = x;
		observed_y = y;
		observed_weight = w;
		memcpy(camParam,param,sizeof(double)*12);
	}
	
	/*template <typename T>
	bool operator()(const T* const pt,
					const T* const dummy,
					T* residuals) const ;*/
	
	template <typename T>
	bool operator()(const T* const pt,
					T* residuals) const ;
	/*
	inline virtual bool Evaluate(double const* const* pt,
		double* residuals,
		double** jacobians) const;
		*/			
	double observed_x;
	double observed_y;
	double observed_weight;
	double camParam[12];
};


#ifdef BUNDLE_EVERYTHING
struct SnavelyReprojectionError {
	SnavelyReprojectionError(double observed_x, double observed_y,double f,bool bFirstCam,double* initParam)
		: observed_x(observed_x), observed_y(observed_y),m_focus(f) {
		m_bFirstCam = bFirstCam;
		memcpy(m_initParam,initParam,sizeof(double)*9);
	}
	
	template <typename T>
		bool operator()(const T* const camera,
		const T* const point,
		T* residuals) const {
			// camera[0,1,2] are the angle-axis rotation.
			T p[3];
			if(m_bFirstCam)
			{
				T tempR[3];
				tempR[0] = T(m_initParam[0]); tempR[1] = T(m_initParam[1]); tempR[2] = T(m_initParam[2]);
				ceres::AngleAxisRotatePoint(tempR, point, p);
			}
			else
				ceres::AngleAxisRotatePoint(camera, point, p);

			// camera[3,4,5] are the translation.
			if(m_bFirstCam)
			{
				p[0] += T(m_initParam[3]); p[1] += T(m_initParam[4]); p[2] += T(m_initParam[5]);
			}
			else
			{
				p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
			}

			// Compute the center of distortion. The sign change comes from
			// the camera model that Noah Snavely¡¯s Bundler assumes, whereby
			// the camera coordinate system has a negative z axis.
			T xp = p[0] / p[2];
			T yp = p[1] / p[2];
			// Compute final projected point position.
			const T& focal = camera[6];
			//T focal = T(m_focus);//camera[6];

			T predicted_x = focal * xp;
			T predicted_y = focal * yp;
			// The error is the difference between the predicted and observed position.
			residuals[0] = predicted_x - T(observed_x);
			residuals[1] = predicted_y - T(observed_y);


			/* with distortion
			T r2; 
			T distortion; 
			T predicted_x = focal * xp;
			T predicted_y = focal * yp;
			// The error is the difference between the predicted and observed position.
			T tempX= T(observed_x)/focal;
			T tempY= T(observed_y)/focal;
			r2 = tempX*tempX+ tempY*tempY;
			//distortion = T(1.0) + r2 * (l1 + l2 * r2);
			const T& l1 = camera[7];
			distortion = T(1.0) + r2 * l1;
			T x = focal * distortion * T(tempX);
			T y = focal * distortion * T(tempY);

			residuals[0] = predicted_x - x;
			residuals[1] = predicted_y - y;*/

			return true;
		}
	double observed_x;
	double observed_y;
	double m_focus;  //used when focus of K is fixed

	bool m_bFirstCam;
	double m_initParam[9];
};
#elif 0
struct SnavelyReprojectionError {
	SnavelyReprojectionError(double observed_x, double observed_y,bool bFirstCam,double* initParam)
		: observed_x(observed_x), observed_y(observed_y) {
		m_bFirstCam = bFirstCam;
		memcpy(m_initParam,initParam,sizeof(double)*9);
	}
	
	template <typename T>
		bool operator()(const T* const camera,
		const T* const point,
		T* residuals) const {
			// camera[0,1,2] are the angle-axis rotation.
			T p[3];
			if(m_bFirstCam)
			{
				T tempR[3];
				tempR[0] = T(m_initParam[0]); tempR[1] = T(m_initParam[1]); tempR[2] = T(m_initParam[2]);
				ceres::AngleAxisRotatePoint(tempR, point, p);
			}
			else
				ceres::AngleAxisRotatePoint(camera, point, p);

			// camera[3,4,5] are the translation.
			if(m_bFirstCam)
			{
				p[0] += T(m_initParam[3]); p[1] += T(m_initParam[4]); p[2] += T(m_initParam[5]);
			}
			else
			{
				p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
			}

			// Compute the center of distortion. The sign change comes from
			// the camera model that Noah Snavely¡¯s Bundler assumes, whereby
			// the camera coordinate system has a negative z axis.
			T xp = p[0] / p[2];
			T yp = p[1] / p[2];

			// Apply second and fourth order radial distortion.
			const T& l1 = camera[7];
			//const T& l2 = camera[8];
			T r2 = xp*xp + yp*yp;
			//T distortion = T(1.0) + r2  * (l1 + l2  * r2);
			T distortion = T(1.0) + r2 * l1 ;
			
			// Compute final projected point position.
			const T& focal = camera[6];
			//T focal = T(m_initParam[6]);//camera[6];

			T predicted_x = focal * distortion * xp;
			T predicted_y = focal * distortion * yp;
			// The error is the difference between the predicted and observed position.
			residuals[0] = predicted_x - T(observed_x);
			residuals[1] = predicted_y - T(observed_y);
			return true;
		}
	double observed_x;
	double observed_y;
	double m_focus;  //used when focus of K is fixed

	bool m_bFirstCam;
	double m_initParam[9];
};
#else
struct SnavelyReprojectionError {
	SnavelyReprojectionError(double observed_x, double observed_y,bool bFirstCam,double* initParam)
		: observed_x(observed_x), observed_y(observed_y) {
			m_bFirstCam = bFirstCam;
			memcpy(m_initParam,initParam,sizeof(double)*9);
	}

	template <typename T>
	bool operator()(const T* const camera,
		const T* const point,
		T* residuals) const {
			// camera[0,1,2] are the angle-axis rotation.
			T p[3];
			if(m_bFirstCam)
			{
				T tempR[3];
				tempR[0] = T(m_initParam[0]); tempR[1] = T(m_initParam[1]); tempR[2] = T(m_initParam[2]);
				ceres::AngleAxisRotatePoint(tempR, point, p);
			}
			else
				ceres::AngleAxisRotatePoint(camera, point, p);

			// camera[3,4,5] are the translation.
			if(m_bFirstCam)
			{
				p[0] += T(m_initParam[3]); p[1] += T(m_initParam[4]); p[2] += T(m_initParam[5]);
			}
			else
			{
				p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
			}

			// Compute the center of distortion. The sign change comes from
			// the camera model that Noah Snavely¡¯s Bundler assumes, whereby
			// the camera coordinate system has a negative z axis.
			T xp = p[0] / p[2];
			T yp = p[1] / p[2];

			// Apply second and fourth order radial distortion.
			const T& focal = camera[6];

			T normalize_x = T(observed_x)/focal;
			T normalize_y = T(observed_y)/focal;
			T r2 = normalize_x*normalize_x+ normalize_y*normalize_y;

			const T& l1 = camera[7];
			//const T& l2 = camera[8];
			
			//T distortion = T(1.0) + r2  * (l1 + l2  * r2);
			T distortion = T(1.0) + r2 * l1 ;

			// Compute final projected point position.
			
			//T focal = T(m_initParam[6]);//camera[6];

			T ideal_normalized_x = distortion * normalize_x;
			T ideal_normalized_y = distortion * normalize_y;
			// The error is the difference between the predicted and observed position.
			residuals[0] = focal*(xp - ideal_normalized_x);
			residuals[1] = focal*(yp - ideal_normalized_y);
			return true;
	}
	double observed_x;
	double observed_y;
	double m_focus;  //used when focus of K is fixed

	bool m_bFirstCam;
	double m_initParam[9];
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//The follwing is used for Trajectory Recon (CVPR14, ICCV15 version)
void PatchOptimziationForTracking_ceres(TrajElement3D* pPts3D, int refImageIdx,vector<int>& warpSequenceIdx,vector<CamViewDT*>& sequences,int stepSIze,bool bDoFastOpt = false);
//Nonlinear Optimization for 3D Patch estimation
class PatchNormalOptCostFunctionUsingTripleRays : public ceres::SizedCostFunction<1, 3> 
{
public:
	virtual ~PatchNormalOptCostFunctionUsingTripleRays(){}
	inline virtual bool Evaluate(double const* const* parameters,
							double* residuals,
							double** jacobians) const;

	PatchNormalOptCostFunctionUsingTripleRays(CamViewDT* ref,CamViewDT* target,TrajElement3D* p3D,cv::Mat_<double>& ray1,cv::Mat_<double>& ray2,cv::Mat_<double>& ray3,int step)
		:pRefImage(ref),pTargetImage(target),p3dPt(p3D)	
	{
		gridStep = step;

		//m_refPatch = Mat_<double>(p3dPt->m_patch.rows,p3dPt->m_patch.cols);
		//m_targetPatch = Mat_<double>(p3dPt->m_patch.rows,p3dPt->m_patch.cols);   //why did I commment out this?
		//m_refPatch = Mat_<double>(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);
		//m_targetPatch = Mat_<double>(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);   //why did I commment out this?

		ray1.copyTo(rayCenter);
		ray2.copyTo(rayArrow1);
		ray3.copyTo(rayArrow2);

		xCenter = PATCH3D_GRID_SIZE/2.0;
		yCenter = PATCH3D_GRID_SIZE/2.0;
	}

	PatchNormalOptCostFunctionUsingTripleRays()
	{
	}

	double xCenter ;
	double yCenter ;

	CamViewDT* pRefImage;
	CamViewDT* pTargetImage;
	TrajElement3D* p3dPt;

	int gridStep;

	cv::Mat_<double> rayCenter;
	cv::Mat_<double> rayArrow1;
	cv::Mat_<double> rayArrow2;

	//Mat_<double> m_refPatch;
	//Mat_<double> m_targetPatch;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//The follwing is patch normal estimation sugin 3 parameters (distance on ray direction of reference cam, yaw and pitch for normal rotation)
//Important:: Assume that pPts3D's axis and normal are orthogonal each other
void PatchOptimziationUsingRayYawPitch_ceres(TrajElement3D* pPts3D, int refImageIdx,vector<int>& warpSequenceIdx,vector<CamViewDT*>& sequences,int stepSIze,bool bDoFastOpt = false);
class PatchNormalOptCostFunctionUsingYawPitch : public ceres::SizedCostFunction<1, 3> 
{
public:
	virtual ~PatchNormalOptCostFunctionUsingYawPitch(){}
	inline virtual bool Evaluate(double const* const* parameters,
							double* residuals,
							double** jacobians) const;
	static void ApplyParametersToPatch(TrajElement3D* p3D,cv::Mat_<double> refCamCenter,double params[3])
	{
		printf("params: %f %f %f\n",params[0],params[1],params[2]);
		cv::Mat_<double> Rot_patchToNorm;  //patch patch original (arrow1,arrow2,normal) coordinate to normalize (x,y,z) coordinate
		cv::Mat_<double> Rot_normToPatch; //patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate

		cv::Mat arrow1_unit,arrow2_unit;
		normalize(p3D->m_arrow1StepVect4by1.rowRange(0,3),arrow1_unit);		//x axis
		normalize(p3D->m_arrow2StepVect4by1.rowRange(0,3),arrow2_unit);		//y axis
		Rot_normToPatch = cv::Mat_<double>(3,3);		//patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate
		Rot_normToPatch(0,0)= arrow1_unit.at<double>(0,0);
			Rot_normToPatch(1,0)= arrow1_unit.at<double>(1,0);
				Rot_normToPatch(2,0)= arrow1_unit.at<double>(2,0);
		Rot_normToPatch(0,1)= arrow2_unit.at<double>(0,0);
			Rot_normToPatch(1,1)= arrow2_unit.at<double>(1,0);
				Rot_normToPatch(2,1)= arrow2_unit.at<double>(2,0);
		Rot_normToPatch(0,2)= p3D->m_normal.at<double>(0,0);
			Rot_normToPatch(1,2)= p3D->m_normal.at<double>(1,0);
				Rot_normToPatch(2,2)= p3D->m_normal.at<double>(2,0);
		
		Rot_patchToNorm= Rot_normToPatch.t();

		double euler[3] ={params[1],params[2],0};
		cv::Mat_<double> Rot_byEulerMat(3,3);
		double* Rot_byEuler = (double*)Rot_byEulerMat.ptr();
		ceres::AngleAxisToRotationMatrix(euler,Rot_byEuler);
		Rot_byEulerMat = Rot_byEulerMat.t();		//opencv is row-major vs ceres is column major

		cv::Mat_<double> rayDirect = p3D->m_ptMat4by1.rowRange(0,3) - refCamCenter;
		normalize(rayDirect,rayDirect);
		cv::Mat patchCenter = rayDirect*params[0] + refCamCenter;
		cv::Point3d cPt = MatToPoint3d(patchCenter) ;
		p3D->SetPos( cPt);

		p3D->LocallyRotatePatch(Rot_normToPatch*Rot_byEulerMat*Rot_patchToNorm);
	}

	PatchNormalOptCostFunctionUsingYawPitch(CamViewDT* ref,CamViewDT* target,TrajElement3D* p3D,int step)
		:m_pRefImage(ref),m_pTargetImage(target),m_p3dPt(p3D)	
	{
		gridStep = step;
		m_xCenter = PATCH3D_GRID_SIZE/2.0;
		m_yCenter = PATCH3D_GRID_SIZE/2.0;
		//m_refPatch = Mat_<double>(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);
		//m_targetPatch = Mat_<double>(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);  

		//Compute rotation of Patch to normalizedCoordinate (tp align x,y,z to arrow1,arrow2,normal)
		cv::Mat arrow1_unit,arrow2_unit;
		normalize(p3D->m_arrow1StepVect4by1.rowRange(0,3),arrow1_unit);		//x axis
		normalize(p3D->m_arrow2StepVect4by1.rowRange(0,3),arrow2_unit);		//y axis
		m_Rot_normToPatch = cv::Mat_<double>(3,3);		//patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate
		m_Rot_normToPatch(0,0)= arrow1_unit.at<double>(0,0);
			m_Rot_normToPatch(1,0)= arrow1_unit.at<double>(1,0);
				m_Rot_normToPatch(2,0)= arrow1_unit.at<double>(2,0);
		m_Rot_normToPatch(0,1)= arrow2_unit.at<double>(0,0);
			m_Rot_normToPatch(1,1)= arrow2_unit.at<double>(1,0);
				m_Rot_normToPatch(2,1)= arrow2_unit.at<double>(2,0);
		m_Rot_normToPatch(0,2)= p3D->m_normal.at<double>(0,0);
			m_Rot_normToPatch(1,2)= p3D->m_normal.at<double>(1,0);
				m_Rot_normToPatch(2,2)= p3D->m_normal.at<double>(2,0);
		
		m_Rot_patchToNorm= m_Rot_normToPatch.t();

		//p3D->LocallyRotatePatch(m_Rot_patchToNorm);		

		//Assume that arrow1 and arrow2 are orthogonal 
		m_arrow1Step_normCoord = cv::Mat_<double>::zeros(3,1);
		m_arrow1Step_normCoord(0,0) = norm(p3D->m_arrow1stHead - p3D->m_pt3D)/PATCH3D_GRID_HALFSIZE;			//should be aligned to x axis after m_Rot_patchToNorm

		m_arrow2Step_normCoord = cv::Mat_<double>::zeros(3,1);
		m_arrow2Step_normCoord(1,0) = norm(p3D->m_arrow2ndHead - p3D->m_pt3D)/PATCH3D_GRID_HALFSIZE;		//should be aligned to y axis after m_Rot_patchToNorm

		m_rayDirect = p3D->m_ptMat4by1.rowRange(0,3) - m_pRefImage->m_CamCenter;
		normalize(m_rayDirect,m_rayDirect);
	}

	PatchNormalOptCostFunctionUsingYawPitch()
	{

	}


	//General information
	double m_xCenter ;
	double m_yCenter ;
	CamViewDT* m_pRefImage;
	CamViewDT* m_pTargetImage;
	TrajElement3D* m_p3dPt;
	int gridStep;

	//Variables to make patch from parameters (ray distance, yaw, pitch)
	cv::Mat_<double> m_rayDirect;			//camCenter to
	cv::Mat_<double> m_arrow1Step_normCoord;		//3x1 mat		//m_arrow1Step*PATCH3D_GRID_HALFSIZE = arrow1 of the patch (with half width)
	cv::Mat_<double> m_arrow2Step_normCoord;		//3x1 mat

	//Original Patch
	cv::Mat_<double> m_Rot_patchToNorm;  //patch patch original (arrow1,arrow2,normal) coordinate to normalize (x,y,z) coordinate
	cv::Mat_<double> m_Rot_normToPatch; //patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate

	/* The following seems like a bug??? in openmp case
	//The following is used in PatchNormalOptCostFunctionUsingYawPitch
	//To avodi repeated mem 
	Mat_<double> m_refPatch;		
	Mat_<double> m_targetPatch;
	*/
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The following is a pilot test
//Given 3D and 2D correspondences, and initial camera - patch relative position, find local optimal new cam position (that is, new R and t)
//Initial values are transmitted thorugh R and T
//Also, return values are saved in R and t
void SingleReconPatchPosEst(vector<cv::Point3d>& corresPt3Ds,vector<cv::Point2d>& corresPt2Ds,cv::Mat& K,cv::Mat& R,cv::Mat& t);
//In testing time
struct SingleReconPatchPosEstCostFunc
{
	SingleReconPatchPosEstCostFunc(double K11,double K22,double K13,double K23,double x,double y,double x_3D,double y_3D,double z_3D)
	{
		KMatrix[0] = K11;
		KMatrix[1] = K22;
		KMatrix[2] = K13;
		KMatrix[3] = K23;

		observed_ImX = x;
		observed_ImY = y;
		observed_3D[0] = x_3D;
		observed_3D[1] = y_3D;
		observed_3D[2] = z_3D;
	}
	
	template <typename T>
	bool operator()(const T* const pt,
					const T* const dummy,
					T* residuals) const ;

	double observed_ImX;
	double observed_ImY;

	/*double observed_3DX;
	double observed_3DY;
	double observed_3DZ;*/
	double observed_3D[3];
	double KMatrix[4];    //K11,K22, K13,K23
};

//normal estimation using diagonal axis
//double FindBestRatioForNormal(Mat& pt_1,Mat& pt_2,Point2d& centerPt,Mat& KMatrix);
/*
struct DiagonalAngleDetector
{
public:
	DiagonalAngleDetector(Mat& pt_1,Mat& pt_2,Mat& K,Point2d center)
	{
		pt1_x = pt_1.at<double>(0,0);
		pt1_y = pt_1.at<double>(1,0);
		pt1_z = pt_1.at<double>(2,0);

		pt2_x = pt_2.at<double>(0,0);
		pt2_y = pt_2.at<double>(1,0);
		pt2_z = pt_2.at<double>(2,0);

		KMatrix[0] = K.at<double>(0,0);
		KMatrix[1] = K.at<double>(1,1);
		KMatrix[2] = K.at<double>(0,2);
		KMatrix[3] = K.at<double>(1,2);

		observed_centerX = center.x;
		observed_centerY = center.y;
	}
	
	template <typename T>
	bool operator()(const T* const ratio,
					const T* const dummy,
					T* residuals) const ;

	double pt1_x;
	double pt1_y;
	double pt1_z;

	double pt2_x;
	double pt2_y;
	double pt2_z;

	double observed_centerX;
	double observed_centerY;

	double KMatrix[4];    //K11,K22, K13,K23

};*/



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//13.0809  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Nonlinear Optimization for 3D Patch estimation
class PatchOptimizeForTrackingCostFunction: public ceres::SizedCostFunction<1, 3> 
{
public:
	virtual ~PatchOptimizeForTrackingCostFunction(){}
	inline virtual bool Evaluate(double const* const* parameters,
							double* residuals,
							double** jacobians) const;

	PatchOptimizeForTrackingCostFunction(CamViewDT* ref,CamViewDT* target,TrajElement3D* p3D,cv::Mat_<double>& ray1,cv::Mat_<double>& ray2,cv::Mat_<double>& ray3,int step)
		:pRefImage(ref),pTargetImage(target),p3dPt(p3D)	
	{
		gridStep = step;

		//m_refPatch = Mat_<double>(p3dPt->m_patch.rows,p3dPt->m_patch.cols);
		//m_targetPatch = Mat_<double>(p3dPt->m_patch.rows,p3dPt->m_patch.cols);   //why did I commment out this?
		m_refPatch = cv::Mat_<double>(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);
		m_targetPatch = cv::Mat_<double>(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);   //why did I commment out this?

		ray1.copyTo(rayCenter);
		ray2.copyTo(rayArrow1);
		ray3.copyTo(rayArrow2);

		xCenter = PATCH3D_GRID_HALFSIZE;
		yCenter = PATCH3D_GRID_HALFSIZE;
	}

	PatchOptimizeForTrackingCostFunction()
	{
	}

	double xCenter ;
	double yCenter ;

	CamViewDT* pRefImage;
	CamViewDT* pTargetImage;
	TrajElement3D* p3dPt;

	int gridStep;

	cv::Mat_<double> rayCenter;
	cv::Mat_<double> rayArrow1;
	cv::Mat_<double> rayArrow2;

	cv::Mat_<double> m_refPatch;
	cv::Mat_<double> m_targetPatch;
};


////////////////////////////////////////////////////////////////////
//// The following is for ICCV 15, Human body pose reconstruction
struct PoseReconJointTrajError {
	PoseReconJointTrajError(double* Trans,vector< pair<cv::Point3d,double> >& relatedPeaks)
	{
		m_peakNum = relatedPeaks.size();
		memcpy(m_Trans,Trans,sizeof(double)*16);

		m_peakValues = new double[m_peakNum];
		m_peaks = new double[3*m_peakNum];
		for(int c=0;c<m_peakNum;++c)
		{
			m_peaks[c*3] = relatedPeaks[c].first.x;
			m_peaks[c*3+1] = relatedPeaks[c].first.y;
			m_peaks[c*3+2] = relatedPeaks[c].first.z;

			m_peakValues[c] = relatedPeaks[c].second;
		}
		
	}

	~PoseReconJointTrajError()
	{
		delete[] m_peakValues;
		delete[] m_peaks;
	}

	template <typename T>
	bool operator()(const T* const startPt,	//size==3
		T* residuals) const {			//size ==1

			//////////////////////////////////////////////////////////////////////////
			// Compute transformed pt
			T predicted[4];
			predicted[0] = T(m_Trans[0])*startPt[0]+ T(m_Trans[1])*startPt[1] + T(m_Trans[2])*startPt[2] + T(m_Trans[3]);
			predicted[1] = T(m_Trans[4])*startPt[0]+ T(m_Trans[5])*startPt[1] + T(m_Trans[6])*startPt[2] + T(m_Trans[7]);
			predicted[2] = T(m_Trans[8])*startPt[0]+ T(m_Trans[9])*startPt[1] + T(m_Trans[10])*startPt[2] + T(m_Trans[11]);
			predicted[3] = T(m_Trans[12])*startPt[0]+ T(m_Trans[13])*startPt[1] + T(m_Trans[14])*startPt[2] + T(m_Trans[15]);

			predicted[0] =predicted[0]/predicted[3];
			predicted[1] =predicted[1]/predicted[3];
			predicted[2] =predicted[2]/predicted[3];


			//////////////////////////////////////////////////////////////////////////
			// Compute least error
			T maxScore=T(0);
			for(int c=0;c<m_peakNum;++c)
			{
				//Compute Gaussian score
				T dist = pow(predicted[0] - T(m_peaks[3*c]),2) +pow(predicted[1] - T(m_peaks[3*c +1]),2)+pow(predicted[2] - T(m_peaks[3*c +2]),2);
				dist = ceres::sqrt(dist);

				T score = T(m_peakValues[c])* ceres::exp(- dist*dist/T(1800.0));		//1800 = 2*sigma*sigma where sigma ==30
				if(maxScore<score)
					maxScore = score;
			}
			residuals[0] = 1e6- maxScore;		//need to minimize cost function
			return true;
	}

	double m_Trans[16];	//4x4 matrix
	double* m_peakValues;
	double* m_peaks;	//3*peakNum 
	int m_peakNum;
};
void PoseReconNodeTrajOptimization_ceres(cv::Point3f startPt,vector< vector<cv::Mat_<double> > >& transformVector,vector< vector<pair<cv::Point3d,double> > >& relatedPeaks,cv::Point3f& optimizedStartPt);



////////////////////////////////////////////////////////////////////
//// The following is new functions


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//The follwing is patch normal estimation sugin 3 parameters (distance on ray direction of reference cam, yaw and pitch for normal rotation)
//Important:: Assume that pPts3D's axis and normal are orthogonal each other
#define PATCH_PHOTOMET_GRID 11
#define PATCH_PHOTOMET_GRID_SQ 121
void PatchOptimziationUsingPhotometricShapeRefine_ceres(TrajElement3D* pPts3D, int refImageIdx,vector<int>& warpSequenceIdx,vector<CamViewDT*>& sequences,int stepSIze,bool bDoFastOpt = false);
//Nonlinear Optimization for 3D Patch estimation
class MicroStrcutureOptCostFunctionUsingRays : public ceres::SizedCostFunction<1, PATCH_PHOTOMET_GRID_SQ>		//assuming 11 by 11
{
public:
	virtual ~MicroStrcutureOptCostFunctionUsingRays  (){}
	inline virtual bool Evaluate(double const* const* parameters,
		double* residuals,
		double** jacobians) const;

	MicroStrcutureOptCostFunctionUsingRays (CamViewDT* ref,CamViewDT* target,TrajElement3D* p3D,vector< cv::Mat_<double> >& rays, int step)
		:pRefImage(ref),pTargetImage(target),p3dPt(p3D)	
	{
		gridStep = step;

		//m_refPatch = Mat_<double>(p3dPt->m_patch.rows,p3dPt->m_patch.cols);
		//m_targetPatch = Mat_<double>(p3dPt->m_patch.rows,p3dPt->m_patch.cols);   //why did I commment out this?
		//m_refPatch = Mat_<double>(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);
		//m_targetPatch = Mat_<double>(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);   //why did I commment out this?
		m_rays  = rays;

		int grid_width = sqrt(m_rays.size());
		xCenter = grid_width/2.0;
		yCenter = grid_width/2.0;
	}

	MicroStrcutureOptCostFunctionUsingRays ()
	{
	}

	double xCenter ;
	double yCenter ;

	CamViewDT* pRefImage;
	CamViewDT* pTargetImage;
	TrajElement3D* p3dPt;

	int gridStep;

	vector< cv::Mat_<double> > m_rays;			//3x1 vectors
};


//Nonlinear Optimization for 3D Patch estimation
class MicroStrcutureOptCostFunctionUsingRays_smooth_single : public ceres::SizedCostFunction<1, 1, 1>		//assuming 11 by 11
{
public:
	virtual ~MicroStrcutureOptCostFunctionUsingRays_smooth_single  (){}
	inline virtual bool Evaluate(double const* const* parameters,
		double* residuals,
		double** jacobians) const;

	MicroStrcutureOptCostFunctionUsingRays_smooth_single (cv::Mat_<double>& ray1,cv::Mat_<double>& ray2)
	{
		m_ray1 = ray1;
		m_ray2 = ray2;
	}

	MicroStrcutureOptCostFunctionUsingRays_smooth_single ()
	{
	}

	cv::Mat_<double> m_ray1;			//3x1 vectors
	cv::Mat_<double> m_ray2;			//3x1 vectors
};



//Nonlinear Optimization for 3D Patch estimation
class MicroStrcutureOptCostFunctionUsingRays_smooth : public ceres::SizedCostFunction<1, PATCH_PHOTOMET_GRID_SQ>		//assuming 11 by 11
{
public:
	virtual ~MicroStrcutureOptCostFunctionUsingRays_smooth  (){}
	inline virtual bool Evaluate(double const* const* parameters,
		double* residuals,
		double** jacobians) const;

	MicroStrcutureOptCostFunctionUsingRays_smooth (vector< cv::Mat_<double> >& rays)
	{
		m_rays = rays;
	}

	MicroStrcutureOptCostFunctionUsingRays_smooth ()
	{
	}

	vector< cv::Mat_<double> >  m_rays;			//3x1 vectors
};


//////////////////////////////////////////////////////////////////////////
// Patch R,t optimization minimizing reprojection errror
void PatchOptimziationMinimizingReproError_ceres(TrajElement3D* pPts3D,
												 vector<cv::Point3d>& sample3DPts,vector< vector<cv::Mat*> >& ProjVect, vector< vector<cv::Point2d> >& target2dPts,
												 int stepSize,bool bVerbose);
struct PatchTrackByReproError
{
	PatchTrackByReproError(cv::Mat& projectMat,
							cv::Mat& rotNorm2Patch,
							cv::Mat& normSamplePt,
							double x,double y)
	{
		m_projMat = projectMat;
		m_rotNormToPatch = rotNorm2Patch;
		m_normSamplePt_mat = normSamplePt;
		m_observed_ImX = x;
		m_observed_ImY = y;
		/*
		//Compute sample pt location in the normalized coordinate
		cv::Mat Rot_patchToNorm= m_rotNormToPatch.t();
		Point3d normalizedPt = samplePt - pPatch3D->m_pt3D;
		Mat normalizedPt_mat;
		Point3dToMat4by1(normalizedPt,normalizedPt_mat);
		normalizedPt_mat = Rot_patchToNorm*normalizedPt_mat;
		m_normSamplePt_mat = normalizedPt_mat/normalizedPt_mat.at<double>(3,0);*/
	};

	static void ComputeNormSampleLocation(TrajElement3D* pPatch3D,const cv::Point3d& originalPt,const cv::Mat& rotPatchToNorm,cv::Mat& normalizedPt_out);		//Rot_normToPatch is the output
	static void ComputeRotNormToPatch(TrajElement3D* pPatch3D,cv::Mat_<double>& Rot_normToPatch);		//Rot_normToPatch is the output
	static void ApplyParametersToPatch(TrajElement3D* pPatch3D,
		cv::Mat_<double> Rot_patchToNorm,  //patch patch original (arrow1,arrow2,normal) coordinate to normalize (x,y,z) coordinate
		cv::Mat_<double> Rot_normToPatch, //patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate		
		double params[6]);


	template <typename T>
	bool operator()(const T* const parameters,
		T* residuals) const;

	double m_observed_ImX;
	double m_observed_ImY;
	cv::Mat m_projMat;				//3x4 double
	cv::Mat m_rotNormToPatch;		//3x3 double
	cv::Mat m_normSamplePt_mat;		//3x1 double
};