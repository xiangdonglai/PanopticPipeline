#pragma once

#include <cv.h>
#include <cxcore.h>
#include <ctype.h>
#include <ctime>
#include <limits>
#include <vector>
#include <string>

#ifdef LINUX_COMPILE
	#include <sys/time.h>
#endif
// typedef unsigned long long ULONGLONG;

//#include "opencv2/nonfree/gpu.hpp"
//#include "opencv2/gpu/gpu.hpp"

using namespace std;

void CreateFolder(const char* folderName);

/*double Distance(Point2d& pt_1,Point2f& pt_2);
double Distance(Point2f& pt_1,Point2f& pt_2);
double Distance(Point2d& pt_1,Point2d& pt_2);
double Distance(Point3d& pt_1,Point3d& pt_2);
double Distance(Point3f& pt_1,Point3f& pt_2);*/

template <typename Type1,typename Type2>
double Distance(const cv::Point_<Type1>& pt_1,const  cv::Point_<Type2>& pt_2)
{
	double dist = pow(double(pt_1.x -pt_2.x),2) +pow(double(pt_1.y - pt_2.y),2);
	return std::sqrt(dist);
}

template <typename Type1,typename Type2>
double Distance(const cv::Point3_<Type1>& pt_1,const cv::Point3_<Type2>& pt_2)
{
	double dist = pow(double(pt_1.x -pt_2.x),2) +pow(double(pt_1.y - pt_2.y),2)+pow(double(pt_1.z - pt_2.z),2);
	return std::sqrt(dist);
}

//Compute distance from pt to a line segment (lineSourcePt -> lineDestPt)
//If the orthogoal intersction from pt does not meet the line segment, it return shorte length between pt->lineSourcePt and pt->lineDestPt
template <typename Type>
float DistancePtToLine(cv::Point3_<Type>& lineSourcePt,cv::Point3_<Type>& lineDestPt,cv::Point3_<Type>& unitLineDirect,float lineSegDist,cv::Point3_<Type>& pt)
{
	cv::Point3_<Type> sourceToPt = pt - lineSourcePt;
	float distAlongBoneDirection = sourceToPt.dot(unitLineDirect);

	if(distAlongBoneDirection<0 ||distAlongBoneDirection>lineSegDist)
	{
		float dist1 = Distance(lineSourcePt,pt);
		float dist2 = Distance(lineDestPt,pt);
		return min(dist1,dist2);
	}
	cv::Point3_<Type> orthoVect = sourceToPt - unitLineDirect*distAlongBoneDirection;
	return norm(orthoVect);
}


/*U = A2-A1;
V = B2-B1;
W = cross(U,V);
X = A1 + dot(cross(B1-A1,V),W)/dot(W,W)*U;
Y = B1 + dot(cross(B1-A1,U),W)/dot(W,W)*V;
d = norm(Y-X);*/

template <typename Type>
float DistanceBetweenLines(cv::Point3_<Type>& lineSourcePt_a,cv::Point3_<Type>& lineDestPt_a,
							cv::Point3_<Type>& lineSourcePt_b,cv::Point3_<Type>& lineDestPt_b)
{

	cv::Point3_<Type> U = lineDestPt_a - lineSourcePt_a;
	cv::Point3_<Type> V = lineDestPt_b - lineSourcePt_b;
	cv::Point3_<Type> W = U.cross(V);
	cv::Point3_<Type> X = lineSourcePt_a + ((lineSourcePt_b-lineSourcePt_a).cross(V)).dot(W)/W.dot(W)*U;
	cv::Point3_<Type> Y = lineSourcePt_b + ((lineSourcePt_b-lineSourcePt_a).cross(U)).dot(W)/W.dot(W)*V;

	if((X-lineSourcePt_a).dot(U)<0)
		return 1e5;
	if((Y-lineSourcePt_b).dot(V)<0)
		return 1e5;

	if( norm(X-lineSourcePt_a) > norm(U))
		return 1e5;

	if( norm(Y-lineSourcePt_b) > norm(V))
		return 1e5;

	double dist = norm(Y-X);
	return dist;
}

double Distance(cv::Mat& pt_2,cv::Point3d& pt_1);
double Distance(cv::Point3d& pt_1,cv::Mat& pt_2);
double Distance(cv::Mat& pt_1,cv::Mat& pt_2);
void writeMatrix(cv::Mat &M, char* outputFileName);
void writeMatrix(char* outputFileName,cv::Mat &M);
void printMatrix(cv::Mat &M, std::string matrix);
void printMatrix(std::string matrix,cv::Mat &M);
void printMatrixFloat(cv::Mat &M, std::string matrix);
cv::Point3d GetCenter(cv::Point3d p1,cv::Point3d p2);
bool LineInterSection(cv::Point2d p1,cv::Point2d p2,cv::Point2d q1,cv::Point2d q2,cv::Point2d& interPt);
void GetRotMatrix(cv::Mat &axis1,cv::Mat& axis2,cv::Mat& returnRotMat);
void MatToPoint3d(cv::Mat& mat,cv::Point3d& returnPt);
cv::Point3d MatToPoint3d(cv::Mat& mat);
cv::Point3f MatToPoint3f(cv::Mat& mat);


template <typename Type>
cv::Point_<Type> RotatePt2D(cv::Mat_<float> r,cv::Point_<Type> pt)
{
	cv::Point_<Type> returnPt(0,0);
	if(r.rows == 2 && r.cols == 3)	//Affine (2x3)
	{
		returnPt.x = pt.x*r(0,0) + pt.y*r(0,1) + r(0,2);
		returnPt.y = pt.x*r(1,0) + pt.y*r(1,1) + r(1,2);
	}
	else if(r.rows == 2 && r.cols == 2)
	{
		returnPt.x = pt.x*r(0,0) + pt.y*r(0,1);
		returnPt.y = pt.x*r(1,0) + pt.y*r(1,1);
	}
	
	return returnPt;
}
/*
template <typename Type>
Point_<Type> InvRotatePt2D(Mat_<float>& r,Point_<Type> pt)
{
	returnPt.x = pt.x*r(0,0) - pt.y*r(0,1);
	returnPt.y = -pt.x*r(1,0) + pt.y*r(1,1);

	return returnPt;
}
*/
//template <typename Type> void Point3ToMat(Point3_<Type>& pt,cv::Mat& returnMat);
//template <typename Type> void Point3ToMat(Point3_<Type>& pt_1,Mat_<double>& returnMat);

template <typename Type>
void Point3ToMatDouble(const cv::Point3_<Type>& pt_1,cv::Mat& returnMat)
{
	returnMat = cv::Mat::zeros(3,1,CV_64F);
	returnMat.at<double>(0,0) = pt_1.x;
	returnMat.at<double>(1,0) = pt_1.y;
	returnMat.at<double>(2,0) = pt_1.z;
}

template <typename Type>
void Point3ToMatDouble(const cv::Point3_<Type>& pt_1,cv::Mat_<double>& returnMat)
{
	returnMat = cv::Mat_<double>(3,1);
	returnMat(0,0) = pt_1.x;
	returnMat(1,0) = pt_1.y;
	returnMat(2,0) = pt_1.z;
}

template <typename Type>
void Point3MutipliedByScalar(cv::Point3_<Type>& pt_1,double s)
{
	pt_1.x = pt_1.x*s;
	pt_1.y = pt_1.y*s;
	pt_1.z = pt_1.z*s;
}

cv::Point3d Point3fTo3d(const cv::Point3f& pt);
cv::Point3f Point3dTo3f(const cv::Point3d& pt);
void Point3fTo3d(const cv::Point3f& pt,cv::Point3d& returnPt);
void Point3dTo3f(const cv::Point3d& pt,cv::Point3f& returnPt);
void Point3fToMat(const cv::Point3f& pt,cv::Mat& returnMat);
void Point3fToMat(const cv::Point3f& pt,cv::Mat_<double>& returnMat);
void Point3dToMat(const cv::Point3d& pt,cv::Mat& returnMat);
void Point3dToMat(const cv::Point3d& pt,cv::Mat_<double>& returnMat);

//void Point3fToMat(Point3f& pt_1,cv::Mat& returnMat);
//void Point3fToMat(Point3f& pt_1,Mat_<double>& returnMat);
//void Point3dToMat(Point3d& pt_1,cv::Mat& returnMat);

template <typename Type>
void Point3xToMat4by1(const cv::Point3_<Type>& pt_1,cv::Mat_<double>& returnMat)
{
	//returnMat = cv::Mat::ones(4,1,CV_64F);
	returnMat = cv::Mat_<double>::ones(4,1);
	returnMat.at<double>(0,0) = pt_1.x;
	returnMat.at<double>(1,0) = pt_1.y;
	returnMat.at<double>(2,0) = pt_1.z;
}

void Point3dToMat4by1(const cv::Point3d& pt_1,cv::Mat& returnMat);
void Point3dToMat4by1(const cv::Point3d& pt_1,cv::Mat_<double>& returnMat);
void Point3fToMat4by1(const cv::Point3f& pt_1,cv::Mat& returnMat);
void Point3fToMat4by1(const cv::Point3f& pt_1,cv::Mat_<double>& returnMat);
void DrawCross(cv::Mat& image,int x,int y,cv::Scalar color,int size);
void DrawDotGray(cv::Mat& image,int x,int y,int grayValue,int size=1);
void DrawDotRGB(cv::Mat& image,int x,int y,cv::Point3f color,int size=1);
void DrawDotBGR(cv::Mat& image,int x,int y,cv::Point3f color,int size=1);
void DrawDotRGB(cv::Mat& image,int x,int y,cv::Scalar& color,int size);

int COLOR_CLAMP(int color);



bool IsOutofBoundary(const cv::Mat& image,double x,double y);
bool IsOutofBoundary(int width,int height,double x, double y);
bool IsOutofBoundary(const cv::Mat& image,cv::Rect& bbox);

void GetFolderPathFromFullPath(const char* fullPath,char folderPath[]);
void GetFileNameFromFullPath(const char* fullPath,char* fileName);
void GetFinalFolderNameInFullPath(const char* fullPath,char* folderName);


cv::Point2f Project3DPtF(cv::Point3f& pt,cv::Mat_<float>& PMat);

//cv::Mat should be double type
cv::Point2d Project3DPt(cv::Point3f& pt,cv::Mat& PMat);
cv::Point2d Project3DPt(cv::Point3d& pt,cv::Mat& PMat);
cv::Point2d Project3DPt(cv::Mat& pt4by1,cv::Mat& PMat);

cv::Point2d Plus(cv::Point2d& pt1,cv::Point2d& pt2);
cv::Point2d Plus(cv::Point2f& pt1,cv::Point2d& pt2);
cv::Point2d Plus(cv::Point2d& pt1,cv::Point2f& pt2);

cv::Point2d Minus(cv::Point2d& pt1,cv::Point2d& pt2);
cv::Point2d Minus(cv::Point2f& pt1,cv::Point2d& pt2);
cv::Point2d Minus(cv::Point2d& pt1,cv::Point2f& pt2);

void printfLog(const char *strLog,...);

class CClock
{
public:
	CClock()
	{
#ifndef LINUX_COMPILE
		initTime = -1;
#endif
		tic();
	}
	void tic()
	{
#ifdef LINUX_COMPILE
		gettimeofday(&initTime, 0);
#else
		initTime = GetTickCount64();
#endif
	};
	int toc()
	{

#ifdef LINUX_COMPILE
		timeval endTime;
		gettimeofday(&endTime, 0);
		double endTimeDouble = endTime.tv_sec + endTime.tv_usec *1e-6;
		double initTimeDouble = initTime.tv_sec + initTime.tv_usec *1e-6;
		double interval = endTimeDouble -initTimeDouble;
		int intervalInt = interval*1e3;
		std::printf("TIME: %d ms\n",intervalInt);
		return intervalInt;

		std::cout << "TIME:: " << (endTime.tv_usec - initTime.tv_usec)/1000 <<" ms" << std::endl;
#else
		 long long timeInverval = GetTickCount64() -initTime;
		 std::printf("TIME: %d ms\n",timeInverval);
		 return (int)timeInverval;
#endif
	};

	int toc(const char* timeString)
	{
#ifdef LINUX_COMPILE
		timeval endTime;
		gettimeofday(&endTime, 0);
		
		double endTimeDouble = endTime.tv_sec + endTime.tv_usec *1e-6;
		double initTimeDouble = initTime.tv_sec + initTime.tv_usec *1e-6;
		double interval = endTimeDouble -initTimeDouble;
		int intervalInt = interval*1e3;
		
		char str[256];
		sprintf(str,"TIME::%s:",timeString);
		printfLog("%s %d ms\n",str,intervalInt);
		return intervalInt;

#else
		 long long timeInverval = GetTickCount64() -initTime;
		 char str[256];
		 sprintf(str,"TIME::%s:",timeString);
		 //std::printf("%s %d ms\n",str,timeInverval);
		 printfLog("%s %d ms\n",str,timeInverval);
		 return (int)timeInverval;
#endif

		 
	};
private:
#ifdef LINUX_COMPILE
	timeval initTime;
#else
	ULONGLONG initTime;
#endif

};

/*int ExtractFrameIdxFromPath(CString filePath);
int ExtractCamIdxFromPath(CString filePath); //only for dome system
int ExtractPanelIdxFromPath(CString filePath); //only for dome system*/

int ExtractFrameIdxFromPath(const char* fullPath);
int ExtractCamIdxFromPath(const char* fullPath); //only for dome system
int ExtractPanelIdxFromPath(const char* fullPath); //only for dome system
int ExtractFrameIdxFromPath_last8digit(const char* fullPath); //assuming last 8 digits means frameIdx

void ShowSubRegion();
void Vec3dToVec3b(cv::Vec3d& source,cv::Vec3b& target);  //double to uchar

//template <typename Type>
//void Vec3dToPoint3(Vec3d& source,Point3_<Type>& target);  
template <typename Type>
void Vec3dToPoint3(cv::Vec3d& source,cv::Point3_<Type>& target,bool bChangeOrder = false) //double to uchar
{
	if(bChangeOrder ==false)
	{
		target.x = source(0);
		target.y = source(1);
		target.z = source(2);
	}
	else
	{
		target.x = source(2);
		target.y = source(1);
		target.z = source(1);
	}
	
}


double myNCC(cv::Mat_<double>& im_1,cv::Mat_<double>& im_2);
double myNCC(cv::Mat_<double>& im_1,cv::Mat_<double>& im_2,int step);
double BilinearInterpolation(cv::Mat_<uchar>& grayImage,double x,double y);
cv::Vec3d BilinearInterpolation(cv::Mat_<cv::Vec3b>& rgbImage,double x,double y);

template <typename Type>
double BilinearInterpolation(cv::Mat_<Type>& image,double x,double y)
{
	int floor_y = (int)floor(y);
	int ceil_y = (int)ceil(y);
	int floor_x = (int)floor(x);
	int ceil_x = (int)ceil(x);

	if( floor_y>=image.rows || ceil_y<0  || floor_x>=image.cols || ceil_x<0 )
		return 0;
	
	if(ceil_y>= image.rows )	//no need to interplation
		ceil_y = floor_y;
	else if(floor_y <0 )	//no need to interplation
		floor_y = ceil_y;
	
	if(ceil_x>= image.cols)
		ceil_x = floor_x;
	else if(floor_x < 0)
		floor_x = ceil_x;

	//interpolation
	double value_lb = image(floor_y,floor_x);   //Vec3d: double vector
	double value_lu = image(ceil_y,floor_x);
	double value_rb = image(floor_y,ceil_x);
	double value_ru = image(ceil_y,ceil_x);

	double alpha = y - floor_y;
	double value_l= (1-alpha) *value_lb + alpha * value_lu;
	double value_r= (1-alpha) *value_rb + alpha * value_ru;
	double beta = x - floor_x;
	double finalValue = (1-beta) *value_l + beta * value_r;

	return finalValue;
}

void randGenerator(int min,int max,int selectionNum, vector<int>& output);
void randGenerator2(int min,int max,int selectionNum, vector<int>& output);
void ParameterSetting(char* filePath);


#ifdef LINUX_COMPILE
#include <pthread.h>

class MyCriticalSection
{
public:
	MyCriticalSection()
	{
		pthread_mutex_init( &mutex, NULL);
		//mutex= PTHREAD_MUTEX_INITIALIZER;
	}
	void Lock()
	{
		pthread_mutex_lock(&mutex);  // lock the critical section
	};
	void Unlock()
	{
		pthread_mutex_unlock(&mutex);  // lock the critical section	
	};

	pthread_mutex_t mutex;
};
#else  //window
class MyCriticalSection
{
public:
	void Lock()
	{
		m_section.Lock();
	};
	void Unlock()
	{

		m_section.Unlock();
	};

	CCriticalSection m_section;
};
#endif

void Rotation2Quaternion(cv::Mat& R, cv::Mat& q);
void Quaternion2Rotation(cv::Mat& q, cv::Mat& R);


void ImageSC(cv::Mat_<double>& target,cv::Mat& resultImage,bool noPopup=true);
void ImageSC(cv::Mat_<float>& target,cv::Mat& resultImage,bool noPopup=true);
//void ImageSC(Mat_<unsigned int>& target);
//void ImageSC(Mat_<double>& target);
//void ImageSC(Mat_<float>& target);
void ImageSC(string name,cv::Mat& target);
void ImageSC(cv::Mat& target);

cv::Point2f DistortPointR1(cv::Point2f& pt, double k1);

//set default value to the non max pixels
//by checking windowSize x windowSize 
void NonMaxSuppression(cv::Mat_<float>& input,int windowSize,float defaultValue);
void NonMinSuppression(cv::Mat_<float>& input,int windowSize,float defaultValue);
void CalVariance(cv::Mat& input,cv::Mat_<float>& output,int windowSize);
//void CalVarianceGPU(cv::cuda::GpuMat& input,cv::Mat_<float>& output,int windowSize);

cv::Rect GetRect(double x,double y,double halfWidth,double halfHeight);
cv::Rect GetRect(cv::KeyPoint& k);


void GetWarppedPoints(cv::Mat& homography,vector<cv::Point2f>& sourcePoints,vector<cv::Point2f>& warppedPoints);
void GetBboxCoveringTriplet(vector<cv::Point2f>& pts,double scaleFactor,cv::Rect& returnBbox);
//void Normalize(Point3d& vect);

template <typename Type>
void Normalize(cv::Point3_<Type>& vect)
{
	double dist = vect.x*vect.x + vect.y*vect.y + vect.z*vect.z;
	dist = std::sqrt(dist);
	vect.x /=(float)dist;
	vect.y /=(float)dist;
	vect.z /=(float)dist;
}

//template<typename _Tp>
void SaveImageWithPoint(cv::Mat& image,cv::Point_<double> pt,int radius,cv::Scalar color= cv::Scalar(0,0,255));

void GetFilesInDirectory(const string &directory,std::vector<string> &out);


#include <fstream>
cv::Mat PlaneFitting(vector<cv::Point3f>& neighborVoxMat);
bool IsFileExist(const char* tempFramePath);

void PutGaussianKernel(cv::Mat_<uchar>& costMap,cv::Point2d centerPt,int bandwidth,double maxValue,float sigmaRatio=0.6);
void PutGaussianKernel(cv::Mat_<float>& costMap,cv::Point2d centerPt,int bandwidth,double maxValue,float sigmaRatio=0.6);
void PutValueKernel(cv::Mat_<int>& indexMap,cv::Point2d centerPt,int bandwidth,int index);			//not used.... due to the following PutGaussianKernelWithIndexLog function
void PutGaussianKernelWithIndexLog(cv::Mat_<float>& costMap,cv::Point2d centerPt,int bandwidth,double maxValue,cv::Mat_<int>& indexMap,int index);

//////////////////////////////////////////////////////////////////////////
/// Point Recon Tools
void VisualizePointReprojection(cv::Mat& targetImage,cv::Point3d pt3D,cv::Mat& PMatrix,cv::Scalar& tempColor);

double CalcReprojectionError_weighted(vector<cv::Mat*>& M,vector<cv::Point2d>& pt2D,vector<double>& weights,cv::Mat& X); //for One 3Dpt
double CalcReprojectionError(vector<cv::Mat*>& M,vector<cv::Point2d>& pt2D,cv::Mat& X); //for One 3Dpt
double CalcReprojectionError_returnErroVect(vector<cv::Mat*>& M,vector<cv::Point2d>& pt2D,cv::Mat& X,vector<double>& errorReturn); //for One 3Dpt
double CalcReprojectionError(vector<cv::Mat*>& M,vector<cv::Point2f>& pt2D,cv::Mat& X); //for One 3Dpt
double CalcReprojectionError(cv::Mat& ptMat4by1,cv::Mat& projMat,cv::Point2f& pt2D);

void triangulate(cv::Mat& M1,cv::Point2d& p1, cv::Mat& M2,cv::Point2d& p2,cv::Mat& X);
void triangulate(vector<cv::Mat*>& M,vector<cv::Point2d*>& p,cv::Mat& X);
void triangulate(vector<cv::Mat*>& M,vector<cv::Point2d>& p,cv::Mat& X);
double triangulateWithRANSAC(vector<cv::Mat*>& M,vector<cv::Point2f>& pt2D,int iterNum,cv::Mat& X,double thresholdForInlier,vector<unsigned int>& inliers);
double triangulateWithRANSAC(vector<cv::Mat*>& M,vector<cv::Point2d>& pt2D,int iterNum,cv::Mat& X,double thresholdForInlier,vector<unsigned int>& inliers);
double triangulateWithRANSACExhaustive(vector<cv::Mat*>& M,vector<cv::Point2d>& pt2D,cv::Mat& X,double thresholdForInlier,vector<unsigned int>& inliers);

//for ThreeDFeatureGeneration such as 1stArrow and 2ndArrow
double triangulateWithOptimization(vector<cv::Mat*>& M,vector<cv::Point2d>& p,cv::Mat& X); //cv::Mat should be double always. 
double triangulateWithOptimizationF(vector<cv::Mat*>& M,vector<cv::Point2f>& p,cv::Mat& X);  //cv::Mat should be double always. for tracked point

cv::Mat_<double> getRigidTransform(vector<cv::Point3f>& src,vector<cv::Point3f>& dst);
cv::Mat_<double> getRigidTransform_RANSAC(vector<cv::Point3f>& src,vector<cv::Point3f>& dst);
cv::Mat_<double> getRigidTransformWithVerification(vector<cv::Point3f>& src,vector<cv::Point3f>& dst);
cv::Mat_<double> getRigidTransformWithIterationWithReject(vector<cv::Point3f>& originalSrc,vector<cv::Point3f>& originalDst,int iterationNum);
cv::Mat_<double> getRigidTransformWithIteration(vector<cv::Point3f>& originalSrc,vector<cv::Point3f>& originalDst,int iterationNum);			//ICP


#include <fstream>

template <typename T>
void FileWritePoint3(std::ofstream& fout,const cv::Point3_<T>& pt)
{
	fout <<pt.x <<" "<<pt.y <<" "<<pt.z<<" ";
}

template <typename T>
void FileReadPoint3(std::ifstream& fin,cv::Point3_<T>& pt)
{
	fin >>pt.x >> pt.y >> pt.z;
}

void PathChangeFromPatchCloudToTrajStream(const char* patchCloudFile,char* trajStreamFile);

bool WorkingVolumeCheck(cv::Mat X);
bool WorkingVolumeCheck(cv::Point3d X);

void Get3DPtfromDist(cv::Mat& Kinv,cv::Mat& Rinv,cv::Mat& t,cv::Point2d& pt,double dist,cv::Point3d& pt3D);


//////////////////////////////////////////////////////////////////////////
/// Image Load
cv::Mat ReadImageFromDisk(const char* folderName,int frameIdx,int panelIdx,int camIdx,bool bLoadAsGray=false);

//To handle vga-hd frame index complication
int GetCurGlobalImgFrame();


//////////////////////////////////////////////////////////////////////////

extern cv::Scalar g_black;
extern cv::Scalar g_blue;
extern cv::Scalar g_green;
extern cv::Scalar g_red;
extern cv::Scalar g_white;
extern cv::Scalar g_cyan;
extern cv::Scalar g_yellow;

extern cv::Point3f g_black_p3f;
extern cv::Point3f g_red_p3f;
extern cv::Point3f g_yellow_p3f;
extern cv::Point3f g_gray_p3f;
extern cv::Point3f g_blue_p3f;
extern cv::Point3f g_magenta_p3f;
extern cv::Point3f g_cyan_p3f;
extern cv::Point3f g_green_p3f;
extern cv::Point3f g_orange_p3f;

extern cv::Point3d g_black_p3d;
extern cv::Point3d g_red_p3d;
extern cv::Point3d g_yellow_p3d;
extern cv::Point3d g_gray_p3d;
extern cv::Point3d g_blue_p3d;
extern cv::Point3d g_magenta_p3d;
extern cv::Point3d g_cyan_p3d;
extern cv::Point3d g_green_p3d;
extern cv::Point3d g_orange_p3d;
