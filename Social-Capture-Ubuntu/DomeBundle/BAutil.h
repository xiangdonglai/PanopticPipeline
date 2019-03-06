#pragma once

#define GLOG_NO_ABBREVIATED_SEVERITIES
#define NOMINMAX

// #include <Windows.h> // need for get num of CPU cores
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#ifdef NDEBUG
#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif
#endif
#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/algorithm/string.hpp>


//options for intrinsic parameters
#define BA_OPT_INTRINSIC_ALL             0 //Default
#define BA_OPT_INTRINSIC_ALL_FIXED       1
#define BA_OPT_INTRINSIC_SKEW_FIXED      2
#define BA_OPT_INTRINSIC_SKEW_ZERO_FIXED 3
#define BA_OPT_INTRINSIC_CENTER_FIXED    4

#define BA_OPT_LOAD_INTRINSICS_BUT_JUST_AS_INITIAL_GUESS 10

//options for lens distortion
#define BA_OPT_LENSDIST_RADIAL_AND_TANGENTIAL 0 //Default
#define BA_OPT_LENSDIST_ALL_FIXED             1
#define BA_OPT_LENSDIST_RADIAL_1ST_ONLY       2
#define BA_OPT_LENSDIST_RADIAL_ONLY           3
//#define BA_OPT_LENSDIST_TANGENTIAL_ZERO       3
//#define BA_OPT_LENSDIST_PRISM_ZERO            4

//#define BA_OPT_CONVERT_1ST_RADIAL_BEFORE_BA   true


//options for extrinsic parameters
#define BA_OPT_EXTRINSIC_ALL        0//Default
#define BA_OPT_EXTRINSIC_ALL_FIXED  1//All extrinsic params are not optimized.
#define BA_OPT_EXTRINSIC_R_FIXED    2
#define BA_OPT_EXTRINSIC_T_FIXED    3

//TODO:
//#define BA_LENS_PINHOLE 0
//#define BA_LENS_FISHEYE 1


namespace BA
{

struct CameraData
{
	std::string filename;
	int imgWidth;
	int imgHeight;

	//intrinsic
	double FocalLength[2];
	double OpticalCenter[2];
	double Skew;

	//lens distortion
	double Radialfirst;
	double Radialothers[2];
	double Tangential[2];
	double Prism[2];

	//extrinsic
	double AngleAxis[3];
	double Translation[3];

	//constraints
	int opt_intrinsic;
	int opt_lensdistortion;
	int opt_extrinsic;


	bool available;//if false, not used in BA
	//int lenstype;


	void copyIntrinsics(CameraData &dst) const
	{
		dst.FocalLength[0]   = this->FocalLength[0];
		dst.FocalLength[1]   = this->FocalLength[1];
		dst.OpticalCenter[0] = this->OpticalCenter[0];
		dst.OpticalCenter[1] = this->OpticalCenter[1];
		dst.Skew             = this->Skew;

		dst.Radialfirst     = this->Radialfirst;
		dst.Radialothers[0] = this->Radialothers[0];
		dst.Radialothers[1] = this->Radialothers[1];
		dst.Tangential[0]   = this->Tangential[0];
		dst.Tangential[1]   = this->Tangential[1];
		dst.Prism[0]        = this->Prism[0];
		dst.Prism[1]        = this->Prism[1];
	}

	void copyExtrinsics(CameraData &dst) const
	{
		for (int i = 0; i < 3; i++)
		{
			dst.AngleAxis[i] = this->AngleAxis[i];
			dst.Translation[i] = this->Translation[i];
		}
	}
	
	void dispParams() const
	{
		std::cout << "\n"
			<< "Name        : " << filename << "\n"
			<< "Available   : " << std::string(available ? "Yes" : "No") << "\n"
			<< "\n"
			<< "Focal Length: " << FocalLength[0] << ", " << FocalLength[1] << "\n"
			<< "Image Center: " << OpticalCenter[0] << ", " << OpticalCenter[1] << "\n"
			<< "Skew        : " << Skew << "\n"
			<< "Radial      : " << Radialfirst << ", " << Radialothers[0] << ", " << Radialothers[1] << ", " << "\n"
			<< "Tangential  : " << Tangential[0] << ", " << Tangential[1] << "\n"
			<< "Prism       : " << Prism[0] << ", " << Prism[1] << "\n"
			<< "\n"
			<< "Angle Axis  : " << AngleAxis[0] << ", " << AngleAxis[1] << ", " << AngleAxis[2] << "\n"
			<< "Translation : " << Translation[0] << ", " << Translation[1] << ", " << Translation[2] << "\n"
			<< std::endl;
	}

};

struct PointData
{
	Eigen::Vector3d              xyz;    // 3D coordinates
	Eigen::Vector3i              rgb;    // color information
	std::vector<int>             camID;  // camID[i]: i-th camera observing this point
	std::vector<bool>            inlier; // true: camID[i] is used for BA
	std::vector<Eigen::Vector2d> uv;     // uv[i]: 2D point (u,v) on i-th camera image
	std::vector<int>             fID;    // ID for uv[i] in camID[i]

	size_t nObserved(void) const { return camID.size(); };
	size_t nInliers(void)  const { return std::count(inlier.begin(), inlier.end(), true); };
	void   disp() const
	{
		std::cout << "\n"
			<< "XYZ: " << xyz.transpose() << "\n"
			<< "RGB: " << rgb.transpose() << "\n"
			<< "camID[inlier]: " << this->nObserved() << " cameras (" << this->nInliers()<< " inliers)\n";
		for (size_t i = 0; i < camID.size(); i++)
			std::cout << camID[i] << "[" << inlier[i] << "], ";

		std::cout << "\n" 
			<<"uv: ";
		for (size_t i = 0; i < uv.size(); i++)
			std::cout << "[" << uv[i].transpose() << "], ";

		std::cout << "\n"
			<<"fID: ";
		// for each(int fid in fID)
		for (int fid: fID)
			std::cout << fid << ", ";

		std::cout << "\n\n";

	};
};

struct NVM
{
	int nCamera;
	std::vector<std::string>           filenames;
	std::vector<double>                focallength;
	std::vector<std::vector<double>>   quaternion;
	std::vector<std::vector<double>>   position;
	std::vector<double>                firstradial;

	std::map<std::string, int>         filename_id;//[filename, camID]

	int n3Dpoints;
	int nInliers;
	std::vector<PointData> pointdata;
};



struct residualData
{
	std::vector<std::pair<int, int>> ID;                 //[pointID, cameraID]
	std::vector<Eigen::Vector2d>     observed_pt;        //[u, v]
	std::vector<Eigen::Vector2d>     reprojected_pt;     //[u, v]
	std::vector<Eigen::Vector2d>     error;              //[err_u, err_v]
	double                           mean_abs_error[2];  //(err_u, err_v)
};


template <typename T>
inline void PinholeReprojection(
	const T* const FocalLength,
	const T* const OpticalCenter,
	const T* const Skew,
	const T* const Radialfirst,
	const T* const Radialothers,
	const T* const Tangential,
	const T* const Prism,
	const T* const AngleAxis,
	const T* const Translation,
	const T* const point3D,
	T* reprojected2D)
{

	T fx = FocalLength[0];
	T fy = FocalLength[1];
	T s  = Skew[0];
	T u0 = OpticalCenter[0];
	T v0 = OpticalCenter[1];
	T a0 = Radialfirst[0], a1 = Radialothers[0], a2 = Radialothers[1];
	T p0 = Tangential[0],  p1 = Tangential[1];
	T s0 = Prism[0],       s1 = Prism[1];

	//rotate 3d point
	T pt[3];
	ceres::AngleAxisRotatePoint(AngleAxis, point3D, pt);

	//translate 3d point
	pt[0] += Translation[0];
	pt[1] += Translation[1];
	pt[2] += Translation[2];

	// Project to normalize coordinate
	T xn = pt[0] / pt[2];
	T yn = pt[1] / pt[2];

	//apply lens distortion
	T r2 = xn*xn + yn*yn;
	T r4 = r2*r2;
	T r6 = r2*r4;
	T X2 = xn*xn;
	T Y2 = yn*yn;
	T XY = xn*yn;

	T radial       = T(1.0) + a0*r2 + a1*r4 + a2*r6;
	T tangential_x = p0*(r2 + T(2.0)*X2) + T(2.0)*p1*XY;
	T tangential_y = p1*(r2 + T(2.0)*Y2) + T(2.0)*p0*XY;
	T prism_x      = s0*r2;
	T prism_y      = s1*r2;

	T xm = radial*xn + tangential_x + prism_x;
	T ym = radial*yn + tangential_y + prism_y;

	reprojected2D[0] = fx*xm + s*ym + u0;
	reprojected2D[1] = fy*ym + v0;
}



struct PinholeReprojectionError 
{
	PinholeReprojectionError(double observed_x, double observed_y)
		: observed_x(observed_x), observed_y(observed_y) {}

	template <typename T>
	bool operator()(
		const T* const FocalLength,
		const T* const OpticalCenter,
		const T* const Skew,
		const T* const Radialfirst,
		const T* const Radialothers,
		const T* const Tangential,
		const T* const Prism,
		const T* const AngleAxis,
		const T* const Translation,
		const T* const point3D,
		T* residuals)
		const
	{
		T reprojected2D[2];
		PinholeReprojection(FocalLength, OpticalCenter, Skew, Radialfirst, Radialothers, Tangential, Prism, AngleAxis, Translation, point3D, reprojected2D);

		residuals[0] = reprojected2D[0] - T(observed_x);
		residuals[1] = reprojected2D[1] - T(observed_y);


		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(const double observed_x, const double observed_y)
	{
		return (new ceres::AutoDiffCostFunction<PinholeReprojectionError, 2, 2, 2, 1, 1, 2, 2, 2, 3, 3, 3>(
			new PinholeReprojectionError(observed_x, observed_y)));
	}

	double observed_x;
	double observed_y;
};




bool loadNVM(const std::string filepath, NVM &nvmdata, const int nMinViews = 2);
bool loadNVMfast(const std::string filepath, NVM &nvmdata, const int nMinViews = 2);
bool loadMultipleNVM(const std::vector<std::string>& filepathVect, NVM &nvmdata, const int nMinViews = 2);
bool loadInitialIntrinsics(const std::string intrinsicfile, const std::map<std::string, int> &filename_id, std::vector<CameraData> &camera);
bool loadIntrinsics(const std::string intrinsicfile, std::vector<CameraData> &camera);
bool loadExtrinsics(const std::string extrinsicfile, std::vector<CameraData> &camera);
bool loadAllCameraParams(const std::string cameraparamfile, std::vector<CameraData> &camera);
bool loadAllCameraParamsfast(const std::string cameraparamfile, std::vector<CameraData> &camera);
bool initCameraData(const NVM &nvmdata, const std::string filepath, std::vector<CameraData> &camera);
void shiftImageCenter(const std::vector<CameraData> &cameradata, std::vector<PointData> &pointdata);




bool saveAllData(const std::string savefolder, const std::vector<CameraData> &camera, const std::vector<PointData> &pointdata, const residualData &res, const std::string prefix, const bool After);
bool saveCameraAllParams(const std::string file_fullpath, const std::string separator, const std::vector<CameraData> &camera);
bool saveCameraIntrinsics(const std::string file_fullpath, const std::string separator, const std::vector<CameraData> &camera);
bool saveCameraExtrinsics(const std::string file_fullpath, const std::string separator, const std::vector<CameraData> &camera);
bool save3Dpoints(const std::string file_fullpath, const std::string separator, const std::vector<PointData> &pointdata);
bool saveReprojectionError(const std::string file_fullpath, const std::string separator, const residualData &res, const std::vector<CameraData> &camera, const int order);
bool saveNVM(const std::string savefolder, const std::string outputnvmname, const std::vector<CameraData> &cameradata, const std::vector<PointData> &pointdata);
bool saveCeresReport(const std::string savefolder, const std::string filename, const ceres::Solver::Summary &summary);


void calcReprojectionError(const std::vector<CameraData> &cameradata, const std::vector<PointData> &pointdata, residualData &res);
void runBundleAdjustment(std::vector<CameraData> &cameradata, std::vector<BA::PointData> &pointdata, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, ceres::LossFunction *loss_funcion = NULL, const double thresh = -1.0);


void setConstantParams(CameraData &cam, ceres::Problem &problem);

void setCeresOption(const NVM &nvmdata, ceres::Solver::Options &options);
void setCeresOption(const int nCameras, ceres::Solver::Options &options);


}