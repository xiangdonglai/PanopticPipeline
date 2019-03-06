#pragma once

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>

#include <omp.h>

#ifdef NDEBUG
#define EIGEN_NO_DEBUG
#endif
#include <Eigen/Dense>


#include "BAutil.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ransac_headeronly.h"

namespace DC
{
	inline void distortPointVSFM_closedform(double &x, double &y, const double d);
	inline void distortPointVSFM_closedform(Eigen::Vector2d &uv, const double d);

	//bool readPCcorresSparse(const std::string filename, std::vector< std::vector<Eigen::Vector2d> > &PCpts);
	//bool readDeviceNames(const std::string filename, std::vector<std::string> &devicename);
	bool readNumProjPoints(const std::string filename, std::map<std::string, int> &nProjPoints);




	//inline bool valid3Dpoint(const std::vector<double> &X, const std::vector<double> &NotValid_X, double thresh = 1e-8);
	//void convertValidPointsToEigenMat(const int nProjectors, const int prjID, const int nPoints, const std::vector<std::vector<double>> &uv, const std::vector<std::vector<double>> &xyz, const std::vector<double> &NotValid_xyz, Eigen::MatrixXd &m, Eigen::MatrixXd &X);
	//void convertValidPointsToEigenMat(const int nPoints, const int camID, const std::vector<std::vector<double>> &uv, const std::vector< std::vector<bool> > visMap, const std::vector<std::vector<double>> &xyz, const std::vector<double> &NotValid_xyz, Eigen::MatrixXd &m, Eigen::MatrixXd &X);

	void copyKRTtoCameraData(const Eigen::Matrix3d &K, const Eigen::Matrix3d &R, const Eigen::Vector3d &t, BA::CameraData & camdata);


	//void normalizePoints(const Eigen::MatrixXd &pts, Eigen::MatrixXd &newpts, Eigen::Matrix3d &T);
	//bool decomposePtoKRT(const Eigen::MatrixXd &P, Eigen::Matrix3d &K, Eigen::Matrix3d &R, Eigen::Vector3d &t);
	//bool estimateProjectionMatrixDLT(const Eigen::MatrixXd &m_, const Eigen::MatrixXd &X_, Eigen::MatrixXd &P);
	//bool optimizeKRTwithDistortion(const Eigen::MatrixXd &m, Eigen::MatrixXd &X, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, BA::CameraData &camera);
	//bool optimizeKRTwithDistortion(const Eigen::MatrixXd &m, Eigen::MatrixXd &X, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, BA::CameraData &camera, const Eigen::VectorXi inlier);


	//void removeDuplicatePoints(std::vector<BA::PointData> &pointdata);
	void disableDuplicatePoints(std::vector<BA::PointData> &pointdata);


	//bool estimateProjectionMatrixRANSAC(const std::vector<std::vector<double>> &xyz, BA::CameraData &camera);
	bool estimateProjectionMatrixRANSAC(const std::vector<Eigen::Vector2d> &m, const std::vector<Eigen::Vector3d> &X, Eigen::Matrix<double, 3, 4> &P, std::vector<bool> &inliers, const double DIST_THRESHOLD = 3.0, const bool verbose = false);
	void ransac_PMat_dist(const Eigen::MatrixXd &alldata, const std::vector< Eigen::MatrixXd > &testModels, const double distanceThreshold, unsigned int &out_bestModelIndex, std::vector<bool> &inliers);

	////bool estimateProjectionMatrixRANSAC(const std::vector<Eigen::Vector2d> &m, const std::vector<Eigen::Vector3d> &X, Eigen::Matrix<double, 3, 4> &P, std::vector<size_t> &inlierPtID, const double DIST_THRESHOLD = 3.0, const bool verbose = false);
	////void ransac_PMat_dist(const Eigen::MatrixXd &alldata, const std::vector< Eigen::MatrixXd > &testModels, const double distanceThreshold, unsigned int &out_bestModelIndex, std::vector<size_t> &out_inlierIndices);
	void ransac_PMat_fit(const Eigen::MatrixXd &alldata, const std::vector<size_t> &useIndices, std::vector<Eigen::MatrixXd> &fitModels);
	void ransac_getparam(const Eigen::Matrix<double, 5, 1> &data, Eigen::Vector2d &m, Eigen::Vector3d &X);



	bool estimateProjectionMatrixDLT(const std::vector<Eigen::Vector2d> &m_, const std::vector<Eigen::Vector3d> &X_, Eigen::Matrix<double, 3, 4> &P);
	bool decomposePtoKRT(const Eigen::Matrix<double, 3, 4> &P, Eigen::Matrix3d &K, Eigen::Matrix3d &R, Eigen::Vector3d &t);
	////bool optimizeKRTwithDistortion(const std::vector<Eigen::Vector2d> &m, std::vector<Eigen::Vector3d> &X, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, ceres::LossFunction *loss_function, BA::CameraData &camera);
	//bool optimizeKRTwithDistortion(std::vector<Eigen::Vector3d> &X, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, ceres::LossFunction *loss_function, BA::CameraData &camera);
	bool optimizeKRTwithDistortion(const int devID, BA::CameraData &camera, const std::vector<BA::PointData> &pointdata, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, ceres::LossFunction *loss_function);
	//bool solvePnPwithDistortion(std::vector<std::vector<double>> &xyz, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, ceres::LossFunction *loss_function, BA::CameraData &camera);

	bool loadProjectorFeaturePoints(const std::string filebase, const int nProjectors, std::vector<Eigen::Vector2d> &ProjPoints, std::vector<int> &numProjPoints);
	//bool loadProjectorFeaturePoints(const std::string filename, std::vector<Eigen::Vector2d> &Proj2Dpoints);
	//bool calibrateProjectors(std::vector<Eigen::Vector2d> &proj2Dpoints, std::vector<BA::CameraData> &projdata, std::vector<BA::PointData> &pointdata, std::vector<std::vector<double>> &xyz, int nCameras);
	//bool calibrateProjectors(std::vector<Eigen::Vector2d> &proj2Dpoints, std::vector<int> &numProjPoints, std::vector<BA::CameraData> &projdata, std::vector<BA::PointData> &pointdata, std::vector<std::vector<double>> &xyz, int nCameras);
	//bool calibrateProjectors(std::vector<Eigen::Vector2d> &proj2Dpoints, std::vector<int> &numProjPoints, std::vector<BA::CameraData> &projdata, std::vector<BA::PointData> &pointdata, int nCameras);
	bool calibrateProjectors(const std::string projpointfilebase, const int nProjectors, const int nCameras, std::vector<BA::CameraData> &projdata, std::vector<BA::PointData> &pointdata);


	void runBAall(const std::string savefolder, const std::vector<BA::CameraData> &camdata, const std::vector<BA::CameraData> &projdata, std::vector<BA::PointData> &pointdata);
}