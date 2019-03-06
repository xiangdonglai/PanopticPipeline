#include "DomeCalibUtil.h"

using namespace std;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
typedef Eigen::Matrix<double, 3, 4> Matrix34d;

typedef long long _Longlong;
typedef unsigned long long _ULonglong;

Eigen::IOFormat MatlabFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "];");


namespace DC
{

inline void distortPointVSFM_closedform(Eigen::Vector2d &uv, const double d)
{
	distortPointVSFM_closedform(uv[0], uv[1], d);
}

inline void distortPointVSFM_closedform(double &x, double &y, const double d)
{
	double zero = 1e-12;

	if (abs(d) <= zero)
		return;


	if (abs(y) <= zero)
		y = zero;


	double new_y;

	double t2 = y*y;
	double t3 = t2*t2*t2;
	double t4 = x*x;
	double t7 = d*(t2 + t4);
	if (d > 0.0)
	{
		double t8  = 1.0 / t7;
		double t10 = t3 / (t7*t7);
		double t14 = sqrt(t10*(0.25 + t8 / 27.0));
		double t15 = t2*t8*y*0.5;
		double t17 = pow(t14 + t15, 1.0 / 3.0);
		double t18 = t17 - t2*t8 / (t17*3.0);

		new_y = t18;

	}
	else
	{
		typedef complex<double> C;

		double t9  = t3 / (t7*t7*4.0);
		double t11 = t3 / (t7*t7*t7*27.0);
		double t12 = t9 + t11;
		C      t13 = sqrt(t12);
		double t14 = t2 / t7;
		double t15 = t14*y*0.5;
		C      t16 = t13 + t15;
		C      t17 = pow(t16, 1.0 / 3.0);
		C      t18 = (t17 + t14 / (t17*3.0)) * C(0.0, sqrt(3.0));
		C      t19 = -0.5*(t17 + t18) + t14 / (t17*6.0);

		new_y = t19.real();
	}


	x = new_y * x / y;
	y = new_y;

}



bool readPCcorresSparse(const string filename, std::vector< std::vector<Eigen::Vector2d> > &PCpts)
{
	ifstream ifs(filename);
	if (ifs.fail())
	{
		cout << "Cannot open : " << filename << endl;
		return false;
	}


	stringstream ss;
	ss << ifs.rdbuf();
	ifs.close();


	//initialize PCpts
	if (!PCpts.empty())
		for (size_t i = 0; i < PCpts.size(); i++)
			PCpts[i].clear();

	int nPoints, nDevices;
	ss >> nPoints >> nDevices;
	ss.get();

	PCpts = vector<vector<Vector2d>>(nPoints, vector<Vector2d>(nDevices, Vector2d(0, 0)));


	string str;
	while (getline(ss, str))
	{
		vector<string> str_vec;
		boost::algorithm::split(str_vec, str, boost::is_space());

		int ptID = stoi(str_vec[0]),
			devID = stoi(str_vec[1]);

		PCpts[ptID][devID][0] = stod(str_vec[2]);
		PCpts[ptID][devID][1] = stod(str_vec[3]);
	}


	return true;
}

bool readDeviceNames(const string filename, vector<string> &devicename)
{
	ifstream ifs(filename);
	if (ifs.fail())
	{
		cerr << "Cannot open " << filename << endl;
		return false;
	}

	string str;
	while (getline(ifs, str))
	{
		devicename.push_back(str);
	}
}

bool readNumProjPoints(const string filename, map<string, int> &nProjPoints)
{
	ifstream ifs(filename);
	if (ifs.fail())
	{
		cerr << "Cannot open " << filename << endl;
		return false;
	}

	nProjPoints.clear();
	string str;
	while (getline(ifs, str))
	{
		stringstream ss(str);
		string projname;
		int npoints;

		ss >> projname >> npoints;
		nProjPoints.insert(make_pair(projname, npoints));
	}

}


//bool read_triangulatePCcorresSparse(const string devicenamefile, const string nProjPointsfile, const string PCcorresSparsefile, const map<string, int> &camname_id, const vector<BA::CameraData> &cameradata, vector<BA::PointData> &projpoints)
//{
//	vector<string> devicename;
//	DC::readDeviceNames(devicenamefile, devicename);
//
//	const int nProjectors = 5;
//	vector<BA::CameraData> projdata(nProjectors);
//	for (int i = 0; i < nProjectors; i++)
//		projdata[i].filename = devicename[i];
//
//
//
//
//
//
//
//	ifstream ifs(PCcorresSparsefile);
//	if (ifs.fail())
//	{
//		cout << "Cannot open : " << PCcorresSparsefile << endl;
//		return false;
//	}
//
//	stringstream ss;  // load on memory
//	ss << ifs.rdbuf();
//	ifs.close();
//
//
//
//	int nPoints, nDevices;
//	ss >> nPoints >> nDevices;
//	ss.get();
//
//	projpoints.resize(nPoints);
//
//
//	string str;
//	while (getline(ss, str))
//	{
//		vector<string> str_vec;
//		boost::algorithm::split(str_vec, str, boost::is_space());
//
//		int ptID  = stoi(str_vec[0]),
//			devID = stoi(str_vec[1]);//device ID on PCcorres
//
//		map<string, int>::const_iterator itr = camname_id.find(devicename[devID]);
//		if (itr == camname_id.end())
//			continue;
//
//		int camID = itr->second; // camera ID on 
//
//		PCpts[ptID][devID][0] = stod(str_vec[2]);
//		PCpts[ptID][devID][1] = stod(str_vec[3]);
//	}
//
//
//	return true;
//}


//bool read_triangulatePCcorresSparse(const string filename, const , const vector<string> &devicename, const map<string, int> &camname_id, const vector<BA::CameraData> &cameradata, vector<BA::PointData> &projpoints)
//{
//	ifstream ifs(filename);
//	if (ifs.fail())
//	{
//		cout << "Cannot open : " << filename << endl;
//		return false;
//	}
//
//
//	stringstream ss;
//	ss << ifs.rdbuf();
//	ifs.close();
//
//
//
//	int nPoints, nDevices;
//	ss >> nPoints >> nDevices;
//	ss.get();
//
//
//
//	string str;
//	while (getline(ss, str))
//	{
//		vector<string> str_vec;
//		boost::algorithm::split(str_vec, str, boost::is_space());
//
//		int ptID = stoi(str_vec[0]),
//			devID = stoi(str_vec[1]);
//
//		PCpts[ptID][devID][0] = stod(str_vec[2]);
//		PCpts[ptID][devID][1] = stod(str_vec[3]);
//	}
//
//
//	return true;
//}

//bool loadProjectorFeaturePoints(const string filename, vector<Vector2d> &ProjPoints)
//{
//	ifstream ifs(filename);
//	if (ifs.fail())
//	{
//		cerr << "Cannot open " << filename << endl;
//		return false;
//	}
//
//
//	int nPoints;
//	ifs >> nPoints;
//	
//	ProjPoints.resize(nPoints);
//	for (int i = 0; i < nPoints; i++)
//	{
//		double u, v;
//		ifs >> u >> v;
//		ProjPoints[i](0) = u;
//		ProjPoints[i](1) = v;
//	}
//
//	return true;
//}
//
bool loadProjectorFeaturePoints(const string filebase, const int nProjectors, vector<Vector2d> &ProjPoints, vector<int> &numProjPoints)
{
	numProjPoints.resize(nProjectors);
	vector<vector<Vector2d>> ProjPoints_i(nProjectors);
	for (int i = 0; i < nProjectors; i++)
	{
		string filename = filebase + to_string(_ULonglong(i)) + ".txt";
		ifstream ifs(filename);
		if (ifs.fail())
		{
			cerr << "Cannot open " << filename << endl;
			return false;
		}


		int nPoints;
		ifs >> nPoints;


		numProjPoints[i] = nPoints;
		ProjPoints_i[i].resize(nPoints);
		for (int j = 0; j < nPoints; j++)
		{
			double u, v;
			ifs >> u >> v;
			ProjPoints_i[i][j](0) = u;
			ProjPoints_i[i][j](1) = v;
		}
	}

	ProjPoints.clear();
	for (int i = 0; i < nProjectors; i++)
		ProjPoints.insert(ProjPoints.end(), ProjPoints_i[i].begin(), ProjPoints_i[i].end());


	return true;
}

//void removeDuplicatePoints(vector<BA::PointData> &pointdata)
//{
//	size_t nPoints = pointdata.size();
//	vector<int>           reg_fid;       reg_fid.reserve(nPoints);
//	vector<int>           reg_views;     reg_views.reserve(nPoints);
//
//	for (size_t i = 0; i < nPoints; i++)
//	{
//		int fid   = pointdata[i].fID[0];
//		int views = pointdata[i].nObserved();
//
//		vector<int>::iterator itr = find(reg_fid.begin(), reg_fid.end(), fid);
//
//		if (itr == reg_fid.end()) // new point
//		{
//			reg_fid.push_back(fid);
//			reg_views.push_back(views);
//		}
//		else                      // same fid exists, overwrite if nviews are larger
//		{
//			int idx = itr - reg_fid.begin();
//			if (views > reg_views[idx])
//			{
//				reg_fid[idx]   = fid;
//				reg_views[idx] = views;
//			}
//		}
//	}
//
//
//	size_t nUnique = reg_fid.size();
//	vector<BA::PointData> new_pointdata(nUnique);
//	for (size_t i = 0; i < nUnique; i++)
//		new_pointdata[i] = pointdata[reg_fid[i]];
//
//
//	pointdata = std::move(new_pointdata);
//}
//
void disableDuplicatePoints(vector<BA::PointData> &pointdata)
{
	size_t nPoints = pointdata.size();
	vector<int>  reg_ptIdx;   reg_ptIdx.reserve(nPoints);
	vector<int>  reg_fid;   reg_fid.reserve(nPoints);
	vector<int>  reg_views; reg_views.reserve(nPoints);

	for (size_t i = 0; i < nPoints; i++)
	{
		int fid   = pointdata[i].fID[0];
		int views = (int)pointdata[i].nObserved();
		
		if(pointdata[i].inlier.front()==false)		//ignore
			continue;

		vector<int>::iterator itr = find(reg_fid.begin(), reg_fid.end(), fid);

		if (itr == reg_fid.end()) // new point
		{
			reg_ptIdx.push_back(i);
			reg_fid.push_back(fid);
			reg_views.push_back(views);
		}
		else                      // same fid exists, overwrite if nviews are larger
		{
			int idx = itr - reg_fid.begin();
			if (views > reg_views[idx])
			{
				reg_ptIdx[idx] = i;
				reg_fid[idx]   = fid;
				reg_views[idx] = views;
			}
		}
	}

	for (size_t i = 0; i < nPoints; i++)
		fill(pointdata[i].inlier.begin(), pointdata[i].inlier.end(), false);

	/*for each(int i in reg_fid)
		fill(pointdata[i].inlier.begin(), pointdata[i].inlier.end(), true);*/

	for (size_t i = 0; i < reg_ptIdx.size(); i++)
	{
		int idx  = reg_ptIdx[i];
		fill(pointdata[idx].inlier.begin(), pointdata[idx].inlier.end(), true);
	}
}


bool calibrateProjectors(const string projpointfilebase, const int nProjectors, const int nCameras, vector<BA::CameraData> &projdata, vector<BA::PointData> &pointdata)
{
	if (projdata.size()!=nProjectors)
		projdata.resize(nProjectors);

	string projnamebase = "p";
	for (int i = 0; i < nProjectors; i++)
	{
		BA::CameraData *proj = &projdata[i];

		proj->filename  = projnamebase + to_string(_ULonglong(i)) + ".jpg";
		proj->available = true;
		proj->imgWidth  = 1280;
		proj->imgHeight = 800;

		proj->opt_intrinsic      = BA_OPT_INTRINSIC_ALL;
		proj->opt_extrinsic      = BA_OPT_EXTRINSIC_ALL;
		proj->opt_lensdistortion = BA_OPT_LENSDIST_RADIAL_AND_TANGENTIAL;
	}




	cout << "\tLoading feature points on projector images...\n";
	vector<Vector2d> Proj2Dpoints;
	vector<int> numProjPoints;
	if (!loadProjectorFeaturePoints(projpointfilebase, nProjectors, Proj2Dpoints, numProjPoints))
		return false;





	// calc projection matrix for each projectors by 2D-3D correspondences
	cout << "\tEstimating Projectors' camera parameters...";

	size_t n3Dpoints = pointdata.size();
	vector<vector<Vector2d>> m(nProjectors); //m[i](u, v)   : 2D points (u, v) on i-th projector image
	vector<vector<Vector3d>> X(nProjectors); //X[i](x, y, z): 3D points (x, y, z) corresponding to m[i](u, v)
	for (size_t pt3D_id = 0; pt3D_id < n3Dpoints; pt3D_id++)
	{
		BA::PointData *ptdata = &pointdata[pt3D_id];

		if (ptdata->nInliers() == 0)
			continue;


		int fID = ptdata->fID[0]; //all featureIDs are same in Projector-Camera calibration
		int pt2D_id = fID;

		int projID;
		for (int i = 0, sumpts = 0; i < nProjectors; i++)
		{
			if (fID > sumpts && fID < sumpts + numProjPoints[i])
				projID = i;

			sumpts += numProjPoints[i];
		}


		m[projID].push_back(Proj2Dpoints[pt2D_id]);
		X[projID].push_back(pointdata[pt3D_id].xyz);


		//add points to PointData
		ptdata->uv.push_back(Proj2Dpoints[pt2D_id]);
		ptdata->camID.push_back(projID + nCameras);
		ptdata->fID.push_back(fID);
		ptdata->inlier.push_back(true);
	}



	cout << "DLT...";
	vector<Matrix34d> P(nProjectors);  //P[i]: projection matrix of i-th projector
	vector<Matrix3d>  R(nProjectors);  //R[i]: rotation matrix
	vector<Matrix3d>  K(nProjectors);  //K[i]: intrinsic parameter matrix
	vector<Vector3d>  t(nProjectors);  //t[i]: translation vector
	for (int projID = 0; projID < nProjectors; projID++)
	{
		vector<bool> status;
		bool success = estimateProjectionMatrixRANSAC(m[projID], X[projID], P[projID], status, 10.0, false);
		if (!success)
		{
			cerr << "Error: failed to calcurate a projection matrix of a projector." << endl;
			return false;
		}

		decomposePtoKRT(P[projID], K[projID], R[projID], t[projID]);
		copyKRTtoCameraData(K[projID], R[projID], t[projID], projdata[projID]);

		//projdata[projID].dispParams();
	}



	cout << "Non-linear Optimization...\n";
	for (int projID = 0; projID < nProjectors; projID++)
	{
		ceres::Solver::Options options;
		ceres::Solver::Summary summary;
		ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);

		BA::setCeresOption(1, options);
		options.minimizer_progress_to_stdout = false;



		BA::CameraData proj_before(projdata[projID]);



		optimizeKRTwithDistortion(projID + nCameras, projdata[projID], pointdata, options, summary, loss_function);

		//cout << summary.FullReport();
		//proj_before.dispParams();
		//projdata[projID].dispParams();
	}

	return true;
}


//
//bool calibrateProjectors(vector<Vector2d> &proj2Dpoints, vector<BA::CameraData> &projdata, vector<BA::PointData> &pointdata, vector<vector<double>> &xyz, int nCameras)
//{
//	size_t nProjectors   = projdata.size();
//	size_t n2Dprojpoints = proj2Dpoints.size(); //num of feature points in a projector image
//	size_t n3Dpoints     = xyz.size();
//	if (n3Dpoints != pointdata.size())
//	{
//		cerr << "the number of 3D points and 2D correspondences must be same.\n" << endl;
//		return false;
//	}
//
//
//	// calc projection matrix for each projectors by 2D-3D correspondences
//	vector<vector<Vector2d>> m(nProjectors); //m[i](u, v)   : 2D points (u, v) on i-th projector image
//	vector<vector<Vector3d>> X(nProjectors); //X[i](x, y, z): 3D points (x, y, z) corresponding to m[i](u, v)
//	for (size_t pt3D_id = 0; pt3D_id < n3Dpoints; pt3D_id++)
//	{
//		BA::PointData *ptdata = &pointdata[pt3D_id];
//
//		if (ptdata->nInliers() == 0)
//			continue;
//
//
//		int fID     = ptdata->featureID[0]; //all featureIDs are same in Projector-Camera calibration
//		int projID  = fID / n2Dprojpoints;
//		int pt2D_id = fID % n2Dprojpoints;
//
//
//		m[projID].push_back(proj2Dpoints[pt2D_id]);
//		X[projID].push_back(Vector3d(xyz[pt3D_id][0], xyz[pt3D_id][1], xyz[pt3D_id][2]));
//
//
//		//add points to PointData
//		vector<double> add_pt2D = { proj2Dpoints[pt2D_id](0), proj2Dpoints[pt2D_id](1) };
//		ptdata->point2D.push_back(add_pt2D);
//		ptdata->camID.push_back(projID + nCameras);
//		ptdata->featureID.push_back(fID);
//		ptdata->inlier.push_back(true);
//
//		
//		//add points to CameraData of projectors
//		BA::CameraData *pjdata = &projdata[projID];
//		pjdata->point2D.push_back(add_pt2D);
//		pjdata->ptID.push_back(pt3D_id);
//		pjdata->inlier.push_back(true);
//
//	}
//
//
//	//DLT
//	vector<Matrix34d> P(nProjectors);  //P[i]: projection matrix of i-th projector
//	vector<Matrix3d>  R(nProjectors);  //R[i]: rotation matrix
//	vector<Matrix3d>  K(nProjectors);  //K[i]: intrinsic parameter matrix
//	vector<Vector3d>  t(nProjectors);  //t[i]: translation vector
//	for (int projID = 0; projID < nProjectors; projID++)
//	{
//		vector<bool> status;
//		estimateProjectionMatrixRANSAC(m[projID], X[projID], P[projID], status, 3.0, true);
//
//		decomposePtoKRT(P[projID], K[projID], R[projID], t[projID]);
//
//
//		copyKRTtoCameraData(K[projID], R[projID], t[projID], projdata[projID]);
//		copy(status.begin(), status.end(), projdata[projID].inlier.begin());
//		BA::dispCameraParams(projdata[projID]);
//	}
//	//exit(0);
//
//
//	for (int projID = 0; projID < nProjectors; projID++)
//	{
//		ceres::Solver::Options options;
//		ceres::Solver::Summary summary;
//		ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
//
//		BA::setCeresOption(1, options);
//		options.minimizer_progress_to_stdout = false;
//		//cout << "\n\n";
//
//
//
//		projdata[projID].opt_intrinsic      = BA_OPT_INTRINSIC_ALL;
//		projdata[projID].opt_extrinsic      = BA_OPT_EXTRINSIC_ALL;
//		projdata[projID].opt_lensdistortion = BA_OPT_LENSDIST_RADIAL_AND_TANGENTIAL;
//
//		//BA::CameraData proj_before(projdata[projID]);
//
//
//
//		//optimizeKRTwithDistortion(m[projID], X[projID], options, summary, loss_function, projdata[projID]);
//		//optimizeKRTwithDistortion(X[projID], options, summary, loss_function, projdata[projID]);
//		solvePnPwithDistortion(xyz, options, summary, loss_function, projdata[projID]);
//
//		//cout << summary.FullReport();
//		//BA::dispCameraParams(proj_before);
//		//BA::dispCameraParams(projdata[projID]);
//	}
//
//
//
//
//	return true;
//}
//
//
//
//inline bool valid3Dpoint(const vector<double> &X, const vector<double> &NotValid_X, double thresh)
//{
//	if (abs(X[0] - NotValid_X[0]) > thresh &&
//		abs(X[1] - NotValid_X[1]) > thresh &&
//		abs(X[2] - NotValid_X[2]) > thresh)
//		return true;
//	else
//		return false;
//}
//
//
//void convertValidPointsToEigenMat(const int nProjectors, const int prjID, const int nPoints, const vector<vector<double>> &uv, const vector<vector<double>> &xyz, const vector<double> &NotValid_xyz, Eigen::MatrixXd &m, Eigen::MatrixXd &X)
//{
//	int nPoints_per_proj = nPoints / nProjectors;
//
//	vector<vector<double>> m_, X_;
//	for (int i = 0; i < nPoints_per_proj; i++)
//	{
//		int ptID = prjID*nPoints_per_proj + i;
//		int devID = 2 * prjID;
//
//		if (!DC::valid3Dpoint(xyz[ptID], NotValid_xyz))
//			continue;
//
//		vector<double> temp2d(2);
//		temp2d[0] = uv[ptID][devID];
//		temp2d[1] = uv[ptID][devID + 1];
//		m_.push_back(temp2d);
//
//		X_.push_back(xyz[ptID]);
//	}
//
//	int nValidPoints = m_.size();
//	m.resize(2, nValidPoints);
//	X.resize(3, nValidPoints);
//	for (int i = 0; i < nValidPoints; i++)
//	{
//		m(0, i) = m_[i][0];
//		m(1, i) = m_[i][1];
//
//		X(0, i) = X_[i][0];
//		X(1, i) = X_[i][1];
//		X(2, i) = X_[i][2];
//	}
//}
//
//void convertValidPointsToEigenMat(const int nPoints, const int camID, const vector<vector<double>> &uv, const vector< vector<bool> > visMap, const vector<vector<double>> &xyz, const vector<double> &NotValid_xyz, Eigen::MatrixXd &m, Eigen::MatrixXd &X)
//{
//
//	vector<vector<double>> m_, X_;
//	for (int ptID = 0; ptID < nPoints; ptID++)
//	{
//
//		if (!visMap[ptID][camID] || !DC::valid3Dpoint(xyz[ptID], NotValid_xyz))
//			continue;
//
//		vector<double> temp2d(2);
//		temp2d[0] = uv[ptID][2 * camID];
//		temp2d[1] = uv[ptID][2 * camID + 1];
//		m_.push_back(temp2d);
//
//		X_.push_back(xyz[ptID]);
//	}
//
//	int nValidPoints = m_.size();
//	m.resize(2, nValidPoints);
//	X.resize(3, nValidPoints);
//	for (int i = 0; i < nValidPoints; i++)
//	{
//		m(0, i) = m_[i][0];
//		m(1, i) = m_[i][1];
//
//		X(0, i) = X_[i][0];
//		X(1, i) = X_[i][1];
//		X(2, i) = X_[i][2];
//	}
//}
//
void copyKRTtoCameraData(const Eigen::Matrix3d &K, const Eigen::Matrix3d &R, const Eigen::Vector3d &t, BA::CameraData & camera)
{
	camera.FocalLength[0]   = K(0, 0);
	camera.FocalLength[1]   = K(1, 1);
	camera.Skew             = K(0, 1);
	camera.OpticalCenter[0] = K(0, 2);
	camera.OpticalCenter[1] = K(1, 2);

	camera.Radialfirst     = 0.0;
	camera.Radialothers[0] = 0.0;
	camera.Radialothers[1] = 0.0;
	camera.Tangential[0]   = 0.0;
	camera.Tangential[1]   = 0.0;
	camera.Prism[0]        = 0.0;
	camera.Prism[1]        = 0.0;

	ceres::RotationMatrixToAngleAxis(R.data(), camera.AngleAxis);
	
	camera.Translation[0] = t(0);
	camera.Translation[1] = t(1);
	camera.Translation[2] = t(2);
}


//void normalizePoints(const Eigen::MatrixXd pts, Eigen::MatrixXd newpts, Eigen::Matrix3d T)
//{
////TODO
//}
//
//// calculate K, R, t from 2D-3D correspondences.
//// m: 2xN, X: 3xN
//bool estimateProjectionMatrixDLT(const Eigen::MatrixXd &m_, const Eigen::MatrixXd &X_, Eigen::MatrixXd &P)
//{
//	int nPoints = m_.cols();
//	if (nPoints != X_.cols())
//	{
//		cerr << "Error: the number of 2D and 3D points must be same." << endl;
//		return false;
//	}
//	if (nPoints < 6)
//	{
//		cerr << "Error: the number of 2D and 3D points must be more than 6." << endl;
//		return false;
//	}
//
//	// normalization: TODO
//	//Eigen::MatrixXd m;
//	//Eigen::MatrixXd X;
//	//Eigen::Matrix3d T1, T2;
//	//normalizePoints(m_, m, T1);
//	//normalizePoints(X_, X, T2);
//	Eigen::MatrixXd m = m_;
//	Eigen::MatrixXd X = X_;
//
//
//	//// DLT
//	//Eigen::MatrixXd A(2 * nPoints, 12); // 2Nx12
//	//Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(1, 4);
//	//for (int ptID = 0; ptID < nPoints; ptID++)
//	//{
//
//	//	Eigen::MatrixXd Xt(1, 4);//homogenious [x,y,z,1]
//	//	Xt << X.col(ptID).transpose(), 1.0;
//
//	//	double u = m(0, ptID);
//	//	double v = m(1, ptID);
//
//	//	A.row(2 * ptID    ) << zero, -Xt,  v*Xt;
//	//	A.row(2 * ptID + 1) << Xt,  zero, -u*Xt;
//	//}
//	//Eigen::MatrixXd p = A.jacobiSvd(Eigen::ComputeThinV).matrixV().col(11); //p = V(:,end)
//	//P = Eigen::Map<Eigen::MatrixXd>(p.data(), 4, 3).transpose(); //P = reshape(p,4,3)'
//
//
//	Eigen::MatrixXd A(2 * nPoints, 12); // 2Nx12
//	Eigen::MatrixXd zero_Nx4 = Eigen::MatrixXd::Zero(nPoints, 4);
//	Eigen::MatrixXd ones_Nx1 = Eigen::MatrixXd::Ones(nPoints, 1);
//
//	Eigen::MatrixXd u = m.row(0).transpose();
//	Eigen::MatrixXd v = m.row(1).transpose();
//	Eigen::MatrixXd x = X.row(0).transpose();
//	Eigen::MatrixXd y = X.row(1).transpose();
//	Eigen::MatrixXd z = X.row(2).transpose();
//
//	A << zero_Nx4,         -x, -y, -z, -ones_Nx1,  v.array()*x.array(),  v.array()*y.array(),  v.array()*z.array(),  v,
//		x, y, z, ones_Nx1,              zero_Nx4, -u.array()*x.array(), -u.array()*y.array(), -u.array()*z.array(), -u;
//
//	Eigen::MatrixXd p = A.jacobiSvd(Eigen::ComputeThinV).matrixV().col(11); //p = V(:,end)
//	P = Eigen::Map<Eigen::MatrixXd>(p.data(), 4, 3).transpose(); //P = reshape(p,4,3)'
//
//
//	return true;
//}
//
bool estimateProjectionMatrixDLT(const vector<Vector2d> &m, const vector<Vector3d> &X, Matrix34d &P)
{
	size_t nPoints = m.size();
	if (nPoints != X.size())
	{
		cerr << "Error: the number of 2D and 3D points must be same." << endl;
		return false;
	}
	if (nPoints < 6)
	{
		cerr << "Error: the number of 2D and 3D points must be more than 6." << endl;
		return false;
	}


	MatrixXd A(2 * nPoints, 12); // 2Nx12
	Eigen::Matrix<double, 1, 4> zero = Eigen::Matrix<double, 1, 4>::Zero();
	for (int ptID = 0; ptID < nPoints; ptID++)
	{

		Eigen::Matrix<double, 1, 4> Xt;//homogenious [x,y,z,1]
		Xt << X[ptID].transpose(), 1.0;

		double u = m[ptID](0);
		double v = m[ptID](1);

		A.row(2 * ptID    ) << zero, -Xt,  v*Xt;
		A.row(2 * ptID + 1) << Xt,  zero, -u*Xt;
	}
	
	Eigen::Matrix<double, 12, 1> p = A.jacobiSvd(Eigen::ComputeThinV).matrixV().col(11); //p = V(:,end)
	P = Eigen::Map<Eigen::Matrix<double, 4, 3>>(p.data(), 4, 3).transpose(); //P = reshape(p,4,3)'


	return true;
}

void ransac_getparam(const Eigen::Matrix<double, 5, 1> &data, Vector2d &m, Vector3d &X)
{
	m[0] = data[0];
	m[1] = data[1];

	X[0] = data[2];
	X[1] = data[3];
	X[2] = data[4];
}

void ransac_PMat_fit(const MatrixXd &alldata, const vector<size_t> &useIndices, vector<MatrixXd> &fitModels)
{
	size_t n = useIndices.size();
	vector<Vector2d> m(n);
	vector<Vector3d> X(n);
	for (size_t i = 0; i < n; i++)
		ransac_getparam(alldata.col(useIndices[i]), m[i], X[i]);

	Matrix34d P;
	estimateProjectionMatrixDLT(m, X, P);

	fitModels.resize(1);
	fitModels[0] = P;
}



void ransac_PMat_dist(const MatrixXd &alldata, const vector< MatrixXd > &testModels, const double distanceThreshold, unsigned int &out_bestModelIndex, vector<bool> &inliers)
{
	out_bestModelIndex = 0;
	const MatrixXd &P = testModels[0];

	const int N = alldata.cols();
	for (int i = 0; i < N; i++)
	{
		Vector2d m;
		Vector3d X;
		ransac_getparam(alldata.col(i), m, X);

		X = P.block<3, 3>(0, 0)*X + P.col(3);
		double err_u = m[0] - X[0] / X[2];
		double err_v = m[1] - X[1] / X[2];

		double dist = sqrt(err_u*err_u + err_v*err_v);
		inliers[i] = (dist < distanceThreshold);
	}
}




bool estimateProjectionMatrixRANSAC(const std::vector<Eigen::Vector2d> &m, const std::vector<Eigen::Vector3d> &X, Eigen::Matrix<double, 3, 4> &P, vector<bool> &inliers, const double DIST_THRESHOLD, const bool verbose)
{
	size_t num_points = m.size();

	MatrixXd alldata(5, num_points);
	for (int i = 0; i < num_points; i++)
		alldata.col(i) << m[i](0), m[i](1), X[i](0), X[i](1), X[i](2);




	MatrixXd best_model;
	int      num_min_samples = 6;
	int      maxIter = 5000;

	bool found_solution = ransac::execute(
		alldata,
		ransac_PMat_fit,
		ransac_PMat_dist,
		NULL,
		DIST_THRESHOLD, num_min_samples, inliers, best_model, verbose, 0.999, maxIter);


	if (found_solution)
	{
		P = best_model;
		return true;
	}
	else
		return false;

}

// QR method for decomposition of P
bool decomposePtoKRT(const Matrix34d &P, Matrix3d &K, Matrix3d &R, Vector3d &t)
{
	Eigen::HouseholderQR<Matrix3d> qr(P.block(0, 0, 3, 3).inverse());
	Matrix3d Q = qr.householderQ();
	Matrix3d U = qr.matrixQR().triangularView<Eigen::Upper>();


	// sign correction
	if (U(0, 0) < 0)
	{
		Q.col(0) = -Q.col(0);
		U.row(0) = -U.row(0);
	}

	if (U(1, 1) < 0)
	{
		Q.col(1) = -Q.col(1);
		U.row(1) = -U.row(1);
	}

	if (U(2, 2) < 0)
	{
		Q.col(2) = -Q.col(2);
		U.row(2) = -U.row(2);
	}


	// recover K
	Matrix3d invU = U.inverse();
	K = invU / invU(2, 2);
	K(1, 0) = 0.0;
	K(2, 0) = 0.0;
	K(2, 1) = 0.0;


	// recover R, t
	R = Q.transpose(); // Q is orthogonal -> Q' == inv(Q)
	t = U * P.col(3);

	if (R.determinant() < 0)
	{
		R = -R;
		t = -t;
	}


	if (abs(R.determinant() - 1.0) > 0.001)
	{
		cerr << "R is not a rotation matrix" << endl;
		return false;
	}


	return true;
}
//
//// QR method for decomposition of P
//bool decomposePtoKRT(const Eigen::MatrixXd &P, Eigen::Matrix3d &K, Eigen::Matrix3d &R, Eigen::Vector3d &t)
//{
//	if (P.rows() != 3 || P.cols() != 4)
//	{
//		cerr << "P is not a 3x4 projection matrix." << endl;
//		return false;
//	}
//
//
//	Eigen::HouseholderQR<Eigen::Matrix3d> qr( P.block(0, 0, 3, 3).inverse() );
//	Eigen::Matrix3d Q = qr.householderQ();
//	Eigen::Matrix3d U = qr.matrixQR().triangularView<Eigen::Upper>();
//
//
//	// sign correction
//	if (U(0, 0) < 0)
//	{
//		//Eigen::Matrix3d S;
//		//S << -1, 0, 0,
//		//	  0, 1, 0,
//		//	  0, 0, 1;
//		//Q = Q*S;
//		//U = S*U;
//
//		Q.col(0) = -Q.col(0);
//		U.row(0) = -U.row(0);
//	}
//
//	if (U(1, 1) < 0)
//	{
//		//Eigen::Matrix3d S;
//		//S << 1,  0, 0,
//		//	 0, -1, 0,
//		//	 0,  0, 1;
//		//Q = Q*S;
//		//U = S*U;
//
//		Q.col(1) = -Q.col(1);
//		U.row(1) = -U.row(1);
//	}
//
//	if (U(2, 2) < 0)
//	{
//		//Eigen::Matrix3d S;
//		//S << 1, 0, 0,
//		//	0, 1,  0,
//		//	0, 0, -1;
//		//Q = Q*S;
//		//U = S*U;
//
//		Q.col(2) = -Q.col(2);
//		U.row(2) = -U.row(2);
//	}
//
//
//	// recover K
//	Eigen::Matrix3d invU = U.inverse();
//	K = invU / invU(2, 2);
//	K(1, 0) = 0.0;
//	K(2, 0) = 0.0;
//	K(2, 1) = 0.0;
//
//
//	// recover R, t
//	R = Q.transpose(); // Q is orthogonal -> Q' == inv(Q)
//	t = U * P.col(3);
//
//	if (R.determinant() < 0)
//	{
//		R = -R;
//		t = -t;
//	}
//
//
//	if (abs(R.determinant() - 1.0) > 0.001)
//	{
//		cerr << "R is not a rotation matrix" << endl;
//		return false;
//	}
//
//
//	return true;
//}
//
//bool solvePnPwithDistortion(vector<vector<double>> &xyz, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, ceres::LossFunction *loss_function, BA::CameraData &camera)
//{
//	size_t nPoints = camera.point2D.size();
//
//	ceres::Problem problem;
//
//	double
//		*focal      = camera.FocalLength,
//		*imcenter   = camera.OpticalCenter,
//		*skew       = &camera.Skew,
//		*rad_1st    = &camera.Radialfirst,
//		*rad_other  = camera.Radialothers,
//		*tangential = camera.Tangential,
//		*prism      = camera.Prism,
//		*angleaxis  = camera.AngleAxis,
//		*trans      = camera.Translation;
//
//	vector<double> *m = camera.point2D.data();
//
//	for (size_t i = 0; i < nPoints; i++)
//	{
//		if (!camera.inlier[i])
//			continue;
//
//
//		double observed_x = m[i][0];
//		double observed_y = m[i][1];
//
//		int ptID = camera.ptID[i];
//		double *pt3D = xyz[ptID].data();
//
//		ceres::CostFunction* cost_function = BA::PinholeReprojectionError::Create(observed_x, observed_y);
//		problem.AddResidualBlock(cost_function, loss_function, focal, imcenter, skew, rad_1st, rad_other, tangential, prism, angleaxis, trans, pt3D);
//
//		problem.SetParameterBlockConstant(pt3D);
//	}
//
//	BA::setConstantParams(camera, problem);
//	ceres::Solve(options, &problem, &summary);
//
//
//	return true;
//}
bool optimizeKRTwithDistortion(const int devID, BA::CameraData &camera, const std::vector<BA::PointData> &pointdata, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, ceres::LossFunction *loss_function)
{
	size_t nPoints = pointdata.size();

	ceres::Problem problem;

	double
		*focal      = camera.FocalLength,
		*imcenter   = camera.OpticalCenter,
		*skew       = &camera.Skew,
		*rad_1st    = &camera.Radialfirst,
		*rad_other  = camera.Radialothers,
		*tangential = camera.Tangential,
		*prism      = camera.Prism,
		*angleaxis  = camera.AngleAxis,
		*trans      = camera.Translation;


	for (size_t ptID = 0; ptID < nPoints; ptID++)
	{
		const BA::PointData *ptdata = &pointdata[ptID];

		vector<int>::const_iterator itr = find(ptdata->camID.begin(), ptdata->camID.end(), devID);
		if (itr == ptdata->camID.end())
			continue;


		int idx = itr - ptdata->camID.begin();
		if (!ptdata->inlier[idx])
			continue;


		double observed_x = ptdata->uv[idx][0];
		double observed_y = ptdata->uv[idx][1];

		double *pt3D = const_cast<double *>(ptdata->xyz.data());

		ceres::CostFunction* cost_function = BA::PinholeReprojectionError::Create(observed_x, observed_y);
		problem.AddResidualBlock(cost_function, loss_function, focal, imcenter, skew, rad_1st, rad_other, tangential, prism, angleaxis, trans, pt3D);

		problem.SetParameterBlockConstant(pt3D);
	}

	BA::setConstantParams(camera, problem);
	ceres::Solve(options, &problem, &summary);


	return true;
}

//bool optimizeKRTwithDistortion(vector<Vector3d> &X, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, ceres::LossFunction *loss_function, BA::CameraData &camera)
//{
//	size_t nPoints = camera.point2D.size();
//	if (nPoints != X.size())
//	{
//		cerr << "Error: the number of 2D and 3D points must be same." << endl;
//		return false;
//	}
//
//	ceres::Problem problem;
//
//	double
//		*focal      = camera.FocalLength,
//		*imcenter   = camera.OpticalCenter,
//		*skew       = &camera.Skew,
//		*rad_1st    = &camera.Radialfirst,
//		*rad_other  = camera.Radialothers,
//		*tangential = camera.Tangential,
//		*prism      = camera.Prism,
//		*angleaxis  = camera.AngleAxis,
//		*trans      = camera.Translation;
//
//	vector<double> *m = camera.point2D.data();
//
//	for (size_t ptID = 0; ptID < nPoints; ptID++)
//	{
//		if (!camera.inlier[ptID])
//			continue;
//
//
//		double observed_x = m[ptID][0];
//		double observed_y = m[ptID][1];
//
//		double *pt3D = X[ptID].data();
//
//		ceres::CostFunction* cost_function = BA::PinholeReprojectionError::Create(observed_x, observed_y);
//		problem.AddResidualBlock(cost_function, loss_function, focal, imcenter, skew, rad_1st, rad_other, tangential, prism, angleaxis, trans, pt3D);
//
//		problem.SetParameterBlockConstant(pt3D);
//	}
//
//	BA::setConstantParams(camera, problem);
//	ceres::Solve(options, &problem, &summary);
//
//
//	return true;
//}
//
//bool optimizeKRTwithDistortion(const vector<Vector2d> &m, vector<Vector3d> &X, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, ceres::LossFunction *loss_function, BA::CameraData &camera)
//{
//	size_t nPoints = m.size();
//	if (nPoints != X.size())
//	{
//		cerr << "Error: the number of 2D and 3D points must be same." << endl;
//		return false;
//	}
//
//	ceres::Problem problem;
//
//	double
//		*focal      = camera.FocalLength,
//		*imcenter   = camera.OpticalCenter,
//		*skew       = &camera.Skew,
//		*rad_1st    = &camera.Radialfirst,
//		*rad_other  = camera.Radialothers,
//		*tangential = camera.Tangential,
//		*prism      = camera.Prism,
//		*angleaxis  = camera.AngleAxis,
//		*trans      = camera.Translation;
//
//
//	for (size_t ptID = 0; ptID < nPoints; ptID++)
//	{
//		if (!camera.inlier[ptID])
//			continue;
//
//		double observed_x = m[ptID](0);
//		double observed_y = m[ptID](1);
//
//		double *pt3D = X[ptID].data();
//
//		ceres::CostFunction* cost_function = BA::PinholeReprojectionError::Create(observed_x, observed_y);
//		problem.AddResidualBlock(cost_function, loss_function, focal, imcenter, skew, rad_1st, rad_other, tangential, prism, angleaxis, trans, pt3D);
//
//		problem.SetParameterBlockConstant(pt3D);
//	}
//
//	BA::setConstantParams(camera, problem);
//	ceres::Solve(options, &problem, &summary);
//
//
//	return true;
//}
//
//
//// refine all camera parameters
//// m: 2xN, X: 3xN
//bool optimizeKRTwithDistortion(const Eigen::MatrixXd &m, Eigen::MatrixXd &X, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, BA::CameraData &camera)
//{
//	int nPoints = m.cols();
//	if (nPoints != X.cols())
//	{
//		cerr << "Error: the number of 2D and 3D points must be same." << endl;
//		return false;
//	}
//
//	ceres::Problem problem;
//
//	double
//		*focal      = camera.FocalLength,
//		*imcenter   = camera.OpticalCenter,
//		*skew       = &camera.Skew,
//		*rad_1st    = &camera.Radialfirst,
//		*rad_other  = camera.Radialothers,
//		*tangential = camera.Tangential,
//		*prism      = camera.Prism,
//		*angleaxis  = camera.AngleAxis,
//		*trans      = camera.Translation;
//
//
//	for (int ptID = 0; ptID < nPoints; ptID++)
//	{
//		double observed_x = m(0, ptID);
//		double observed_y = m(1, ptID);
//
//		double *pt3D = &X(0, ptID);
//
//		ceres::CostFunction* cost_function = BA::PinholeReprojectionError::Create(observed_x, observed_y);
//		problem.AddResidualBlock(cost_function, NULL, focal, imcenter, skew, rad_1st, rad_other, tangential, prism, angleaxis, trans, pt3D);
//
//		problem.SetParameterBlockConstant(pt3D);
//	}
//
//
//	BA::setConstantParams(camera, problem);
//	ceres::Solve(options, &problem, &summary);
//
//
//	return true;
//}
//
//
//bool optimizeKRTwithDistortion(const Eigen::MatrixXd &m, Eigen::MatrixXd &X, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, BA::CameraData &camera, const Eigen::VectorXi inlier)
//{
//	int nPoints = m.cols();
//	if (nPoints != X.cols())
//	{
//		cerr << "Error: the number of 2D and 3D points must be same." << endl;
//		return false;
//	}
//
//	ceres::Problem problem;
//
//	double
//		*focal      = camera.FocalLength,
//		*imcenter   = camera.OpticalCenter,
//		*skew       = &camera.Skew,
//		*rad_1st    = &camera.Radialfirst,
//		*rad_other  = camera.Radialothers,
//		*tangential = camera.Tangential,
//		*prism      = camera.Prism,
//		*angleaxis  = camera.AngleAxis,
//		*trans      = camera.Translation;
//
//	int ninliers = inlier.rows();
//	for (int i = 0; i < ninliers; i++)
//	{
//		int ptID = inlier(i);
//
//		double observed_x = m(0, ptID);
//		double observed_y = m(1, ptID);
//
//		double *pt3D = &X(0, ptID);
//
//		ceres::CostFunction* cost_function = BA::PinholeReprojectionError::Create(observed_x, observed_y);
//		problem.AddResidualBlock(cost_function, NULL, focal, imcenter, skew, rad_1st, rad_other, tangential, prism, angleaxis, trans, pt3D);
//
//		problem.SetParameterBlockConstant(pt3D);
//	}
//
//
//	BA::setConstantParams(camera, problem);
//	ceres::Solve(options, &problem, &summary);
//
//
//	return true;
//}
//
//
void runBAall(const std::string savefolder, const std::vector<BA::CameraData> &camdata, const std::vector<BA::CameraData> &projdata, std::vector<BA::PointData> &pointdata)
{
	char buf[512];
	stringstream ss;
	size_t nCameras    = camdata.size();
	//size_t nProjectors = projdata.size();

	vector<BA::CameraData> allDevices(nCameras);
	copy(camdata.begin(),  camdata.end(),  allDevices.begin());
	//copy(projdata.begin(), projdata.end(), allDevices.begin() + nCameras);

	
	BA::residualData resdata_pre;
	BA::calcReprojectionError(allDevices, pointdata, resdata_pre);
	sprintf(buf,"Before BA: (%f, %f)=> %f \n",resdata_pre.mean_abs_error[0],resdata_pre.mean_abs_error[1],
		sqrt(pow(resdata_pre.mean_abs_error[0],2.0) + pow(resdata_pre.mean_abs_error[1],2.0) ));
	printf("%s",buf);
	ss <<buf;


	ceres::LossFunction *loss_func = new ceres::CauchyLoss(1.0);
	ceres::Solver::Summary summary;
	ceres::Solver::Options options;
	BA::setCeresOption(nCameras, options);

	BA::runBundleAdjustment(allDevices, pointdata, options, summary, loss_func);
	cout << summary.FullReport() << endl;

	BA::saveCeresReport(savefolder, "BA_CeresReport.txt", summary);
	BA::saveNVM(savefolder, "BA_caliball.nvm", allDevices, pointdata);
	
	BA::residualData resdata;
	BA::calcReprojectionError(allDevices, pointdata, resdata);
	sprintf(buf,"After BA: (%f, %f)=> %f \n",resdata.mean_abs_error[0],resdata.mean_abs_error[1],
		sqrt(pow(resdata.mean_abs_error[0],2.0) + pow(resdata.mean_abs_error[1],2.0) ));
	printf("%s",buf);
	ss <<buf;

	char reproErrorCompareFileName[512];
	sprintf(reproErrorCompareFileName,"%s/BA_ReprojectionError_changes.txt",savefolder.c_str());
	ofstream fout(reproErrorCompareFileName);
	fout << ss.str();
	fout.close();

	BA::saveAllData(savefolder, allDevices, pointdata, resdata, "BA_", true);
}

//void runBAall(std::string savefolder, std::vector<BA::CameraData> &camdata, std::vector<BA::CameraData> &projdata, BA::NVM &nvmdata)
//{
//	size_t nCameras    = camdata.size();
//	size_t nProjectors = projdata.size();
//	
//	vector<BA::CameraData> allDevices(nCameras + nProjectors);
//	copy(camdata.begin(),  camdata.end(),  allDevices.begin());
//	copy(projdata.begin(), projdata.end(), allDevices.begin() + nCameras);
//
//
//	ceres::LossFunction *loss_func = new ceres::CauchyLoss(1.0);
//	ceres::Solver::Summary summary;
//	ceres::Solver::Options options;
//	BA::setCeresOption(nCameras + nProjectors, options);
//
//	BA::runBundleAdjustment(allDevices, nvmdata.xyz, options, summary, nvmdata.pointdata, loss_func);
//	cout << summary.FullReport() << endl;
//
//
//
//	// save data
//	string ceres_report = savefolder + "/BA_CeresReport.txt";
//	ofstream ofs(savefolder + "/BA_CeresReport.txt");
//	if (ofs.fail())
//		cerr << "Cannot write " << ceres_report << endl;
//	else
//		ofs << summary.FullReport();
//	ofs.close();
//
//
//
//	BA::saveNVM(savefolder, "BA_caliball.nvm", allDevices, nvmdata.xyz, nvmdata.rgb, nvmdata.pointdata);
//
//
//
//	BA::residualData resdata;
//	BA::calcReprojectionError(allDevices, nvmdata.xyz, resdata);
//	BA::saveAllData(savefolder, allDevices, nvmdata.xyz, resdata, "BA_", true);
//
//
//}


}