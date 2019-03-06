#include "BAutil.h"
#include <thread>

using namespace std;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector3i;

namespace BA
{

void calcReprojectionError(const vector<CameraData> &cameradata, const vector<PointData> &pointdata, residualData &res)
{

	const size_t n3Dpoints = pointdata.size();
	const size_t nCameras  = cameradata.size();


	res.ID.clear();
	res.ID.reserve(n3Dpoints*nCameras);
	
	res.error.clear();
	res.error.reserve(n3Dpoints*nCameras);
	
	res.observed_pt.clear();
	res.observed_pt.reserve(n3Dpoints*nCameras);

	res.reprojected_pt.clear();
	res.reprojected_pt.reserve(n3Dpoints*nCameras);



	double mean_abs_error[2] = { 0.0, 0.0 };

	for (size_t ptID = 0; ptID < n3Dpoints; ptID++)
	{
		const PointData *pt = &pointdata[ptID];
		if (pt->nInliers() < 2) //required at least 2 observations
			continue;


		const double *xyz = pt->xyz.data();


		for (size_t i = 0; i < pt->nObserved(); i++)
		{
			if (!pt->inlier[i])
				continue;

			int camID = pt->camID[i];

			const CameraData *cam = &cameradata[camID];

			const double
				*focal      = cam->FocalLength,
				*imcenter   = cam->OpticalCenter,
				*skew       = &cam->Skew,
				*rad_1st    = &cam->Radialfirst,
				*rad_other  = cam->Radialothers,
				*tangential = cam->Tangential,
				*prism      = cam->Prism,
				*angleaxis  = cam->AngleAxis,
				*trans      = cam->Translation;

			double
				observed_u = pt->uv[i][0],
				observed_v = pt->uv[i][1];


			Vector2d reprojected_pt;
			PinholeReprojection(focal, imcenter, skew, rad_1st, rad_other, tangential, prism, angleaxis, trans, xyz, reprojected_pt.data());


			Vector2d residual;
			residual[0] = reprojected_pt[0] - observed_u;
			residual[1] = reprojected_pt[1] - observed_v;
			res.error.push_back(residual);


			mean_abs_error[0] += abs(residual[0]);
			mean_abs_error[1] += abs(residual[1]);


			pair<int, int> ID(static_cast<int>(ptID), camID);
			res.ID.push_back(ID);

			res.reprojected_pt.push_back(reprojected_pt);
			res.observed_pt.push_back(pt->uv[i]);
		}
	}

	size_t nReprojections = res.error.size();

	res.mean_abs_error[0] = mean_abs_error[0] / nReprojections;
	res.mean_abs_error[1] = mean_abs_error[1] / nReprojections;

}


void setCeresOption(const NVM &nvmdata, ceres::Solver::Options &options)
{
	setCeresOption(nvmdata.nCamera, options);
}

void setCeresOption(const int nCameras, ceres::Solver::Options &options)
{
	// SYSTEM_INFO sysinfo;
	// GetSystemInfo(&sysinfo);
	// int nCPUs = sysinfo.dwNumberOfProcessors;
	int nCPUs = std::thread::hardware_concurrency();

	options.num_threads                  = nCPUs;
	options.num_linear_solver_threads    = nCPUs;
	options.max_num_iterations           = 200;
	options.minimizer_progress_to_stdout = true;
	//options.function_tolerance         = 1e-10;
	//options.gradient_tolerance         = 1e-10;

	if (nCameras < 200)
	{
		options.linear_solver_type         = ceres::DENSE_SCHUR;
		options.trust_region_strategy_type = ceres::DOGLEG;
		options.use_nonmonotonic_steps     = true;
	}
	else
	{
#if 0
		options.linear_solver_type         = ceres::ITERATIVE_SCHUR;
		options.preconditioner_type        = ceres::PreconditionerType::CLUSTER_JACOBI;
		options.visibility_clustering_type = ceres::VisibilityClusteringType::SINGLE_LINKAGE;
#else

		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.dynamic_sparsity   = true;
#endif

	}
}

void runBundleAdjustment(std::vector<CameraData> &cameradata, std::vector<BA::PointData> &pointdata, const ceres::Solver::Options &options, ceres::Solver::Summary &summary, ceres::LossFunction *loss_funcion, const double thresh)
{
	if (loss_funcion != NULL && thresh > 0)
	{
		cerr << "Warning: loss_function and outlier thresholding are used together. loss_function is disabled in BA.\n" << endl;

		loss_funcion = NULL;
	}

	ceres::Problem problem;
	const size_t n3Dpoints = pointdata.size();

	for (size_t ptID = 0; ptID < n3Dpoints; ptID++)
	{
		PointData *pt = &pointdata[ptID];
		if (pt->nInliers() < 2) //required at least 2 observations
			continue;


		double *xyz = pt->xyz.data();

		for (size_t i = 0; i < pt->nObserved(); i++)
		{
			if (!pt->inlier[i])
				continue;

			int camID = pt->camID[i];
			CameraData *cam = &cameradata[camID];
			double
				*focal      = cam->FocalLength,
				*imcenter   = cam->OpticalCenter,
				*skew       = &cam->Skew,
				*rad_1st    = &cam->Radialfirst,
				*rad_other  = cam->Radialothers,
				*tangential = cam->Tangential,
				*prism      = cam->Prism,
				*angleaxis  = cam->AngleAxis,
				*trans      = cam->Translation;

			double 
				observed_u = pt->uv[i][0],
				observed_v = pt->uv[i][1];


			//---check outlier----
			if (thresh > 0) 
			{
				double reprojected_pt[2];
				PinholeReprojection(focal, imcenter, skew, rad_1st, rad_other, tangential, prism, angleaxis, trans, xyz, reprojected_pt);

				double residual_x = reprojected_pt[0] - observed_u;
				double residual_y = reprojected_pt[1] - observed_v;
				double dist       = sqrt(residual_x*residual_x + residual_y*residual_y);

				if (dist > thresh)
				{
					pt->inlier[i] = false;
					if (pt->nInliers() < 2) //check again.
						continue;
					else
						break; //next point
				}
			}
			//--------------------



			ceres::CostFunction* cost_function = PinholeReprojectionError::Create(observed_u, observed_v);
			problem.AddResidualBlock(cost_function, loss_funcion, focal, imcenter, skew, rad_1st, rad_other, tangential, prism, angleaxis, trans, xyz);

			setConstantParams(*cam, problem);

		}
	}

	ceres::Solve(options, &problem, &summary);

}



void setConstantParams(CameraData &cam, ceres::Problem &problem)
{

	double
		*focal      = cam.FocalLength,
		*center     = cam.OpticalCenter,
		*skew       = &cam.Skew,
		*rad_1st    = &cam.Radialfirst,
		*rad_other  = cam.Radialothers,
		*tangential = cam.Tangential,
		*prism      = cam.Prism,
		*angleaxis  = cam.AngleAxis,
		*trans      = cam.Translation;


	switch (cam.opt_intrinsic)
	{
		case BA_OPT_INTRINSIC_ALL:
			break;

		case BA_OPT_INTRINSIC_ALL_FIXED:
			problem.SetParameterBlockConstant(focal);
			problem.SetParameterBlockConstant(center);
			problem.SetParameterBlockConstant(skew);
			break;

		case BA_OPT_INTRINSIC_CENTER_FIXED:
			problem.SetParameterBlockConstant(center);
			break;

		case BA_OPT_INTRINSIC_SKEW_ZERO_FIXED:
		case BA_OPT_INTRINSIC_SKEW_FIXED:
			problem.SetParameterBlockConstant(skew);
			break;
	}



	switch (cam.opt_lensdistortion)
	{
		case BA_OPT_LENSDIST_RADIAL_AND_TANGENTIAL:
			problem.SetParameterBlockConstant(prism);
			break;

		case BA_OPT_LENSDIST_ALL_FIXED:
			problem.SetParameterBlockConstant(rad_1st);
			problem.SetParameterBlockConstant(rad_other);
			problem.SetParameterBlockConstant(tangential);
			problem.SetParameterBlockConstant(prism);
			break;


		case BA_OPT_LENSDIST_RADIAL_1ST_ONLY:
			problem.SetParameterBlockConstant(rad_other);
			problem.SetParameterBlockConstant(tangential);
			problem.SetParameterBlockConstant(prism);
			break;

		case BA_OPT_LENSDIST_RADIAL_ONLY:
			problem.SetParameterBlockConstant(tangential);
			problem.SetParameterBlockConstant(prism);
			break;
	}


	switch (cam.opt_extrinsic)
	{
		case BA_OPT_EXTRINSIC_ALL:
			break;

		case BA_OPT_EXTRINSIC_ALL_FIXED:
			problem.SetParameterBlockConstant(angleaxis);
			problem.SetParameterBlockConstant(trans);
			break;

		case BA_OPT_EXTRINSIC_R_FIXED:
			problem.SetParameterBlockConstant(angleaxis);
			break;

		case BA_OPT_EXTRINSIC_T_FIXED:
			problem.SetParameterBlockConstant(trans);
			break;
	}

}


bool loadNVM(const string filepath, NVM &nvmdata, const int nMinViews)
{
	ifstream ifs(filepath);
	if (ifs.fail())
	{
		cerr << "Cannot load " << filepath << endl;
		return false;
	}




	string token;
	ifs >> token; //NVM_V3
	if (token != "NVM_V3")
	{
		cerr << "\n" 
			 << filepath <<" is not an NVM_V3 file." << endl;
		return false;
	}



	//loading camera parameters
	int nCameras;
	ifs >> nCameras;
	if (nCameras <= 1)
	{
		cerr << "# of cameras must be more than 1." << endl;
		return false;
	}
	nvmdata.nCamera = nCameras;
	for (int camID = 0; camID < nCameras; camID++)
	{
		string filename;
		double f;
		vector<double> q(4), c(3), d(2);
		ifs >> filename >> f >> q[0] >> q[1] >> q[2] >> q[3] >> c[0] >> c[1] >> c[2] >> d[0] >> d[1];

		nvmdata.filenames.push_back(filename);
		nvmdata.focallength.push_back(f);
		nvmdata.quaternion.push_back(q);
		nvmdata.position.push_back(c);
		nvmdata.firstradial.push_back(d[0]);

		nvmdata.filename_id[filename] = camID;
	}


	//loading 2D and 3D points
	int nPoints;
	ifs >> nPoints;
	if (nPoints <= 0)
	{
		cerr << "# of 3D points is 0." << endl;
		return false;
	}



	int nInliers = 0;
	nvmdata.n3Dpoints = nPoints;
	nvmdata.pointdata.resize(nPoints);
	for (int i = 0; i < nPoints; i++)
	{
		int nObserved;
		Vector3d xyz;
		Vector3i rgb;
		ifs >> xyz[0] >> xyz[1] >> xyz[2]
			>> rgb[0] >> rgb[1] >> rgb[2]
			>> nObserved;

		bool inlier = (nObserved >= nMinViews);
		if (inlier) nInliers++;

		//if (nObserved < nMinViews)
		//	continue;

		PointData *pdata = &nvmdata.pointdata[i];
		pdata->xyz = xyz;
		pdata->rgb = rgb;

		pdata->camID.resize(nObserved);
		pdata->inlier.resize(nObserved);
		pdata->uv.resize(nObserved);
		pdata->fID.resize(nObserved);
		for (int i = 0; i < nObserved; i++)
		{
			int camID, fID;
			double observed_x, observed_y;
			ifs >> camID >> fID >> observed_x >> observed_y;

			pdata->camID[i]  = camID;
			pdata->inlier[i] = inlier;
			pdata->fID[i]    = fID;

			pdata->uv[i][0] = observed_x;
			pdata->uv[i][1] = observed_y;
		}
	}

	nvmdata.nInliers = nInliers;
	return true;
}


bool loadNVMfast(const string filepath, NVM &nvmdata, const int nMinViews)
{
	ifstream ifs(filepath);
	if (ifs.fail())
	{
		cerr << "Cannot load " << filepath << endl;
		return false;
	}

	stringstream ss;
	ss << ifs.rdbuf();//load all data on memory
	ifs.close();

	string token;
	ss >> token; //NVM_V3
	if (token != "NVM_V3")
	{
		cerr << "\n"
			<< filepath << " is not an NVM_V3 file." << endl;
		return false;
	}

	//loading camera parameters
	int nCameras;
	ss >> nCameras;
	if (nCameras <= 1)
	{
		cerr << "# of cameras must be more than 1." << endl;
		return false;
	}

	ss.get();
	nvmdata.nCamera = nCameras;
	for (int camID = 0; camID < nCameras; camID++)
	{

		string line;
		vector<string> str_vec;

		getline(ss, line);
		boost::algorithm::split(str_vec, line, boost::is_space());


		string filename = str_vec[0];
		double f        = stod(str_vec[1]);
		vector<double> q;
		vector<double> c;
		q.push_back(stod(str_vec[2]));
		q.push_back(stod(str_vec[3]));
		q.push_back(stod(str_vec[4]));
		q.push_back(stod(str_vec[5]));


		c.push_back(stod(str_vec[6]));
		c.push_back(stod(str_vec[7]));
		c.push_back(stod(str_vec[8]));
		double d = stod(str_vec[9]);

		nvmdata.filenames.push_back(filename);
		nvmdata.focallength.push_back(f);
		nvmdata.quaternion.push_back(q);
		nvmdata.position.push_back(c);
		nvmdata.firstradial.push_back(d);
		nvmdata.filename_id[filename] = camID;
	}

	//loading 2D and 3D points
	int nNVMpoints;
	ss >> nNVMpoints;
	if (nNVMpoints <= 0)
	{
		cerr << "# of 3D points is 0." << endl;
		return false;
	}
	ss.get();

	nvmdata.n3Dpoints = nNVMpoints;
	nvmdata.pointdata.resize(nNVMpoints);

	int nInliers = 0;
	for (int i = 0; i < nNVMpoints; i++)
	{
		string         line;
		vector<string> str_vec;

		getline(ss, line);
		boost::algorithm::split(str_vec, line, boost::is_space());

		int nObserved = stoi(str_vec[6]);

		bool inlier = (nObserved >= nMinViews);
		if (inlier) nInliers++;

		PointData *pdata = &nvmdata.pointdata[i];
		pdata->camID.resize(nObserved);
		pdata->inlier.resize(nObserved);
		pdata->uv.resize(nObserved);
		pdata->fID.resize(nObserved);


		pdata->xyz[0] = stod(str_vec[0]);
		pdata->xyz[1] = stod(str_vec[1]);
		pdata->xyz[2] = stod(str_vec[2]);

		pdata->rgb[0] = stoi(str_vec[3]);
		pdata->rgb[1] = stoi(str_vec[4]);
		pdata->rgb[2] = stoi(str_vec[5]);

		for (int j = 0, k = 7; j < nObserved; j++)
		{
			int 
				camID = stoi(str_vec[k++]),
				fID   = stoi(str_vec[k++]);

			double
				observed_x = stod(str_vec[k++]),
				observed_y = stod(str_vec[k++]);

			pdata->camID[j]  = camID;
			pdata->inlier[j] = inlier;
			pdata->fID[j]    = fID;

			pdata->uv[j][0] = observed_x;
			pdata->uv[j][1] = observed_y;
		}
	}

	nvmdata.nInliers = nInliers;

	return true;
}


//////////////////////////////////////////////////////////////////////////
/// Start: Modified by Hanbyul Joo


//////////////////////////////////////////////////////////////////////////
/// Assuming name is pp_cc.xxx
int ExtractCamIdxFromPath(const char* fullPath)
{
	int leng = strlen(fullPath);

	char camStr[64];
	memcpy(camStr, fullPath + (leng - 6), sizeof(char) * 2);  //copying from 0~ folderLeng
	camStr[2] = 0;  //folderLeng < actual folderPath array size

	int numberInt = atoi(camStr);
	return numberInt;
}

int ExtractPanelIdxFromPath(const char* fullPath)
{
	int leng = strlen(fullPath);

	char panelStr[64];
	memcpy(panelStr, fullPath + (leng - 9), sizeof(char) * 2);  //copying from 0~ folderLeng
	panelStr[2] = 0;  //folderLeng < actual folderPath array size

	int numberInt = atoi(panelStr);
	return numberInt;
}


//By Hanbyul Joo
//Given a already loaded nvm, load more points
//Only assume Panoptic studio's camera naming rule. 
bool loadAdditionalNVM(const string filepath, vector<pair <int, int> > &camNameTable, NVM &nvmdata, const int nMinViews)
{
	ifstream ifs(filepath);
	if (ifs.fail())
	{
		cerr << "Cannot load " << filepath << endl;
		return false;
	}


	string token;
	ifs >> token; //NVM_V3
	if (token != "NVM_V3")
	{
		cerr << "\n"
			<< filepath << " is not an NVM_V3 file." << endl;
		return false;
	}

	//loading camera parameters
	int nCameras;
	ifs >> nCameras;
	if (nCameras <= 1)
	{
		cerr << "# of cameras must be more than 1." << endl;
		return false;
	}

	//nvmdata.nCamera = nCameras;
	vector<int> cameraIdConverter;		//given camIdx of current nvm, return the cam idx of the first nvm file
	cameraIdConverter.resize(nCameras, -1);
	for (int camID = 0; camID < nCameras; camID++)
	{
		string filename;
		double f;
		vector<double> q(4), c(3), d(2);
		ifs >> filename >> f >> q[0] >> q[1] >> q[2] >> q[3] >> c[0] >> c[1] >> c[2] >> d[0] >> d[1];


		int panelIdx  = ExtractPanelIdxFromPath( filename.c_str() );
		int cameraIdx = ExtractCamIdxFromPath(   filename.c_str() );

		//Find the corresponding cameras.. we should have used map structures...
		int univCamIdx = -1;			//univCamIdx means the cam idx of the first nvm file
		for (int t = 0; t<camNameTable.size(); ++t)
		{
			if (camNameTable[t].first == panelIdx && camNameTable[t].second == cameraIdx)
			{
				univCamIdx = t;		//found the corresponding index
				break;
			}
		}
		if (univCamIdx == -1)
		{
			cerr << "There is a camera which was not in the first NVM file. Check camera names" << endl;
			return false;
		}
		cameraIdConverter[camID] = univCamIdx;		//camIdx ->		univCamIdx
		/*nvmdata.filenames.push_back(filename);
		nvmdata.focallength.push_back(f);
		nvmdata.quaternion.push_back(q);
		nvmdata.position.push_back(c);
		nvmdata.firstradial.push_back(d[0]);

		nvmdata.filename_id[filename] = camID;*/
	}


	//loading 2D and 3D points
	int nPoints;
	ifs >> nPoints;
	if (nCameras <= 0)
	{
		cerr << "# of 3D points is 0." << endl;
		return false;
	}

	int nInliers = 0;
	int originalPtNum = nvmdata.pointdata.size();
	nvmdata.n3Dpoints = originalPtNum + nPoints;	//updated the number of pts
	//nvmdata.pointdata.reserve(nPoints);		    //this is a bug. It should be resize
	nvmdata.pointdata.resize(nvmdata.n3Dpoints);

	for (int i = originalPtNum; i < nvmdata.n3Dpoints; i++)
	{
		int nObserved;
		Vector3d xyz;
		Vector3i rgb;
		ifs >> xyz[0] >> xyz[1] >> xyz[2]
			>> rgb[0] >> rgb[1] >> rgb[2]
			>> nObserved;

		bool inlier = (nObserved >= nMinViews);
		if (inlier) nInliers++;

		//if (nObserved < nMinViews)
		//	continue;

		PointData *pdata = &nvmdata.pointdata[i];
		pdata->xyz = xyz;
		pdata->rgb = rgb;

		pdata->camID.resize(nObserved);
		pdata->inlier.resize(nObserved);
		pdata->uv.resize(nObserved);
		pdata->fID.resize(nObserved);
		for (int i = 0; i < nObserved; i++)
		{
			int camID, fID;
			double observed_x, observed_y;
			ifs >> camID >> fID >> observed_x >> observed_y;

			pdata->camID[i]  = cameraIdConverter[camID];			//camId is converted according to the first NVM's cam order
			pdata->inlier[i] = inlier;
			pdata->fID[i]    = fID;

			pdata->uv[i][0] = observed_x;
			pdata->uv[i][1] = observed_y;
		}
	}

	nvmdata.nInliers += nInliers;


	return true;
}

bool loadMultipleNVM(const vector<string>& filepathVect, NVM &nvmdata, const int nMinViews)
{
	if (filepathVect.size() == 0)
		return false;

	else if (filepathVect.size() == 1)
		return loadNVM(filepathVect.front(), nvmdata, nMinViews);

	else
	{
		loadNVM(filepathVect.front(), nvmdata, nMinViews);

		//Generate name table
		vector<pair <int, int>  > camNameTable;
		camNameTable.reserve(nvmdata.filenames.size());
		for (int c = 0; c<nvmdata.filenames.size(); ++c)
		{
			int panelIdx  = ExtractPanelIdxFromPath( nvmdata.filenames[c].c_str() );
			int cameraIdx = ExtractCamIdxFromPath(   nvmdata.filenames[c].c_str() );

			camNameTable.push_back(make_pair(panelIdx, cameraIdx));
		}

		for (int i = 1; i<filepathVect.size(); ++i)
			loadAdditionalNVM(filepathVect[i], camNameTable, nvmdata, nMinViews);
	}
}

//////////////////////////////////////////////////////////////////////////
/// End: Modified by Hanbyul Joo
//////////////////////////////////////////////////////////////////////////






bool checkOptionIntrinsics(const int option)
{
	bool ok = true;

	switch (option)
	{
	case BA_OPT_INTRINSIC_ALL:
	case BA_OPT_INTRINSIC_ALL_FIXED:
	case BA_OPT_INTRINSIC_SKEW_FIXED:
	case BA_OPT_INTRINSIC_SKEW_ZERO_FIXED:
	case BA_OPT_INTRINSIC_CENTER_FIXED:
	case BA_OPT_LOAD_INTRINSICS_BUT_JUST_AS_INITIAL_GUESS:
		break;

	default:
		ok = false;
		break;
	}

	return ok;
}

bool checkOptionExtrinsics(const int option)
{
	bool ok = true;

	switch (option)
	{
	case BA_OPT_EXTRINSIC_ALL:
	case BA_OPT_EXTRINSIC_ALL_FIXED:
	case BA_OPT_EXTRINSIC_R_FIXED:
	case BA_OPT_EXTRINSIC_T_FIXED:
		break;

	default:
		ok = false;
		break;
	}

	return ok;
}

bool checkOptionLensDist(const int option)
{
	bool ok = true;

	switch (option)
	{
	case BA_OPT_LENSDIST_ALL_FIXED:
	case BA_OPT_LENSDIST_RADIAL_1ST_ONLY:
	case BA_OPT_LENSDIST_RADIAL_AND_TANGENTIAL:
	case BA_OPT_LENSDIST_RADIAL_ONLY:
		break;

	default:
		ok = false;
		break;
	}

	return ok;
}

bool initCameraData(const NVM &nvmdata, const string filepath, vector<CameraData> &camera)
{
	//load image info
	/*ifstream ifs(filepath);
	if (ifs.fail())
	{
		cerr << "Cannot open " << filepath << endl;
		return false;
	}
	*/
	vector<int>    imgWidth, imgHeight, opt_intrinsic, opt_lensdistortion, opt_extrinsic;
	vector<bool>   available;
	vector<string> filenames;
	string str;
	for(int p=0;p<=20;++p)
	{
		int startCamIdx = 1;
		int endCamIdx = 24;
		if(p==0)
		{
			startCamIdx = 0;
			endCamIdx = 30;
		}
		

		for(int c=startCamIdx;c<=endCamIdx;++c)
		{
			string file;
			int w, h, option1, option2, option3, avail;

			if(p==0)
			{
				w = 1920;
				h = 1080;
			}
			else
			{
				w = 640;
				h = 480;
			}
			char buf[512];
			sprintf(buf,"%02d_%02d.jpg",p,c);
			filenames.push_back(buf);
			imgWidth.push_back(w);
			imgHeight.push_back(h);
			opt_intrinsic.push_back(3);
			opt_lensdistortion.push_back(3);
			opt_extrinsic.push_back(0);
			available.push_back(true);
		}
	}
	/*while (getline(ifs, str))
	{
		string file;
		int w, h, option1, option2, option3, avail;

		stringstream ss(str);
		ss >> file >> w >> h >> option1 >> option2 >> option3 >> avail;

		if (!checkOptionIntrinsics(option1))
		{
			cerr << "\n"
				<< "Option for intrinsic parameters is not valid in " << file << endl;
			return false;
		}
		if(!checkOptionLensDist(option2))
		{
			cerr << "\n"
				<< "Option for lens distortion is not valid in " << file << endl;
			return false;
		}
		if(!checkOptionExtrinsics(option3))
		{
			cerr << "\n"
				<< "Option for extrinsic parameters is not valid in " << file << endl;
			return false;
		}

		filenames.push_back(file);
		imgWidth.push_back(w);
		imgHeight.push_back(h);
		opt_intrinsic.push_back(option1);
		opt_lensdistortion.push_back(option2);
		opt_extrinsic.push_back(option3);

		bool flag = avail > 0 ? true : false;
		available.push_back(flag);
	}
	*/



	int nCameras = nvmdata.nCamera;

	if (filenames.size() > nCameras)
	{
		cerr << "\n"
			<< "Warning: "
			<< "The number of cameras in the ini file is more than that in the NVM file."
			<< "The ini file may not be valid."
			<< endl;
	}
	else if (filenames.size() < nCameras)
	{
		cerr << "\n"
			<< "Error: "
			<< "The number of cameras in the ini file is less than that in the NVM file.\n"
			<< "The ini file is not valid."
			<< endl;
		return false;
	}



	//initialize cmaera data
	camera.resize(nCameras);

	size_t nImages = filenames.size();
	int    found   = 0;
	vector<bool> load_ok(nCameras, false);
	
	bool first_warning = true;

	for (size_t i = 0; i < nImages; i++)
	{

		map<string, int>::const_iterator itr = nvmdata.filename_id.find( filenames[i] );
		if (itr == nvmdata.filename_id.end())
		{
			if (first_warning)
			{
				cerr << "\n";
				first_warning = false;
			}

			cerr << "Warning: "
				 << "Cannot find " << filenames[i] << " in the NVM file." << endl;
			continue;
		}

		//string imgfile = itr->first; <- is equal to filenames[i]
		int camID      = itr->second;

		CameraData *cam = &camera[camID];
		found++;
		load_ok[camID] = true;


		cam->filename           = filenames[i];
		cam->imgWidth           = imgWidth[i];
		cam->imgHeight          = imgHeight[i];

		cam->opt_intrinsic      = opt_intrinsic[i];
		cam->opt_lensdistortion = opt_lensdistortion[i];
		cam->opt_extrinsic      = opt_extrinsic[i];
		cam->available          = available[i];


		cam->FocalLength[0]   = nvmdata.focallength[camID];
		cam->FocalLength[1]   = nvmdata.focallength[camID];
		cam->OpticalCenter[0] = 0.5*(cam->imgWidth - 1.0);
		cam->OpticalCenter[1] = 0.5*(cam->imgHeight - 1.0);
		cam->Skew             = 0.0;


		cam->Radialfirst = -1.0*nvmdata.firstradial[camID]; //good approximation
		for (int j = 0; j < 2; j++){
			cam->Radialothers[j] = 0.0;
			cam->Tangential[j]   = 0.0;
			cam->Prism[j]        = 0.0;
		}


		//quaternion to angle-axis
		double q[4] = { nvmdata.quaternion[camID][0], nvmdata.quaternion[camID][1], nvmdata.quaternion[camID][2], nvmdata.quaternion[camID][3] };
		double angle[3];
		ceres::QuaternionToAngleAxis(q, angle);
		cam->AngleAxis[0] = angle[0];
		cam->AngleAxis[1] = angle[1];
		cam->AngleAxis[2] = angle[2];

		//position to translation t=-R*c
		double c[3] = { nvmdata.position[camID][0], nvmdata.position[camID][1], nvmdata.position[camID][2] };
		double t[3];
		ceres::QuaternionRotatePoint(q, c, t);

		cam->Translation[0] = -t[0];
		cam->Translation[1] = -t[1];
		cam->Translation[2] = -t[2];
	}


	if (found != nCameras)
	{
		cerr << "Error: "
			<< "The ini file does not have information for following cameras:"<< endl;
		for (int camID = 0; camID < nCameras; camID++)
		{
			if (!load_ok[camID])
				cerr << nvmdata.filenames[camID] << endl;

		}

		return false;
	}

	return true;
}

//shift 2D coordinates (camera center in the NVM file is assumed to be image center)
void shiftImageCenter(const std::vector<CameraData> &cameradata, std::vector<PointData> &pointdata)
{
	const size_t n3Dpoints = pointdata.size();
	for (size_t ptID = 0; ptID < n3Dpoints; ptID++)
	{
		PointData *ptdata = &pointdata[ptID];
		for (size_t i = 0; i < ptdata->nObserved(); i++)
		{
			if (!ptdata->inlier[i])
				continue;

			int camID = ptdata->camID[i];
			const CameraData *cam = &cameradata[camID];
			if (!cam->available)
				continue;

			ptdata->uv[i][0] += 0.5*(cam->imgWidth - 1.0);
			ptdata->uv[i][1] += 0.5*(cam->imgHeight - 1.0);

		}
	}
}

bool loadInitialIntrinsics(const string intrinsicfile, const map<string, int> &filename_id, vector<CameraData> &camera)
{
	ifstream ifs(intrinsicfile);
	if (ifs.fail())
	{
		cerr << "Cannot open " << intrinsicfile << endl;
		return false;
	}


	bool first_warning = true;
	string str;
	while (getline(ifs, str))
	{
		string file;
		double fx, fy, s, u0, v0, a0, a1, a2, p0, p1, s0, s1;
		stringstream ss(str);
		ss  >> file
			>> fx >> fy >> s >> u0 >> v0
			>> a0 >> a1 >> a2
			>> p0 >> p1 >> s0 >> s1;

		map<string, int>::const_iterator itr = filename_id.find(file);
		if (itr == filename_id.end())
		{
			if (first_warning)
			{
				cerr << "\n";
				first_warning = false;
			}

			cerr << "Warning: "
				 << "Cannot find " << file << " in the NVM file. Ignore it and continue." << endl;
			continue;
		}

		int camID = itr->second;

		camera[camID].FocalLength[0]   = fx;
		camera[camID].FocalLength[1]   = fy;
		camera[camID].OpticalCenter[0] = u0;
		camera[camID].OpticalCenter[1] = v0;
		camera[camID].Skew             = s;

		camera[camID].Radialfirst     = a0;
		camera[camID].Radialothers[0] = a1;
		camera[camID].Radialothers[1] = a2;
		camera[camID].Tangential[0]   = p0;
		camera[camID].Tangential[1]   = p0;
		camera[camID].Prism[0]        = s0;
		camera[camID].Prism[1]        = s1;


		if (camera[camID].opt_intrinsic == BA_OPT_LOAD_INTRINSICS_BUT_JUST_AS_INITIAL_GUESS)
		{
			camera[camID].opt_intrinsic = BA_OPT_INTRINSIC_ALL;
		}
		else
		{
			camera[camID].opt_intrinsic      = BA_OPT_INTRINSIC_ALL_FIXED;
			camera[camID].opt_lensdistortion = BA_OPT_LENSDIST_ALL_FIXED;
		}

	}


	return true;
}

bool loadIntrinsics(const string intrinsicfile, vector<CameraData> &camera)
{
	ifstream ifs(intrinsicfile);
	if (ifs.fail())
	{
		cerr << "Cannot open " << intrinsicfile << endl;
		return false;
	}

	bool empty = camera.empty() ? true : false;
	
	int camID = 0;
	string str;
	while (getline(ifs, str))
	{
		string filename;
		int w, h;
		double fx, fy, s, u0, v0, a0, a1, a2, p0, p1, s0, s1;
		stringstream ss(str);
		ss >> filename >> w >> h
			>> fx >> fy >> s >> u0 >> v0
			>> a0 >> a1 >> a2
			>> p0 >> p1 >> s0 >> s1;

		CameraData cam;

		cam.filename         = filename;
		cam.imgWidth         = w;
		cam.imgHeight        = h;
		cam.FocalLength[0]   = fx;
		cam.FocalLength[1]   = fy;
		cam.OpticalCenter[0] = u0;
		cam.OpticalCenter[1] = v0;
		cam.Skew             = s;

		cam.Radialfirst     = a0;
		cam.Radialothers[0] = a1;
		cam.Radialothers[1] = a2;
		cam.Tangential[0]   = p0;
		cam.Tangential[1]   = p0;
		cam.Prism[0]        = s0;
		cam.Prism[1]        = s1;
		
		cam.available       = true;

		if (empty)
			camera.push_back(cam);
		else
			camera.at(camID) = cam;


		camID++;
	}


	return true;
}

bool loadExtrinsics(const string extrinsicfile, vector<CameraData> &camera)
{
	ifstream ifs(extrinsicfile);
	if (ifs.fail())
	{
		cerr << "Cannot open " << extrinsicfile << endl;
		return false;
	}

	bool empty = camera.empty() ? true : false;

	int camID = 0;
	string str;
	while (getline(ifs, str))
	{
		string filename;
		double r0, r1, r2, t0, t1, t2;
		stringstream ss(str);
		ss >> filename
			>> r0 >> r1 >> r2
			>> t0 >> t1 >> t2;


		CameraData cam;

		cam.AngleAxis[0]   = r0;
		cam.AngleAxis[1]   = r1;
		cam.AngleAxis[2]   = r2;
		cam.Translation[0] = t0;
		cam.Translation[1] = t1;
		cam.Translation[2] = t2;

		cam.available = true;

		if (empty)
			camera.push_back(cam);
		else
			camera.at(camID) = cam;


		camID++;
	}


	return true;
}

bool loadAllCameraParams(const string cameraparamfile, vector<CameraData> &camera)
{
	ifstream ifs(cameraparamfile);
	if (ifs.fail())
	{
		cerr << "Cannot open " << cameraparamfile << endl;
		return false;
	}


	bool empty = camera.empty() ? true : false;

	int camID = 0;
	string str;
	while (getline(ifs, str))
	{
		string cameraname;
		int w, h;
		double
			fx, fy, s, u0, v0, // focal length (x, y), skew, optical center (x, y)
			a0, a1, a2,        // radial distortion
			p0, p1,            // tangential distortion
			s0, s1,            // prism distortion
			r0, r1, r2,        // angle-axis rotation parameters
			t0, t1, t2;        // translation


		stringstream ss(str);
		ss >> cameraname >> w >> h
			>> fx >> fy >> s >> u0 >> v0
			>> a0 >> a1 >> a2
			>> p0 >> p1
			>> s0 >> s1
			>> r0 >> r1 >> r2
			>> t0 >> t1 >> t2;

		BA::CameraData cam;
		cam.filename         = cameraname;
		cam.imgWidth         = w;
		cam.imgHeight        = h;
		cam.FocalLength[0]   = fx;
		cam.FocalLength[1]   = fy;
		cam.Skew             = s;
		cam.OpticalCenter[0] = u0;
		cam.OpticalCenter[1] = v0;

		cam.Radialfirst     = a0;
		cam.Radialothers[0] = a1;
		cam.Radialothers[1] = a2;
		cam.Tangential[0]   = p0;
		cam.Tangential[1]   = p1;
		cam.Prism[0]        = s0;
		cam.Prism[1]        = s1;

		cam.AngleAxis[0]   = r0;
		cam.AngleAxis[1]   = r1;
		cam.AngleAxis[2]   = r2;
		cam.Translation[0] = t0;
		cam.Translation[1] = t1;
		cam.Translation[2] = t2;

		cam.available = true;

		if (empty)
			camera.push_back(cam);
		else
			camera.at(camID) = cam;

		camID++;
	}

	return true;

}

bool loadAllCameraParamsfast(const string cameraparamfile, vector<CameraData> &camera)
{
	ifstream ifs(cameraparamfile);
	if (ifs.fail())
	{
		cerr << "Cannot open " << cameraparamfile << endl;
		return false;
	}

	stringstream ss;
	ss << ifs.rdbuf();
	ifs.close();



	bool empty = camera.empty() ? true : false;

	int camID = 0;
	string str;
	while (getline(ss, str))
	{
		vector<string> str_vec;
		boost::algorithm::split(str_vec, str, boost::is_space());


		BA::CameraData cam;
		cam.filename         = str_vec[0];
		cam.imgWidth         = stoi(str_vec[1]);
		cam.imgHeight        = stoi(str_vec[2]);
		cam.FocalLength[0]   = stod(str_vec[3]);
		cam.FocalLength[1]   = stod(str_vec[4]);
		cam.Skew             = stod(str_vec[5]);
		cam.OpticalCenter[0] = stod(str_vec[6]);
		cam.OpticalCenter[1] = stod(str_vec[7]);

		cam.Radialfirst     = stod(str_vec[8]);
		cam.Radialothers[0] = stod(str_vec[9]);
		cam.Radialothers[1] = stod(str_vec[10]);
		cam.Tangential[0]   = stod(str_vec[11]);
		cam.Tangential[1]   = stod(str_vec[12]);
		cam.Prism[0]        = stod(str_vec[13]);
		cam.Prism[1]        = stod(str_vec[14]);

		cam.AngleAxis[0]   = stod(str_vec[15]);
		cam.AngleAxis[1]   = stod(str_vec[16]);
		cam.AngleAxis[2]   = stod(str_vec[17]);
		cam.Translation[0] = stod(str_vec[18]);
		cam.Translation[1] = stod(str_vec[19]);
		cam.Translation[2] = stod(str_vec[20]);

		cam.available = true;

		if (empty)
			camera.push_back(cam);
		else
			camera.at(camID) = cam;

		camID++;
	}

	return true;

}



bool saveCameraAllParams(const string filename, const string sep, const vector<CameraData> &camera)
{
	ofstream ofs(filename);
	if (ofs.fail())
	{
		cerr << "Cannot write " << filename << endl;
		return false;
	}

	ofs << scientific << setprecision(16);
	for (int camID = 0; camID < camera.size(); camID++)
	{
		const CameraData *cam = &camera[camID];
		if (!cam->available)
			continue;

		string file = cam->filename;

		int    w = cam->imgWidth;
		int    h = cam->imgHeight;

		double fx = cam->FocalLength[0];
		double fy = cam->FocalLength[1];
		double s  = cam->Skew;
		double u0 = cam->OpticalCenter[0];
		double v0 = cam->OpticalCenter[1];

		double a0 = cam->Radialfirst;
		double a1 = cam->Radialothers[0];
		double a2 = cam->Radialothers[1];
		double p0 = cam->Tangential[0];
		double p1 = cam->Tangential[1];
		double s0 = cam->Prism[0];
		double s1 = cam->Prism[1];

		double r0 = cam->AngleAxis[0];
		double r1 = cam->AngleAxis[1];
		double r2 = cam->AngleAxis[2];
		double t0 = cam->Translation[0];
		double t1 = cam->Translation[1];
		double t2 = cam->Translation[2];

		ofs << file << sep << w << sep << h << sep
			<< fx << sep << fy << sep << s << sep << u0 << sep << v0 << sep
			<< a0 << sep << a1 << sep << a2 << sep
			<< p0 << sep << p1 << sep << s0 << sep << s1 << sep
			<< r0 << sep << r1 << sep << r2 << sep
			<< t0 << sep << t1 << sep << t2 << endl;
	}

	return true;
}
bool saveCameraExtrinsics(const string filename, const string sep, const vector<CameraData> &camera)
{
	ofstream ofs(filename);
	if (ofs.fail())
	{
		cerr << "Cannot write " << filename << endl;
		return false;
	}

	ofs << scientific << setprecision(16);
	for (int camID = 0; camID < camera.size(); camID++)
	{
		if (!camera[camID].available)
			continue;

		string file = camera[camID].filename;

		double r0 = camera[camID].AngleAxis[0];
		double r1 = camera[camID].AngleAxis[1];
		double r2 = camera[camID].AngleAxis[2];
		double t0 = camera[camID].Translation[0];
		double t1 = camera[camID].Translation[1];
		double t2 = camera[camID].Translation[2];

		ofs << file << sep
			<< r0 << sep << r1 << sep << r2 << sep
			<< t0 << sep << t1 << sep << t2 << endl;
	}

	return true;
}

bool saveCameraIntrinsics(const string filename, const string sep, const vector<CameraData> &camera)
{
	ofstream ofs(filename);
	if (ofs.fail())
	{
		cerr << "Cannot write " << filename << endl;
		return false;
	}

	ofs << scientific << setprecision(16);
	for (int camID = 0; camID < camera.size(); camID++)
	{
		const CameraData *cam = &camera[camID];

		if (!camera[camID].available)
			continue;

		string file = cam->filename;

		int    w  = cam->imgWidth;
		int    h  = cam->imgHeight;

		double fx = cam->FocalLength[0];
		double fy = cam->FocalLength[1];
		double s  = cam->Skew;
		double u0 = cam->OpticalCenter[0];
		double v0 = cam->OpticalCenter[1];

		double a0 = cam->Radialfirst;
		double a1 = cam->Radialothers[0];
		double a2 = cam->Radialothers[1];
		double p0 = cam->Tangential[0];
		double p1 = cam->Tangential[1];
		double s0 = cam->Prism[0];
		double s1 = cam->Prism[1];

		ofs << file << sep << w << sep << h << sep
			<< fx << sep << fy << sep << s << sep << u0 << sep << v0 << sep
			<< a0 << sep << a1 << sep << a2 << sep
			<< p0 << sep << p1 << sep << s0 << sep << s1 << endl;
	}

	return true;
}

bool save3Dpoints(const string filename, const string sep, const vector<PointData> &pointdata)
{
	ofstream ofs(filename);
	if (ofs.fail())
	{
		cerr << "Cannot write " << filename << endl;
		return false;
	}

	ofs << scientific << setprecision(16);
	for (size_t ptID = 0; ptID < pointdata.size(); ptID++)
	{
		const double *xyz = pointdata[ptID].xyz.data();

		ofs << xyz[0] << sep << xyz[1] << sep << xyz[2] << endl;
	}

	return true;
}


bool saveReprojectionError(const string filename, const string sep, const residualData &res, const vector<CameraData> &camera, const int order)
{
	ofstream ofs(filename);
	ofs << scientific << setprecision(5);
	if (ofs.fail())
	{
		cerr << "Cannot write " << filename << endl;
		return false;
	}

	size_t numdata = res.error.size();

	for (int i = 0; i < numdata; i++)
	{
		int    ptID  = res.ID[i].first;
		int    camID = res.ID[i].second;

		string cameraname = camera[camID].filename;

		double x0 = res.observed_pt[i][0];
		double y0 = res.observed_pt[i][1];

		double x1 = res.reprojected_pt[i][0];
		double y1 = res.reprojected_pt[i][1];

		double res_x = res.error[i][0];
		double res_y = res.error[i][1];

		if (order == 0)
			ofs << cameraname << sep << ptID << sep;
		else
			ofs << ptID << sep << cameraname << sep;


		ofs	<< x0 << sep << y0 << sep
			<< x1 << sep << y1 << sep
			<< res_x << sep << res_y << endl;
	}

	return true;
}


bool saveAllData(const string savefolder, const vector<CameraData> &camera, const vector<PointData> &pointdata, const residualData &res, const string prefix, const bool After)
{
	string postfix;
	if (After)
		postfix = "_after";
	else
		postfix = "_before";


	string sep = "\t";  //separator
	string ext = ".txt";//extension

	string filename;


	filename = savefolder + "/" + prefix + "Camera_AllParams" + postfix + ext;
	saveCameraAllParams(filename, sep, camera);


	filename = savefolder + "/" + prefix + "Camera_Extrinsics" + postfix + ext;
	saveCameraExtrinsics(filename, sep, camera);


	filename = savefolder + "/" + prefix + "Camera_Intrinsics" + postfix + ext;
	saveCameraIntrinsics(filename, sep, camera);

	
	filename = savefolder + "/" + prefix + "3Dpoints" + postfix + ext;
	save3Dpoints(filename, sep, pointdata);


	filename = savefolder + "/" + prefix + "ReprojectionError" + postfix + ext;
	saveReprojectionError(filename, sep, res, camera, 0);

	
	return true;
}


bool saveCeresReport(const std::string savefolder, const std::string filename, const ceres::Solver::Summary &summary)
{
	string ceres_report = savefolder + "/" + filename;
	ofstream ofs(ceres_report);
	if (ofs.fail())
	{
		cerr << "Cannot write " << ceres_report << endl;
		return false;
	}

	ofs << summary.FullReport();

	return true;
}


bool saveNVM(const string savefolder, const string outputnvmname, const vector<CameraData> &camera, const vector<PointData> &pointdata)
{
	string filename = savefolder + "/" + outputnvmname;
	ofstream ofs(filename);
	if (ofs.fail())
	{
		cerr << "Cannot write " << filename << endl;
		return false;
	}

	int digits = 12;

	stringstream ss_cam;
	ss_cam << fixed << setprecision(digits);

	//write camera data
	size_t nCameras      = camera.size();
	int    nCamAvailable = 0;
	for (size_t camID = 0; camID < nCameras; camID++)
	{
		const CameraData *cam = &camera[camID];
		if (!cam->available)
			continue;

		string file = cam->filename;
		double fx   = cam->FocalLength[0];
		double rad  = -cam->Radialfirst;


		// angle axis -> quaternion
		double angle[3] = { cam->AngleAxis[0], cam->AngleAxis[1], cam->AngleAxis[2] };
		double q[4];
		ceres::AngleAxisToQuaternion(angle, q);


		// translation -> camera position c=-R'*t
		double t[3] = { cam->Translation[0], cam->Translation[1], cam->Translation[2] };
		double c[3];
		angle[0] = -angle[0];
		angle[1] = -angle[1];
		angle[2] = -angle[2];
		ceres::AngleAxisRotatePoint(angle, t, c);
		c[0] = -c[0];
		c[1] = -c[1];
		c[2] = -c[2];


		ss_cam << file << '\t'
			<< fx << ' '
			<< q[0] << ' ' << q[1] << ' ' << q[2] << ' ' << q[3] << ' '
			<< c[0] << ' ' << c[1] << ' ' << c[2] << ' '
			<< rad << ' ' << 0 << '\n';

		nCamAvailable++;
	}




	// write point data
	const size_t nPoints = pointdata.size();
	stringstream ss_pts;
	ss_pts << fixed << setprecision(digits);

	size_t nOutliers = 0;
	for (size_t ptID = 0; ptID < nPoints; ptID++)
	{
		const PointData *pdata = &pointdata[ptID];

		size_t nInliers = pdata->nInliers();
		if (nInliers == 0)
		{
			nOutliers++;
			continue;
		}



		stringstream ss;
		ss << fixed << setprecision(digits);
		for (size_t i = 0; i < pdata->nObserved(); i++)
		{
			if (!pdata->inlier[i])
				continue;

			int camID = pdata->camID[i];
			if (!camera[camID].available)
				continue;


			double u   = pdata->uv[i][0];
			double v   = pdata->uv[i][1];
			int    fID = pdata->fID[i];

			ss << camID << ' ' << fID << ' ' << u << ' ' << v << ' ';
		}

		double x = pdata->xyz[0];
		double y = pdata->xyz[1];
		double z = pdata->xyz[2];

		int r = pdata->rgb[0];
		int g = pdata->rgb[1];
		int b = pdata->rgb[2];

		ss_pts << x << ' ' << y << ' ' << z << ' ' << r << ' ' << g << ' ' << b << ' ' << nInliers << ' ' << ss.str() << '\n';
	}
	size_t nInlier3Dpts = nPoints - nOutliers;



	ofs << "NVM_V3 \n"
		<< '\n'
		<< nCamAvailable << '\n'
		<< ss_cam.str()
		<< '\n'
		<< nInlier3Dpts << '\n'
		<< ss_pts.str()
		<< "\n"
		<< "\n"
		<< "\n"
		<< "0\n"
		<< "\n"
		<< "#the last part of NVM file points to the PLY files\n"
		<< "#the first number is the number of associated PLY files\n"
		<< "#each following number gives a model - index that has PLY\n"
		<< "0";

	return true;
}


}