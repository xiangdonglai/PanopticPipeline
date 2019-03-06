#include "DomeCalibUtil.h"


#include <iostream>
#include <omp.h>

using namespace std;
using Eigen::Vector2d;
#include <boost/filesystem.hpp>

#if 0

int main(int argc, char *argv[])
{
	string rootfolder      = "D:/codes/ProjectorBasedCalibration-master/pattern1/vsfm";//argv[1];
	string nvmfile         = "151125_calib.nvm";//argv[2];
	string projptsfilebase = "ProjPoints";
	string savefolder      = "D:/codes/ProjectorBasedCalibration-master/pattern1/vsfm/out";//argv[3];
	//string imginfofile     = argv[4];
	string intrinsicfile;
	if(argc==6)
	   intrinsicfile= rootfolder + "/" + argv[5];

	double start_time;


	//-------load NVM file------------
	cout << "Loading NVM file and camera parameters...";
	start_time = omp_get_wtime();

	int minViews = 3;
	cout << "(at least " << minViews << " matches)...";

	BA::NVM nvmdata;
	vector<BA::CameraData> camdata;

	if (!BA::loadNVMfast(rootfolder + "/" + nvmfile, nvmdata, minViews))
		return false;

	cout << "Done. (" << omp_get_wtime() - start_time << " sec)" << endl;
	//-------------------------------


	//-------remove duplicate points (which have same featureID)-----------
	cout << "Disabling duplicate points in the NVM data...";
	start_time = omp_get_wtime();


	vector<BA::PointData>  pointdata(nvmdata.pointdata);
	DC::disableDuplicatePoints(pointdata);

	cout << "Done. (" << omp_get_wtime() - start_time << " sec)" << endl;
	//-------------------------------




	//---------convert NVM data to CameraData---------
	cout << "Converting NVM data to CameraData...";

	if (!BA::initCameraData(nvmdata, "", camdata))
		return 1;
	BA::shiftImageCenter(camdata, pointdata);

	cout << "Done.\n" << endl;
	//-------------------------------------


	if (!intrinsicfile.empty())
	{
		if (!BA::loadInitialIntrinsics(intrinsicfile, nvmdata.filename_id, camdata))
			return 1;
	}



	//---------projectors calibration---------
	cout << "Calibrating projectors from 2D-3D correspondences...\n";
	start_time = omp_get_wtime();
	int nProjectors = 5;
	int nCameras    = (int)camdata.size();

	vector<BA::CameraData> projdata;
	DC::calibrateProjectors(rootfolder + "/"+ projptsfilebase, nProjectors, nCameras, projdata, pointdata);
	cout << "Done. (" << omp_get_wtime() - start_time << " sec)\n" << endl;
	//-------------------------------------





	//---------BA for all data---------
	size_t availabelCameras    = count_if(camdata.begin(),  camdata.end(),  [](BA::CameraData &c){return c.available; });
	size_t availabelPeojectors = count_if(projdata.begin(), projdata.end(), [](BA::CameraData &c){return c.available; });
	size_t inlierPoints        = pointdata.size() - count_if(pointdata.begin(), pointdata.end(), [](BA::PointData &p){return p.nInliers()==0; });

	cout << "Running BA for all devices and all inliers...\n"
		<< "\tNum of Devices: " << availabelCameras + availabelPeojectors << "\n"
		<< "\tNum of Points : " << inlierPoints << "\n";

	start_time = omp_get_wtime();

	DC::runBAall(savefolder, camdata, projdata, pointdata);

	cout << "Done. (" << omp_get_wtime() - start_time << " sec)" << endl;
	//-------------------------------------


	return 0;
}


#else

int main(int argc, char *argv[])
{
	if(argc<4)
	{
		printf("Usage: ./DomeBundle.exe folderContainsNVM savefolder minView\n");
		printf("ex> ./DomeCorres.exe C:/151125_calib/full C:/151125_calib/full/out 3\n");
		return 0;
	}

	string rootfolder(argv[1]);//      = "C:/151125_calib/full";//argv[1];
	string savefolder(argv[2]);//      =  "C:/151125_calib/full/out";//argv[3];
	int minViews = atoi(argv[3]);

	bool bDuplicateCheck = true;
	if(argc>=5)
	{
		if(strcmp(argv[4],"1")==0)
		{
			bDuplicateCheck = false;
			printf("Do not performe feature duplication check\n");
		}
		else
			bDuplicateCheck = true;
	}

	string nvmfile         = "result.nvm";//argv[2];
		
	printf("rootFolder: %s\n",rootfolder.c_str());
	printf("savefolder: %s\n",savefolder.c_str());
	printf("nViewThreshold: %d\n",minViews);
	printf("bDuplicateCheck: %d\n",bDuplicateCheck);

	/*string intrinsicfile;
	if(argc==6)
	   intrinsicfile= rootfolder + "/" + argv[5];*/
	// CreateDirectory(savefolder.c_str(),NULL);
	boost::filesystem::create_directory(boost::filesystem::path(savefolder));
	double start_time;

	//-------load NVM file------------
	cout << "Loading NVM file and camera parameters...";
	start_time = omp_get_wtime();

	cout << "(at least " << minViews << " matches)..." <<endl;

	BA::NVM nvmdata;
	vector<BA::CameraData> camdata;

	if (!BA::loadNVMfast(rootfolder + "/" + nvmfile, nvmdata, minViews))
		return false;

	cout << "Done. (" << omp_get_wtime() - start_time << " sec)" << endl;
	//-------------------------------

	//-------remove duplicate points (which have same featureID)-----------
	cout << "Disabling duplicate points in the NVM data...";
	start_time = omp_get_wtime();

	
	vector<BA::PointData>  pointdata(nvmdata.pointdata);
	if(bDuplicateCheck)
		DC::disableDuplicatePoints(pointdata);

	cout << "Done. (" << omp_get_wtime() - start_time << " sec)" << endl;
	//-------------------------------

	//---------convert NVM data to CameraData---------
	cout << "Converting NVM data to CameraData...";

	if (!BA::initCameraData(nvmdata, "", camdata))
		return 1;
	BA::shiftImageCenter(camdata, pointdata);

	cout << "Done.\n" << endl;
	//-------------------------------------

	/*if (!intrinsicfile.empty())
	{
		if (!BA::loadInitialIntrinsics(intrinsicfile, nvmdata.filename_id, camdata))
			return 1;
	}*/

	/*
	//---------projectors calibration---------
	cout << "Calibrating projectors from 2D-3D correspondences...\n";
	start_time = omp_get_wtime();
	int nProjectors = 5;
	int nCameras    = (int)camdata.size();

	vector<BA::CameraData> projdata;
	DC::calibrateProjectors(rootfolder + "/"+ projptsfilebase, nProjectors, nCameras, projdata, pointdata);
	cout << "Done. (" << omp_get_wtime() - start_time << " sec)\n" << endl;
	//-------------------------------------
	*/

	//---------BA for all data---------
	size_t availabelCameras    = count_if(camdata.begin(),  camdata.end(),  [](BA::CameraData &c){return c.available; });
	//size_t availabelPeojectors = count_if(projdata.begin(), projdata.end(), [](BA::CameraData &c){return c.available; });
	size_t inlierPoints        = pointdata.size() - count_if(pointdata.begin(), pointdata.end(), [](BA::PointData &p){return p.nInliers()==0; });
	size_t availabelPeojectors =0;

	cout << "Running BA for all devices and all inliers...\n"
		<< "\tNum of Devices: " << availabelCameras + availabelPeojectors << "\n"
		<< "\tNum of Points : " << inlierPoints << "\n";

	start_time = omp_get_wtime();

	vector<BA::CameraData> projdata;
	DC::runBAall(savefolder, camdata, projdata, pointdata);

	cout << "Done. (" << omp_get_wtime() - start_time << " sec)" << endl;
	//-------------------------------------

	return 0;
}

#endif