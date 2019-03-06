// #include <windows.h>
// #include <ImageHlp.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <omp.h>

#include "DomeCorresSIFT.h"
#include "BAutil.h"

#include "strlib.h"

#include <Eigen/Core>
using Eigen::Vector2d;
#include <boost/filesystem.hpp>


#if 0

int main(int argc, char *argv[])
{
	string camfolder;
	string projimgfile;
	if(argc>1)
	{
		camfolder   = argv[1];                   // example: "D:/Programs/DomeCalib/testdata/03_random_141027";
		projimgfile = camfolder + "/" + argv[2]; // example: "random2_half.png";
	}
	else
	{
		camfolder   = "D:/codes/ProjectorBasedCalibration-master/pattern1";                   // example: "D:/Programs/DomeCalib/testdata/03_random_141027";
		projimgfile = camfolder + "/random2_half.png"; // example: "random2_half.png";
	}


	int widthP  = 1280,
		heightP =  800;

	int nProjectors  = 5;
	int nVGAs        = 24;
	//int nHDs          = 30 + 3;//30 HDs and 3 Kinects
	int nHDs         = 31;
	int nPanels      = 20;

	int nCameras = nHDs + nPanels*nVGAs;
	int nDevices = nProjectors + nCameras;


	//-----generate strings for the device names------
	vector<string> devicenames;
	for (int projID = 0; projID < nProjectors; projID++)
	{
		string projname = strsprintf("p%d", projID);
		devicenames.push_back(projname);
	}

	for (int panelID = 0; panelID <= nPanels; panelID++)
	{
		int camID_start       = (panelID == 0) ? 0 : 1;
		int nCameras_on_panel = (panelID == 0) ? nHDs : nVGAs + 1;
	
		for (int camID = camID_start; camID < nCameras_on_panel; camID++)
		{
			string camname = strsprintf("%02d_%02d", panelID, camID);
			/*if(frameIdx>=0)
			{
				camname = strsprintf("%08d/%08d_%02d_%02d", frameIdx, frameIdx, panelID, camID);
			}*/
			devicenames.push_back(camname);
		}
	}


	vector< vector<Vector2d> > PCpts;
	int equalize_method = 2;

	double start_time;


#if 1

	FindCorresSIFT(projimgfile, devicenames, nProjectors, nDevices, camfolder, PCpts, equalize_method);

#else

	cout << "Loading PCcorresSparse.txt...";
	start_time = omp_get_wtime();
	readPCcorresSparse(camfolder + "/SIFT_points/PCcorresSparse.txt", PCpts);           

	cout << "Done. (" << omp_get_wtime() - start_time << " sec)" << endl;
#endif


	int min_nViews = 20;
	if (min_nViews > 0)
	{
		cout << "Removing poor matches (<" << min_nViews << " views)...";
		cout << PCpts.size() << "->";

		start_time = omp_get_wtime();
		removeLessObserevation(nProjectors, nDevices, min_nViews, PCpts);

		cout << PCpts.size() << " points...";
		cout << "Done. (" << omp_get_wtime() - start_time << " sec)" << endl;
	}

	GenerateVisualSFMinput(camfolder, devicenames, PCpts, nProjectors, nHDs, nVGAs, nPanels);

	return 0;
}
#else
void GetFolderList(string rootFolder,vector<string>& folderList)
{
	char search_path[200];
    sprintf(search_path, "%s/*.*", rootFolder.c_str());

	// WIN32_FIND_DATA fd; 
 //    HANDLE hFind = ::FindFirstFile(search_path, &fd); 
 //    if(hFind != INVALID_HANDLE_VALUE) { 
 //        do { 
 //            // read all (real) files in current folder
 //            // , delete '!' read other 2 default folder . and ..
 //            if(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY ) {
	// 			 if( strcmp(fd.cFileName,".")==0 || strcmp(fd.cFileName,"..")==0 || strcmp(fd.cFileName,"full")==0 )
	// 				 continue;
    
 //                folderList.push_back(rootFolder + "/" + string(fd.cFileName));
 //            }
 //        }while(::FindNextFile(hFind, &fd)); 
 //        ::FindClose(hFind); 
 //    } 

    boost::filesystem::path p(rootFolder);
    if(boost::filesystem::is_directory(p)) {

        for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {}))
        	folderList.emplace_back(entry.path().string());
    }
}

int main(int argc, char *argv[])
{
	if(argc<2)
	{
		printf("Usage: ./DomeCorres.exe dataRootFolder\n");
		printf("ex> ./DomeCorres.exe D:/calibrationTest/pairVismMat\n");
		return 0;
	}
	printf("Start Maching Merger\n");

	string rootfolder(argv[1]);//      = "D:/calibrationTest/pairVismMat";//argv[1];	
	int nVGAs        = 480;
	int nHDs         = 31;
	int nKinect      = 31;
	
	string nvmfile("result.nvm");

	vector<string> folderList;
	GetFolderList(rootfolder,folderList);
	int datasetNum = folderList.size();
	for(int i=0;i<folderList.size();++i)
		printf("Include folder: %s\n",folderList[i].c_str());
	
	printf("rootFolder: %s\n",rootfolder.c_str());
	printf("nvmfileName: %s\n",nvmfile.c_str());
	printf("datasetNum: %d\n",datasetNum);
	
	vector<BA::NVM> nvmdataVect(datasetNum);
	//Load NVM files
	for(int p=0;p<folderList.size();++p)
	{
		BA::NVM& nvmdata = nvmdataVect[p];
		vector<BA::CameraData> camdata;
		int minViews = 3;

		char fileName[512];
		sprintf(fileName,"%s/%s",folderList[p].c_str(),nvmfile.c_str());
		printf("Read NVM file: %s\n",fileName);
		if (!BA::loadNVMfast(fileName, nvmdata, minViews))
			continue;
	}

	char outputFolderName[512];
	sprintf(outputFolderName,"%s/full",rootfolder.c_str());
	// CreateDirectory(outputFolderName,NULL);
	boost::filesystem::create_directory(boost::filesystem::path(outputFolderName));
	GenerateVisualSFMinput(nvmdataVect,outputFolderName);

	return 0;
}

#endif