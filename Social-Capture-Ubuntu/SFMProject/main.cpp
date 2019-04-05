#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <boost/filesystem.hpp>
#include "Constants.h"
#include "ImportExportDlg.h"
#include "SkeletonGeneratorDlg.h"
#include "Utility.h"
#include "FaceReconByPMDlg.h"

void GetFirstLastFrameIdx_last8digit(char* dirPath,int& firstFrameReturn,int& lastFrameReturn)
{
	boost::filesystem::path dirname(dirPath);
	vector<boost::filesystem::path> v;
	copy(boost::filesystem::directory_iterator(dirname), boost::filesystem::directory_iterator(), std::back_inserter(v));
	if(v.size()==0)
	{
		firstFrameReturn=-1;
		lastFrameReturn=-1;
		return;
	}
	std::sort(v.begin(),v.end());
	char initFile[512];
	char lastFile[512];
	
	sprintf(initFile,"%s",v.front().filename().string().c_str() );
	int initFrameIdx = ExtractFrameIdxFromPath_last8digit(initFile);
	printf("GetFirstLastFrameIdx:: %s: %d\n",initFile,initFrameIdx );


	sprintf(lastFile,"%s",v.back().filename().string().c_str() );
	int lastFrameIdx = ExtractFrameIdxFromPath_last8digit(lastFile);
	printf("GetFirstLastFrameIdx:: %s: %d\n",lastFile,lastFrameIdx );
	firstFrameReturn = initFrameIdx;
	lastFrameReturn  = lastFrameIdx;
}

// imitate command line parsing of Hanbyul's original code
int main(int argc, char** argv)
{
	std::vector<std::string> paramVect;
	for(auto i = 0; i < argc; i++) paramVect.emplace_back(argv[i]);

	std::string& option = paramVect[1];
	printf("-- Option: %s\n",option.c_str());

	if(strcmp(option.c_str(),"calibNorm")==0)
	{
		sprintf(g_calibrationFolder, "%s", paramVect[2].c_str()); // defined in Constants.cpp
		ImportExportDlg::OnBnTransform2NormalizedCalibrationCoord();
	}
	else if (strcmp(option.c_str(),"skel_mpm_undist_19pts") == 0)
	{
		// run undistortion
		sprintf(g_dataMainFolder, "%s", paramVect[2].c_str());  // defined in Constants.cpp
		sprintf(g_calibrationFolder, "%s", paramVect[3].c_str());

		g_dataFrameStartIdx = atoi(paramVect[4].c_str());
		int frameEndIdx = atoi(paramVect[5].c_str());
		g_dataFrameNum = frameEndIdx - g_dataFrameStartIdx + 1;

		printf("## input: g_dataMainFolder: %s\n",g_dataMainFolder);
		printf("## input: g_calibrationFolder: %s\n",g_calibrationFolder);
		printf("## input: g_dataFrameStartIdx: %d\n",g_dataFrameStartIdx);
		printf("## input: g_dataFrameNum: %d\n",g_dataFrameNum);

		SkeletonGeneratorDlg::Script_Util_Undistort_PoseMachine2DResult_mpm_19joints(false);
	}
	else if (strcmp(option.c_str(),"skel_all_vga_mpm19") == 0)
	{
		sprintf(g_dataMainFolder, "%s", paramVect[2].c_str());
		sprintf(g_calibrationFolder, "%s", paramVect[3].c_str());

		g_dataFrameStartIdx = atoi(paramVect[4].c_str());
		int frameEndIdx = atoi(paramVect[5].c_str());
		g_dataFrameNum = frameEndIdx - g_dataFrameStartIdx + 1;

		g_askedVGACamNum = atoi(paramVect[6].c_str());			//additional parameter...  usually 140 or 480
		sprintf(g_dataSpecificFolder,"%s/coco19_body3DPSRecon_updated/%04d",g_dataMainFolder,g_askedVGACamNum);

		g_poseEstLoadingDataFirstFrameIdx = g_dataFrameStartIdx;
		g_poseEstLoadingDataNum = g_dataFrameNum;

		printf("## input: g_dataMainFolder: %s\n",g_dataMainFolder);
		printf("## input: g_dataSpecificFolder: %s\n",g_dataSpecificFolder);
		printf("## input: g_calibrationFolder: %s\n",g_calibrationFolder);
		printf("## input: g_dataFrameStartIdx: %d\n",g_dataFrameStartIdx);
		printf("## input: g_dataFrameNum: %d\n",g_dataFrameNum);

		SkeletonGeneratorDlg::Script_NodePartProposalRecon_fromPoseMachine_coco19();
		SkeletonGeneratorDlg::Script_3DPS_Reconstruct_PoseMachine_coco19();
		g_poseEstLoadingDataFirstFrameIdx = g_dataFrameStartIdx;
		g_poseEstLoadingDataNum = g_dataFrameNum;
		SkeletonGeneratorDlg::Script_Load_body3DPS_byFrame(true);
		SkeletonGeneratorDlg::Script_3DPS_Optimization_usingDetectionPeaks(true);
	}
	else if (strcmp(option.c_str(),"skel_export_vga_mpm19") == 0)
	{
		sprintf(g_dataMainFolder, "%s", paramVect[2].c_str());
		sprintf(g_calibrationFolder, "%s", paramVect[3].c_str());

		g_dataFrameStartIdx = atoi(paramVect[4].c_str());
		int frameEndIdx = atoi(paramVect[5].c_str());
		g_dataFrameNum = frameEndIdx - g_dataFrameStartIdx + 1;

		g_askedVGACamNum = atoi(paramVect[6].c_str());			//additional parameter...  usually 140 or 480
		sprintf(g_dataSpecificFolder,"%s/coco19_body3DPSRecon_updated/%04d",g_dataMainFolder,g_askedVGACamNum);

		g_poseEstLoadingDataFirstFrameIdx = g_dataFrameStartIdx;
		g_poseEstLoadingDataNum = g_dataFrameNum;
		SkeletonGeneratorDlg::Script_Load_body3DPS_byFrame_folderSpecify();
		SkeletonGeneratorDlg::Script_Export_3DPS_Json(true);
	}
	else if (strcmp(option.c_str(),"skel_convert_vga2hd") == 0)
	{
		sprintf(g_dataMainFolder, "%s", paramVect[2].c_str());
		sprintf(g_calibrationFolder, "%s", paramVect[3].c_str());

		g_dataFrameStartIdx = atoi(paramVect[4].c_str());
		int frameEndIdx = atoi(paramVect[5].c_str());
		g_dataFrameNum = frameEndIdx - g_dataFrameStartIdx + 1;

		g_askedVGACamNum = atoi(paramVect[6].c_str());
		sprintf(g_dataSpecificFolder,"%s/coco19_body3DPSRecon_updated/%04d",g_dataMainFolder,g_askedVGACamNum);

		g_poseEstLoadingDataFirstFrameIdx = g_dataFrameStartIdx;
		g_poseEstLoadingDataNum = g_dataFrameNum;
		printf("## input: g_dataMainFolder: %s\n",g_dataMainFolder);
		printf("## input: g_dataSpecificFolder: %s\n",g_dataSpecificFolder);
		printf("## input: g_calibrationFolder: %s\n",g_calibrationFolder);
		printf("## input: g_dataFrameStartIdx: %d\n",g_dataFrameStartIdx);
		printf("## input: g_dataFrameNum: %d\n",g_dataFrameNum);

		g_fpsType = FPS_VGA_25;
		SkeletonGeneratorDlg::Script_Load_body3DPS_byFrame_folderSpecify();
		SkeletonGeneratorDlg::Script_VGA_SaveAsHDFrameIdxViaInterpolation();
	}
	else if(strcmp(option.c_str(),"skel_export_hd")==0)
	{
		sprintf(g_dataMainFolder,"%s",paramVect[2].c_str());
		sprintf(g_calibrationFolder,"%s",paramVect[3].c_str());
		sprintf(g_dataSpecificFolder,"%s/coco19_body3DPSRecon_updated_vga_hdidx",g_dataMainFolder);

		int firstFrame,lastFrame;
		GetFirstLastFrameIdx_last8digit(g_dataSpecificFolder,firstFrame,lastFrame);
		printf("firstFrame: %d, lastFrame: %d",firstFrame,lastFrame);
		g_dataFrameStartIdx = firstFrame;//atoi(paramVect[4].c_str());
		int frameEndIdx = lastFrame;//atoi(paramVect[5].c_str());
		g_dataFrameNum = frameEndIdx-g_dataFrameStartIdx+1;

		g_poseEstLoadingDataFirstFrameIdx = g_dataFrameStartIdx;
		g_poseEstLoadingDataNum = g_dataFrameNum;
		printf("## input: g_dataMainFolder: %s\n",g_dataMainFolder);
		printf("## input: g_dataSpecificFolder: %s\n",g_dataSpecificFolder);
		printf("## input: g_calibrationFolder: %s\n",g_calibrationFolder);
		printf("## input: g_dataFrameStartIdx: %d\n",g_dataFrameStartIdx);
		printf("## input: g_dataFrameNum: %d\n",g_dataFrameNum);

		g_fpsType = FPS_HD_30;
		//Load HD Skeleton Version
		SkeletonGeneratorDlg::Script_HD_Load_body3DPS_byFrame(true);
		//Save as interpolated version
		SkeletonGeneratorDlg::Script_Export_3DPS_Json(true); //this automatically handle coco19
	}
	else if(strcmp(option.c_str(),"face_pm_undistort_hd")==0 || strcmp(option.c_str(), "face_pm_recon_hd")==0 || strcmp(option.c_str(),"face_pm_export_hd")==0)
	{
		sprintf(g_dataMainFolder,"%s",paramVect[2].c_str());
		sprintf(g_calibrationFolder,"%s",paramVect[3].c_str());
		g_dataFrameStartIdx = atoi(paramVect[4].c_str());
		int frameEndIdx = atoi(paramVect[5].c_str());
		g_dataFrameNum = frameEndIdx - g_dataFrameStartIdx + 1;

		printf("%s\n",g_dataMainFolder);
		printf("%s\n",g_calibrationFolder);
		printf("%d\n",g_dataFrameStartIdx);
		printf("%d\n",g_dataFrameNum);
		g_fpsType = FPS_HD_30;
		g_askedVGACamNum = 31;
		// merge undistort + Recon

		if (strcmp(option.c_str(),"face_pm_undistort_hd")==0)
			Module_Face_pm::UndistortDetectionResult();
		else if (strcmp(option.c_str(), "face_pm_recon_hd")==0)
			Module_Face_pm::ReconFacePM70();
		else
		{
			g_poseEstLoadingDataFirstFrameIdx = g_dataFrameStartIdx;
			g_poseEstLoadingDataNum = g_dataFrameNum;
			Module_Face_pm::Load3DFace();
			Module_Face_pm::ExportToJson();
		}
	}
}