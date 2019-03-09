#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include "Constants.h"
#include "ImportExportDlg.h"
#include "SkeletonGeneratorDlg.h"

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
		g_dataFrameNum = frameEndIdx-g_dataFrameStartIdx+1;

		printf("## input: g_dataMainFolder: %s\n",g_dataMainFolder);
		printf("## input: g_calibrationFolder: %s\n",g_calibrationFolder);
		printf("## input: g_dataFrameStartIdx: %d\n",g_dataFrameStartIdx);
		printf("## input: g_dataFrameNum: %d\n",g_dataFrameNum);

		SkeletonGeneratorDlg::Script_Util_Undistort_PoseMachine2DResult_mpm_19joints(false);
	}
}