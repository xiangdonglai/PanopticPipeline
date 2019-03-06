#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include "Constants.h"
#include "ImportExportDlg.h"

// imitate command line parsing of Hanbyul's original code
int main(int argc, char** argv)
{
	std::vector<std::string> paramVect;
	for(auto i = 0; i < argc; i++) paramVect.emplace_back(argv[i]);

	std::string& option = paramVect[1];
	printf("-- Option: %s\n",option.c_str());

	if(strcmp(paramVect[1].c_str(),"calibNorm")==0)
	{
		sprintf(g_calibrationFolder, "%s", paramVect[2].c_str()); // defined in Constants.cpp
		ImportExportDlg::OnBnTransform2NormalizedCalibrationCoord();
	}
}