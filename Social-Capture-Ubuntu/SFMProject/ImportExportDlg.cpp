#include "ImportExportDlg.h"
#include "DomeImageManager.h"

namespace ImportExportDlg
{
	void OnBnTransform2NormalizedCalibrationCoord()
	{
		CDomeImageManager fullDome;
		fullDome.SetCalibFolderPath(g_calibrationFolder);
		fullDome.InitDomeCamVgaHdKinect(480, CDomeImageManager::LOAD_SENSORS_VGA_HD_KINECT);

		//Normalized 
		char saveFolder[512];
		sprintf(saveFolder,"%s/normalized/",g_calibrationFolder);
		boost::filesystem::create_directory(boost::filesystem::path(saveFolder));
		fullDome.TransformExtrisicToNormCoord(saveFolder);
	}
}