#include <cstdio>

namespace SkeletonGeneratorDlg
{
	void Script_Util_Undistort_PoseMachine2DResult_mpm_19joints(bool bHD);
	void Script_NodePartProposalRecon_fromPoseMachine_coco19();
	void Script_3DPS_Reconstruct_PoseMachine_coco19();
	void Script_Load_body3DPS_byFrame(bool bCoco19=false);
	void Script_Load_body3DPS_byFrame_folderSpecify();
	//3D Pictorial Structure from NodeProposals
	void Script_3DPS_Optimization_usingDetectionPeaks(bool bCoco19=false);
	void Script_Export_3DPS_Json(bool bNormCoord);
}