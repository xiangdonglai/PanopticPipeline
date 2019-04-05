#pragma once

#include "DomeImageManager.h"
#include "Module_VisualHull/VisualHullModule.h"
//#include "DataStructures.h"
#include "UtilityGPU.h"

#include "FaceReconByPMDT.h"

namespace Module_Face_pm
{

	bool Load_Undist_FaceDetectMultipleCamResult_face70_PoseMachine(const char* poseDetectFolder,const char* poseDetectSaveFolder,
		const int currentFrame, CDomeImageManager& domeImMan,bool isHD);

	bool Load_Undist_FaceDetectMultipleCamResult_faceCoco_PoseMachine(const char* poseDetectFolder,const char* poseDetectSaveFolder,
		const int currentFrame, CDomeImageManager& domeImMan,bool isHD);
	
class CFaceRecon_pm
{
public:
	CFaceRecon_pm(void);
	~CFaceRecon_pm(void){};

	vector<SFaceReconMemory_pm> m_faceReconMem;		//Save reconstructed 3D faces. Each element for each time instance

	void Face_Landmark_Reconstruction_hd(const char* faceDetectFilePath, CDomeImageManager& domeImageManager,std::vector<SFace3D_pm>& faceReconResult);
	
	
	//Visualization	
	void SetCurMemIdxByFrameIdx(int curImgFrameIdx);

	void ComputeFaceNormals();

	//Load/Save
	void SaveFaceReconResult(const char* folderPath,vector<SFace3D_pm>& faceRecon,int frameIdx,bool bIsHD);
	void SaveFaceReconResult_json(const char* folderPath,vector<SFace3D_pm>& faceRecon,int frameIdx,bool bIsHD);
	void LoadFace3DByFrame(const char* folderPath,const int firstFrameIdx,const int frameNum,bool bIsHD);
	

	//Parameters
	float m_vis_avgDetectThresh;
	float m_vis_reproErrorThresh;
	int m_vis_visibilityThresh;

	int m_fpsType;
private:
	int m_currentSelectedLocalVectIdx;		//face vector element idx at current time t
	int m_curImgFrameIdx;

	void GenerateHumanColors();
	vector<cv::Point3f> m_colorSet;
};

extern CFaceRecon_pm g_faceReconManager_pm;
}	//end of namespace Module_Hand
