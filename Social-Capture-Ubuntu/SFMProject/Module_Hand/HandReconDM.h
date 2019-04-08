#pragma once

#include "DomeImageManager.h"
#include "Module_VisualHull/VisualHullModule.h"
//#include "DataStructures.h"
#include "UtilityGPU.h"

#include "HandReconDT.h"

namespace Module_Hand
{

	bool Load_Undist_HandDetectMultipleCamResult_PoseMachine(const char* poseDetectFolder,const char* poseDetectSaveFolder,
		const int currentFrame, CDomeImageManager& domeImMan,bool isHD);

	bool LoadFingerDetectResult_MultiCams_Json(const char* loadFolderName,const int currentFrame, const vector<CamViewDT*>& camViews,vector< vector<SHand2D> >& detectedHandVect,bool isHD );

class CHandRecon
{
public:
	CHandRecon(void);
	~CHandRecon(void){};

	vector<SHandReconMemory> m_handReconMem;		//Save reconstructed 3D faces. Each element for each time instance

	void Finger_Landmark_Reconstruction_Voting_hd(const char* handDetectFilePath, CDomeImageManager& domeImageManager,std::vector<SHand3D>& handReconResult);
	
	// //Visualization	
	// void SetCurMemIdxByFrameIdx(int curImgFrameIdx);
	// void ShowHandReconResult(VisualizedData& vis,bool bCompareData);
	// void ShowHandRecon3DResult(VisualizedData& vis, vector<SHand3D>& handRecon,bool bCompareData);
	// void VisHandsOnSelectedCam_3DPS_givenImg(CamViewDT* targetCam, cv::Mat& givenImg,bool bPopup=false);
	// void VisHandsOnGivenImg(CamViewDT* targetCam, cv::Mat& img);

	//Load/Save
	void SaveHandReconResult(const char* folderPath,vector<SHand3D>& faceRecon,int frameIdx,bool bIsHD);
	void SaveHandReconResult_json(const char* folderPath,vector<SHand3D>& faceRecon,int frameIdx,bool bIsHD);
	void LoadHand3DByFrame(const char* folderPath,const int firstFrameIdx,const int frameNum,bool bIsHD);

	//Utilities
	void ComputeHandNormals();
	//void Script_Quantify_Detector(const char* handDetectFileDir,const char* saveFolderDir);	

	void Scrit_handOutlier_rejection();

	//Parameters
	float m_vis_avgDetectThresh;
	float m_vis_reproErrorThresh;
	int m_vis_visibilityThresh;

	int m_fpsType;
private:

	//outlier detection
	void OutlierRejection_length(SHand3D* pHand);
	void OutlierRejection_overlap(SHand3D* pLHand,SHand3D* pRHand);

	int m_currentSelectedLocalVectIdx;		//face vector element idx at current time t
	int m_curImgFrameIdx;

	vector< vector<int> > m_fingerGroupIdxVect;

	void GenerateHumanColors();
	vector<cv::Point3f> m_colorSet;
};

extern CHandRecon g_handReconManager;
extern CHandRecon g_handReconManager_compare;

}	//end of namespace Module_Hand
