#pragma once
#include "DataStructures.h"



////////////////////////////////////////////////////////////////////
// Handling image loading and calibration data loading
// Has functions to load HD cameras, but mainly used for VGA camera loading
class CDomeImageManager
{
public:

	enum EnumLoadSensorTypes
	{
		LOAD_SENSORS_VGA=0,
		LOAD_SENSORS_HD,
		LOAD_SENSORS_KINECT,
		LOAD_SENSORS_VGA_HD,
		LOAD_SENSORS_VGA_HD_KINECT,
		LOAD_SENSORS_VGA_KINECT,
		LOAD_SENSORS_HD_KINECT,
		LOAD_SENSORS_VGA_SINGLE_PANEL
	};

	CDomeImageManager(void);
	~CDomeImageManager(void);
	void Clear()
	{
		for(unsigned int i=0;i<m_domeViews.size();++i)
		{
			if(m_domeViews[i]!=NULL)
				delete m_domeViews[i];
		}
		m_domeViews.clear();
		m_camNameToIdxTable.clear();
	};
	void DeleteMemory()		//Todo: is it just same as Clear()??
	{
		for(unsigned int i=0;i<m_domeViews.size();++i)
			delete m_domeViews[i];
		m_domeViews.clear();
	};

	//Set/Get parameters
	void SetFrameIdx(int idx);// {m_currentFrame = idx;};
	void SetCalibFolderPath(const char* path) {strcpy(m_memoryCaliDataDirPath,path);};
	void SetImageFolderPath(const char* path) {strcpy(m_memoryImageDirPath,path);};
	void SetHistEqFlag(bool flag) {m_bDoHistEqualize = flag;};
	int GetFrameIdx() const {return m_currentFrame;};
	int GetCameraNum() const {return m_domeViews.size();};
	const char* GetCalibFolderPath() {return m_memoryCaliDataDirPath;};
	pair<int,int> GetImageSize(int viewIdx);
	
	//Camera parameter setting after initialization
	static void CamParamSettingByKQuatCenter(CamViewDT& element);
	static void CamParamSettingByDecomposingP(CamViewDT& element,cv::Mat& P);
	static void CamParamSettingByPKRt(CamViewDT& element);
	static void CamParamSettingByPKRt(CamViewDT& element,cv::Mat& P,cv::Mat& K,cv::Mat& R,cv::Mat& t);

	//Camera Sampling
	static void VGACamSampling_naive(int numCams,vector< pair<int,int> >& sampledIdx);
	static void VGACamSampling_furthest(int numCams,vector< pair<int,int> >& sampledIdx);
	static void VGACamSampling(int numCams,vector< pair<int,int> >& sampledIdx);

	//void SetFrame
	void InitDomeCamVgaHdLabelOnly(vector<CamViewDT*>& refVect);
	void InitDomeCamVgaHdLabelOnly(int askedVGACameraNum=480, bool bLoadHD=false);	//No cliabration data is needed. Just add labels (predefined since it's a dome)

	void InitDomeCamVgaHdKinect(int askedVGACameraNum =480,EnumLoadSensorTypes bSensorTypeOp=LOAD_SENSORS_VGA); //CalibPatch should be set correctly, Set cameraLabel, Load calibration data
	void InitDomeCamVgaHd(int askedVGACameraNum =480,bool bLoadHD=false,bool bLoadOnlySinglePanel=false); //CalibPatch should be set correctly, Set cameraLabel, Load calibration data
	bool InitDomeCamOnlyByExtrinsic(const char* calibFolderPath,bool bLoadHD=true);		//Only load extrinsic parameters
	void LoadDomeImagesCurrentFrame(EnumLoadMode loadMode = LOAD_MODE_GRAY);	//Load images from the saved camerIdx, frameIdx, and m_memoryImageDirPath
	void LoadDomeImagesNextFrame(EnumLoadMode loadMode = LOAD_MODE_GRAY);		//Incread frameIdx and call LoadDomeImagesCurrentFrame
	bool AddCameraNImageLoad(const char* inputDirPath,int frameIdx,int panelIdx,int camIdx,bool bLoadExtrinsic);
	bool AddCameraButDontLoadImage(const char* inputDirPath,int frameIdx,int panelIdx,int camIdx,bool bLoadExtrinsic);		//don't load image. but load all others. To save time. such as using already reconstructed face detect
	bool AddCameraLabelOnly(int frameIdx,int panelIdx,int camIdx);		//no calibration, no actual image

	int GetViewIdxFromPanelCamIdx(int panelIdx,int camIdx);
	CamViewDT* GetViewDTFromPanelCamIdx(int panelIdx,int camIdx);
	void GenerateCamNameToIdxTable();
	void CalcWorldScaleParameters();	//This function update global variables
	void SaveExtrinsicParam(char* dirPath);
	void SaveIntrinsicParam(char* dirPath,bool withoutDistortion );

	//Remove cameras
	void DeleteSelectedCamerasByNonValidVect(vector<int>& camerasTobeDeleted,vector<int>& newOrders);
	void DeleteSelectedCamerasByValidVect(vector<int>& validCameras,vector<int>& newOrders);
	

	vector<CamViewDT*> m_domeViews;
	map< pair<int,int>, int > m_camNameToIdxTable;	//map (panelIdx, camIdx) -> idx of m_domeViews
	
	//to simulate non-synchronization case (by adding random offset)
	bool m_doRandomOff;
	vector<int> m_randomOff;	

	//VGA Panel Orders	
	int GetPanelOrderDistance(int idx1,int idx2);	
	static void ComputePanelOrders();

	//Made independently defined "vector<CamViewDT*> camVect"
	static void InitDomeCamVgaHdKinect(vector<CamViewDT*>& camVect,const char* calibFolder,int askedVGACameraNum =480,EnumLoadSensorTypes bSensorTypeOp=LOAD_SENSORS_VGA); //CalibPatch should be set correctly, Set cameraLabel, Load calibration data
	static bool LoadDomeImages(vector<CamViewDT*>& camVect, const char* imgDirPath,int frameIdx,EnumLoadMode loadMode = LOAD_MODE_GRAY,bool bVerbose=false);	
	static bool AddCameraButDontLoadImage(vector<CamViewDT*>& camVect,const char* calibDirPath,int frameIdx,int panelIdx,int camIdx,bool bLoadExtrinsic);
	void ComputeRtForNormCoord(cv::Mat_<double>& R,cv::Mat_<double>& t,double& scale);				//using global variables
	void TransformExtrisicToNormCoord(const char* saveDirPath);				//using global variables

private:
	bool LoadPhysicalImagesByFrameIdx(int frameIdx,EnumLoadMode loadingMode,bool bVerbose=true);
	
	//The following function update global variables
	void CalculatePhysicalScale();
	void CalculateFloorPlane();

	//VGA Panel Orders
	static vector<int> m_panelOrderIdx;
	static vector<int> m_nearPanelOfHD;
	static vector<int> m_panelIdxToOrder;

	//Parameters
	char m_memoryImageDirPath[512];
	char m_memoryCaliDataDirPath[512];
	int m_currentFrame;
	bool m_bDoHistEqualize;
};

void MergeToGridImageByPanel(const vector<CamViewDT*>& m_domeViews,vector<cv::Mat> &gridImagesByPanel, bool bUseRGBImage);
void MergeToGridImage(vector<cv::Mat>& imgVect,int maxWidth,vector< pair<int,int>>& camNameVect, cv::Mat& outputImg);


