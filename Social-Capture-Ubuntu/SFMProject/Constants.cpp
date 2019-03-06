// #include "stdafx.h"
#include "Constants.h"

using namespace cv;

// General settting //////////////////////////////////////////////////////////////////////////////////////////////////////////////

//VGA and HD frameIdx interchangeble handling
FPStype g_fpsType = FPS_VGA_25;

//char CALIB_DATA_FOLDER[512];// ="calibDataHao130323_usingIdealImage_10Cam";
char g_calibrationFolder[512];
char g_testCaliDataDirPath[512];

// Memory Data Generation //////////////////////////////////////////////////////////////////////////////////////////////////////////////
char g_dataImageFolder[512];// = "D:/Research_Window/3_data/130314_DomeData/0314Sooyeon2_images_ideal";;
char g_manualRenderImgSaveFolder[512];
char g_dataMainFolder[512];
char g_panopticDataFolder[512];
char g_sequenceName[512];		//extracted from g_dataMainFolder by choosing last folder name
char g_backgroundImagePath[512];// = "D:/Research_Window/3_datg_memoryLoadingDataNuma/130314_DomeData/0314Sooyeon2_images_ideal";;
int g_backgroundImageStartIdx = 1;
int g_backgroundImageNum = 1;
int g_dataFrameStartIdx = 0;  //all first idx for Monocular SFM
int g_dataFrameInterval=1;
int DEBUG_PATCH_HOMOGRPHY_TARGET_IDX = 570;//433//1004
int g_trackingFrameLimitNum =10;
// Memory Data Generation //////////////////////////////////////////////////////////////////////////////////////////////////////////////
int g_memoryLoadingDataNum =5;//30;
int g_memoryLoadingDataFirstFrameIdx;		
char g_dataSpecificFolder[512];
char g_GTposeEstLoadingDataFolderPath[512];
int g_GTSubjectNum;
int g_dataFrameNum=1;
int g_noVisSubject=-1;
int g_noVisSubject2=-1;
char g_maskFolderPathForVisualHull[512];

int g_faceLoadingDataFirstFrameIdx= 0;
int g_faceLoadingDataNum = 1;
char g_faceMemoryLoadingDataFolderPath[512];

char g_poseEstLoadingDataFolderComparePath[512];
int g_poseEstLoadingDataFirstFrameIdx;
int g_poseEstLoadingDataNum;
int g_poseEstLoadingDataInterval=1;

int g_visualHullLoadingDataFirstFrameIdx =1;
int g_visualHullLoadingDataNum =1;
char g_visualHullLoadingDataFolderPath[512];

int g_generalCounter=0; //for general purpose
//char g_memoryLoadingDataFirstFramePath[512];// = "D:/Research_Window/3_data/130314_DomeData/0314Sooyeon2_images_ideal/PatchRecon/recon_00000165.hbj";

// Test Data settting //////////////////////////////////////////////////////////////////////////////////////////////////////////////
//char g_testDataDirPath[] = "C:/Research_Window/sourceCode/3_data/130307_DomeData/Flea/SooyeonAtDome_ideal";
char g_testDataDirPath[512];// = "D:/Research_Window/3_data/130314_DomeData/Flea/0314Sooyeon2";
int g_testDataStartFrame= 128;
int g_testDataImageNum = 5;


//VGA frame index
int g_imgFrameIdxSlider_vga =-1;
int g_trackedFrameIdx_first_vga =0;
int g_trackedFrameIdx_last_vga =100;

//HD frame idx
int g_imgFrameIdxSlider_hd =-1;
int g_trackedFrameIdx_first_hd =0;
int g_trackedFrameIdx_last_hd =100;

// SFM settting /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*int FIRST_DOME_PANEL_IDX =1;
int FIRST_DOME_CAM_IDX =1;
int SECOND_DOME_PANEL_IDX =1;
int SECOND_DOME_CAM_IDX =2;*/
int FIRST_DOME_PANEL_IDX =1;
int FIRST_DOME_CAM_IDX =2;
int SECOND_DOME_PANEL_IDX =1;
int SECOND_DOME_CAM_IDX =3;
//int DOMEDATA_CAMNUM = 469;

int FIRST_IMAGE_IDX = 0;
int SECOND_IMAGE_IDX = 1;
int INPUT_IMAGE_NUM =50;
int INPUT_IMAGE_START_IDX=158;
int INPUT_IMAGE_STEP =10;

bool DO_HISTOGRAM_EQUALIZE = false;
bool DONT_USE_SAVED_FEATURE = false;
DatabaseReconMethod g_DBreconMethod = RECON_POINT_CLOUD;

// ETC //////////////////////////////////////////////////////////////////////////////////////////////////////////////

// for whole camera
double DOME_VOLUMECUT_X_MIN = -100;
double DOME_VOLUMECUT_X_MAX = 60;
double DOME_VOLUMECUT_Y_MAX = 70;//100;
double DOME_VOLUMECUT_Y_MIN = -100;
double DOME_VOLUMECUT_Z_MIN = 60;
double DOME_VOLUMECUT_Z_MAX = 230;

///// Dome Floor parameters //////////////////////////////////////////////////////
vector<Point3f> g_floorPts;
Point3f g_floorCenter;
Point3f g_floorNormal;
Point3f g_floorAxis1;
Point3f g_floorAxis2;

// FOR DISPLAY //////////////////////////////////////////////////////////////////
int MAX_TRAJ_SHOW_LENGTH = 100;

// Not that important Parameters ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Point3d DOME_VOLUMECUT_CENTER_PT(-23, -1 ,158);
double DOME_VOLUMECUT_RADIOUS = 50;//130;
double OPTICALFLOW_BIDIRECT_DIST_THRESH = 3;

// Parameters ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double PATCH_3D_ARROW_SIZE_CM = 1; //2;		
double PATCH_3D_ARROW_SIZE_WORLD_UNIT=0;  //Have to be recalculated using PATCH_HALF_SIZE_CM * WORLD_TO_CM_RATIO
//double INIT_3D_TRIPLET_SIZE = 4.0;

//For optimization
int PATCH3D_GRID_SIZE = 11;//11;
//int PATCH3D_GRID_SIZE = 21;
int PATCH3D_GRID_HALFSIZE = PATCH3D_GRID_SIZE/2;

//For normalizing Data terms
double VISIBLE_MOTION_LIMIT = 5.0;
//double VISIBLE_MOTION_LIMIT = 10.0;
double VISIBLE_NORMAL_ANGLE_LIMIT = 1;

//Data Term Ratios
double g_dataterm_ratio_motion=1;
double g_dataterm_ratio_appearance=0;
double g_dataterm_ratio_normal=0;
bool g_bDataterm_ratio_auto = true;
bool g_enableMRF = true;
bool g_enableRANSAC = true;
double g_MRF_smoothRatio = 0;

int g_PoseReconTargetSubject = 1;
int g_PoseReconTargetJoint = -1;
int g_PoseReconTargetRequestedNum = 10;		//how many more joint do you want to find
int g_askedVGACamNum  = 1000;

double MAGIC_VISIBILITY_COST_THRESHOLD = 0.8;
int MAGIC_RANSAC_ITER_NUM = 1000;

double MAGIC_TRIPLET_TRIANGULATION_REPRO_ERROR_LIMIT = 2.0;
double TRACKING3D_RANSAC_TH = 1;//0.5;

int g_pointSelectionByInputParam =-1;

//Filtering points by Floor plane
//floor plane
double g_floorPlane_xCoef = -0.448511;
double g_floorPlane_yCoef = -10.778207;
double g_floorPlane_thread = -25;
double g_floorPlane_const =1030.592200;

int g_ShowLongTermThreshDuration = 50;
int g_memoryLoadingDataInterval=1;

double WORLD_TO_CM_RATIO =1;//1.843742;   //garbage value
double  DOME_CAMERA_DIST_CM_UNIT = 22*0.95715;  //I measured this by ruler. 0.95715 is computed by checkerboard reconstruction
bool g_bColorEnhanced = false;
bool USE_PNG_IMAGE = true;

bool g_bDoPhotometOptimization = true;
bool g_bDoRejection_PhotoCons = true;
bool g_bDoRejection_PhotoCons_onlySmallMotion = true;
bool g_bDoRejection_MotionMagnitude = true;		
bool g_bDoRejection_VisCamNum = true;
bool g_bDoRejection_PatchSize = true;
bool g_bDoRejection_RANSAC_ERROR = true;

float g_reject_PhotometricConsist_thresh = 0.8;
float g_reject_MotionMagnitude_thresh = 30;	//CM unit
int g_reject_VisibilityNum_thresh = 5;	//CM unit

float g_nearPlaneForDepthRender = 0.001;
float g_farPlaneForDepthRender = 100;		//GLfloat 

char* SELECTED_PT_FILE_NAME ="c:/tempResult/selectedPts/selectedPts.txt";

int g_partPropScoreMode=PART_TRAJ_SCORE_WEIGHTED;

int g_gpu_device_id=0;

char g_renderSubFolderName[512];
bool g_render_skeletonWithText_trigger=false;


//Mesh Models
char g_smpl_model_path[512];
char g_face_model_path[512];
char g_handr_model_path[512];
char g_handl_model_path[512];
