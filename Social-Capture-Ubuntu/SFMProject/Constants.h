#pragma once
#//include <vld.h>
#include <vector>
#include <opencv2/core/core.hpp>
using namespace std;


// Important options //////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define WITHOUD_DISTORTION_COEEF

//  Debug options  //////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define DEBUG_PATCH_HOMOGRPHY_EST 0
//#define SINGLE_IMAGE_RECOND_DEBUG

#define SHOW_PATCH_OPTIMIZATION 0
#define TARGET_PATCH_NUM 930

#define SHOW_FEATURE_EXTRACT_RESULT 0
#define FIRST_TWO_CAM_DISTANCE (10)
//#define FIRST_TWO_CAM_DISTANCE (1)
#define SHOW_WORST_PATCH 0
#define SAVE_ESTIMATED_PATCH_PROJECTION_FOR_SINGLE_RECON 0 

#define PANELDIFF_THRSH (1)
#define CAMDISTANCE_THRSH (300)		//300cm

//ETC //////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum DatabaseReconMethod
{
	RECON_POINT_CLOUD,
	RECON_3D_PATCH
};


//Related to SyncMan.h
enum FPStype
{
	FPS_VGA_25 =0,
	FPS_HD_30  //1
};
extern FPStype g_fpsType;


extern char g_dataImageFolder[];
extern char g_testDataDirPath[];
extern DatabaseReconMethod g_DBreconMethod;
extern int DEBUG_PATCH_HOMOGRPHY_TARGET_IDX;//433//1004
//extern char CALIB_DATA_FOLDER[];
extern char g_manualRenderImgSaveFolder[];
extern int FIRST_DOME_PANEL_IDX ;
extern int FIRST_DOME_CAM_IDX ;
extern int SECOND_DOME_PANEL_IDX ;
extern int SECOND_DOME_CAM_IDX ;
extern int g_dataFrameStartIdx;
extern int g_dataFrameInterval;
//extern int DOMEDATA_CAMNUM;
extern int MATCHING_SUFFICEINTBOUND;
extern int MATCHING_LOWERBOUND;
//#define INPUT_DIGIT_4
//#define INPUT_JPG
//#define INPUT_JPG
extern int FIRST_IMAGE_IDX ;
extern int SECOND_IMAGE_IDX ;
extern int INPUT_IMAGE_NUM;
extern int INPUT_IMAGE_START_IDX;
extern int INPUT_IMAGE_STEP;
extern bool DONT_USE_SAVED_FEATURE;
extern double DOME_VOLUMECUT_X_MIN;
extern double DOME_VOLUMECUT_X_MAX;
extern double DOME_VOLUMECUT_Y_MAX;
extern double DOME_VOLUMECUT_Y_MIN;
extern double DOME_VOLUMECUT_Z_MIN;
extern double DOME_VOLUMECUT_Z_MAX;
extern int g_testDataImageNum;
extern int g_testDataStartFrame;
extern int g_memoryLoadingDataNum;
extern int g_memoryLoadingDataFirstFrameIdx;
extern char g_dataSpecificFolder[];
extern char g_GTposeEstLoadingDataFolderPath[];
extern int g_GTSubjectNum;


//Global frame range
extern int g_imgFrameIdxSlider_vga;		//current frame by sliderbar in GUI
extern int g_trackedFrameIdx_first_vga;	 //very initial frame from all loaded data
extern int g_trackedFrameIdx_last_vga;	//very last frame from all loaded data

extern int g_imgFrameIdxSlider_hd;
extern int g_trackedFrameIdx_first_hd;
extern int g_trackedFrameIdx_last_hd;

//Dome Floor Paramters
extern vector<cv::Point3f> g_floorPts;
extern cv::Point3f g_floorCenter;
extern cv::Point3f g_floorNormal;
extern cv::Point3f g_floorAxis1;
extern cv::Point3f g_floorAxis2;

//face
extern int g_faceLoadingDataFirstFrameIdx;
extern int g_faceLoadingDataNum;
extern char g_faceMemoryLoadingDataFolderPath[];

//human pose estimation
extern char g_poseEstLoadingDataFolderComparePath[];
extern int g_poseEstLoadingDataFirstFrameIdx;
extern int g_poseEstLoadingDataNum;
extern int g_poseEstLoadingDataInterval;

//visualHull
extern int g_visualHullLoadingDataFirstFrameIdx;
extern int g_visualHullLoadingDataNum;
extern char g_visualHullLoadingDataFolderPath[];


extern char g_dataMainFolder[];
extern char g_panopticDataFolder[];
extern char g_sequenceName[];	
extern char g_calibrationFolder[];
extern char g_testCaliDataDirPath[];
extern char g_backgroundImagePath[];
extern char g_maskFolderPathForVisualHull[];
extern int g_backgroundImageStartIdx;
extern int g_backgroundImageNum;
extern int g_trackingFrameLimitNum;
extern int g_dataFrameNum;
extern int g_noVisSubject;
extern int g_noVisSubject2;

extern int g_PoseReconTargetSubject;
extern int g_PoseReconTargetJoint;
extern int g_PoseReconTargetRequestedNum;
extern int g_askedVGACamNum;
extern int g_generalCounter;		//for general purpose

//For Dome system data
#define DOME_DATA 
#define DOME_PANEL_NUM (20)
#define DOME_VGA_CAMNUM_EACHPANEL (24)
#define DOME_HD_CAMNUM (31)		//max possible number
#define DOME_KINECT_CAMNUM (10)		//max possible number

//Panel Index 
#define PANEL_HD (0)
#define PANEL_KINECT (50)

#define SENSOR_TYPE_VGA (1)		//note that this is same as panel label for vga case, since vga panel index can be 1-20. 
#define SENSOR_TYPE_HD	(0)
#define SENSOR_TYPE_KINECT (50)
#define SENSOR_TYPE_UNDEFINED (-1)

#ifndef PI
#define PI 3.14159265
#endif

extern cv::Point3d DOME_VOLUMECUT_CENTER_PT;
extern double DOME_VOLUMECUT_RADIOUS;
extern double WORLD_TO_CM_RATIO;
#define world2cm(X) (X*WORLD_TO_CM_RATIO)
#define cm2world(X) (X/WORLD_TO_CM_RATIO)

//Parameters
extern double PATCH_3D_ARROW_SIZE_CM;		
extern double PATCH_3D_ARROW_SIZE_WORLD_UNIT;
extern double VISIBLE_MOTION_LIMIT;
extern double VISIBLE_NORMAL_ANGLE_LIMIT;extern int PATCH3D_GRID_SIZE;
extern int PATCH3D_GRID_HALFSIZE;
extern int MAX_TRAJ_SHOW_LENGTH;
extern double OPTICALFLOW_BIDIRECT_DIST_THRESH;
extern int MAGIC_RANSAC_ITER_NUM;


extern double MAGIC_TRIPLET_TRIANGULATION_REPRO_ERROR_LIMIT;
extern double g_dataterm_ratio_motion;
extern double g_dataterm_ratio_appearance;
extern double g_dataterm_ratio_normal;
extern double TRACKING3D_RANSAC_TH;
extern double MAGIC_VISIBILITY_COST_THRESHOLD;



extern int g_pointSelectionByInputParam; 

extern double g_floorPlane_xCoef;
extern double g_floorPlane_yCoef;
extern double g_floorPlane_const ;
extern double g_floorPlane_thread ;

#define FLOOR_FILTER(x,y,z)  ( (x)*g_floorPlane_xCoef + (y)* g_floorPlane_yCoef - (z) + g_floorPlane_const + g_floorPlane_thread)


extern int g_ShowLongTermThreshDuration;
extern int g_memoryLoadingDataInterval;

extern double DOME_CAMERA_DIST_CM_UNIT;  //22
extern bool DO_HISTOGRAM_EQUALIZE;

extern bool g_bColorEnhanced;
extern bool USE_PNG_IMAGE;




//Flag for tracking
extern bool g_enableRANSAC;
extern bool g_enableMRF;
extern double g_MRF_smoothRatio;
extern bool g_bDataterm_ratio_auto;
extern bool g_bDoPhotometOptimization;
extern bool g_bDoRejection_PhotoCons;
extern bool g_bDoRejection_PhotoCons_onlySmallMotion;
extern bool g_bDoRejection_MotionMagnitude;		
extern bool g_bDoRejection_VisCamNum;
extern bool g_bDoRejection_PatchSize;
extern bool g_bDoRejection_RANSAC_ERROR;

extern float g_reject_PhotometricConsist_thresh;
extern float g_reject_MotionMagnitude_thresh;
extern int g_reject_VisibilityNum_thresh;

//openglRendering and depth Rendering
extern float g_nearPlaneForDepthRender;
extern float g_farPlaneForDepthRender;		//GLfloat 

extern char* SELECTED_PT_FILE_NAME; 

extern char g_renderSubFolderName[];
extern bool g_render_skeletonWithText_trigger;

enum EnumLoadMode
{
	LOAD_MODE_GRAY_RGB =0,
	LOAD_MODE_GRAY,
	LOAD_MODE_RGB,
	LOAD_MODE_NO_IMAGE
};


extern int g_partPropScoreMode;
#define PART_TRAJ_SCORE_WEIGHTED (0)
#define PART_TRAJ_SCORE_NON_WEIGHTED (1)
#define PART_TRAJ_SCORE_WEIGHTED_DIV_CNT (2)

extern int g_gpu_device_id;

//Mesh Models
extern char g_smpl_model_path[];
extern char g_face_model_path[];
extern char g_handr_model_path[];
extern char g_handl_model_path[];