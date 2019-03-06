// #include "stdafx.h"
#include <string.h>
#include <sstream>
#include "DomeImageManager.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>	//RQDecomp3x3
#include <list>
#define IGNORE_BROKEN_CAM 1


using namespace cv;


void CDomeImageManager::SetFrameIdx(int idx)
{
	m_currentFrame = idx;
	for(int i=0;i<m_domeViews.size();++i)
	{
		m_domeViews[i]->m_actualImageFrameIdx = idx;

	}
}

CDomeImageManager::CDomeImageManager(void)
{
	//Set up Folder Path from global variables
	//WARNING: global variable may not be valid for static members which is made before reading INputParameter.txt
	strcpy(m_memoryImageDirPath,g_dataImageFolder);
	strcpy(m_memoryCaliDataDirPath,g_calibrationFolder);
	m_currentFrame = g_dataFrameStartIdx;

	//Generate random synchronization (used for off-syncronization test)
	for(int i=0;i<550;++i)		//550 is arbitrary big number than camera num
	{
		int randOff = rand()%3-2;			//-2 ~ 2
		m_randomOff.push_back(randOff);
	}
	m_doRandomOff = false;

	m_bDoHistEqualize = DO_HISTOGRAM_EQUALIZE;
}

CDomeImageManager::~CDomeImageManager(void)
{
	for(unsigned int i=0;i<m_domeViews.size();++i)
	{
		if(m_domeViews[i]!=NULL)
			delete m_domeViews[i];
	}
	m_domeViews.clear();
}

//Load image data (and calibration data) for the m_currentFrame
void CDomeImageManager::LoadDomeImagesCurrentFrame(EnumLoadMode loadMode)
{
	printf("## LoadDomeImagesCurrentFrame started: frame %d: camNum %d\n",m_currentFrame,m_domeViews.size());

	if(m_domeViews.size()==0)
	{
		printf("## ERROR: CDomeImageManager: Please Init first\n");
		return;
	}
	/*if(m_domeViews.size()==0)
		InitDomeCamVgaHd(loadMode,bLoadHD,bDebugMode);		//Need to change to apply loadMode here. In current settup, the first loaded images are loadMode 2 (gray  & rgb)
	else*/
		LoadPhysicalImagesByFrameIdx(m_currentFrame,loadMode,false);

	printf("## LoadDomeImagesCurrentFrame has been finished: camNum %d\n",m_domeViews.size());
}

void CDomeImageManager::LoadDomeImagesNextFrame(EnumLoadMode loadMode)
{
	m_currentFrame++;
	LoadDomeImagesCurrentFrame(loadMode);
}

//Set cameraLabel (predefined. only camNum matters)
//Load calibration data (m_memoryCaliDataDirPath should be set first)
void CDomeImageManager::InitDomeCamVgaHdKinect(int askedVGACameraNum, EnumLoadSensorTypes bSensorTypeOp)
{
	Clear();
	//int vgaCameraNum = askedVGACameraNum;			
	stringstream ss;
	ss << "## Init DomeImageManager ::\n";
	int nKinect=0,nVGA=0,nHD=0;
	//HD
	if(bSensorTypeOp==LOAD_SENSORS_HD || bSensorTypeOp==LOAD_SENSORS_VGA_HD ||  bSensorTypeOp==LOAD_SENSORS_VGA_HD_KINECT ||bSensorTypeOp==LOAD_SENSORS_HD_KINECT)
	{
		int panelIdx  = PANEL_HD;
		int camNum  = DOME_HD_CAMNUM;
		int camStartIdx = 0;
		//for(int camIdx=camStartIdx;camIdx<=camNum ;++camIdx)
		for(int camIdx=camStartIdx;camIdx<camNum ;++camIdx)
		{
			/*if(camIdx== 1 || camIdx== 4  || camIdx== 7 || camIdx== 2 || camIdx== 28 || camIdx== 13 || camIdx== 17 || camIdx== 6 || camIdx== 10 || camIdx== 19 || camIdx==26)
				continue;

			if(camIdx==21)		//160422 ultimatum1 frame drop
				continue;*/

			bool bSuccess = AddCameraButDontLoadImage(m_memoryImageDirPath,m_currentFrame,panelIdx,camIdx,true);
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",panelIdx,camIdx);
				//m_domeViews.pop_back();
				continue;
			}

			nHD++;
		}
	}

	//Kinect
	if(bSensorTypeOp==LOAD_SENSORS_KINECT || bSensorTypeOp==LOAD_SENSORS_HD_KINECT ||  bSensorTypeOp==LOAD_SENSORS_VGA_KINECT || bSensorTypeOp==LOAD_SENSORS_VGA_HD_KINECT)
	{
		int panelIdx  = PANEL_KINECT;
		int camNum  = DOME_KINECT_CAMNUM;
		int camStartIdx =1;
		for(int camIdx=camStartIdx;camIdx<=camNum ;++camIdx)		
		{
			bool bSuccess = AddCameraButDontLoadImage(m_memoryImageDirPath,m_currentFrame,panelIdx,camIdx,true);
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",panelIdx,camIdx);
				//m_domeViews.pop_back();
				continue;
			}
			nKinect++;
		}

	}
	
	//VGA
	if(bSensorTypeOp==LOAD_SENSORS_VGA_SINGLE_PANEL)
	{
		int panelIdx  = 1;
		int camNum  = DOME_VGA_CAMNUM_EACHPANEL;
		int camStartIdx =1;
		for(int camIdx=camStartIdx;camIdx<=camNum ;++camIdx)		//cam
		{
			bool bSuccess = AddCameraButDontLoadImage(m_memoryImageDirPath,m_currentFrame,panelIdx,camIdx,true);
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",panelIdx,camIdx);
				//m_domeViews.pop_back();
				continue;
			}
			nVGA++;
		}
	}
	else if(bSensorTypeOp==LOAD_SENSORS_VGA || bSensorTypeOp==LOAD_SENSORS_VGA_HD ||  bSensorTypeOp==LOAD_SENSORS_VGA_HD_KINECT ||bSensorTypeOp==LOAD_SENSORS_VGA_KINECT)
	{
		/*int order[] = {1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
		vector<int> panelOrderIdx(order,order + sizeof(order)/sizeof(int) );
			
		vector< pair<int,int> > camLabels;
		for(unsigned int t=0;t<panelOrderIdx.size();++t)
		{
			for(int camIdx=1;camIdx<=24 ;++camIdx)
			{
#ifdef IGNORE_BROKEN_CAM
				if(panelOrderIdx[t] ==14 && camIdx == 18)
					continue;
#endif
				camLabels.push_back( make_pair(panelOrderIdx[t],camIdx));
			}
		}

		vector<pair<int,int> > sampledIdx;  
		if(askedVGACameraNum>=camLabels.size())
			sampledIdx = camLabels;
		else
		{
			float ratio = float(camLabels.size())/float(askedVGACameraNum);
			for(int i=1;i<=askedVGACameraNum;++i)
			{
				int idx = ratio*i - 1 ;
				sampledIdx.push_back( camLabels[idx]);
			}
		}*/
		vector<pair<int,int> > sampledIdx;  
		VGACamSampling(askedVGACameraNum,sampledIdx);

		for(int i=0;i<sampledIdx.size();++i)
		{
			bool bSuccess;
			bSuccess = AddCameraButDontLoadImage(m_memoryImageDirPath,m_currentFrame,sampledIdx[i].first,sampledIdx[i].second,true);
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",sampledIdx[i].first,sampledIdx[i].second);
				//m_domeViews.pop_back();
				continue;
			}
			nVGA++;
		}
	}

	
	printf("## Init DomeImageManager :: \n\tcalibrationPath: %s \n\t VGA: %d (asked %d)\n\t HD %d\n\t kinect %d\n",m_memoryCaliDataDirPath,nVGA,askedVGACameraNum,nHD,nKinect);
}



//Set cameraLabel (predefined. only camNum matters)
//Load calibration data (m_memoryCaliDataDirPath should be set first)
void CDomeImageManager::InitDomeCamVgaHdKinect(vector<CamViewDT*>& camVect,const char* calibFolder,int askedVGACameraNum, EnumLoadSensorTypes bSensorTypeOp)
{
	for(unsigned int i=0;i<camVect.size();++i)
	{
		if(camVect[i]!=NULL)
			delete camVect[i];
	}
	//m_camNameToIdxTable.clear();

	//int vgaCameraNum = askedVGACameraNum;			
	stringstream ss;
	ss << "## Init DomeImageManager ::\n";
	int nKinect=0,nVGA=0,nHD=0;
	//HD
	if(bSensorTypeOp==LOAD_SENSORS_HD || bSensorTypeOp==LOAD_SENSORS_VGA_HD ||  bSensorTypeOp==LOAD_SENSORS_VGA_HD_KINECT ||bSensorTypeOp==LOAD_SENSORS_HD_KINECT)
	{
		int panelIdx  = PANEL_HD;
		int camNum  = DOME_HD_CAMNUM;
		int camStartIdx = 0;
		//for(int camIdx=camStartIdx;camIdx<=camNum ;++camIdx)
		for(int camIdx=camStartIdx;camIdx<camNum ;++camIdx)
		{
			bool bSuccess = AddCameraButDontLoadImage(camVect,calibFolder,-1,panelIdx,camIdx,true);
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",panelIdx,camIdx);
				//m_domeViews.pop_back();
				continue;
			}

			nHD++;
		}
	}

	//Kinect
	if(bSensorTypeOp==LOAD_SENSORS_KINECT || bSensorTypeOp==LOAD_SENSORS_HD_KINECT ||  bSensorTypeOp==LOAD_SENSORS_VGA_KINECT || bSensorTypeOp==LOAD_SENSORS_VGA_HD_KINECT)
	{
		int panelIdx  = PANEL_KINECT;
		int camNum  = DOME_KINECT_CAMNUM;
		int camStartIdx =1;
		for(int camIdx=camStartIdx;camIdx<=camNum ;++camIdx)		
		{
			bool bSuccess = AddCameraButDontLoadImage(camVect,calibFolder,-1,panelIdx,camIdx,true);
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",panelIdx,camIdx);
				//m_domeViews.pop_back();
				continue;
			}
			nKinect++;
		}

	}

	//VGA
	if(bSensorTypeOp==LOAD_SENSORS_VGA_SINGLE_PANEL)
	{
		int panelIdx  = 1;
		int camNum  = DOME_VGA_CAMNUM_EACHPANEL;
		int camStartIdx =1;
		for(int camIdx=camStartIdx;camIdx<=camNum ;++camIdx)		//cam
		{
			bool bSuccess = AddCameraButDontLoadImage(camVect,calibFolder,-1,panelIdx,camIdx,true);
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",panelIdx,camIdx);
				//m_domeViews.pop_back();
				continue;
			}
			nVGA++;
		}
	}
	else if(bSensorTypeOp==LOAD_SENSORS_VGA || bSensorTypeOp==LOAD_SENSORS_VGA_HD ||  bSensorTypeOp==LOAD_SENSORS_VGA_HD_KINECT ||bSensorTypeOp==LOAD_SENSORS_VGA_KINECT)
	{
		/*int order[] = {1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
		vector<int> panelOrderIdx(order,order + sizeof(order)/sizeof(int) );

		vector< pair<int,int> > camLabels;
		for(unsigned int t=0;t<panelOrderIdx.size();++t)
		{
			for(int camIdx=1;camIdx<=24 ;++camIdx)
			{
#ifdef IGNORE_BROKEN_CAM
				if(panelOrderIdx[t] ==14 && camIdx == 18)
					continue;
#endif
				camLabels.push_back( make_pair(panelOrderIdx[t],camIdx));
			}
		}

		vector<pair<int,int> > sampledIdx;  
		if(askedVGACameraNum>=camLabels.size())
			sampledIdx = camLabels;
		else
		{
			float ratio = float(camLabels.size())/float(askedVGACameraNum);
			for(int i=1;i<=askedVGACameraNum;++i)
			{
				int idx = ratio*i - 1 ;
				sampledIdx.push_back( camLabels[idx]);
			}
		}*/
		vector<pair<int,int> > sampledIdx;  
		VGACamSampling(askedVGACameraNum,sampledIdx);


		for(int i=0;i<sampledIdx.size();++i)
		{
			bool bSuccess;
			bSuccess = AddCameraButDontLoadImage(camVect,calibFolder,-1,sampledIdx[i].first,sampledIdx[i].second,true);
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",sampledIdx[i].first,sampledIdx[i].second);
				//m_domeViews.pop_back();
				continue;
			}
			nVGA++;
		}
	}
	printf("## Init DomeImageManager :: \n\tcalibrationPath: %s \n\t VGA: %d (asked %d)\n\t HD %d\n\t kinect %d\n",calibFolder,nVGA,askedVGACameraNum,nHD,nKinect);
}

void CDomeImageManager::VGACamSampling(int numCams,vector< pair<int,int> >& sampledIdx)
{
	if(numCams==20)
	{
		sampledIdx.push_back(make_pair(6,24));
		sampledIdx.push_back(make_pair(8,20));
		sampledIdx.push_back(make_pair(18,22));
		sampledIdx.push_back(make_pair(4,19));
		sampledIdx.push_back(make_pair(7,19));
		sampledIdx.push_back(make_pair(9,24));
		sampledIdx.push_back(make_pair(19,22));
		sampledIdx.push_back(make_pair(11,20));
		sampledIdx.push_back(make_pair(10,19));
		sampledIdx.push_back(make_pair(12,24));
		sampledIdx.push_back(make_pair(20,22));
		sampledIdx.push_back(make_pair(14,20));
		sampledIdx.push_back(make_pair(13,19));
		sampledIdx.push_back(make_pair(16,22));
		sampledIdx.push_back(make_pair(15,24));
		sampledIdx.push_back(make_pair(2,20));
		sampledIdx.push_back(make_pair(1,19));
		sampledIdx.push_back(make_pair(3,24));
		sampledIdx.push_back(make_pair(5,20));
		sampledIdx.push_back(make_pair(17,22));
		return;
	}
	else if(numCams==19)
	{
		//sampledIdx.push_back(make_pair(6,24));
		sampledIdx.push_back(make_pair(8,20));
		sampledIdx.push_back(make_pair(18,22));
		sampledIdx.push_back(make_pair(4,19));
		sampledIdx.push_back(make_pair(7,19));
		sampledIdx.push_back(make_pair(9,24));
		sampledIdx.push_back(make_pair(19,22));
		sampledIdx.push_back(make_pair(11,20));
		sampledIdx.push_back(make_pair(10,19));
		sampledIdx.push_back(make_pair(12,24));
		sampledIdx.push_back(make_pair(20,22));
		sampledIdx.push_back(make_pair(14,20));
		sampledIdx.push_back(make_pair(13,19));
		sampledIdx.push_back(make_pair(16,22));
		sampledIdx.push_back(make_pair(15,24));
		sampledIdx.push_back(make_pair(2,20));
		sampledIdx.push_back(make_pair(1,19));
		sampledIdx.push_back(make_pair(3,24));
		sampledIdx.push_back(make_pair(5,20));
		sampledIdx.push_back(make_pair(17,22));
		return;
	}
	//VGACamSampling_naive(numCams,sampledIdx);
	VGACamSampling_furthest(numCams,sampledIdx);
}

void CDomeImageManager::VGACamSampling_naive(int numCams,vector< pair<int,int> >& sampledIdx)
{
	printf("##### VGACamSampling_naive ###\n");
	int order[] = {1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
	vector<int> panelOrderIdx(order,order + sizeof(order)/sizeof(int) );

	if(numCams>=480)
	{
		sampledIdx.clear();  
		for(unsigned int t=0;t<panelOrderIdx.size();++t)
		{
			for(int camIdx=1;camIdx<=24 ;++camIdx)
			{
				sampledIdx.push_back( make_pair(panelOrderIdx[t],camIdx));
			}
		}
	}
	else
	{
		vector< pair<int,int> > allIdx;
		for(unsigned int t=0;t<panelOrderIdx.size();++t)
		{
			for(int camIdx=1;camIdx<=24 ;++camIdx)
			{
				if(panelOrderIdx[t] ==14 && camIdx == 18)			//always ignore this guy, since it is broken. Because numCams<480, it is fine.
					continue;

				allIdx.push_back( make_pair(panelOrderIdx[t],camIdx));
			}
		}

		sampledIdx.clear();  
		float ratio = float(allIdx.size())/float(numCams);
		for(int i=1;i<=numCams;++i)
		{
			int idx = ratio*i - 1 ;
			sampledIdx.push_back( allIdx[idx]);
		}
		printf("######################## Debug: check sampling result: %d (asked) vs %d (sampled)\n\ ########################\n", numCams,sampledIdx.size());
	}
}



void CDomeImageManager::VGACamSampling_furthest(int numCams,vector< pair<int,int> >& sampledIdx)
{
	printf("##### VGACamSampling_furthest ###\n");
	int order[] = {1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
	vector<int> panelOrderIdx(order,order + sizeof(order)/sizeof(int) );

	if(numCams>=480)
	{
		sampledIdx.clear();  
		for(unsigned int t=0;t<panelOrderIdx.size();++t)
		{
			for(int camIdx=1;camIdx<=24 ;++camIdx)
			{
				sampledIdx.push_back( make_pair(panelOrderIdx[t],camIdx));
			}
		}
	}
	else
	{
		CDomeImageManager fullDome;
		fullDome.SetCalibFolderPath(g_calibrationFolder);
		fullDome.InitDomeCamOnlyByExtrinsic(g_calibrationFolder,false);
		


		list< pair<int,int> > allIdx;
		for(unsigned int t=0;t<panelOrderIdx.size();++t)
		{
			for(int camIdx=1;camIdx<=24 ;++camIdx)
			{
				if(panelOrderIdx[t] ==14 && camIdx == 18)			//always ignore this guy, since it is broken. Because numCams<480, it is fine.
					continue;
				pair<int,int> tempPair = make_pair(panelOrderIdx[t],camIdx);
				allIdx.push_back( tempPair);
			}
		}

		sampledIdx.clear();
		sampledIdx.push_back(allIdx.front());
		allIdx.pop_front();
		while(sampledIdx.size()<numCams)
		{
			list< pair<int,int> >::iterator iter =  allIdx.begin();
			list< pair<int,int> >::iterator iter_furthest =  allIdx.begin();
			double distMax =-1;
			while(iter!=allIdx.end())
			{
				double curClosestDist=1e5;
				CamViewDT* camDT_target = fullDome.GetViewDTFromPanelCamIdx(iter->first,iter->second);

				for(int i=0;i<sampledIdx.size();++i)
				{
					CamViewDT* camDT_prev = fullDome.GetViewDTFromPanelCamIdx(sampledIdx[i].first,sampledIdx[i].second);
					double centerDist = Distance(camDT_target->m_CamCenter,camDT_prev->m_CamCenter);
					
					curClosestDist = min(curClosestDist,centerDist);
				}
				
				if(curClosestDist >distMax)
				{
					distMax =curClosestDist ;
					iter_furthest = iter;
				}
				iter++;
			}
			//printf("distMax: %f\n",distMax);

			sampledIdx.push_back(*iter_furthest);
			allIdx.erase(iter_furthest);
		}
		printf("######################## Debug: check sampling result: %d (asked) vs %d (sampled)\n\ ########################\n", numCams,sampledIdx.size());
	}
}


//Set cameraLabel (predefined. only camNum matters)
//Load calibration data (m_memoryCaliDataDirPath should be set first)
void CDomeImageManager::InitDomeCamVgaHd(int askedVGACameraNum,bool bLoadHD,bool bLoadOnlySinglePanel)
{
	Clear();
	//int vgaCameraNum = askedVGACameraNum;			

	printf("## Init DomeImageManager :: cameraNum %d\n## Init DomeImageManager ::  calibrationPath: %s\n",askedVGACameraNum,m_memoryCaliDataDirPath);
	int orderHD[] = {0,1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
	vector<int> panelOrderIdx_HD(orderHD,orderHD+ sizeof(orderHD)/sizeof(int) );

	int order[] = {1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
	vector<int> panelOrderIdx(order,order + sizeof(order)/sizeof(int) );

	if(bLoadHD)
		panelOrderIdx = panelOrderIdx_HD;

	if(bLoadOnlySinglePanel)
	{
		panelOrderIdx.clear();
		panelOrderIdx.push_back(1);		//test only a single panel
	}

	if(bLoadHD ==false && askedVGACameraNum<480)
	{
		/*vector< pair<int,int> > allIdx;
		for(unsigned int t=0;t<panelOrderIdx.size();++t)
		{
			for(int camIdx=1;camIdx<=24 ;++camIdx)
			{
#ifdef IGNORE_BROKEN_CAM
				if(panelOrderIdx[t] ==14 && camIdx == 18)
					continue;
#endif
				allIdx.push_back( make_pair(panelOrderIdx[t],camIdx));
			}
		}

		vector<pair<int,int> > sampledIdx;  
		if(askedVGACameraNum>=allIdx.size())
			sampledIdx = allIdx;
		else
		{
			float ratio = float(allIdx.size())/float(askedVGACameraNum);
			for(int i=1;i<=askedVGACameraNum;++i)
			{
				int idx = ratio*i - 1 ;
				sampledIdx.push_back( allIdx[idx]);
			}
		}*/
		vector<pair<int,int> > sampledIdx;  
		VGACamSampling(askedVGACameraNum,sampledIdx);
		
		//Load
		for(int i=0;i<sampledIdx.size();++i)
		{
			bool bSuccess;
			//if(loadMode==LOAD_MODE_NO_IMAGE)
				bSuccess = AddCameraButDontLoadImage(m_memoryImageDirPath,m_currentFrame,sampledIdx[i].first,sampledIdx[i].second,true);
			/*else
				bSuccess = AddCameraNImageLoad(m_memoryImageDirPath,m_currentFrame+randomOff,sampledIdx[i].first,sampledIdx[i].second,true);*/
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",sampledIdx[i].first,sampledIdx[i].second);
				//m_domeViews.pop_back();
				continue;
			}
		}
		return;
	}

	int imageNum = -1;  //for debug
	for(unsigned int t=0;t<panelOrderIdx.size();++t)
	{
		int panelIdx  = panelOrderIdx[t];
		int camNum  = (panelIdx == 0) ? DOME_HD_CAMNUM:DOME_VGA_CAMNUM_EACHPANEL;
		for(int camIdx=1;camIdx<=camNum ;++camIdx)
		{
			int camIdxTemp = camIdx;
			if(panelIdx ==0)	//HD starts from 0
				camIdxTemp = camIdxTemp-1;
			bool bSuccess;
			//if(loadMode==LOAD_MODE_NO_IMAGE)
				bSuccess = AddCameraButDontLoadImage(m_memoryImageDirPath,m_currentFrame,panelIdx,camIdxTemp,true);
			/*else
				bSuccess = AddCameraNImageLoad(m_memoryImageDirPath,m_currentFrame+randomOff,panelIdx,camIdx,true);*/
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",panelIdx,camIdxTemp);
				//m_domeViews.pop_back();
				continue;
			}
			if(m_domeViews.size() == imageNum)
				break;
		}
		if(m_domeViews.size() == imageNum)
			break;
	}

	//printf("## Load Dome VGA data: finished\n");
}

//Set cameraLabel (predefined. only camNum matters)
void CDomeImageManager::InitDomeCamVgaHdLabelOnly(int askedVGACameraNum,bool bLoadHD)
{
	Clear();
	//int vgaCameraNum = askedVGACameraNum;			

	if(bLoadHD)
		printf("## Init DomeImageManager :: askedVGACameraNum %d with HD\n",askedVGACameraNum);	
	else
		printf("## Init DomeImageManager :: askedVGACameraNum %d WITHOUT HD\n",askedVGACameraNum);	

	int orderHD[] = {0,1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
	vector<int> panelOrderIdx_HD(orderHD,orderHD+ sizeof(orderHD)/sizeof(int) );

	int order[] = {1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
	vector<int> panelOrderIdx(order,order + sizeof(order)/sizeof(int) );

	if(bLoadHD)
		panelOrderIdx = panelOrderIdx_HD;

	if(bLoadHD ==false && askedVGACameraNum<480)
	{
		/*vector< pair<int,int> > allIdx;

		for(unsigned int t=0;t<panelOrderIdx.size();++t)
		{
			for(int camIdx=1;camIdx<=24 ;++camIdx)
			{
#ifdef IGNORE_BROKEN_CAM
				if(panelOrderIdx[t] ==14 && camIdx == 18)
					continue;
#endif
				allIdx.push_back( make_pair(panelOrderIdx[t],camIdx));
			}
		}

		vector<pair<int,int> > sampledIdx;  
		if(askedVGACameraNum>=allIdx.size())
			sampledIdx = allIdx;
		else
		{
			float ratio = float(allIdx.size())/float(askedVGACameraNum);
			for(int i=1;i<=askedVGACameraNum;++i)
			{
				int idx = ratio*i - 1 ;
				sampledIdx.push_back( allIdx[idx]);
			}
		}*/
		vector<pair<int,int> > sampledIdx;  
		VGACamSampling(askedVGACameraNum,sampledIdx);
		
		//Load
		for(int i=0;i<sampledIdx.size();++i)
		{
			bool bSuccess;
			bSuccess = AddCameraLabelOnly(m_currentFrame,sampledIdx[i].first,sampledIdx[i].second);
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",sampledIdx[i].first,sampledIdx[i].second);
				//m_domeViews.pop_back();
				continue;
			}
		}
		return;
	}

	int imageNum = -1;  //for debug
	for(unsigned int t=0;t<panelOrderIdx.size();++t)
	{
		int panelIdx  = panelOrderIdx[t];
		int camNum  = (panelIdx == 0) ? DOME_HD_CAMNUM:DOME_VGA_CAMNUM_EACHPANEL;
		for(int camIdx=1;camIdx<=camNum ;++camIdx)
		{
			int camIdxTemp = camIdx;
			if(panelIdx ==0)	//HD starts from 0
				camIdxTemp = camIdxTemp-1;

			bool bSuccess;
			bSuccess = AddCameraLabelOnly(m_currentFrame,panelIdx,camIdxTemp);
			if(bSuccess == false)
			{
				printf("Failed in image loading %d_%d\n",panelIdx,camIdxTemp);
				//m_domeViews.pop_back();
				continue;
			}
			if(m_domeViews.size() == imageNum)
				break;
		}
		if(m_domeViews.size() == imageNum)
			break;

		if(askedVGACameraNum==0)
			break;
	}

	//printf("## Load Dome VGA data: finished\n");
}

//Set cameraLabel (predefined. only camNum matters)
//camera order is same as "refVect"
void CDomeImageManager::InitDomeCamVgaHdLabelOnly(vector<CamViewDT*>& refVect)
{
	Clear();

	for(unsigned int t=0;t<refVect.size();++t)
	{
		int panelIdx  = refVect[t]->m_actualPanelIdx;
		int camIdx = refVect[t]->m_actualCamIdx;
		AddCameraLabelOnly(m_currentFrame,panelIdx,camIdx);
	}
	//printf("## Load Dome VGA data: finished\n");
}


/*void CDomeImageManager::SettingRMatrixGL(CamViewDT& element)
{
	Mat invR = element.m_R.inv();

	//colwise
	element.m_RMatrixGL[0] = invR.at<double>(0,0);
	element.m_RMatrixGL[1] = invR.at<double>(1,0);
	element.m_RMatrixGL[2] = invR.at<double>(2,0);
	element.m_RMatrixGL[3] = 0;
	element.m_RMatrixGL[4] = invR.at<double>(0,1);
	element.m_RMatrixGL[5] = invR.at<double>(1,1);
	element.m_RMatrixGL[6] = invR.at<double>(2,1);
	element.m_RMatrixGL[7] = 0;
	element.m_RMatrixGL[8] = invR.at<double>(0,2);
	element.m_RMatrixGL[9] = invR.at<double>(1,2);
	element.m_RMatrixGL[10] = invR.at<double>(2,2);
	element.m_RMatrixGL[11] = 0;
	element.m_RMatrixGL[12] = 0; //4th col
	element.m_RMatrixGL[13] = 0;
	element.m_RMatrixGL[14] = 0;
	element.m_RMatrixGL[15] = 1;
}*/

//K, quat, center are already saved
void CDomeImageManager::CamParamSettingByKQuatCenter(CamViewDT& element)
{
	//K,invK
	element.m_invK = element.m_K.inv();

	//R,invR
	element.m_R = Mat::zeros(3,3,CV_64F);
	//	Quaternion2Rotation(&(CvMat)element.m_R_Quat,&(CvMat)element.m_R);
	Quaternion2Rotation(element.m_R_Quat,element.m_R);

	element.m_invR = element.m_R.inv();

	//t
	element.m_t = - element.m_R*element.m_CamCenter;

	//P
	Mat M = Mat::zeros(3,4,CV_64F);
	element.m_R.copyTo(M.colRange(0,3));
	element.m_t.copyTo(M.colRange(3,4));
	element.m_P = element.m_K*M;

	//ETC
	//opticalAxis
	Mat imageCenter = Mat::zeros(3,1,CV_64F);
	imageCenter.at<double>(2,0) = 1;
	imageCenter = element.m_R*(imageCenter-element.m_t);
	element.m_opticalAxis = imageCenter - element.m_CamCenter;
	normalize(element.m_opticalAxis,element.m_opticalAxis);
	//element.m_CamCenterPt3d = MatToPoint3d(element.m_CamCenter);

	element.InitBasicInfo();
	//element.SettingRMatrixGL();
	//element.SettingModelViewMatrixGL();
	
	/*
	if(element.m_actualPanelIdx ==1 && element.m_actualCamIdx  ==1)
	{
		printf("%s\n",element.m_fullPath.c_str());
		printMatrix("K",element.m_K);
		printMatrix("P",element.m_P);
		for (int i=0;i<16;++i)
			printf("%f ",element.m_modelViewMatGL[i]);
		printf("\n");
		for (int i=0;i<16;++i)
			printf("%f ",element.m_mvpMatGL[i]);
		for (int i=0;i<16;++i)
			printf("%f ",element.m_projMatGL[i]);
		printf("\n");
	}*/
}
void CDomeImageManager::CamParamSettingByPKRt(CamViewDT& element,Mat& P,Mat& K,Mat& R,Mat& t)
{
	P.copyTo(element.m_P);
	R.copyTo(element.m_R);
	K.copyTo(element.m_K);
	t.copyTo(element.m_t);

	Mat invR = R.inv();
	invR.copyTo(element.m_invR);
	Mat camCenter = - invR*t;
	camCenter.copyTo(element.m_CamCenter);

	Mat invK = K.inv();
	invK.copyTo(element.m_invK);

	//quaternion
	Mat quat(4,1,CV_64F);
	//Rotation2Quaternion(&(CvMat)element.m_R,&(CvMat)quat);
	Rotation2Quaternion(element.m_R,quat);
	quat.copyTo(element.m_R_Quat);

	//opticalAxis
	Mat imageCenter = Mat::zeros(3,1,CV_64F);
	imageCenter.at<double>(2,0) = 1;
	imageCenter = invR*(imageCenter-t);
	element.m_opticalAxis = imageCenter - camCenter;
	normalize(element.m_opticalAxis,element.m_opticalAxis);
	//element.m_CamCenterPt3d = MatToPoint3d(element.m_CamCenter);

	element.InitBasicInfo();
	//element.SettingRMatrixGL();
	//element.SettingModelViewMatrixGL();
	/*
	if(element.m_actualPanelIdx ==1 && element.m_actualCamIdx  ==1)
	{
		printf("%s\n",element.m_fullPath.c_str());
		printMatrix("K",K);
		printMatrix("P",P);
		for (int i=0;i<16;++i)
			printf("%f ",element.m_modelViewMatGL[i]);
		printf("\n");
		for (int i=0;i<16;++i)
			printf("%f ",element.m_mvpMatGL[i]);
		for (int i=0;i<16;++i)
			printf("%f ",element.m_projMatGL[i]);
		printf("\n");
	}*/
}
//P,K,R,t are already saved in element
void CDomeImageManager::CamParamSettingByPKRt(CamViewDT& element)
{
	Mat invR = element.m_R.inv();
	invR.copyTo(element.m_invR);
	Mat camCenter = - invR*element.m_t;
	camCenter.copyTo(element.m_CamCenter);

	Mat invK = element.m_K.inv();
	invK.copyTo(element.m_invK);
	
	
	//quaternion
	Mat quat(4,1,CV_64F);
	//Rotation2Quaternion(&(CvMat)element.m_R,&(CvMat)quat);
	Rotation2Quaternion(element.m_R,quat);

	//printMatrix(quat,"Quat");
	quat.copyTo(element.m_R_Quat);
	
	//opticalAxis
	Mat imageCenter = Mat::zeros(3,1,CV_64F);
	imageCenter.at<double>(2,0) = 1;
	imageCenter = invR*(imageCenter-element.m_t);
	element.m_opticalAxis = imageCenter - camCenter;
	normalize(element.m_opticalAxis,element.m_opticalAxis);
	//element.m_CamCenterPt3d = MatToPoint3d(element.m_CamCenter);

	element.InitBasicInfo();
	//element.SettingRMatrixGL();
	//element.SettingModelViewMatrixGL();
}

//Used for the first pair of SfM
//RQ decomposition Method
//K can be changed
void CDomeImageManager::CamParamSettingByDecomposingP(CamViewDT& element,Mat& P)
{
	P.copyTo(element.m_P);
	Mat KR = element.m_P.colRange(0,3);
	Mat K(3,3,CV_64F);
	Mat R(3,3,CV_64F);
	RQDecomp3x3(KR,K,R);   //R is orthonormal, KR = K*R
	Mat invK = K.inv();
	Mat invR = R.inv();
	Mat Kt = element.m_P.colRange(3,4);
	Mat t = invK*Kt;
	Mat camCenter = - invR*t;

	K.copyTo(element.m_K);
	R.copyTo(element.m_R);
	camCenter.copyTo(element.m_CamCenter);

	//quaternion
	Mat quat(4,1,CV_64F);
	//Rotation2Quaternion(&(CvMat)R,&(CvMat)quat);
	Rotation2Quaternion(element.m_R,quat);
	//printMatrix(quat,"Quat");
	quat.copyTo(element.m_R_Quat);

	t.copyTo(element.m_t);
	invK.copyTo(element.m_invK);
	invR.copyTo(element.m_invR);

	//opticalAxis
	Mat imageCenter = Mat::zeros(3,1,CV_64F);
	imageCenter.at<double>(2,0) = 1;
	imageCenter = invR*(imageCenter-t);
	element.m_opticalAxis = imageCenter - camCenter;
	normalize(element.m_opticalAxis,element.m_opticalAxis);

	//element.m_CamCenterPt3d = MatToPoint3d(element.m_CamCenter);

	element.InitBasicInfo();
	//element.SettingRMatrixGL();
	//element.SettingModelViewMatrixGL();
}


bool CDomeImageManager::AddCameraButDontLoadImage(const char* inputDirPath,int frameIdx,int panelIdx,int camIdx,bool bLoadExtrinsic)
{
	char fullPath[256];
	char fileName[512];
	
	if(USE_PNG_IMAGE)
		sprintf(fileName,"%08d_%02d_%02d.png",frameIdx,panelIdx,camIdx);
	else
		sprintf(fileName,"%08d_%02d_%02d.bmp",frameIdx,panelIdx,camIdx);
	sprintf(fullPath,"%s/%08d/%s",inputDirPath,frameIdx,fileName);
	if(IsFileExist(fullPath) == false)
		sprintf(fullPath,"%s/%s",inputDirPath,fileName);
	if(IsFileExist(fullPath) == false)
		sprintf(fullPath,"%s/%03dXX/%08d/%08d_%02d_%02d.png",m_memoryImageDirPath,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
	
	//printfLog("Load image file from (%s) \n",fullPath);
	//Do not check file exist even.
	/*if(IsFileExist(fullPath) == false)
	{
		printfLog("Image Read Failed: %s\n",fullPath);
		return false;
	}*/
		
	m_domeViews.push_back(new CamViewDT);
	m_domeViews.back()->m_sequenceIdx = (int)m_domeViews.size()-1;
	m_domeViews.back()->m_fullPath = fullPath;
	m_domeViews.back()->m_fileName = fileName;
	m_domeViews.back()->m_actualImageFrameIdx = frameIdx;//ExtractFrameIdxFromPath(fullPath);	//will used to find already save feature file
	m_domeViews.back()->m_actualPanelIdx = panelIdx;//ExtractPanelIdxFromPath(fullPath);	//will used to find already save feature file
	m_domeViews.back()->m_actualCamIdx = camIdx;//ExtractCamIdxFromPath(fullPath);	//will used to find already save feature file

	/* Do not Load Images
	m_domeViews.back()->m_rgbInputImage = imread(fullPath);

	if(m_domeViews.back()->m_rgbInputImage.rows ==0)
	{
		printfLog("Image Read Failed: %s\n",fullPath);
		delete m_domeViews.back();
		m_domeViews.pop_back();
		return false;
	}
	//
	if(DO_HISTOGRAM_EQUALIZE)
	{
		Mat grayImage = imread(fullPath,CV_LOAD_IMAGE_GRAYSCALE);
		equalizeHist(grayImage,m_domeViews.back()->m_inputImage);
	}
	else
		m_domeViews.back()->m_inputImage = imread(fullPath,CV_LOAD_IMAGE_GRAYSCALE);
		
	printfLog("Success: Image Loading: %s\n",fullPath);
	*/

		//Disabled this, because usually I checked the size of m_inputImage to check validity
	if(panelIdx==0 || panelIdx==50)
		m_domeViews.back()->m_inputImage = Mat::zeros(1080,1920,CV_8UC1);
	else
		m_domeViews.back()->m_inputImage = Mat::zeros(480,640,CV_8UC1);
		
	//printfLog("Success: Fake Image Loading: %s\n",fullPath);

	bool bSuccess = m_domeViews.back()->LoadIntrisicParameter(m_memoryCaliDataDirPath,panelIdx,camIdx);
	if(bSuccess==false)	
	{
		delete m_domeViews.back();
		m_domeViews.pop_back();
		return false;
	}

	//Load Extrinsic Parameter
	if(bLoadExtrinsic)
	{
		char calibParamPath[256];
		sprintf(calibParamPath,"%s/%02d_%02d_ext.txt",m_memoryCaliDataDirPath,panelIdx,camIdx); //load same K for all cam

		ifstream fin(calibParamPath);
		if(fin)
		{
			m_domeViews.back()->m_R_Quat = Mat(4,1,CV_64F);
			m_domeViews.back()->m_CamCenter = Mat(3,1,CV_64F);

			double  quat[4];
			double  center[3];
			for(int i=0;i<4;++i)
			{
				fin >> quat[i];
			}
			for(int i=0;i<3;++i)
			{
				fin >> center[i];
			}
			memcpy(m_domeViews.back()->m_R_Quat.data,quat,4*sizeof(double));
			memcpy(m_domeViews.back()->m_CamCenter.data,center,3*sizeof(double));
			CamParamSettingByKQuatCenter(*m_domeViews.back());
			//printMatrix("R",m_domeViews.back()->m_R);
		}
		else
		{
			sprintf(calibParamPath,"%s/%02d_%02d_extRT.txt",m_memoryCaliDataDirPath,panelIdx,camIdx); //load same K for all cam
			ifstream fin(calibParamPath);
			if(fin.is_open())
			{
				m_domeViews.back()->m_R_Quat = Mat(4,1,CV_64F);
				m_domeViews.back()->m_CamCenter = Mat(3,1,CV_64F);

				double  r[9];
				double  t[3];
				for(int i=0;i<9;++i)
				{
					fin >> r[i];
				}
				for(int i=0;i<3;++i)
				{
					fin >> t[i];
				}
				Mat_<double> Rmat(3,3);
				Mat_<double> tmat(3,1);
				memcpy(Rmat.data,r,9*sizeof(double));
				memcpy(tmat.data,t,3*sizeof(double));

				//cout << Rmat <<endl;
				//cout <<tmat <<endl;
				//cout << m_domeViews.back()->m_K <<endl;
				Rotation2Quaternion(Rmat,m_domeViews.back()->m_R_Quat);
				m_domeViews.back()->m_CamCenter= -Rmat.inv()*tmat;
				//memcpy(m_domeViews.back()->m_R_Quat.data,quat,4*sizeof(double));
				//memcpy(m_domeViews.back()->m_CamCenter.data,center,3*sizeof(double));
				CamParamSettingByKQuatCenter(*m_domeViews.back());
				m_domeViews.back()->m_R = Rmat;		//sometimes quaternion conversion failed. 
				m_domeViews.back()->m_t = tmat;		//sometimes quaternion conversion failed. 
				Mat M = Mat::zeros(3,4,CV_64F);
				m_domeViews.back()->m_R.copyTo(M.colRange(0,3));
				m_domeViews.back()->m_t.copyTo(M.colRange(3,4));
				m_domeViews.back()->m_P = m_domeViews.back()->m_K*M;

				//printMatrix("R",m_domeViews.back()->m_R);
			}
		}

		fin.close();
	}
	return true;
}


//static function
bool CDomeImageManager::AddCameraButDontLoadImage(vector<CamViewDT*>& camVect,const char* calibDirPath,int frameIdx,int panelIdx,int camIdx,bool bLoadExtrinsic)
{
	char fullPath[256];
	char fileName[512];
	
	if(USE_PNG_IMAGE)
		sprintf(fileName,"%08d_%02d_%02d.png",frameIdx,panelIdx,camIdx);
	else
		sprintf(fileName,"%08d_%02d_%02d.bmp",frameIdx,panelIdx,camIdx);
	/*sprintf(fullPath,"%s/%08d/%s",calibDirPath,frameIdx,fileName);
	if(IsFileExist(fullPath) == false)
		sprintf(fullPath,"%s/%s",calibDirPath,fileName);
	if(IsFileExist(fullPath) == false)
		sprintf(fullPath,"%s/%03dXX/%08d/%08d_%02d_%02d.png",m_memoryImageDirPath,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);*/
	
	//printfLog("Load image file from (%s) \n",fullPath);
	//Do not check file exist even.
	/*if(IsFileExist(fullPath) == false)
	{
		printfLog("Image Read Failed: %s\n",fullPath);
		return false;
	}*/
		
	camVect.push_back(new CamViewDT);
	camVect.back()->m_sequenceIdx = (int)camVect.size()-1;
	camVect.back()->m_fullPath = fullPath;
	camVect.back()->m_fileName = fileName;
	camVect.back()->m_actualImageFrameIdx = frameIdx;//ExtractFrameIdxFromPath(fullPath);	//will used to find already save feature file
	camVect.back()->m_actualPanelIdx = panelIdx;//ExtractPanelIdxFromPath(fullPath);	//will used to find already save feature file
	camVect.back()->m_actualCamIdx = camIdx;//ExtractCamIdxFromPath(fullPath);	//will used to find already save feature file

	/* Do not Load Images
	camVect.back()->m_rgbInputImage = imread(fullPath);

	if(camVect.back()->m_rgbInputImage.rows ==0)
	{
		printfLog("Image Read Failed: %s\n",fullPath);
		delete camVect.back();
		camVect.pop_back();
		return false;
	}
	//
	if(DO_HISTOGRAM_EQUALIZE)
	{
		Mat grayImage = imread(fullPath,CV_LOAD_IMAGE_GRAYSCALE);
		equalizeHist(grayImage,camVect.back()->m_inputImage);
	}
	else
		camVect.back()->m_inputImage = imread(fullPath,CV_LOAD_IMAGE_GRAYSCALE);
		
	printfLog("Success: Image Loading: %s\n",fullPath);
	*/

		//Disabled this, because usually I checked the size of m_inputImage to check validity
	if(panelIdx==0 || panelIdx==50)
		camVect.back()->m_inputImage = Mat::zeros(1080,1920,CV_8UC1);
	else
		camVect.back()->m_inputImage = Mat::zeros(480,640,CV_8UC1);
		
	//printfLog("Success: Fake Image Loading: %s\n",fullPath);

	bool bSuccess = camVect.back()->LoadIntrisicParameter(calibDirPath,panelIdx,camIdx);
	if(bSuccess==false)	
	{
		delete camVect.back();
		camVect.pop_back();
		return false;
	}

	//Load Extrinsic Parameter
	if(bLoadExtrinsic)
	{
		char calibParamPath[256];
		sprintf(calibParamPath,"%s/%02d_%02d_ext.txt",calibDirPath,panelIdx,camIdx); //load same K for all cam

		ifstream fin(calibParamPath);
		camVect.back()->m_R_Quat = Mat(4,1,CV_64F);
		camVect.back()->m_CamCenter = Mat(3,1,CV_64F);

		double  quat[4];
		double  center[3];
		for(int i=0;i<4;++i)
		{
			fin >> quat[i];
		}
		for(int i=0;i<3;++i)
		{
			fin >> center[i];
		}
		memcpy(camVect.back()->m_R_Quat.data,quat,4*sizeof(double));
		memcpy(camVect.back()->m_CamCenter.data,center,3*sizeof(double));
		CamParamSettingByKQuatCenter(*camVect.back());
		//printMatrix("R",camVect.back()->m_R);

		fin.close();
	}
	return true;
}

bool CDomeImageManager::AddCameraLabelOnly(int frameIdx,int panelIdx,int camIdx)
{
	m_domeViews.push_back(new CamViewDT);
	m_domeViews.back()->m_sequenceIdx = (int)m_domeViews.size()-1;
	m_domeViews.back()->m_actualImageFrameIdx = frameIdx;//ExtractFrameIdxFromPath(fullPath);	//will used to find already save feature file
	m_domeViews.back()->m_actualPanelIdx = panelIdx;//ExtractPanelIdxFromPath(fullPath);	//will used to find already save feature file
	m_domeViews.back()->m_actualCamIdx = camIdx;//ExtractCamIdxFromPath(fullPath);	//will used to find already save feature file

	return true;
}

bool CDomeImageManager::AddCameraNImageLoad(const char* inputDirPath,int frameIdx,int panelIdx,int camIdx,bool bLoadExtrinsic)
{
	char fullPath[256];
	char fileName[512];
	
	if(USE_PNG_IMAGE)
		sprintf(fileName,"%08d_%02d_%02d.png",frameIdx,panelIdx,camIdx);
	else
		sprintf(fileName,"%08d_%02d_%02d.bmp",frameIdx,panelIdx,camIdx);
	sprintf(fullPath,"%s/%08d/%s",inputDirPath,frameIdx,fileName);
	if(IsFileExist(fullPath) == false)
		sprintf(fullPath,"%s/%s",inputDirPath,fileName);
	if(IsFileExist(fullPath) == false)
		sprintf(fullPath,"%s/%03dXX/%08d/%08d_%02d_%02d.png",m_memoryImageDirPath,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
	
	//printfLog("Load image file from (%s) \n",fullPath);

	if(IsFileExist(fullPath) == false)
	{
		printfLog("Image Read Failed: %s\n",fullPath);
		return false;
	}
		
	m_domeViews.push_back(new CamViewDT);
	m_domeViews.back()->m_sequenceIdx = (int)m_domeViews.size()-1;
	m_domeViews.back()->m_fullPath = fullPath;
	m_domeViews.back()->m_fileName = fileName;
	m_domeViews.back()->m_actualImageFrameIdx = frameIdx;//ExtractFrameIdxFromPath(fullPath);	//will used to find already save feature file
	m_domeViews.back()->m_actualPanelIdx = panelIdx;//ExtractPanelIdxFromPath(fullPath);	//will used to find already save feature file
	m_domeViews.back()->m_actualCamIdx = camIdx;//ExtractCamIdxFromPath(fullPath);	//will used to find already save feature file


	m_domeViews.back()->m_rgbInputImage = imread(fullPath);

	if(m_domeViews.back()->m_rgbInputImage.rows ==0)
	{
		printfLog("Image Read Failed: %s\n",fullPath);
		delete m_domeViews.back();
		m_domeViews.pop_back();
		return false;
	}
	//
	/*if(m_bDoHistEqualize) 
	{
		Mat grayImage = imread(fullPath,CV_LOAD_IMAGE_GRAYSCALE);
		equalizeHist(grayImage,m_domeViews.back()->m_inputImage);
	}
	else*/
		m_domeViews.back()->m_inputImage = imread(fullPath,CV_LOAD_IMAGE_GRAYSCALE);

	//printfLog("Succeess: Image Loading: %s\n",fullPath);
	bool bSuccess = m_domeViews.back()->LoadIntrisicParameter(m_memoryCaliDataDirPath,panelIdx,camIdx);
	if(bSuccess==false)	
	{
		delete m_domeViews.back();
		m_domeViews.pop_back();
		return false;
	}
	
	//Load Extrinsic Parameter
	if(bLoadExtrinsic)
	{
		char calibParamPath[256];
		//sprintf(calibParamPath,"%s/%s/%02d_%02d_ext.txt",inputDirPath,CALIB_DATA_FOLDER,panelIdx,camIdx); //load same K for all cam
		sprintf(calibParamPath,"%s/%02d_%02d_ext.txt",m_memoryCaliDataDirPath,panelIdx,camIdx); //load same K for all cam

		ifstream fin(calibParamPath);
		m_domeViews.back()->m_R_Quat = Mat(4,1,CV_64F);
		m_domeViews.back()->m_CamCenter = Mat(3,1,CV_64F);

		double  quat[4];
		double  center[3];
		for(int i=0;i<4;++i)
		{
			fin >> quat[i];
		}
		for(int i=0;i<3;++i)
		{
			fin >> center[i];
			//center[i] *=100;
		}
		memcpy(m_domeViews.back()->m_R_Quat.data,quat,4*sizeof(double));
		memcpy(m_domeViews.back()->m_CamCenter.data,center,3*sizeof(double));
		CamParamSettingByKQuatCenter(*m_domeViews.back());
		//printMatrix("R",m_domeViews.back()->m_R);

		fin.close();
	}

	return true;
}

//loadingMode ==0: load both rgb and gray
//loadingMode ==1: load only gray
//loadingMode ==2: load only rgb
bool CDomeImageManager::LoadPhysicalImagesByFrameIdx(int frameIdx,EnumLoadMode  loadingMode,bool bVerbose)
{
	return LoadDomeImages(m_domeViews,m_memoryImageDirPath,frameIdx,loadingMode,bVerbose);
#if 0
	//if(bVerbose)
		printf("## Load Dome VGA Images: path:%s, frame %d\n",m_memoryImageDirPath,frameIdx);

	//char dirPath[512];
	char targetImagePath[512];
	//GetFolderPathFromFullPath(m_memoryVector[0]->m_camViewVect[0]->m_fullPath.c_str(),dirPath);
	bool completelyFailed =true;
	for(unsigned int i=0;i<m_domeViews.size();++i)
	{
		//int targetFrameIdx = frameIdx;
		int targetFrameIdx = frameIdx;		
		int panelIdx = m_domeViews[i]->m_actualPanelIdx;
		int camIdx = m_domeViews[i]->m_actualCamIdx;
		if(USE_PNG_IMAGE)
		{
			if(panelIdx==0)
			{
				sprintf(targetImagePath,"%s/hd_30/%08d/hd%08d_%02d_%02d.png",m_memoryImageDirPath,targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
					sprintf(targetImagePath,"%s/hd_30/hd%08d_%02d_%02d.png",m_memoryImageDirPath,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
					sprintf(targetImagePath,"%s/hd_30/%03dXX/%08d/hd%08d_%02d_%02d.png",m_memoryImageDirPath,int(targetFrameIdx/100),targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
				{
					//return false;
					printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
					continue;
				}
			}
			else
			{
				sprintf(targetImagePath,"%s/vga_25/%08d/%08d_%02d_%02d.png",m_memoryImageDirPath,targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
					sprintf(targetImagePath,"%s/vga_25/%08d_%02d_%02d.png",m_memoryImageDirPath,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
					sprintf(targetImagePath,"%s/vga_25/%03dXX/%08d/%08d_%02d_%02d.png",m_memoryImageDirPath,int(targetFrameIdx/100),targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
				{
					printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
					//return false;
						continue;
				}
			}
			
		}
		else
		{
			sprintf(targetImagePath,"%s/%08d/%08d_%02d_%02d.bmp",m_memoryImageDirPath,targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
			if(IsFileExist(targetImagePath) == false)
				sprintf(targetImagePath,"%s/%08d_%02d_%02d.bmp",m_memoryImageDirPath,targetFrameIdx,panelIdx,camIdx);
			if(IsFileExist(targetImagePath) == false)
				sprintf(targetImagePath,"%s/%03dXX/%08d/%08d_%02d_%02d.bmp",m_memoryImageDirPath,int(targetFrameIdx/100),targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
			if(IsFileExist(targetImagePath) == false)
			{
				printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
				//return false;
				continue;
			}
		}

		m_domeViews[i]->m_fullPath = string(targetImagePath);
		char fileName[128];
		GetFileNameFromFullPath(targetImagePath,fileName);
		m_domeViews[i]->m_fileName = string(fileName);

		if(loadingMode==LOAD_MODE_NO_IMAGE)
			return true;

		else if(loadingMode==LOAD_MODE_GRAY_RGB )
		{
			m_domeViews[i]->m_rgbInputImage = imread(targetImagePath);
			if(m_domeViews[i]->m_rgbInputImage.rows==0)
			{
				printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
				//return false;
				continue;
			}

			/*if(DO_HISTOGRAM_EQUALIZE)
			{
				Mat grayImage;
				cvtColor(m_domeViews[i]->m_rgbInputImage,grayImage,CV_RGB2GRAY);
				equalizeHist(grayImage,m_domeViews[i]->m_inputImage);
			}
			else*/
				cvtColor(m_domeViews[i]->m_rgbInputImage,m_domeViews[i]->m_inputImage,CV_RGB2GRAY);
		}
		else if(loadingMode==LOAD_MODE_GRAY)
		{
			/*if(DO_HISTOGRAM_EQUALIZE)
			{
				Mat grayImage= imread(targetImagePath,CV_LOAD_IMAGE_GRAYSCALE);
				equalizeHist(grayImage,m_domeViews[i]->m_inputImage);
			}
			else*/
				m_domeViews[i]->m_inputImage = imread(targetImagePath,CV_LOAD_IMAGE_GRAYSCALE);
			
			if(m_domeViews[i]->m_inputImage.rows==0)
				//return false;
				continue;
		}
		else if(loadingMode==LOAD_MODE_RGB)
		{
			m_domeViews[i]->m_rgbInputImage = imread(targetImagePath);
			if(m_domeViews[i]->m_rgbInputImage.rows==0)
			{
				printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
				//return false;
				continue;
			}
		}
		completelyFailed = false;
	}
	if(bVerbose)
		printf("## Load Dome VGA Images: finished\n");
	
	if(completelyFailed)
	{
		printf("## Failed in loading images: final path %s\n",targetImagePath);
		return false;
	}
	else
		return true;
#endif
}

//Calculate 1cm in world scale using the distance between cam_1_1 to cam_1_2
void CDomeImageManager::CalculatePhysicalScale()
{
	//serarch same panel cam1- cam2

	vector<double> distVector;
	int idx1=-1;
	int idx2=-1;
	int targetPanel=-1;
	for(int i=0;i<m_domeViews.size();++i)
	{
		if( m_domeViews[i]->m_actualPanelIdx == PANEL_HD || m_domeViews[i]->m_actualPanelIdx == PANEL_KINECT)
			continue;
		if(idx1==-1 && m_domeViews[i]->m_actualCamIdx==1)
		{
			idx1 = i;
			targetPanel = m_domeViews[i]->m_actualPanelIdx;
		}	

		if( idx2 == -1 && m_domeViews[i]->m_actualPanelIdx == targetPanel && m_domeViews[i]->m_actualCamIdx ==2)
		{
			idx2 = i;
			break;
		}
	}

	if(idx1<0 || idx2<0)
	{
		printfLog("FAILURE: WORLD_TO_CM_RATIO Initialization. You Must load cam_1_1 and cam_1_2 \n");
		return;
	}

	double dist = Distance ( m_domeViews[idx1]->m_CamCenter, m_domeViews[idx2]->m_CamCenter);
	distVector.push_back(dist);


	sort(distVector.begin(),distVector.end());

	double cameraDist = distVector[distVector.size()/2];  //median
	//printfLog("WorldParameters :: Distance %f\n",cameraDist);

	WORLD_TO_CM_RATIO = DOME_CAMERA_DIST_CM_UNIT/cameraDist;	//cm/WORLD
	printfLog("WorldParameters :: %f -> 22 cm : WORLD to cm Ratio: %f \n",cameraDist,WORLD_TO_CM_RATIO);
	printfLog("DOME_CAMERA_DIST_CM_UNIT: %f \n", DOME_CAMERA_DIST_CM_UNIT);

	//Initialize other parameter
	PATCH_3D_ARROW_SIZE_WORLD_UNIT = PATCH_3D_ARROW_SIZE_CM/WORLD_TO_CM_RATIO;
	//printfLog("WorldParameters :: PATCH_3D_ARROW_SIZE_WORLD_UNIT : %f\n",PATCH_3D_ARROW_SIZE_WORLD_UNIT);
	//printf("WorldParameters :: 1Cm in World Scale: %f \n",cm2world(1));
	
	/*// Accurate World Scale calculation
	double prev2CmInWorld= 2/ WORLD_TO_CM_RATIO;

	WORLD_TO_CM_RATIO = 120.0 / 127.062749 ;
	printf("WorldParameters :: More Accurate Measurment : WORLD to cm Ratio: %f \n",WORLD_TO_CM_RATIO);
	printf("WorldParameters :: 2Cm in World Scale(corrected): %f \n\n",prev2CmInWorld / WORLD_TO_CM_RATIO);	*/
}

#include <limits>
void CDomeImageManager::CalcWorldScaleParameters()
{
	//printf("## CalcWorldScaleParam: Camera Num %d\n",m_domeViews.size());
	CalculatePhysicalScale();

	Point3f minCorner(1e5,1e5,1e5);
	Point3f maxCorner(-1e5,-1e5,-1e5);
	for(unsigned int i=0;i<m_domeViews.size();++i)
	{
		if(m_domeViews[i]->m_actualPanelIdx==0)
			continue;
		minCorner.x = min((float)m_domeViews[i]->m_CamCenter.at<double>(0,0),minCorner.x);
		minCorner.y = min((float)m_domeViews[i]->m_CamCenter.at<double>(1,0),minCorner.y);
		minCorner.z = min((float)m_domeViews[i]->m_CamCenter.at<double>(2,0),minCorner.z);

		maxCorner.x = max((float)m_domeViews[i]->m_CamCenter.at<double>(0,0),maxCorner.x);
		maxCorner.y = max((float)m_domeViews[i]->m_CamCenter.at<double>(1,0),maxCorner.y);
		maxCorner.z = max((float)m_domeViews[i]->m_CamCenter.at<double>(2,0),maxCorner.z);
	}

	double twoCamDist = 10;
	if(m_domeViews.size()>2)
	{
		//select two vga camera
		int firstCamIdx=-1;
		int secondCamIdx=-1;
		for(unsigned int i=0;i<m_domeViews.size();++i)
		{
			if(m_domeViews[i]->m_actualPanelIdx!=0)
			{
				if(firstCamIdx<0)
					firstCamIdx = i;
				else
					secondCamIdx = i;
			}

			if(secondCamIdx>0)
				break;
		}
		if(secondCamIdx<0)
		{
			twoCamDist = Distance(m_domeViews[0]->m_CamCenter,m_domeViews[1]->m_CamCenter);
		}
		else
			twoCamDist = Distance(m_domeViews[firstCamIdx]->m_CamCenter,m_domeViews[secondCamIdx]->m_CamCenter);

		//printfLog("WorldParameters :: twoCamDist %f\n",twoCamDist );
	}
	Point3f center = minCorner+ maxCorner;
	center.x /=2;
	center.y /=2;
	center.z /=2;

	//set up working volume parameters //used also in visualhull recon 
	DOME_VOLUMECUT_CENTER_PT = center;
	DOME_VOLUMECUT_CENTER_PT.y += cm2world(50);
	//DOME_VOLUMECUT_CENTER_PT.y =+ 50.0/WORLD_TO_CM_RATIO;		//50 cm 

	//printf("DOME_VOLUMECUT_CENTER_PT:: %f, %f, %f",DOME_VOLUMECUT_CENTER_PT.x,DOME_VOLUMECUT_CENTER_PT.y,DOME_VOLUMECUT_CENTER_PT.z);
	float length = Distance(minCorner,maxCorner);
	DOME_VOLUMECUT_RADIOUS = length*0.28;

	DOME_VOLUMECUT_X_MIN = center.x - DOME_VOLUMECUT_RADIOUS;
	DOME_VOLUMECUT_X_MAX = center.x + DOME_VOLUMECUT_RADIOUS;
	DOME_VOLUMECUT_Y_MAX = center.y + DOME_VOLUMECUT_RADIOUS;
	DOME_VOLUMECUT_Y_MIN = center.y - DOME_VOLUMECUT_RADIOUS;
	DOME_VOLUMECUT_Z_MAX = center.z + DOME_VOLUMECUT_RADIOUS;
	DOME_VOLUMECUT_Z_MIN = center.z - DOME_VOLUMECUT_RADIOUS;

	CalculateFloorPlane();

	printf("WORLD_TO_CM_RATIO: %f\n", WORLD_TO_CM_RATIO);
	//The following is used for depthMap generation
	g_nearPlaneForDepthRender = cm2world(0.1);		//30cm
	g_farPlaneForDepthRender = cm2world(650);		//650 cm
	printf("SetWorldScaleForVisualize: g_near %f, g_far : %f\n",g_nearPlaneForDepthRender,g_farPlaneForDepthRender);
	for(int c=0;c<m_domeViews.size();++c)
		m_domeViews[c]->SettingModelViewMatrixGL();		//since g_near and g_far is recomputed
}

void CDomeImageManager::CalculateFloorPlane()
{
	vector<Point3f> vect;
	Point3f centerOfMass(0,0,0);
	for(unsigned int i=0;i<m_domeViews.size();++i)
	{
		if(m_domeViews[i]->m_actualPanelIdx<16)
			continue;

		
		//if(m_domeViews[i]->m_actualCamIdx==1)
		if(m_domeViews[i]->m_actualCamIdx==15)
		{
			Point3f center = MatToPoint3d(m_domeViews[i]->m_CamCenter);
			vect.push_back(center);
			centerOfMass= centerOfMass+center;
		}
		//PlaneFitting()
	}
	if(vect.size()<2)
		return;

	centerOfMass.x = centerOfMass.x/vect.size();
	centerOfMass.y = centerOfMass.y/vect.size();
	centerOfMass.z = centerOfMass.z/vect.size();

	Point3f direct_1 =  vect[1] - centerOfMass;
	Point3f direct_2 =  vect[0] - centerOfMass;
	Point3f normal = direct_1.cross(direct_2);
	Normalize(normal);  //- y direct
	Normalize(direct_1); //x direct
	Point3f zDirect = direct_1.cross(normal); //z direct
	double radius = Distance(direct_1,centerOfMass) *0.7; 
	centerOfMass = centerOfMass - (120.0/WORLD_TO_CM_RATIO)*normal ;

	Mat_<double> FrameRot  = Mat_<double>::zeros(3,3);
	Mat tempMat = FrameRot.colRange(0,1);
	Point3ToMatDouble(direct_1,tempMat); //x
	tempMat = FrameRot.colRange(1,2);
	Point3ToMatDouble(normal,tempMat); //y
	tempMat = FrameRot.colRange(2,3);
	Point3ToMatDouble(zDirect,tempMat); //z

	//cout << FrameRot <<"\n";
	vect.clear();
	
	Mat_<double> radusPt = (Mat_<double>(3,1) << radius,0,0);
	double twoPI = 2*PI;
	double interval = twoPI/360.0;
	for(double theta=0;theta<twoPI;theta += interval)
	{
		Mat_<double> rot = (Mat_<double>(3,3) << cos(theta),0,-sin(theta),0,1,0,sin(theta),0,cos(theta));  //row-wise
		Mat tempPt =  rot*radusPt;
		tempPt = FrameRot *tempPt;
		Point3f ptPos = MatToPoint3d(tempPt);
		ptPos = ptPos+centerOfMass;
		vect.push_back(ptPos);
 	}
	vect.push_back(vect.back());
	
	g_floorCenter = centerOfMass;
	g_floorNormal = normal;
	g_floorPts = vect;

	//floor height adjust
	g_floorCenter.y -= cm2world(50);
	for(int t=0;t<g_floorPts.size();++t)
	{
		g_floorPts[t].y -= cm2world(50);
	}
	printf("g_cloorCenter: %f %f %f\n",g_floorCenter.x,g_floorCenter.y,g_floorCenter.z);
}

//Find all the extrinsic parameters (ending with ext.txt) in the folder of "calibFolderPath"
//Add them as a new element. 
//Each element only has extrinsic parameters (used only for camera visualization or working volume computation)
bool CDomeImageManager::InitDomeCamOnlyByExtrinsic(const char* calibFolderPath,bool bLoadHD)
{
	Clear();

	printfLog("## InitDomeCamOnlyByExtrinsic: from %s\n",calibFolderPath);

	//GetFilesInDirectory(calibFolderPath,out);
	int orderHD[] = {0,1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
	vector<int> panelOrderIdx_HD(orderHD,orderHD+ sizeof(orderHD)/sizeof(int) );
	int order[] = {1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
	vector<int> panelOrderIdx(order,order + sizeof(order)/sizeof(int) );
	if(bLoadHD)
		panelOrderIdx = panelOrderIdx_HD;

	for(unsigned int t=0;t<panelOrderIdx.size();++t)
	{
		int panelIdx  = panelOrderIdx[t];
		int camNum  = (panelIdx == 0) ? DOME_HD_CAMNUM:DOME_VGA_CAMNUM_EACHPANEL;
		for(int camIdx=1;camIdx<=camNum ;++camIdx)
		{
#ifdef IGNORE_BROKEN_CAM
			if(panelIdx ==14 && camIdx == 18)
				continue;
#endif
			int camIdxTemp = camIdx;
			if(panelIdx ==0)	//HD starts from 0
				camIdxTemp = camIdxTemp-1;
			char path[512];
			sprintf(path,"%s/%02d_%02d_ext.txt",calibFolderPath,panelIdx,camIdxTemp);

			ifstream fin(path);
			if(fin.is_open()==false)
				continue;
		
			m_domeViews.push_back(new CamViewDT);
			m_domeViews.back()->m_actualPanelIdx = panelIdx;
			m_domeViews.back()->m_actualCamIdx= camIdx;

			m_domeViews.back()->m_K = Mat(3,3,CV_64F);
			m_domeViews.back()->m_R_Quat = Mat(4,1,CV_64F);
			m_domeViews.back()->m_CamCenter = Mat(3,1,CV_64F);

			double  quat[4];
			double  center[3];
			for(int i=0;i<4;++i)
			{
				fin >> quat[i];
			}
			for(int i=0;i<3;++i)
			{
				fin >> center[i];
				//center[i] *=100;
			}
			memcpy(m_domeViews.back()->m_R_Quat.data,quat,4*sizeof(double));
			memcpy(m_domeViews.back()->m_CamCenter.data,center,3*sizeof(double));
			CamParamSettingByKQuatCenter(*m_domeViews.back());
		
			fin.close();	
		}
	}
	
	return true;
}


int CDomeImageManager::GetViewIdxFromPanelCamIdx(int panelIdx,int camIdx)
{
	if(m_camNameToIdxTable.size()==0)
		GenerateCamNameToIdxTable();

	map<  pair<int,int>, int >::iterator mi = m_camNameToIdxTable.find(make_pair(panelIdx,camIdx));  

	if(mi!=m_camNameToIdxTable.end())
		return mi->second;
	else 
		return -1;
}


CamViewDT* CDomeImageManager::GetViewDTFromPanelCamIdx(int panelIdx,int camIdx)
{
	if(m_camNameToIdxTable.size()==0)
		GenerateCamNameToIdxTable();

	map<  pair<int,int>, int >::iterator mi = m_camNameToIdxTable.find(make_pair(panelIdx,camIdx));  

	if(mi!=m_camNameToIdxTable.end())
		return m_domeViews[mi->second];
	else 
		return NULL;
}

void CDomeImageManager::GenerateCamNameToIdxTable()
{
	m_camNameToIdxTable.clear();

	for(int i=0;i<m_domeViews.size();++i)
	{
		m_camNameToIdxTable.insert( make_pair( make_pair(m_domeViews[i]->m_actualPanelIdx,m_domeViews[i]->m_actualCamIdx) , i));
	}
}


void CDomeImageManager::DeleteSelectedCamerasByNonValidVect(vector<int>& camerasTobeDeleted,vector<int>& newOrders)
{
	m_camNameToIdxTable.clear();


	vector<bool> bDeleteChecker(m_domeViews.size(),false);
	for(int i=0;i<camerasTobeDeleted.size();++i)
	{
		bDeleteChecker[camerasTobeDeleted[i]] = true;		//There should be deleted
	}

	vector<CamViewDT*> updated_domeViews;
	newOrders.clear();
	updated_domeViews.reserve(bDeleteChecker.size());
	for(int i=0;i<bDeleteChecker.size();++i)
	{
		if(bDeleteChecker[i])
			delete m_domeViews[i];
		else
		{
			updated_domeViews.push_back(m_domeViews[i]);
			newOrders.push_back(i);
		}
	}
	m_domeViews = updated_domeViews;
}


//newOrders has ramining old indices in new vector
void CDomeImageManager::DeleteSelectedCamerasByValidVect(vector<int>& validCameras,vector<int>& newOrders)
{
	m_camNameToIdxTable.clear();


	vector<bool> bDeleteChecker(m_domeViews.size(),true);
	for(int i=0;i<validCameras.size();++i)
	{
		bDeleteChecker[validCameras[i]] = false;		//There should be NOT deleted
	}

	vector<CamViewDT*> updated_domeViews;
	updated_domeViews.reserve(bDeleteChecker.size());
	newOrders.clear();
	for(int i=0;i<bDeleteChecker.size();++i)
	{
		if(bDeleteChecker[i])
			delete m_domeViews[i];
		else
		{
			updated_domeViews.push_back(m_domeViews[i]);
			newOrders.push_back(i);
		}
	}
	m_domeViews = updated_domeViews;
}

//Return Image size
//If images are loaded, return the value
//If images are not loaded (LOAD_MODE_NO_IMAGE), return the value based on panelIdx (e.g., 2D cost map generation from pose detector
//return pair<xWidth,yWidth>
pair<int,int> CDomeImageManager::GetImageSize(int viewIdx)
{
	if(m_domeViews.size()<=viewIdx)
	{
		printf("## WARNING: viewIdx %d is bigger than viewNum %d\n",viewIdx,m_domeViews.size());
		return make_pair(-1,-1);
	}
	else if(m_domeViews[viewIdx]->m_inputImage.rows>0)
		return make_pair(m_domeViews[viewIdx]->m_inputImage.cols,m_domeViews[viewIdx]->m_inputImage.rows);
	else if(m_domeViews[viewIdx]->m_rgbInputImage.rows>0)
		return make_pair(m_domeViews[viewIdx]->m_rgbInputImage.cols,m_domeViews[viewIdx]->m_rgbInputImage.rows);
	else 
	{
		//Return values based on prior knowledge about our system
		//Panel 0: HD Cam: 1920 x 1080
		//Panel 1~20: VGA Cam: 640x480
		if(m_domeViews[viewIdx]->m_sensorType==SENSOR_TYPE_VGA)
			return make_pair(640,480);
		else if(m_domeViews[viewIdx]->m_sensorType==SENSOR_TYPE_HD)
			return make_pair(1920,1080);
		else if(m_domeViews[viewIdx]->m_sensorType==SENSOR_TYPE_KINECT)
			return make_pair(1920,1080);
		else 
			return make_pair(-1,-1);		
	}
	return make_pair(-1,-1);		//not reacheable

}


int CDomeImageManager::GetPanelOrderDistance(int idx1,int idx2)
{
	ComputePanelOrders();

	int panelDiff;
	if(m_domeViews[idx1]->m_actualPanelIdx>0 && m_domeViews[idx2]->m_actualPanelIdx>0 )		//Both of them are VGA
	{
		int assoPanel_1 = m_domeViews[idx1]->m_actualPanelIdx;		
		int assoPanel_2 = m_domeViews[idx2]->m_actualPanelIdx;

		int panelOrder_1 = m_panelIdxToOrder[assoPanel_1 - 1];
		int panelOrder_2 = m_panelIdxToOrder[assoPanel_2 - 1];
		panelDiff =  std::abs(panelOrder_1 - panelOrder_2);
		panelDiff = min(panelDiff,20-panelDiff);

		double camDist = Distance( m_domeViews[idx1]->m_CamCenter,m_domeViews[idx2]->m_CamCenter);
		//if(panelDiff<5)
//			printfLog("PanelDiff %d, camCenterDist %f\n",panelDiff,camDist*WORLD_TO_CM_RATIO);
	}
	else if(m_domeViews[idx1]->m_actualPanelIdx>0 || m_domeViews[idx2]->m_actualPanelIdx>0 )		//Either one is VGA
	{
		double camDist = Distance( m_domeViews[idx1]->m_CamCenter,m_domeViews[idx2]->m_CamCenter);
		camDist*=WORLD_TO_CM_RATIO;
		//printfLog("HD: %f\n",camDist);
		if(camDist > CAMDISTANCE_THRSH)		
			panelDiff = PANELDIFF_THRSH+1;
		else
			panelDiff =0;
	}
	else	//Both of them are HD
	{
		double camDist = Distance( m_domeViews[idx1]->m_CamCenter,m_domeViews[idx2]->m_CamCenter);
		camDist*=WORLD_TO_CM_RATIO;
		printfLog("HD: %f\n",camDist);
		if(camDist > 1000)		//10M
			panelDiff = PANELDIFF_THRSH+1;
		else
			panelDiff =0;
	}

	return panelDiff;
}

//Static function (only need to be computed once)
//Setup static variables: m_panelOrderIdx, m_nearPanelOfHD, m_panelIdxToOrder
vector<int> CDomeImageManager::m_panelOrderIdx;
vector<int> CDomeImageManager::m_nearPanelOfHD;
vector<int> CDomeImageManager::m_panelIdxToOrder;
void CDomeImageManager::ComputePanelOrders()
{
	if(m_panelOrderIdx.size()>0)
		return;

	int panelOrder[] = {1,3,17,5,4,6,18,8,7,9,19,11,10,12,20,14,13,15,16,2};
	//m_panelOrderIdx = vector<int>((panelOrder,panelOrder+ sizeof(panelOrder)/sizeof(int) ));
	m_panelOrderIdx.insert(m_panelOrderIdx.end(),panelOrder,panelOrder+ sizeof(panelOrder)/sizeof(int) );
	m_panelIdxToOrder.resize(m_panelOrderIdx.size());
	for(unsigned int k=0;k<m_panelOrderIdx.size();++k)		//to avoid matching between panels which have too long baseline
		m_panelIdxToOrder[ m_panelOrderIdx[k]-1]  = k; 

	//For Hd
	int nearPanleOfHD[] = {1,2,4,5,7,8,10,11,13,14,3,6,9,12,15,5,8,11,14};  // from P0 ~ P 18 
	m_nearPanelOfHD.insert(m_nearPanelOfHD.end(),nearPanleOfHD,nearPanleOfHD+ sizeof(nearPanleOfHD)/sizeof(int) );
}

//input:m_domeViews
//output:gridImagesByPanel
void MergeToGridImageByPanel(const vector<CamViewDT*>& m_domeViews,vector<Mat> &gridImagesByPanel, bool bUseRGBImage)
{
	if(m_domeViews.size()==0)
		return;

	int panelWidth = m_domeViews.front()->m_inputImage.cols*6;
	int panelHeight = m_domeViews.front()->m_inputImage.rows*4;
	int imageWidth = m_domeViews.front()->m_inputImage.cols;
	int imageHeight = m_domeViews.front()->m_inputImage.rows;
	
	//Mat panelImage[20];
	gridImagesByPanel.resize(20);
	bool panelIsValid[20];
	for(int i=0;i<20;++i)
	{
		gridImagesByPanel[i] = Mat::zeros(panelHeight,panelWidth,CV_8UC3);			//rgb channel
		panelIsValid[i] = false;
	}

	for(unsigned int i=0;i<m_domeViews.size();++i)
	{
		int panelIdx = m_domeViews[i]->m_actualPanelIdx-1;
		if(panelIdx<0)
			continue;
		int camIdx = m_domeViews[i]->m_actualCamIdx-1;
		int leftTopX  = camIdx%6 * imageWidth;
		int leftTopY  = int(camIdx/6) * imageHeight;
		Mat tmpRegion = gridImagesByPanel[panelIdx](Rect(leftTopX,leftTopY,imageWidth,imageHeight));

		if(bUseRGBImage)
			m_domeViews[i]->m_rgbInputImage.copyTo(tmpRegion);
		else
		{
			if(m_domeViews[i]->m_debugImage.type() != tmpRegion.type())
				cvtColor(m_domeViews[i]->m_inputImage,tmpRegion,CV_GRAY2BGR);
			else
				m_domeViews[i]->m_inputImage.copyTo(tmpRegion);
		}
		panelIsValid[panelIdx] = true;
	}

	//Eliminate meaningless panels
	for(int i=0;i<20;++i)
	{
		if(panelIsValid[i]==false)
		{
			Mat dummy;
			gridImagesByPanel[i] = dummy;	
		}
	}
}

void MergeToGridImage(vector<Mat>& imgVect,int maxWidth,vector< pair<int,int>>& camNameVect,Mat& outputImg)
{
	int imgNum = imgVect.size();
	int gridRow = sqrt(imgNum);
	float gridColf = float(imgNum)/gridRow;
	int gridCol;
	if(gridColf>int(gridColf))
		gridCol = int(gridColf)+1; 
	else
		gridCol = int(gridColf);
	
	int imageWidth_org = imgVect.front().cols;
	int imageHeight_org = imgVect.front().rows;

	int imageWidth = int(maxWidth/gridCol);
	int imageHeight = int( float(imageWidth)/imageWidth_org * imageHeight_org ) ;

	int maxHeight = imageHeight * gridRow;
	outputImg = Mat::zeros(maxHeight,maxWidth,CV_8UC3);			//rgb channel

	for(unsigned int i=0;i<imgVect.size();++i)
	{
		int leftTopX  = i%gridCol * imageWidth;
		int leftTopY  = int(i/gridCol) * imageHeight;
		Mat tmpRegion = outputImg(Rect(leftTopX,leftTopY,imageWidth,imageHeight));

		if(imgVect[i].rows==0)
			continue;
		Mat dst;
		resize(imgVect[i],dst,Size(imageWidth,imageHeight));

		//Draw Skeleton Names
		if(camNameVect.size()>i)
		{
			char text[512];
			sprintf(text,"%02d_%02d",camNameVect[i].first,camNameVect[i].second);
			putText(dst,text,cvPoint(20,20),FONT_HERSHEY_COMPLEX_SMALL,1,cvScalar(255,255,0),2) ;
		}


		dst.copyTo(tmpRegion);
	}
	//imshow("test",outputImg);
}

bool CDomeImageManager::LoadDomeImages(vector<CamViewDT*>& camVect, const char* imgDirPath, int frameIdx, EnumLoadMode loadingMode,bool bVerbose)
{
	//if(bVerbose)
		printf("## Load Dome VGA Images: path:%s, frame %d\n",imgDirPath,frameIdx);

	//char dirPath[512];
	char targetImagePath[512];
	//GetFolderPathFromFullPath(m_memoryVector[0]->m_camViewVect[0]->m_fullPath.c_str(),dirPath);
	bool completelyFailed =true;
	for(unsigned int i=0;i<camVect.size();++i)
	{
		//int targetFrameIdx = frameIdx;
		int targetFrameIdx = frameIdx;		
		camVect[i]->m_actualImageFrameIdx = frameIdx;
		int panelIdx = camVect[i]->m_actualPanelIdx;
		int camIdx = camVect[i]->m_actualCamIdx;

		/*
		//termporary for 160224 calibration data
		if(panelIdx ==0 && camIdx ==8)
		{
			printf("!!!!!!!! TEMPORARY: ignore 00_08 cam !!!!!!\n");
			continue;
		}*/

		camVect[i]->m_inputImage = cv::Mat();		//Initialize as a blank.
		camVect[i]->m_rgbInputImage = cv::Mat();		//Initialize as a blank.
		if(USE_PNG_IMAGE)
		{
			if(panelIdx==0)
			{
				sprintf(targetImagePath,"%s/hd_30/%08d/hd%08d_%02d_%02d.png",imgDirPath,targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
					sprintf(targetImagePath,"%s/hd_30/hd%08d_%02d_%02d.png",imgDirPath,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
					sprintf(targetImagePath,"%s/hd_30/%03dXX/%08d/hd%08d_%02d_%02d.png",imgDirPath,int(targetFrameIdx/100),targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
				{
					//return false;
					printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
					continue;
				}
			}
			else
			{
				sprintf(targetImagePath,"%s/vga_25/%08d/%08d_%02d_%02d.png",imgDirPath,targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
					sprintf(targetImagePath,"%s/vga_25/%08d_%02d_%02d.png",imgDirPath,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
					sprintf(targetImagePath,"%s/vga_25/%03dXX/%08d/%08d_%02d_%02d.png",imgDirPath,int(targetFrameIdx/100),targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
				if(IsFileExist(targetImagePath) == false)
				{
					printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
					//return false;
						continue;
				}
			}
			
		}
		else
		{
			sprintf(targetImagePath,"%s/%08d/%08d_%02d_%02d.bmp",imgDirPath,targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
			if(IsFileExist(targetImagePath) == false)
				sprintf(targetImagePath,"%s/%08d_%02d_%02d.bmp",imgDirPath,targetFrameIdx,panelIdx,camIdx);
			if(IsFileExist(targetImagePath) == false)
				sprintf(targetImagePath,"%s/%03dXX/%08d/%08d_%02d_%02d.bmp",imgDirPath,int(targetFrameIdx/100),targetFrameIdx,targetFrameIdx,panelIdx,camIdx);
			if(IsFileExist(targetImagePath) == false)
			{
				printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
				//return false;
				continue;
			}
		}

		camVect[i]->m_fullPath = string(targetImagePath);
		char fileName[128];
		GetFileNameFromFullPath(targetImagePath,fileName);
		camVect[i]->m_fileName = string(fileName);

		if(loadingMode==LOAD_MODE_NO_IMAGE)
			return true;

		else if(loadingMode==LOAD_MODE_GRAY_RGB )
		{
			camVect[i]->m_rgbInputImage = imread(targetImagePath);
			if(camVect[i]->m_rgbInputImage.rows==0)
			{
				printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
				//return false;
				continue;
			}

			/*if(DO_HISTOGRAM_EQUALIZE)
			{
				Mat grayImage;
				cvtColor(camVect[i]->m_rgbInputImage,grayImage,CV_RGB2GRAY);
				equalizeHist(grayImage,camVect[i]->m_inputImage);
			}
			else*/
				cvtColor(camVect[i]->m_rgbInputImage,camVect[i]->m_inputImage,CV_RGB2GRAY);
		}
		else if(loadingMode==LOAD_MODE_GRAY)
		{
			/*if(DO_HISTOGRAM_EQUALIZE)
			{
				Mat grayImage= imread(targetImagePath,CV_LOAD_IMAGE_GRAYSCALE);
				equalizeHist(grayImage,camVect[i]->m_inputImage);
			}
			else*/
				camVect[i]->m_inputImage = imread(targetImagePath,CV_LOAD_IMAGE_GRAYSCALE);
			
			if(camVect[i]->m_inputImage.rows==0)
			{
				printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
				//return false;
				continue;
			}
		}
		else if(loadingMode==LOAD_MODE_RGB)
		{
			camVect[i]->m_rgbInputImage = imread(targetImagePath);
			if(camVect[i]->m_rgbInputImage.rows==0)
			{
				printf("## Warning: image load failure: finally tried path: %s\n",targetImagePath);
				//return false;
				continue;
			}
		}
		completelyFailed = false;
	}
	if(bVerbose)
		printf("## Load Dome VGA Images: finished\n");
	
	if(completelyFailed)
	{
		printf("## Completely Failed in loading images: final path %s\n",targetImagePath);
		return false;
	}
	else
		return true;
}


//Generate R,t by which you can transform current world to normalize world coord
//X_norm = R*X_ori+t
//Normalize world means 
//	1. origin is on the center of the floor
//	2. world's y direction is aligned with normal of the floor
//	3. world's y direction is aligned with normal of the floor
//  4. 1 in world scale is 1cm
void CDomeImageManager::ComputeRtForNormCoord(cv::Mat_<double>& R_return,cv::Mat_<double>& t_return,double& scale)
{
	Point3f y_axis = - g_floorNormal;
	Point3f x_axis(1,0,0);
	Point3f z_axis = y_axis.cross(x_axis);
	Normalize(z_axis);
	x_axis = y_axis.cross(z_axis);
	Normalize(x_axis);

	//Compute Rotation Matrix
	cv::Mat_<double> rot_oriToNorm = cv::Mat_<double>(3,3);		//patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate
	rot_oriToNorm(0,0)= x_axis.x;
	rot_oriToNorm(1,0)= x_axis.y;
	rot_oriToNorm(2,0)= x_axis.z;
	rot_oriToNorm(0,1)= y_axis.x;
	rot_oriToNorm(1,1)= y_axis.y;
	rot_oriToNorm(2,1)= y_axis.z;
	rot_oriToNorm(0,2)= z_axis.x;
	rot_oriToNorm(1,2)= z_axis.y;
	rot_oriToNorm(2,2)= z_axis.z;

	cv::Mat_<double> rot_normToOri = rot_oriToNorm.t();

	//rot_normToOri * X_world + 
	Mat_<double> t(3,1);
	t(0,0) = -g_floorCenter.x;
	t(1,0) = -g_floorCenter.y;
	t(2,0) = -g_floorCenter.z;

	/*double scale = world2cm(1);
	R_return = scale*rot_normToOri;		
	t_return = scale*rot_normToOri*t;*/


	//Return values
	scale = world2cm(1);
	R_return = rot_normToOri;
	t_return = rot_normToOri*t;
	//x_norm  = s(rot_oriToNorm*X-g_floorCenter)
	//= s*rot_oriToNorm  + s*(-g_floorCenter);
}

void CDomeImageManager::TransformExtrisicToNormCoord(const char* saveDirPath)
{
	CalcWorldScaleParameters();
	Mat_<double> R_wn,t_wn;
	double scale;
	ComputeRtForNormCoord(R_wn,t_wn,scale);				//using global variables


	//Debug 
	
	
	//s( R_wn  x + t_wn)
	//I ignored s. S only needs to be multiplied to t_new (and meaningless after projection).
	//R_o *x+ t_o == R_new s (R_wn *x + t_wn) + t_new
	//			  == (R_new* R_wn) *x + (R_new *t_wn + t_new)
	//R_o == R_new * R_wn --> R_new == R_o * R_wn'
	//t_o == R_new * t_wn + t_new  --> t_new = t_o - R_new*t_wn
	for(int i=0;i<m_domeViews.size();++i)
	{
		Mat R_new = m_domeViews[i]->m_R* R_wn.t();
		Mat t_new = m_domeViews[i]->m_t - R_new*t_wn;
		t_new   = scale * t_new;		//to apply scale, only have to multiply on t.  //K[R t] X =dot= K[R st] sX
		
		printf("------------\n");
		printf("check: det(R) = %f\n",determinant(R_new));
		Mat_<double> rand(3,1);
		rand(0,0) = 123.12;
		rand(1,0) = -12.52;
		rand(2,0) =  42.52;
		Mat_<double> re = (m_domeViews[i]->m_R*rand + m_domeViews[i]->m_t);
		re = re * 1.0/re(2.0);
		cout <<re <<endl;

		Mat_<double> re2 = R_new*scale*(R_wn*rand + t_wn )+ t_new;
		re2 = re2 * 1.0/re2(2.0);
		cout <<re2 <<endl;

		printf("------------\n");
		printf("check: floor centers\n");

		Mat_<double> debug_center(3,1);
		debug_center(0,0) = g_floorCenter.x;
		debug_center(1,0) = g_floorCenter.y;
		debug_center(2,0) = g_floorCenter.z;
		debug_center = (R_wn*debug_center+ t_wn);
		cout << "scale: " << scale<< endl;
		cout << "center : "<<endl;
		cout << debug_center<<endl;

		printf("------------\n");

		char calibParamPath[256];
		//sprintf(calibParamPath,"%s/%s/%02d_%02d_ext.txt",dirPath,CALIB_DATA_FOLDER,(*m_pMajorCamViewVect)[i]->m_actualPanelIdx,(*m_pMajorCamViewVect)[i]->m_actualCamIdx); //load same K for all cam
		sprintf(calibParamPath,"%s/%02d_%02d_ext.txt",saveDirPath,(m_domeViews)[i]->m_actualPanelIdx,(m_domeViews)[i]->m_actualCamIdx); //load same K for all cam

		Mat quat(4,1,CV_64F);
		Mat center;
		center = -R_new.t() * t_new;
		Rotation2Quaternion(R_new,quat);
		ofstream fout(calibParamPath,ios_base::trunc);
		for(int k=0;k<4;++k)
			fout << quat.at<double>(k,0) <<" ";
		for(int k=0;k<3;++k)
			fout << center.at<double>(k,0) <<" ";
		printfLog("%s has been saved\n",calibParamPath);
		fout.close();
	}
}


void CDomeImageManager::SaveExtrinsicParam(char* dirPath)
{
	for(unsigned int i=0;i<m_domeViews.size();++i)
	{
		char calibParamPath[256];
		//sprintf(calibParamPath,"%s/%s/%02d_%02d_ext.txt",dirPath,CALIB_DATA_FOLDER,(*m_pMajorCamViewVect)[i]->m_actualPanelIdx,(*m_pMajorCamViewVect)[i]->m_actualCamIdx); //load same K for all cam
		sprintf(calibParamPath,"%s/%02d_%02d_ext.txt",dirPath,(m_domeViews)[i]->m_actualPanelIdx,(m_domeViews)[i]->m_actualCamIdx); //load same K for all cam

		ofstream fout(calibParamPath,ios_base::trunc);
		for(int k=0;k<4;++k)
			fout << (m_domeViews)[i]->m_R_Quat.at<double>(k,0) <<" ";
		for(int k=0;k<3;++k)
			fout << (m_domeViews)[i]->m_CamCenter.at<double>(k,0) <<" ";
		printfLog("%s has been saved\n",calibParamPath);
		fout.close();
	}
	printfLog("Done\n");
}

void CDomeImageManager::SaveIntrinsicParam(char* dirPath,bool withoutDistortion )
{
	for(unsigned int i=0;i<m_domeViews.size();++i)
	{
		char calibParamPath[256];
		sprintf(calibParamPath,"%s/%02d_%02d.txt",dirPath,(m_domeViews)[i]->m_actualPanelIdx,(m_domeViews)[i]->m_actualCamIdx); //load same K for all cam

		ofstream fout(calibParamPath,ios_base::trunc);

		double* pt = (m_domeViews)[i]->m_K.ptr<double>(0);
		for(int j=0;j<9;++j)
		{
			fout << pt[j] <<" ";
		}
		if(withoutDistortion==false && (m_domeViews)[i]->m_distortionParams.size()>=2)
		{
			for(int d=0;d<(m_domeViews)[i]->m_distortionParams.size(); ++d)
				fout << (m_domeViews)[i]->m_distortionParams[d] <<" " ;//<< (*m_pMajorCamViewVect)[i]->m_distortionParams[1] << "\n";
			//fout << (*m_pMajorCamViewVect)[i]->m_camParamForBundle[7] <<" " << (*m_pMajorCamViewVect)[i]->m_camParamForBundle[8] << "\n";
		}
		else
			fout << "0 0 0 0 0\n";
		fout.close();
	}
}


