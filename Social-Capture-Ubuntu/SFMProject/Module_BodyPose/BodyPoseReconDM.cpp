#include "BodyPoseReconDM.h"
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <cstdio>
#include "Utility.h"

using namespace std;
using namespace cv;
//domeViews is required to load only the corresponding view's results
bool Module_BodyPose::Load_Undist_PoseDetectMultipleCamResult_MultiPoseMachine_19jointFull(
	const char* poseDetectFolder,const char* poseDetectSaveFolder,
	const int currentFrame, CDomeImageManager& domeImMan,bool isHD)
{
	//Export Face Detect Results
	char fileName[512];
	char savefileName[512];
	if(isHD==false)
	{
		sprintf(fileName,"%s/poseDetectMC_%08d.txt",poseDetectFolder,currentFrame);
		sprintf(savefileName,"%s/poseDetectMC_%08d.txt",poseDetectSaveFolder,currentFrame);
	}
	else
	{
		sprintf(fileName,"%s/poseDetectMC_hd%08d.txt",poseDetectFolder,currentFrame);
		sprintf(savefileName,"%s/poseDetectMC_hd%08d.txt",poseDetectSaveFolder,currentFrame);
	}
	ifstream fin(fileName);
	if(fin.is_open()==false)
	{
		printf("LoadPoseDetectMultipleCamResult_PoseMachine:: Failed from %s\n\n",fileName);
		return false;
	}
	CreateFolder(poseDetectSaveFolder);
	ofstream fout(savefileName);

	printf("%s\n",fileName);
	char buf[512];
	fin >> buf;
	//fout << buf;
	fout << "ver_mpm_19";
	float version;	
	fin >> version;	
	fout << " " <<version <<"\n";

	if(version<0.1)
	{
		printf("## ERROR: version information is wrong: %f\n",version);
		fin.close();
		fout.close();
		return false;
	}

	int processedViews;	
	if(version>0.49)
	{
		fin >> buf;	//processedViews
		fin >> processedViews;	
		fout << buf << " " << processedViews <<"\n";
	}

	//for(int i=0;i<domeViews.size();++i)
	for(int i=0;   ;++i)
	{ 
		int frameIdx,panelIdx,camIdx,peopleNum,jointNum;
		fin >> frameIdx >> panelIdx >>camIdx >> peopleNum >> jointNum;
		if(fin.eof())
			break;

		fout << frameIdx <<" "<< panelIdx <<" " <<camIdx <<" "<< peopleNum <<" "<< jointNum <<"\n";

		CamViewDT* pCamDT = domeImMan.GetViewDTFromPanelCamIdx(panelIdx,camIdx);
		for(int p=0;p<peopleNum;++p)
		{
			for(int j=0;j<jointNum;++j)
			{
				Point2f tempPt;
				double score;
				fin >>tempPt.x >> tempPt.y >> score;  
				//printf("%f %f %f\n",tempPt.x,tempPt.y,score);
				// if(j>=18) // need to handle 25 pts from Op now
				//	continue;
				if(panelIdx ==14 && camIdx==18 || pCamDT==NULL)		//Not a valid parameters
				{
					fout << -1.0 << " " << -1.0 << " " << -1.0 <<" ";  
					continue;
				}
				Point2f idealPt = pCamDT->ApplyUndistort(tempPt);
				fout << idealPt.x << " " << idealPt.y << " " << score <<" ";  
			}
			fout <<"\n";
			if(version<0.31)
			{
				double dummy;
				fin >>dummy >> dummy >> dummy>> dummy;		//scale information. Not useful
			}
		}
	}
	fin.close();
	fout.close();
	return true;
}