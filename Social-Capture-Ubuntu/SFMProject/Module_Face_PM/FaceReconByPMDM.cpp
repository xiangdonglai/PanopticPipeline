#include "FaceReconByPMDM.h"
#include <omp.h>

using namespace std;
using namespace cv;

namespace Module_Face_pm
{

Module_Face_pm::CFaceRecon_pm g_faceReconManager_pm;			

//Tomas's json format
//Loading path should be: (loadFolderName)/faceDetect_(frameIdx).txt
//camViews is used to verify camLabel and save projectMat
bool LoadFaceDetectResult_MultiCams_Json(const char* loadFolderName,const int currentFrame, const vector<CamViewDT*>& camViews,vector< vector<SFace2D_pm> >& detectedHandVect,bool isHD )
{
	vector<SFace2D_pm> dummy;
	detectedHandVect.resize(camViews.size(),dummy);

	char fileName[512];
	if(isHD==false)
		sprintf(fileName,"%s/vga_25/faceDetectMC_%08d.txt",loadFolderName,currentFrame);
	else
		sprintf(fileName,"%s/hd_30/faceDetectMC_hd%08d.txt",loadFolderName,currentFrame);
	//sprintf(fileName,"d:/hand/handDetectMC_hd%08d.txt",currentFrame);
	//sprintf(fileName,"d:/hand_colloScene4/handDetectMC_hd%08d.txt",currentFrame);
	printf("## Loading Hand Detect Result (MultiCams) from %s\n",fileName);

	ifstream fin(fileName);
	if(fin.is_open()==false)
	{
		printf("## WARNING: file not found %s\n",fileName);
		detectedHandVect.clear();
		return false;
	}
	int camNum;
	fin >> camNum;
	printf("## Num of cameras: %d\n",camNum);
	if(camNum == camViews.size())			//Load without sampling
	{
		while(true)
		{
			//Load from images
			int handNum,viewIdx,panelIdx,camIdx;
			fin >> handNum >> viewIdx >> panelIdx >> camIdx;
			//printf("%d %d %d %d\n",handNum,viewIdx,panelIdx,camIdx);
			if(fin.eof())
				break;

			if(handNum==0)
				continue;
			if(!(camViews[viewIdx]->m_actualCamIdx == camIdx && camViews[viewIdx]->m_actualPanelIdx== panelIdx))
			{
				printf("HANDDETEC_LOADER:ERROR: camera order mismatch !!!!!!!!\n");
				return false;
			}
			
			if(handNum>0)
				detectedHandVect[viewIdx].resize(handNum);
			for(int j=0;j<handNum;++j)
			{
				int subjectIdx, landmarkNum;
				fin >> subjectIdx>>landmarkNum;

				detectedHandVect[viewIdx][j].m_viewIdx = viewIdx;
				detectedHandVect[viewIdx][j].m_panelIdx= panelIdx ;
				detectedHandVect[viewIdx][j].m_camIdx = camIdx;
				detectedHandVect[viewIdx][j].m_faceIdx = j;
				detectedHandVect[viewIdx][j].m_P = camViews[viewIdx]->m_P;
				detectedHandVect[viewIdx][j].m_camCenter = MatToPoint3d(camViews[viewIdx]->m_CamCenter);

				detectedHandVect[viewIdx][j].m_subjectIdx = subjectIdx;
				/*if(whichside=='l')
					detectedHandVect[viewIdx][j].m_whichSide = HAND_LEFT;
				else
					detectedHandVect[viewIdx][j].m_whichSide = HAND_RIGHT;*/

				detectedHandVect[viewIdx][j].m_faceLandmarkVect.resize(landmarkNum);
				detectedHandVect[viewIdx][j].m_detectionScore.resize(landmarkNum);
				for(int t=0;t<landmarkNum;++t)
				{
					Point2d tempPt;
					float score;
					fin >>tempPt.x >>tempPt.y >>score;
					detectedHandVect[viewIdx][j].m_faceLandmarkVect[t] = tempPt;
					detectedHandVect[viewIdx][j].m_detectionScore[t] = score;
				}
			}
		}
		fin.close();
		return true;
	}
	detectedHandVect.clear();
	return false;
}

//output: currentHand3D
void ransac(vector<SFace2D_pm*>& face2dVect,vector<int>& fingerGroups,SFace3D_pm& currentHand3D)
{
	double initPair_repro_thresh = 5;
	double relatedView_repro_thresh = 5;
	double final_repro_thresh = 5;
	int minimum_vis_thresh = 3;
	double per_point_conf_thresh = 0.1;
	
	/*//printf("\n\nGroupIdx %d\n",gIdx);
	//SFace3D_pm& currentHand3D = handReconInit[gIdx];
	currentHand3D.m_landmark3D.clear();
	currentHand3D.m_faceInfo.clear();
	currentHand3D.m_landmark3D.resize(FACE_PM_70_LANDMARK_NUM);
	currentHand3D.m_faceInfo.resize(FACE_PM_70_LANDMARK_NUM);*/

	if(face2dVect.size()<minimum_vis_thresh)
			return;

	int bestInlierNum =0;
	float bestReproError =1e5;
	for(int i=0;i<face2dVect.size();++i)
	{
		for(int j=i+1;j<face2dVect.size();++j)
		{
			//printf("pair: %d, %d\n",i,j);
			double camBaseLine = Distance(face2dVect[i]->m_camCenter,face2dVect[j]->m_camCenter);
			//printf("baseline: %f\n",camBaseLine);
			if(cm2world(camBaseLine)<1e-3)
				continue;
			SFace3D_pm candHand3D;

			//face2dVect[i],face2dVect[j]  are the candidate pair
			double reprErrSum =0;
			vector<Point3d> refPt3DVect;
			refPt3DVect.resize(fingerGroups.size());
			bool bFailed =false;
			for(int localIdx=0;localIdx<fingerGroups.size();++localIdx)
			{
				int ptIdx = fingerGroups[localIdx];
				if(face2dVect[i]->m_detectionScore[ptIdx]<per_point_conf_thresh || face2dVect[j]->m_detectionScore[ptIdx]<per_point_conf_thresh)
				{
					bFailed = true;
					break;
				}

				vector<Mat*> MVec; 
				vector<Point2d> pt2DVec;
				
				MVec.push_back(&face2dVect[i]->m_P);
				pt2DVec.push_back(face2dVect[i]->m_faceLandmarkVect[ptIdx]);

				MVec.push_back(&face2dVect[j]->m_P);
				pt2DVec.push_back(face2dVect[j]->m_faceLandmarkVect[ptIdx]);

				vector<unsigned int> inliers;
				Mat X;
				triangulateWithOptimization(MVec,pt2DVec,X);
				double reprErrAvg = CalcReprojectionError(MVec,pt2DVec,X);
				Point3d pt3D = MatToPoint3d(X);
				refPt3DVect[localIdx] = pt3D;
				reprErrSum+=reprErrAvg;
			}
			if(bFailed)
				continue;
			double reprErrAvg = reprErrSum/fingerGroups.size();

			//printf("InitPair: GroupIdx %d: reprError %f, baseline %f \n",gIdx,reprErrAvg,world2cm(camBaseLine));
			if(reprErrAvg>initPair_repro_thresh)
			{
				//printf("Rejected by initPairRepError: %f>%f\n",reprErrAvg,initPair_repro_thresh);
				continue;
			}

			//Find corresponding face 2Ds
			vector<SFace2D_pm*> selectedHand2Ds;
			for(int i=0;i<face2dVect.size();++i)
			{
				double avg2DError =0;
				bool bFailed = false;
				for(int localIdx=0;localIdx<fingerGroups.size();++localIdx)
				{
					int ptIdx = fingerGroups[localIdx];
					Point2d projected2D =  Project3DPt(refPt3DVect[localIdx],face2dVect[i]->m_P);
					double dist2D = Distance(projected2D,face2dVect[i]->m_faceLandmarkVect[ptIdx]);
					avg2DError += dist2D;
					if(face2dVect[i]->m_detectionScore[ptIdx]<per_point_conf_thresh)
					{
						bFailed = true;
						break;
					}
				}
				if(bFailed)
					continue;
				avg2DError /=fingerGroups.size();
				
				if(avg2DError<relatedView_repro_thresh)
				{
					selectedHand2Ds.push_back(face2dVect[i]);
				}
			}

			if(selectedHand2Ds.size()<2)
			{
				//printf("GroupIdx %d: Rejected by selectedHand2Ds.size()<2\n",gIdx);
				continue;
			}

			//printf("This is a good candidate. I will start triangulation\n");
				
			//printf("Triangulate again\n");
			//Triangulate again 
			double reproErrorAvg=0;
			vector<Point3d> reconPt3DVect;
			reconPt3DVect.resize(fingerGroups.size());
			bFailed = false;
			for(int localIdx=0;localIdx<fingerGroups.size();++localIdx)
			{
				int ptIdx = fingerGroups[localIdx];

				vector<Mat*> MVec; 
				vector<Point2d> pt2DVec;
				for(int i=0;i<selectedHand2Ds.size();++i)
				{
					//Calculate reprojection error
					MVec.push_back(&selectedHand2Ds[i]->m_P);
					pt2DVec.push_back(selectedHand2Ds[i]->m_faceLandmarkVect[ptIdx]);
				}

				vector<unsigned int> inliers;
				Mat X;
				triangulateWithOptimization(MVec,pt2DVec,X);
				double reproError = CalcReprojectionError(MVec,pt2DVec,X);

				//printf("inlierNum %d, reproErro %f\n",selectedHand2Ds.size(),reproError);
				//if(inliers.size()<minimum_vis_thresh)
					//continue;

				if(X.rows==0)
				{
					//printf("X is empty: error %f, inliers %d\n",error,inliers);
					bFailed = true;
					break;
				}
				Point3d pt3D = MatToPoint3d(X);

				reconPt3DVect[localIdx] = pt3D;
				reproErrorAvg+=reproError;
			}
			reproErrorAvg/=fingerGroups.size();
			if(bFailed)
				continue;

			if(bestInlierNum <  selectedHand2Ds.size()
				|| (bestInlierNum ==  selectedHand2Ds.size() && reproErrorAvg<bestReproError) )
			{
			//	printf("Update: new inlier num %d,%f (old %d,%f)\n",selectedHand2Ds.size(),reproErrorAvg,bestInlierNum,bestReproError);
				bestInlierNum = selectedHand2Ds.size();
				bestReproError = reproErrorAvg;
				for(int localIdx=0;localIdx<fingerGroups.size();++localIdx)
				{
					int ptIdx = fingerGroups[localIdx];

					currentHand3D.m_landmark3D[ptIdx] = reconPt3DVect[localIdx];						
					currentHand3D.m_faceInfo[ptIdx].m_averageReproError = reproErrorAvg;

					currentHand3D.m_faceInfo[ptIdx].m_visibility.clear();
					currentHand3D.m_faceInfo[ptIdx].m_visibility.reserve(selectedHand2Ds.size());
					double avgDetectScore=0;
					for(int kk=0;kk<selectedHand2Ds.size();++kk)
					{
						currentHand3D.m_faceInfo[ptIdx].m_visibility.push_back( selectedHand2Ds[kk]->m_camIdx);
						avgDetectScore+=selectedHand2Ds[kk]->m_detectionScore[ptIdx];
						//printf("	pt: %d, cam: %d, score %f)\n",ptIdx, selectedHand2Ds[kk]->m_camIdx, selectedHand2Ds[kk]->m_detectionScore[ptIdx]);
					}
					currentHand3D.m_faceInfo[ptIdx].m_averageScore = avgDetectScore/selectedHand2Ds.size();

					if(ptIdx==0)
					{
						//currentHand3D.m_whichSide = selectedHand2Ds.front()->m_whichSide;
						currentHand3D.m_identityIdx = selectedHand2Ds.front()->m_subjectIdx;
					}
				}
			}
		}		//end of j
	}	//end of i
}

void FacePointsTriangulation_face70(vector<vector<SFace2D_pm> >& hand2DVect_multiCam,vector<vector<SFace2D_pm*> >& hand2DGroupVect,vector<SFace3D_pm>& handRecon,bool isHD)
{
	//printf("FacePointsTriangulation_face70 is called\n");

	vector< vector<int> > fingerGroups;
	fingerGroups.resize(11);
	
	for(int i=0;i<FACE_PM_70_LANDMARK_NUM;++i)
	{
		int groupIdx = 0;
		if(i<=6)
			groupIdx = 0;		//group 0: 0-6
		else if(i<=9)			//group 2: 7-9
			groupIdx = 0;
		else if(i<=16)			//group 2: 9-16
			groupIdx = 0;
		else if(i<=21)		
			groupIdx = 3;
		else if(i<=26)
			groupIdx = 4;
		else if(i<=35)
			groupIdx = 5;
		else if(i<=41)
			groupIdx = 6;
		else if(i<=47)
			groupIdx = 7;
		else if(i<=67)
			groupIdx = 8;
		else if(i<=68)
			groupIdx = 9;
		else if(i<=69)
			groupIdx = 10;

		fingerGroups[groupIdx].push_back(i);
	}

	double bestPair_repro_thresh = 2;
	double relatedView_repro_thresh = 3;
	double final_repro_thresh = 3;
	int minimum_vis_thresh = 2;
	double per_point_conf_thresh = 0.2;
	if(isHD)
	{
		bestPair_repro_thresh *=3;
		relatedView_repro_thresh *=3;
	}
	vector<SFace3D_pm> handReconInit;
	handReconInit.resize(hand2DGroupVect.size());

	//hand2DGroupVect means each human's face (which is already known by skeletons)
	#pragma omp parallel for
	for(int gIdx=0;gIdx<hand2DGroupVect.size();++gIdx)
	{
		printf("\n\nGroupIdx %d\n",gIdx);
		vector<SFace2D_pm*>& currentGroup =  hand2DGroupVect[gIdx];

		int landmarkNum = currentGroup.front()->m_faceLandmarkVect.size();
		SFace3D_pm& currentHand3D = handReconInit[gIdx];
		currentHand3D.m_landmark3D.resize(landmarkNum);
		currentHand3D.m_faceInfo.resize(landmarkNum);
		//currentHand3D.avgReprError = 1e5;

		if(currentGroup.size()<minimum_vis_thresh)
			continue;

		#pragma omp parallel for //num_threads(8)
		for(int j=0;j<fingerGroups[0].size();++j)		//for (jaws)
		{
			int ptIdx = fingerGroups[0][j];
			vector<int> tempGroup;
			tempGroup.push_back(ptIdx);
			ransac(currentGroup,tempGroup,currentHand3D);
		}

		#pragma omp parallel for //num_threads(8)
		for(int i=1;i<fingerGroups.size();++i)		//ignore group 0  (jaws)
		{
		//	printf("group: %d\n",i);
			ransac(currentGroup,fingerGroups[i],currentHand3D);
		}
	}	//end of gIdx

	for(unsigned int i=0;i<handReconInit.size();++i)
	{
		if(handReconInit[i].m_landmark3D.size()>0)
			handRecon.push_back(handReconInit[i]);
	}
}

void FacePointsTriangulation_faceCoco(vector<vector<SFace2D_pm> >& hand2DVect_multiCam,vector<vector<SFace2D_pm*> >& hand2DGroupVect,vector<SFace3D_pm>& handRecon,bool isHD)
{
	vector< vector<int> > fingerGroups;
	fingerGroups.resize(12);

	//Left right is defined in viewer's perspective
	//left eyebrow
	fingerGroups[0].push_back(0);
	fingerGroups[0].push_back(1);
	fingerGroups[0].push_back(2);

	//right eyebrow
	fingerGroups[1].push_back(3);
	fingerGroups[1].push_back(4);
	fingerGroups[1].push_back(5);

	//left eye
	fingerGroups[2].push_back(6);
	fingerGroups[2].push_back(7);
	fingerGroups[2].push_back(8);

	//right eye
	fingerGroups[3].push_back(9);
	fingerGroups[3].push_back(10);
	fingerGroups[3].push_back(11);

	//left ear
	fingerGroups[4].push_back(12);

	//nose
	fingerGroups[5].push_back(13);
	fingerGroups[5].push_back(14);
	fingerGroups[5].push_back(15);

	//lip
	fingerGroups[6].push_back(17);
	fingerGroups[6].push_back(18);
	fingerGroups[6].push_back(19);

	//chin
	fingerGroups[7].push_back(20);

	//headTop
	fingerGroups[8].push_back(21);

	//leftEarCo
	fingerGroups[9].push_back(22);

	//rightEarCo
	fingerGroups[10].push_back(23);

	//right ear 
	fingerGroups[11].push_back(16);


	double bestPair_repro_thresh = 2;
	double relatedView_repro_thresh = 3;
	double final_repro_thresh = 3;
	int minimum_vis_thresh = 2;
	double per_point_conf_thresh = 0.2;
	if(isHD)
	{
		bestPair_repro_thresh *=3;
		relatedView_repro_thresh *=3;
	}
	vector<SFace3D_pm> handReconInit;
	handReconInit.resize(hand2DGroupVect.size());

	//g_visData.m_debugPt2.clear();
	for(int gIdx=0;gIdx<hand2DGroupVect.size();++gIdx)
	{
		printf("\n\nGroupIdx %d\n",gIdx);
		vector<SFace2D_pm*>& currentGroup =  hand2DGroupVect[gIdx];

		int landmarkNum = currentGroup.front()->m_faceLandmarkVect.size();
		SFace3D_pm& currentHand3D = handReconInit[gIdx];
		currentHand3D.m_landmark3D.resize(landmarkNum);
		currentHand3D.m_faceInfo.resize(landmarkNum);
		//currentHand3D.avgReprError = 1e5;

		if(currentGroup.size()<minimum_vis_thresh)
			continue;

		#pragma omp parallel for num_threads(8)
		for(int i=0;i<fingerGroups.size();++i)		//ignore group 0  (jaws)
		{
			printf("group: %d\n",i);
			ransac(currentGroup,fingerGroups[i],currentHand3D);
		}
	}	//end of gIdx

	for(unsigned int i=0;i<handReconInit.size();++i)
	{
		if(handReconInit[i].m_landmark3D.size()>0)
			handRecon.push_back(handReconInit[i]);
	}
}

//Input: domeImageManager should contains all the information (camLabels, image, calibration)
void CFaceRecon_pm::Face_Landmark_Reconstruction_hd(const char* handDetectFilePath, CDomeImageManager& domeImageManager,std::vector<SFace3D_pm>& handReconResult)
{
	//Load finger detection result
	vector<vector<SFace2D_pm> > hand2DVect_multiCam;
	bool bLoadingSuccess = LoadFaceDetectResult_MultiCams_Json(handDetectFilePath,domeImageManager.GetFrameIdx(),domeImageManager.m_domeViews,hand2DVect_multiCam,true);
	if(bLoadingSuccess ==false)
	{
		printf("## ERROR: cannot load hand detection results\n");
		exit(1);  // throw error and exit
	}
	
	//Grouping
	vector<vector<SFace2D_pm*> > hand2DGroupVect;
	vector<vector<SFace2D_pm*> > detectedHandVectRemain;			//outer for each view, inner for each hand in a view
	for(int camIdx=0;camIdx<domeImageManager.GetCameraNum();++camIdx)
	{
		vector<SFace2D_pm*> handVect;
		for(int f=0;f<hand2DVect_multiCam[camIdx].size();++f)
		{
			handVect.push_back(&hand2DVect_multiCam[camIdx][f]);		//used for grouping
		}
		detectedHandVectRemain.push_back(handVect);
	}
	
	bool bUpdated = false; 
	while(true)
	{
		vector<vector<SFace2D_pm*> > remains_updated;			//outer for each view, inner for each hand in a view
		remains_updated.resize(detectedHandVectRemain.size());		//detectedHandVectRemain.size()==number of camera
		vector<SFace2D_pm*> relatedHand2Ds;
		for(int camIdx=0;camIdx<detectedHandVectRemain.size();++camIdx)
		{
			for(int hIdx =0; hIdx < detectedHandVectRemain[camIdx].size();hIdx++)
			{
				if(relatedHand2Ds.size()==0)
					relatedHand2Ds.push_back(detectedHandVectRemain[camIdx][hIdx]);
				else
				{
					if(relatedHand2Ds.front()->m_subjectIdx == detectedHandVectRemain[camIdx][hIdx]->m_subjectIdx)
						//&& relatedHand2Ds.front()->m_whichSide == detectedHandVectRemain[camIdx][hIdx]->m_whichSide)
						relatedHand2Ds.push_back(detectedHandVectRemain[camIdx][hIdx]);
					else
						remains_updated[camIdx].push_back(detectedHandVectRemain[camIdx][hIdx]);
				}
			}
		}
		if(relatedHand2Ds.size()==0)
			break;		//no more data
		else if(relatedHand2Ds.size()<2)
		{
			detectedHandVectRemain = remains_updated;
			continue;		//ignore this
		}

		hand2DGroupVect.push_back(relatedHand2Ds);
		detectedHandVectRemain = remains_updated;
	}

	if(hand2DVect_multiCam.size()>0 && hand2DVect_multiCam.front().size()>0)
	{
		int landmakrNum = hand2DVect_multiCam.front().front().m_faceLandmarkVect.size();
		if(landmakrNum==70)
		{
			FacePointsTriangulation_face70(hand2DVect_multiCam, hand2DGroupVect,handReconResult,true);
		}
		else if(landmakrNum==24)
			FacePointsTriangulation_faceCoco(hand2DVect_multiCam, hand2DGroupVect,handReconResult,true);

	}
}

CFaceRecon_pm::CFaceRecon_pm(void)
{
	m_curImgFrameIdx = m_currentSelectedLocalVectIdx = -1;
	m_vis_avgDetectThresh = 1.0;
	m_vis_reproErrorThresh = 1e5;
	m_vis_visibilityThresh =0;

	m_fpsType = FPS_HD_30;	//default
}

//Save Face
void CFaceRecon_pm::SaveFaceReconResult(const char* folderPath,vector<SFace3D_pm>& handRecon,int frameIdx,bool bIsHD)
{
	/*//Save Face Reconstruction Result
	char folderPath[512];
	sprintf(folderPath,"%s/faceRecon_%d",g_dataMainFolder,g_askedVGACamNum);
	CreateFolder(folderPath);
	*/

	char fullPath[512];
	if(bIsHD ==false)
		sprintf(fullPath,"%s/faceRecon3Dpm_%08d.txt",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
	else
		sprintf(fullPath,"%s/faceRecon3Dpm_hd%08d.txt",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
	printf("Save to %s\n",fullPath);
	ofstream fout(fullPath,std::ios_base::trunc);
	fout << "ver 0.3\n";			//0.2: added identity, l-r info 0.3: added fingerreconInfo
	fout << handRecon.size() <<"\n";

	for(unsigned int i=0;i<handRecon.size();++i)
	{
		/*fout << handRecon[i].centerPt3D.x << " " << handRecon[i].centerPt3D.y << " " << handRecon[i].centerPt3D.z << " " <<
			handRecon[i].faceNormal.x << " " << handRecon[i].faceNormal.y << " " << handRecon[i].faceNormal.z <<" " <<
			handRecon[i].avgReprError << " " << handRecon[i].avgDetectScore <<"\n";*/
		/*fout <<handRecon[i].visibility.size() <<" ";
		for(unsigned int j=0;j<handRecon[i].visibility.size();++j)
		{
			fout << handRecon[i].visibility[j].first << " ";
		}
		fout <<"\n";*/
		fout <<handRecon[i].m_identityIdx <<" ";//
		fout <<handRecon[i].m_landmark3D.size() <<" ";
		for(unsigned int j=0;j<handRecon[i].m_landmark3D.size();++j)
		{
			fout << handRecon[i].m_landmark3D[j].x << " " << handRecon[i].m_landmark3D[j].y << " " << handRecon[i].m_landmark3D[j].z << " ";
		}
		fout <<"\n";
		for(unsigned int j=0;j<handRecon[i].m_faceInfo.size();++j)
		{
			fout << handRecon[i].m_faceInfo[j].m_averageScore << " " << handRecon[i].m_faceInfo[j].m_averageReproError << " ";
			fout << handRecon[i].m_faceInfo[j].m_visibility.size() <<" ";
			for(unsigned int k=0;k<handRecon[i].m_faceInfo[j].m_visibility.size();++k)
			{
				fout<<handRecon[i].m_faceInfo[j].m_visibility[k]<< " ";
			}
			fout<<"\n";
		}
		fout <<"\n";
	}
	fout.close();
}

//ver 0.2
void CFaceRecon_pm::LoadFace3DByFrame(const char* folderPath,const int firstFrameIdx,const int frameNum,bool bIsHD)
{ 
	m_faceReconMem.clear();
	for(int f=0;f<frameNum;++f)
	{
		//Save Face Reconstruction Result
		char fullPath[512];
		if(bIsHD==false)
			sprintf(fullPath,"%s/faceRecon3Dpm_%08d.txt",folderPath,firstFrameIdx + f);
		else
			sprintf(fullPath,"%s/faceRecon3Dpm_hd%08d.txt",folderPath,firstFrameIdx + f);
		if(f%1000==0)
			printf("Load FaceReconResult from %s\n",fullPath);
		ifstream fin(fullPath, ios::in);
		char dummy[512];
		if(fin)
		{
			m_faceReconMem.resize(m_faceReconMem.size()+1);
			SFaceReconMemory_pm& newMem = m_faceReconMem.back();
			newMem.frameIdx = firstFrameIdx + f;

			float ver;
			fin >> dummy >> ver; //version			
			int handNum;
			fin >>handNum;
			//printf("Face Num: %d\n",faceNum);
			newMem.faceReconVect.resize(handNum);
			for(int i=0;i<handNum;++i)
			{
				SFace3D_pm currentHand;

				if(ver>0.15)
				{
					int subjectIdx;
					fin >>subjectIdx;
					//char whichSide;
					//fin >>whichSide;
					newMem.faceReconVect[i].m_identityIdx = subjectIdx;
				}

				int landmarkNum;
				fin >>landmarkNum;
				Point3d pt;
				newMem.faceReconVect[i].m_landmark3D.resize(landmarkNum);
				for(int j=0;j<landmarkNum;++j)
				{
					fin >>pt.x >> pt.y >> pt.z;
					newMem.faceReconVect[i].m_landmark3D[j] = pt;
				}
				newMem.faceReconVect[i].m_faceInfo.resize(landmarkNum);
				if(ver>0.25)
				{
					for(int j=0;j<landmarkNum;++j)
					{
						fin >> newMem.faceReconVect[i].m_faceInfo[j].m_averageScore >> newMem.faceReconVect[i].m_faceInfo[j].m_averageReproError;
						int visibilityNum;
						fin >> visibilityNum;
						newMem.faceReconVect[i].m_faceInfo[j].m_visibility.resize(visibilityNum);
						int camIdx;
						for(int k=0;k<visibilityNum;++k)
						{
							fin >> camIdx;
							newMem.faceReconVect[i].m_faceInfo[j].m_visibility[k] = camIdx;
						}
					}
				}
			}
			//printf("Load Face has been finished %s\n",fullPath);
		}
		else
		{
			printf("Failure in Loading from %s\n",fullPath);
			exit(1);
		}
		fin.close();
	}
	GenerateHumanColors();
	ComputeFaceNormals();
	/*
	if(m_faceReconMem.size()>0)
	{
		if(m_faceReconMem.front().size()>0 && m_faceReconMem.front().faceReconVect.front().landmark3D.size()>0)
			ShowFaceRecon3DResult(g_visData,m_faceReconMem.front().faceReconVect);
		else
			ShowFaceRecon3DBySphere(g_visData,m_faceReconMem.front().faceReconVect);
	}*/
}

void CFaceRecon_pm::SetCurMemIdxByFrameIdx(int curImgFrameIdx)
{
	m_curImgFrameIdx = curImgFrameIdx;

	if(m_faceReconMem.size()==0)
	{
		m_currentSelectedLocalVectIdx = -1;
	}
	else
	{
		int initFrameIdx = m_faceReconMem.front().frameIdx;
		int targetIdx = curImgFrameIdx - initFrameIdx;
		m_currentSelectedLocalVectIdx = targetIdx;	//it could be nagative number. But, doenn't matter. Show function will take care of this.
	}
}

//Save Face
void SaveFaceReconResult_json_singeSubject(ofstream& fout,SFace3D_pm& faceRecon)
{
	fout <<"{";
	//fout <<handRecon[i].m_landmark3D.size() <<" ";
	fout <<"\"landmarks\": [";
	for(unsigned int j=0;j<faceRecon.m_landmark3D.size();++j)
	{
		//fout << handRecon.m_landmark3D[j].x << " " << handRecon.m_landmark3D[j].y << " " << handRecon.m_landmark3D[j].z << " ";
		fout << faceRecon.m_landmark3D[j].x << ", " << faceRecon.m_landmark3D[j].y << ", " << faceRecon.m_landmark3D[j].z;//
			
		if(j+1!=faceRecon.m_landmark3D.size())
			fout<< ", ";
	}
	fout <<"],\n";
	fout <<"\"averageScore\": [";
	for(unsigned int j=0;j<faceRecon.m_faceInfo.size();++j)
	{
		fout << faceRecon.m_faceInfo[j].m_averageScore;// << ", ";
		if(j+1!=faceRecon.m_faceInfo.size())
			fout << ", ";
	}
	fout <<"],\n";

	fout <<"\"averageReproError\": [";
	for(unsigned int j=0;j<faceRecon.m_faceInfo.size();++j)
	{
		fout << faceRecon.m_faceInfo[j].m_averageReproError;//
		if(j+1!=faceRecon.m_faceInfo.size())
			fout << ", ";
	}
	fout <<"],\n";
	fout <<"\"visibility\": [";
	for(unsigned int j=0;j<faceRecon.m_faceInfo.size();++j)
	{
		fout <<"[";

	//	fout << faceRecon.m_faceInfo[j].m_visibility.size() <<" ";
		for(unsigned int k=0;k<faceRecon.m_faceInfo[j].m_visibility.size();++k)
		{
			fout<<faceRecon.m_faceInfo[j].m_visibility[k];//<< ", ";
			if(k+1!=faceRecon.m_faceInfo[j].m_visibility.size())
				fout << ", ";
		}
		if(j+1!=faceRecon.m_faceInfo.size())
			fout <<"],";
		else
			fout <<"]";
	}
	fout <<"\n] }";
}

void CFaceRecon_pm::SaveFaceReconResult_json(const char* folderPath,vector<SFace3D_pm>& faceRecon,int frameIdx,bool bIsHD)
{
	char fullPath[512];
	if(bIsHD ==false)
		sprintf(fullPath,"%s/faceRecon3D_%08d.json",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
	else
		sprintf(fullPath,"%s/faceRecon3D_hd%08d.json",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
	if(frameIdx%100==0)
		printf("Save to %s\n",fullPath);
	ofstream fout(fullPath,std::ios_base::trunc);

	fout << "{ \"version\": 0.5, \n";		//ver 0.5: added outlier infor
	//fout << handRecon.size() <<"\n";

	fout << "\"people\" :\n";
	fout << "[";

	for(int p=0;p<faceRecon.size();++p)
	{
		if(p!=0)
			fout <<",";
		fout <<"\n{ \"id\": "<< faceRecon[p].m_identityIdx <<",\n";

		if(faceRecon[p].m_landmark3D.size()==FACE_PM_70_LANDMARK_NUM)
			fout <<"\"face70\": \n";
		else
			fout <<"\"face_coco\": \n";

		SaveFaceReconResult_json_singeSubject(fout,faceRecon[p]);
		fout <<"}\n";
	}


	fout <<"\n] \n}";
	fout.close();
}

//domeViews is required to load only the corresponding view's results
bool Load_Undist_FaceDetectMultipleCamResult_face70_PoseMachine(
	const char* poseDetectFolder,const char* poseDetectSaveFolder,
	const int currentFrame, CDomeImageManager& domeImMan,bool isHD)
{
	//Export Face Detect Results
	char fileName[512];
	char savefileName[512];
	if(isHD==false)
	{
		sprintf(fileName,"%s/faceDetectMC_%08d.txt",poseDetectFolder,currentFrame);
		sprintf(savefileName,"%s/faceDetectMC_%08d.txt",poseDetectSaveFolder,currentFrame);
	}
	else
	{
		sprintf(fileName,"%s/faceDetectMC_hd%08d.txt",poseDetectFolder,currentFrame);
		sprintf(savefileName,"%s/faceDetectMC_hd%08d.txt",poseDetectSaveFolder,currentFrame);
	}
	ifstream fin(fileName);
	if(fin.is_open()==false)
	{
		printf("Load_Undist_FaceDetectMultipleCamResult_PoseMachine:: Failed from %s\n\n",fileName);
		return false;
	}
	CreateFolder(poseDetectSaveFolder);
	ofstream fout(savefileName);

	printf("%s\n",fileName);
	char buf[512];	
	int processedViews;	
	fin >> processedViews;	
	fout << processedViews <<"\n";
	
	//for(int i=0;i<domeViews.size();++i)
	for(int i=0;   ;++i)
	{ 
		int faceNum,viewIdx,panelIdx,camIdx;
		fin >> faceNum >> viewIdx >> panelIdx >> camIdx;
		if(fin.eof())
			break;
		fout << faceNum <<" "<< viewIdx <<" " <<panelIdx <<" "<< camIdx <<"\n";

		CamViewDT* pCamDT = domeImMan.GetViewDTFromPanelCamIdx(panelIdx,camIdx);
		for(int j=0;j<faceNum;++j)
		{
			//char whichside;
			int subjectIdx, landmarkNum;
			fin >> subjectIdx>>landmarkNum;
			fout << subjectIdx <<" " <<landmarkNum <<"\n";
			for(int t=0;t<landmarkNum;++t)
			{
				Point2d tempPt;
				float score;
				fin >>tempPt.x >>tempPt.y >>score;
				Point2f idealPt = pCamDT->ApplyUndistort(tempPt);
				fout << idealPt.x << " " << idealPt.y << " " << score <<" ";  
			}
			fout <<"\n";
		}
	}
	fin.close();
	fout.close();
	return true;
}

//domeViews is required to load only the corresponding view's results
bool Load_Undist_FaceDetectMultipleCamResult_faceCoco_PoseMachine(
	const char* poseDetectFolder,const char* poseDetectSaveFolder,
	const int currentFrame, CDomeImageManager& domeImMan,bool isHD)
{
	//Export Face Detect Results
	char fileName[512];
	char savefileName[512];
	if(isHD==false)
	{
		sprintf(fileName,"%s/faceDetectMC_%08d.txt",poseDetectFolder,currentFrame);
		sprintf(savefileName,"%s/faceDetectMC_%08d.txt",poseDetectSaveFolder,currentFrame);
	}
	else
	{
		sprintf(fileName,"%s/faceDetectMC_hd%08d.txt",poseDetectFolder,currentFrame);
		sprintf(savefileName,"%s/faceDetectMC_hd%08d.txt",poseDetectSaveFolder,currentFrame);
	}
	ifstream fin(fileName);
	if(fin.is_open()==false)
	{
		printf("Load_Undist_FaceDetectMultipleCamResult_faceCoco_PoseMachine:: Failed from %s\n\n",fileName);
		return false;
	}
	CreateFolder(poseDetectSaveFolder);
	ofstream fout(savefileName);

	printf("%s\n",fileName);
	char buf[512];	
	int processedViews;	
	fin >> processedViews;	
	fout << processedViews <<"\n";

	//for(int i=0;i<domeViews.size();++i)
	for(int i=0;   ;++i)
	{ 
		int faceNum,viewIdx,panelIdx,camIdx;
		fin >> faceNum >> viewIdx >> panelIdx >> camIdx;
		if(fin.eof())
			break;
		fout << faceNum <<" "<< viewIdx <<" " <<panelIdx <<" "<< camIdx <<"\n";

		CamViewDT* pCamDT = domeImMan.GetViewDTFromPanelCamIdx(panelIdx,camIdx);
		for(int j=0;j<faceNum;++j)
		{
			//char whichside;
			int subjectIdx, landmarkNum;
			fin >> subjectIdx>>landmarkNum;
			fout << subjectIdx <<" " <<landmarkNum <<"\n";
			for(int t=0;t<landmarkNum;++t)
			{
				Point2d tempPt;
				float score;
				fin >>tempPt.x >>tempPt.y >>score;
				Point2f idealPt = pCamDT->ApplyUndistort(tempPt);
				fout << idealPt.x << " " << idealPt.y << " " << score <<" ";  
			}
			fout <<"\n";
		}
	}
	fin.close();
	fout.close();
	return true;
}

void CFaceRecon_pm::ComputeFaceNormals()
{
	for(int f=0;f<m_faceReconMem.size();++f)
	{
		vector<SFace3D_pm>& curFaceVect = m_faceReconMem[f].faceReconVect;
		for(int i=0;i<curFaceVect.size();++i)
		{
			SFace3D_pm& curFace = curFaceVect[i];

			if(curFace.m_landmark3D.size()==FACE_PM_70_LANDMARK_NUM)
			{
				Point3d vec1 = curFace.m_landmark3D[45]-curFace.m_landmark3D[36]; //leftEye - rightEye
				Normalize(vec1);
				Point3d vec2;//	up vector
				vec2 = curFace.m_landmark3D[27]-curFace.m_landmark3D[51];	//nosetop - noseBottomCenter
				Normalize(vec2);

				curFace.m_normal = vec1.cross(vec2);		
				Normalize(curFace.m_normal);		//handback to palm direction

				curFace.m_faceUp =  vec2;
				curFace.m_center= curFace.m_landmark3D[27];	//noseTop

				curFace.m_faceX =  curFace.m_faceUp.cross(curFace.m_normal);
			}
			else if(curFace.m_landmark3D.size()==FACE_PM_COCO_LANDMARK_NUM)
			{
				Point3d vec1 = curFace.m_landmark3D[11]-curFace.m_landmark3D[6]; //leftEye - rightEye
				Normalize(vec1);
				Point3d vec2;//	up vector
				vec2 = (curFace.m_landmark3D[2]+curFace.m_landmark3D[3])*0.5 -
							(curFace.m_landmark3D[13]+curFace.m_landmark3D[15])*0.5  ;	//nosetop - noseBottomCenter
				Normalize(vec2);

				curFace.m_normal = vec1.cross(vec2);		
				Normalize(curFace.m_normal);		//handback to palm direction

				curFace.m_faceUp =  vec2;
				curFace.m_center= (curFace.m_landmark3D[8]+curFace.m_landmark3D[9])*0.5;	//noseTop

				curFace.m_faceX =  curFace.m_faceUp.cross(curFace.m_normal);
			}
		}
	}
}

void CFaceRecon_pm::GenerateHumanColors()
{
	if(m_colorSet.size()>0)
		return;
	/*
	m_colorSet.push_back(Point3f(128,0,0));
	m_colorSet.push_back(Point3f(0,0,128));
	m_colorSet.push_back(Point3f(0,128,128));
	m_colorSet.push_back(Point3f(0,100,0));
	m_colorSet.push_back(Point3f(218,165,32));
	m_colorSet.push_back(Point3f(199,21,133));
	m_colorSet.push_back(Point3f(243,56,115));
	m_colorSet.push_back(Point3f(39,87,66));
	m_colorSet.push_back(Point3f(2,120,120));
	m_colorSet.push_back(Point3f(100,54,68));
	m_colorSet.push_back(Point3f(42,116,117));
	m_colorSet.push_back(Point3f(253,50,182));
	m_colorSet.push_back(Point3f(200,95,77));

	for(int i=0;i<m_colorSet.size();++i)
	{
		m_colorSet[i].x =m_colorSet[i].x/255.0;
		m_colorSet[i].y =m_colorSet[i].y/255.0;
		m_colorSet[i].z =m_colorSet[i].z/255.0;
	}*/



	/*
	m_colorSet.push_back(Point3f(1.0000000e+00,   7.0296114e-01 ,  3.9235639e-01));
	m_colorSet.push_back(Point3f(7.0943169e-02,   9.9370462e-01 ,  2.8551294e-01));
	m_colorSet.push_back(Point3f(1.2698682e-01,   8.6994103e-01 ,  9.9700327e-01));
	m_colorSet.push_back(Point3f(1.0000000e+00,   3.1773483e-01 ,  2.6900580e-01));
	m_colorSet.push_back(Point3f(9.4853887e-01 ,  4.7711111e-01 ,  9.7867661e-01));
	m_colorSet.push_back(Point3f(1.6581869e-01  , 2.0266472e-01 ,  1.0000000e+00));
	m_colorSet.push_back(Point3f(2.7849822e-01 ,  9.3982947e-01 ,  3.8724543e-01));
	m_colorSet.push_back(Point3f(5.4688152e-01 ,  6.4555187e-01 ,  1.4218716e-01));
	int num = m_colorSet.size();
	m_colorSet.push_back(Point3f(51,160,44));
	m_colorSet.push_back(Point3f(255,127,0));
	m_colorSet.push_back(Point3f(227,26,28));
	m_colorSet.push_back(Point3f(166,206,227));
	m_colorSet.push_back(Point3f(106,61,154));
	m_colorSet.push_back(Point3f(202,178,214));
	m_colorSet.push_back(Point3f(251,154,153));
	m_colorSet.push_back(Point3f(178,223,138));
	//m_colorSet.push_back(Point3f(253,191,111));
	//m_colorSet.push_back(Point3f(31,120,180));
	m_colorSet.push_back(Point3f(177,89,40));
	for(int i=num;i<m_colorSet.size();++i)
	{
		m_colorSet[i].x =m_colorSet[i].x/255.0;
		m_colorSet[i].y =m_colorSet[i].y/255.0;
		m_colorSet[i].z =m_colorSet[i].z/255.0;
	}*/

	//Color table from http://www.rapidtables.com/web/color/RGB_Color.htm


/*	m_colorSet.push_back(Point3f(240,128,128));	//pink
	m_colorSet.push_back(Point3f(0,206,209));	//similar to cyan
	*/
	
	m_colorSet.push_back(Point3f(127,255,0));
	m_colorSet.push_back(Point3f(0,206,209));
	
	m_colorSet.push_back(Point3f(128,0,0));
	m_colorSet.push_back(Point3f(153,50,204));


	m_colorSet.push_back(Point3f(220,20,60));
	m_colorSet.push_back(Point3f(0,128,0));
	m_colorSet.push_back(Point3f(70,130,180));
	m_colorSet.push_back(Point3f(255,20,147));


	m_colorSet.push_back(Point3f(240,128,128));
	m_colorSet.push_back(Point3f(0,250,154));
	m_colorSet.push_back(Point3f(0,0,128));
	m_colorSet.push_back(Point3f(210,105,30));

	m_colorSet.push_back(Point3f(255,165,0));
	m_colorSet.push_back(Point3f(32,178,170));
	m_colorSet.push_back(Point3f(123,104,238));
	

/*	m_colorSet.push_back(Point3f(0,206,209));
	m_colorSet.push_back(Point3f(153,50,204));
	m_colorSet.push_back(Point3f(127,255,0));
	m_colorSet.push_back(Point3f(128,0,0));
	m_colorSet.push_back(Point3f(220,20,60));*/
	for(int i=0;i<m_colorSet.size();++i)
	{
		m_colorSet[i].x =m_colorSet[i].x/255.0;
		m_colorSet[i].y =m_colorSet[i].y/255.0;
		m_colorSet[i].z =m_colorSet[i].z/255.0;
	}
}

}	//end of namespace Module_Face_pm