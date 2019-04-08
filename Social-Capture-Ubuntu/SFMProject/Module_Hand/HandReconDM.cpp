#include "HandReconDM.h"
#include <omp.h>

using namespace std;
using namespace cv;
namespace Module_Hand
{

Module_Hand::CHandRecon g_handReconManager;			
Module_Hand::CHandRecon g_handReconManager_compare;

void ransac(vector<SHand2D*>& currentGroup,vector<int>& fingerIdxVect,SHand3D& currentHand3D);
//Tomas's json format
//Loading path should be: (loadFolderName)/faceDetect_(frameIdx).txt
//camViews is used to verify camLabel and save projectMat
bool LoadFingerDetectResult_MultiCams_Json(const char* loadFolderName,const int currentFrame, const vector<CamViewDT*>& camViews,vector< vector<SHand2D> >& detectedHandVect,bool isHD )
{
	vector<SHand2D> dummy;
	detectedHandVect.resize(camViews.size(),dummy);

	char fileName[512];
	if(isHD==false)
		sprintf(fileName,"%s/vga_25/handDetectMC_%08d.txt",loadFolderName,currentFrame);
	else
		sprintf(fileName,"%s/hd_30/handDetectMC_hd%08d.txt",loadFolderName,currentFrame);
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
				char whichside;
				int subjectIdx, landmarkNum;
				fin >>whichside >>subjectIdx>>landmarkNum;

				detectedHandVect[viewIdx][j].m_viewIdx = viewIdx;
				detectedHandVect[viewIdx][j].m_panelIdx= panelIdx ;
				detectedHandVect[viewIdx][j].m_camIdx = camIdx;
				detectedHandVect[viewIdx][j].m_handIdx = j;
				detectedHandVect[viewIdx][j].m_P = camViews[viewIdx]->m_P;
				detectedHandVect[viewIdx][j].m_camCenter = MatToPoint3d(camViews[viewIdx]->m_CamCenter);

				
				detectedHandVect[viewIdx][j].m_subjectIdx = subjectIdx;
				if(whichside=='l')
					detectedHandVect[viewIdx][j].m_whichSide = HAND_LEFT;
				else
					detectedHandVect[viewIdx][j].m_whichSide = HAND_RIGHT;

				detectedHandVect[viewIdx][j].m_fingerLandmarkVect.resize(landmarkNum);
				detectedHandVect[viewIdx][j].m_detectionScore.resize(landmarkNum);
				for(int t=0;t<landmarkNum;++t)
				{
					Point2d tempPt;
					float score;
					fin >>tempPt.x >>tempPt.y >>score;
					detectedHandVect[viewIdx][j].m_fingerLandmarkVect[t] = tempPt;
					detectedHandVect[viewIdx][j].m_detectionScore[t] = score;
				}
			}
		}
		fin.close();
		return true;
	}
/*	else		//sampled: loading the detection result only for the data corresponding to provided camViews
	{
		bool unexpectedFileEnd = false;
		for(int imIdx=0;imIdx<camViews.size();imIdx++)
		{
			while(true)
			{
				//Load from images
				int handNum,viewIdx,panelIdx,camIdx;
				fin >> handNum >> viewIdx >> panelIdx >> camIdx;
				//printf("%d %d %d %d\n",handNum,viewIdx,panelIdx,camIdx);
				if(fin.eof())
				{
					unexpectedFileEnd = true;
					break;
				}

				if(handNum>0)
					detectedHandVect[imIdx].resize(handNum);

				for(int j=0;j<handNum;++j)
				{
					//detectedHandVect[imIdx].back().viewIdx = viewIdx;		//wrong
					detectedHandVect[imIdx][j].viewIdx = imIdx;			//save sampled idx instead

					detectedHandVect[imIdx][j].panelIdx= panelIdx ;
					detectedHandVect[imIdx][j].camIdx = camIdx;
					detectedHandVect[imIdx][j].faceIdx = j;
					detectedHandVect[imIdx][j].P = camViews[imIdx]->m_P;
					detectedHandVect[imIdx][j].camCenter = MatToPoint3d(camViews[imIdx]->m_CamCenter);


					int faceLandmarkNum;
					fin >> detectedHandVect[imIdx][j].detectionScore >> detectedHandVect[imIdx][j].faceRect.x >>detectedHandVect[imIdx][j].faceRect.y
						>> detectedHandVect[imIdx][j].faceRect.width >>detectedHandVect[imIdx][j].faceRect.height
						>> detectedHandVect[imIdx][j].centerPt.x >> faceLandmarkNum;
					for(int t=0;t<faceLandmarkNum;++t)
					{
						Point2d tempPt;
						fin >>tempPt.x >>tempPt.y;
						detectedHandVect[imIdx][j].m_fingerLandmarkVect.push_back(tempPt);
					}
				}

				if(!(camViews[imIdx]->m_actualCamIdx == camIdx && camViews[imIdx]->m_actualPanelIdx== panelIdx))
				{
					detectedHandVect[imIdx].clear();		//skip this. (Assuming same order between face detect and current image loading order)
				}
				else
					break;		//find corresponding face detection result
			}

			if(unexpectedFileEnd==true)
				break;
		}
		fin.close();

		if(unexpectedFileEnd)
		{
			printf("Error:Face Detection Order is different from current image loading order!!!\n");
			return false;
		}
		return true;
	}*/
	detectedHandVect.clear();
	return false;
}

void HandPointsTriangulation_perHand(vector<vector<SHand2D> >& hand2DVect_multiCam,vector<vector<SHand2D*> >& hand2DGroupVect,vector<SHand3D>& handRecon,bool isHD)
{
	vector< vector<int> > fingerIdxVect;
	fingerIdxVect.resize(6);
	fingerIdxVect[0].push_back(0);
	for(int i=1;i<HAND_LANDMARK_NUM;++i)
	{
		int groupIdx = 0;
		fingerIdxVect[groupIdx].push_back(i);
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
	vector<SHand3D> handReconInit;
	handReconInit.resize(hand2DGroupVect.size());

	//g_visData.m_debugPt2.clear();
	for(int gIdx=0;gIdx<hand2DGroupVect.size();++gIdx)
	{
		printf("\n\nGroupIdx %d\n",gIdx);
		vector<SHand2D*>& currentGroup =  hand2DGroupVect[gIdx];

		int landmarkNum = currentGroup.front()->m_fingerLandmarkVect.size();
		SHand3D& currentHand3D = handReconInit[gIdx];
		currentHand3D.m_landmark3D.resize(landmarkNum);
		currentHand3D.m_fingerInfo.resize(landmarkNum);
		//currentHand3D.avgReprError = 1e5;

		if(currentGroup.size()<minimum_vis_thresh)
			continue;

		#pragma omp parallel for
		for(int i=0;i<fingerIdxVect.size();++i)
		{	
			ransac(currentGroup,fingerIdxVect[i],currentHand3D);
		}
	}	//end of gIdx


	for(unsigned int i=0;i<handReconInit.size();++i)
	{
		if(handReconInit[i].m_landmark3D.size()>0)
			handRecon.push_back(handReconInit[i]);
	}
}

//output: currentHand3D
void ransac(vector<SHand2D*>& currentGroup,vector<int>& fingerIdxVect,SHand3D& currentHand3D)
{
	double initPair_repro_thresh = 4;
	double relatedView_repro_thresh = 4;
	double final_repro_thresh = 6;
	int minimum_vis_thresh = 2;
	double per_point_conf_thresh = 0.2;
	
	/*//printf("\n\nGroupIdx %d\n",gIdx);
	//SHand3D& currentHand3D = handReconInit[gIdx];
	currentHand3D.m_landmark3D.clear();
	currentHand3D.m_fingerInfo.clear();
	currentHand3D.m_landmark3D.resize(HAND_LANDMARK_NUM);
	currentHand3D.m_fingerInfo.resize(HAND_LANDMARK_NUM);*/

	if(currentGroup.size()<minimum_vis_thresh)
			return;

	int bestInlierNum =0;
	for(int i=0;i<currentGroup.size();++i)
	{
		for(int j=i+1;j<currentGroup.size();++j)
		{
			//printf("pair: %d, %d\n",i,j);
			double camBaseLine = Distance(currentGroup[i]->m_camCenter,currentGroup[j]->m_camCenter);
			//printf("baseline: %f\n",camBaseLine);
			if(cm2world(camBaseLine)<1e-3)
				continue;
			SHand3D candHand3D;

			//currentGroup[i],currentGroup[j]  are the candidate pair
			double reprErrSum =0;
			vector<Point3d> refPt3DVect;
			refPt3DVect.resize(fingerIdxVect.size());
			bool bFailed =false;
			for(int localIdx=0;localIdx<fingerIdxVect.size();++localIdx)
			{
				int ptIdx = fingerIdxVect[localIdx];
				if(currentGroup[i]->m_detectionScore[ptIdx]<per_point_conf_thresh || currentGroup[j]->m_detectionScore[ptIdx]<per_point_conf_thresh)
				{
					bFailed = true;
					break;
				}

				vector<Mat*> MVec; 
				vector<Point2d> pt2DVec;
				
				MVec.push_back(&currentGroup[i]->m_P);
				pt2DVec.push_back(currentGroup[i]->m_fingerLandmarkVect[ptIdx]);

				MVec.push_back(&currentGroup[j]->m_P);
				pt2DVec.push_back(currentGroup[j]->m_fingerLandmarkVect[ptIdx]);

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
			double reprErrAvg = reprErrSum/fingerIdxVect.size();

			//printf("InitPair: GroupIdx %d: reprError %f, baseline %f \n",gIdx,reprErrAvg,world2cm(camBaseLine));
			if(reprErrAvg>initPair_repro_thresh)
			{
				//printf("Rejected by initPairRepError: %f>%f\n",reprErrAvg,initPair_repro_thresh);
				continue;
			}

			//Find corresponding face 2Ds
			vector<SHand2D*> selectedHand2Ds;
			for(int i=0;i<currentGroup.size();++i)
			{
				double avg2DError =0;
				bool bFailed = false;
				for(int localIdx=0;localIdx<fingerIdxVect.size();++localIdx)
				{
					int ptIdx = fingerIdxVect[localIdx];
					Point2d projected2D =  Project3DPt(refPt3DVect[localIdx],currentGroup[i]->m_P);
					double dist2D = Distance(projected2D,currentGroup[i]->m_fingerLandmarkVect[ptIdx]);
					avg2DError += dist2D;
					if(currentGroup[i]->m_detectionScore[ptIdx]<per_point_conf_thresh)
					{
						bFailed = true;
						break;
					}
				}
				if(bFailed)
					continue;
				avg2DError /=fingerIdxVect.size();
				
				if(avg2DError<relatedView_repro_thresh)
				{
					selectedHand2Ds.push_back(currentGroup[i]);
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
			reconPt3DVect.resize(fingerIdxVect.size());
			bFailed = false;
			for(int localIdx=0;localIdx<fingerIdxVect.size();++localIdx)
			{
				int ptIdx = fingerIdxVect[localIdx];

				vector<Mat*> MVec; 
				vector<Point2d> pt2DVec;
				for(int i=0;i<selectedHand2Ds.size();++i)
				{
					//Calculate reprojection error
					MVec.push_back(&selectedHand2Ds[i]->m_P);
					pt2DVec.push_back(selectedHand2Ds[i]->m_fingerLandmarkVect[ptIdx]);
				}

				vector<unsigned int> inliers;
				Mat X;
				triangulateWithOptimization(MVec,pt2DVec,X);
				double reproError = CalcReprojectionError(MVec,pt2DVec,X);

				//printf("inlierNum %d\n",inliers.size());
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
			reproErrorAvg/=fingerIdxVect.size();
			if(bFailed)
				continue;

			if(bestInlierNum <  selectedHand2Ds.size() && reproErrorAvg<final_repro_thresh)
			{
				bestInlierNum = selectedHand2Ds.size();
						
				for(int localIdx=0;localIdx<fingerIdxVect.size();++localIdx)
				{
					int ptIdx = fingerIdxVect[localIdx];

					currentHand3D.m_landmark3D[ptIdx] = reconPt3DVect[localIdx];						
					currentHand3D.m_fingerInfo[ptIdx].m_averageReproError = reproErrorAvg;

					currentHand3D.m_fingerInfo[ptIdx].m_visibility.clear();
					currentHand3D.m_fingerInfo[ptIdx].m_visibility.reserve(selectedHand2Ds.size());
					double avgDetectScore=0;
					for(int kk=0;kk<selectedHand2Ds.size();++kk)
					{
						currentHand3D.m_fingerInfo[ptIdx].m_visibility.push_back( selectedHand2Ds[kk]->m_camIdx);
						avgDetectScore+=selectedHand2Ds[kk]->m_detectionScore[ptIdx];
						//if(ptIdx==4 && selectedHand2Ds.front()->m_whichSide==HAND_LEFT)
							//printf(" (%d,%f) ",selectedHand2Ds[kk]->m_camIdx,selectedHand2Ds[kk]->m_detectionScore[ptIdx]);
					}
					//if(ptIdx==4 && selectedHand2Ds.front()->m_whichSide==HAND_LEFT)
						//printf("\n");
					currentHand3D.m_fingerInfo[ptIdx].m_averageScore = avgDetectScore/selectedHand2Ds.size();

					if(ptIdx==0)
					{
						currentHand3D.m_whichSide = selectedHand2Ds.front()->m_whichSide;
						currentHand3D.m_identityIdx = selectedHand2Ds.front()->m_subjectIdx;
					}
				}
			}
		}		//end of j
	}	//end of i
}
void HandPointsTriangulation(vector<vector<SHand2D> >& hand2DVect_multiCam,vector<vector<SHand2D*> >& hand2DGroupVect,vector<SHand3D>& handRecon,bool isHD)
{
	vector< vector<int> > fingerIdxVect;
	fingerIdxVect.resize(6);
	fingerIdxVect[0].push_back(0);
	for(int i=1;i<HAND_LANDMARK_NUM;++i)
	{
		int groupIdx = 0;
		if(i<5)
			groupIdx = 1;
		else if(i<9)
			groupIdx = 2;
		else if(i<13)
			groupIdx = 3;
		else if(i<17)
			groupIdx = 4;
		else
			groupIdx = 5;
		fingerIdxVect[groupIdx].push_back(i);
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
	vector<SHand3D> handReconInit;
	handReconInit.resize(hand2DGroupVect.size());

	//g_visData.m_debugPt2.clear();
	for(int gIdx=0;gIdx<hand2DGroupVect.size();++gIdx)
	{
		printf("\n\nGroupIdx %d\n",gIdx);
		vector<SHand2D*>& currentGroup =  hand2DGroupVect[gIdx];

		int landmarkNum = currentGroup.front()->m_fingerLandmarkVect.size();
		SHand3D& currentHand3D = handReconInit[gIdx];
		currentHand3D.m_landmark3D.resize(landmarkNum);
		currentHand3D.m_fingerInfo.resize(landmarkNum);
		//currentHand3D.avgReprError = 1e5;

		if(currentGroup.size()<minimum_vis_thresh)
			continue;

		#pragma omp parallel for
		for(int i=0;i<fingerIdxVect.size();++i)
		{	
			ransac(currentGroup,fingerIdxVect[i],currentHand3D);
		}
	}	//end of gIdx

	
	for(unsigned int i=0;i<handReconInit.size();++i)
	{
		if(handReconInit[i].m_landmark3D.size()>0)
			handRecon.push_back(handReconInit[i]);
	}
}

//output: two fingers (specified by fingerIdxVect) from lefthand and right hand
void ransac_twohands(vector<SHand2D*>& leftHand2ds,vector<SHand2D*>& rightHand2ds,vector<int>& fingerIdxVect
				,SHand3D& leftHand3D,SHand3D& rightHand3D,vector< vector<Point3d> >& alreadyReconPt3DVect
				,bool bFingerOrderFlip)		//assume index finger can be pinky
{
	double initPair_repro_thresh = 6;
	double relatedView_repro_thresh = 10;
	double final_repro_thresh = 6;
	int minimum_vis_thresh = 2;
	double per_point_conf_thresh = 0.2;
	
	/*//printf("\n\nGroupIdx %d\n",gIdx);
	//SHand3D& currentHand3D = handReconInit[gIdx];
	currentHand3D.m_landmark3D.clear();
	currentHand3D.m_fingerInfo.clear();
	currentHand3D.m_landmark3D.resize(HAND_LANDMARK_NUM);
	currentHand3D.m_fingerInfo.resize(HAND_LANDMARK_NUM);*/
	vector<SHand2D*> currentGroup;	//merged 2D detection results
	currentGroup = leftHand2ds;
	currentGroup.resize(leftHand2ds.size()+rightHand2ds.size());

	for(int i=0;i<rightHand2ds.size();++i)
	{
		currentGroup[leftHand2ds.size()+i] = rightHand2ds[i];
	}


	vector<SHand2D> fakeHands;
	if(bFingerOrderFlip)
	{
		fakeHands.resize(currentGroup.size());

		for(int i=0;i<currentGroup.size();++i)
		{
			fakeHands[i] = *currentGroup[i];		//just copy


			//flipFinger orders
			for(int l=5;l<=8;++l)		//swap, index and pinky
			{
				int other_l = l+12;
				Point2d buf = fakeHands[i].m_fingerLandmarkVect[l];
				fakeHands[i].m_fingerLandmarkVect[l] = fakeHands[i].m_fingerLandmarkVect[other_l];
				fakeHands[i].m_fingerLandmarkVect[other_l] = buf;

				double bufScore = fakeHands[i].m_detectionScore[l];
				fakeHands[i].m_detectionScore[l] = fakeHands[i].m_detectionScore[other_l];
				fakeHands[i].m_detectionScore[other_l] = bufScore;
			}

			for(int l=9;l<=12;++l)		//swap, 3rd and 4th fingers
			{
				int other_l = l+4;
				Point2d buf = fakeHands[i].m_fingerLandmarkVect[l];
				fakeHands[i].m_fingerLandmarkVect[l] = fakeHands[i].m_fingerLandmarkVect[other_l];
				fakeHands[i].m_fingerLandmarkVect[other_l] = buf;

				double bufScore = fakeHands[i].m_detectionScore[l];
				fakeHands[i].m_detectionScore[l] = fakeHands[i].m_detectionScore[other_l];
				fakeHands[i].m_detectionScore[other_l] = bufScore;
			}
		}
		currentGroup.reserve(currentGroup.size()*2);

		for(int i=0;i<fakeHands.size();++i)
		{
			currentGroup.push_back(&fakeHands[i]);
		}
	}

	if(currentGroup.size()<minimum_vis_thresh)
		return;

	vector<bool> bAlreadyUsed;
	bAlreadyUsed.resize(currentGroup.size(),false);
	//vector<Point3d> alreadyReconPt3DVect;
	for(int iterIdx=0;iterIdx<2;++iterIdx)
	{
		SHand3D* pCurrentHand3D;
		if(iterIdx==0)
			pCurrentHand3D = &leftHand3D;
		else
			pCurrentHand3D = &rightHand3D;
		int bestInlierNum =0;
		vector<int> bestCorresHand2DsIdx;		//to remember already used hand2ds
		vector<Point3d> bestReconPt3DVect;
		for(int i=0;i<currentGroup.size();++i)
		{
			if(bAlreadyUsed[i])
				continue;
			for(int j=i+1;j<currentGroup.size();++j)
			{
				if(bAlreadyUsed[j])
					continue;

				//printf("pair: %d, %d\n",i,j);
				double camBaseLine = Distance(currentGroup[i]->m_camCenter,currentGroup[j]->m_camCenter);
				//printf("baseline: %f\n",camBaseLine);
				if(cm2world(camBaseLine)<1e-3)
					continue;
				SHand3D candHand3D;

				//currentGroup[i],currentGroup[j]  are the candidate pair
				double reprErrSum =0;
				vector<Point3d> refPt3DVect;
				refPt3DVect.resize(fingerIdxVect.size());
				bool bFailed =false;
				for(int localIdx=0;localIdx<fingerIdxVect.size();++localIdx)
				{
					int ptIdx = fingerIdxVect[localIdx];
					if(currentGroup[i]->m_detectionScore[ptIdx]<per_point_conf_thresh || currentGroup[j]->m_detectionScore[ptIdx]<per_point_conf_thresh)
					{
						bFailed = true;
						break;
					}

					vector<Mat*> MVec; 
					vector<Point2d> pt2DVec;
				
					MVec.push_back(&currentGroup[i]->m_P);
					pt2DVec.push_back(currentGroup[i]->m_fingerLandmarkVect[ptIdx]);

					MVec.push_back(&currentGroup[j]->m_P);
					pt2DVec.push_back(currentGroup[j]->m_fingerLandmarkVect[ptIdx]);

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
				double reprErrAvg = reprErrSum/fingerIdxVect.size();

				//printf("InitPair: GroupIdx %d: reprError %f, baseline %f \n",gIdx,reprErrAvg,world2cm(camBaseLine));
				if(reprErrAvg>initPair_repro_thresh)
				{
					//printf("Rejected by initPairRepError: %f>%f\n",reprErrAvg,initPair_repro_thresh);
					continue;
				}

				//Find corresponding face 2Ds
				vector<SHand2D*> selectedHand2Ds;
				vector<int> selectedHand2DsIdx;
				for(int i=0;i<currentGroup.size();++i)
				{
					if(bAlreadyUsed[i])
						continue;
					double avg2DError =0;
					bool bFailed = false;
					for(int localIdx=0;localIdx<fingerIdxVect.size();++localIdx)
					{
						int ptIdx = fingerIdxVect[localIdx];
						Point2d projected2D =  Project3DPt(refPt3DVect[localIdx],currentGroup[i]->m_P);
						double dist2D = Distance(projected2D,currentGroup[i]->m_fingerLandmarkVect[ptIdx]);
						avg2DError += dist2D;
						if(currentGroup[i]->m_detectionScore[ptIdx]<per_point_conf_thresh)
						{
							bFailed = true;
							break;
						}
					}
					if(bFailed)
						continue;
					avg2DError /=fingerIdxVect.size();
				
					if(avg2DError<relatedView_repro_thresh)
					{
						selectedHand2Ds.push_back(currentGroup[i]);
						selectedHand2DsIdx.push_back(i);
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
				reconPt3DVect.resize(fingerIdxVect.size());
				bFailed = false;
				for(int localIdx=0;localIdx<fingerIdxVect.size();++localIdx)
				{
					int ptIdx = fingerIdxVect[localIdx];

					vector<Mat*> MVec; 
					vector<Point2d> pt2DVec;
					for(int i=0;i<selectedHand2Ds.size();++i)
					{
						//Calculate reprojection error
						MVec.push_back(&selectedHand2Ds[i]->m_P);
						pt2DVec.push_back(selectedHand2Ds[i]->m_fingerLandmarkVect[ptIdx]);
					}

					vector<unsigned int> inliers;
					Mat X;
					triangulateWithOptimization(MVec,pt2DVec,X);
					double reproError = CalcReprojectionError(MVec,pt2DVec,X);

					//printf("inlierNum %d\n",inliers.size());
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
				reproErrorAvg/=fingerIdxVect.size();
				if(bFailed)
					continue;


				//Check whether this is the same 3D finger as iterIdx cases
				if(iterIdx==1)
				{
					bool bSkip =false;
					for(int kk=0;kk<alreadyReconPt3DVect.size();++kk)
					{
						double distAvg=0;
						for(int k=0;k<alreadyReconPt3DVect[kk].size();++k)
						{
							distAvg += Distance(reconPt3DVect[k],alreadyReconPt3DVect[kk][k]);
						}
						distAvg/=alreadyReconPt3DVect[kk].size();
						if(distAvg<5)
						{
							//printf("!!!!!!!!!! Distance from the pre-reconstructed finger: %f\n",distAvg);
							bSkip = true;
							break;
						}
					}
					if(bSkip)
						continue;
					
				}


				if(bestInlierNum <  selectedHand2Ds.size() && reproErrorAvg<final_repro_thresh)
				{
					bestInlierNum = selectedHand2Ds.size();
						
					for(int localIdx=0;localIdx<fingerIdxVect.size();++localIdx)
					{
						int ptIdx = fingerIdxVect[localIdx];

						pCurrentHand3D->m_landmark3D[ptIdx] = reconPt3DVect[localIdx];						
						pCurrentHand3D->m_fingerInfo[ptIdx].m_averageReproError = reproErrorAvg;

						pCurrentHand3D->m_fingerInfo[ptIdx].m_visibility.clear();
						pCurrentHand3D->m_fingerInfo[ptIdx].m_visibility.reserve(selectedHand2Ds.size());
						double avgDetectScore=0;
						for(int kk=0;kk<selectedHand2Ds.size();++kk)
						{
							pCurrentHand3D->m_fingerInfo[ptIdx].m_visibility.push_back( selectedHand2Ds[kk]->m_camIdx);
							avgDetectScore+=selectedHand2Ds[kk]->m_detectionScore[ptIdx];
							//if(ptIdx==4 && selectedHand2Ds.front()->m_whichSide==HAND_LEFT)
								//printf(" (%d,%f) ",selectedHand2Ds[kk]->m_camIdx,selectedHand2Ds[kk]->m_detectionScore[ptIdx]);
						}
						//if(ptIdx==4 && selectedHand2Ds.front()->m_whichSide==HAND_LEFT)
							//printf("\n");
						pCurrentHand3D->m_fingerInfo[ptIdx].m_averageScore = avgDetectScore/selectedHand2Ds.size();

						if(ptIdx==0)
						{
							//pCurrentHand3D->m_whichSide = selectedHand2Ds.front()->m_whichSide;
							pCurrentHand3D->m_identityIdx = selectedHand2Ds.front()->m_subjectIdx;
						}
					}

					//remember for next finger iteration
					bestCorresHand2DsIdx = selectedHand2DsIdx;
					bestReconPt3DVect = reconPt3DVect;
				}
			}		//end of j
		}	//end of i

		if(iterIdx==0)
		{
			for(int i=0;i<bestCorresHand2DsIdx.size();++i)
				bAlreadyUsed[bestCorresHand2DsIdx[i]] = true;
		}
		alreadyReconPt3DVect.push_back(bestReconPt3DVect);

	}  //end of iterIdx
}


//Consider both hands' detection result togeter and generate best both hands
void HandPointsTriangulation_interference(vector<vector<SHand2D> >& hand2DVect_multiCam,vector<vector<SHand2D*> >& hand2DGroupVect,vector<SHand3D>& handRecon,bool isHD)
{
	vector< vector<int> > fingerIdxVect;
	//fingerIdxVect.resize(6);
	fingerIdxVect.resize(3);
	fingerIdxVect[0].push_back(0);
	for(int i=1;i<HAND_LANDMARK_NUM;++i)
	{
		int groupIdx = 0;
		if(i<5)
			groupIdx = 1;
		else
			groupIdx = 2;
		/*if(i<5)
			groupIdx = 1;
		else if(i<9)
			groupIdx = 2;
		else if(i<13)
			groupIdx = 3;
		else if(i<17)
			groupIdx = 4;
		else
			groupIdx = 5;*/
		fingerIdxVect[groupIdx].push_back(i);
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
	vector<SHand3D> handReconInit;
	handReconInit.resize(hand2DGroupVect.size());

	//Find corresponding pairs
	vector< pair<int,int> > handPairs;		//gIdx, gIdx
	vector<bool> checker;
	checker.resize(hand2DGroupVect.size(),false);
	for(int gIdx=0;gIdx<hand2DGroupVect.size();++gIdx)
	{
		if(checker[gIdx])
			continue;
		vector<SHand2D*>& currentGroup =  hand2DGroupVect[gIdx];
		int curSubjectIdx = currentGroup.front()->m_subjectIdx;

		//find corresponding hands
		int otherHandIdx =-1;
		for(int i=gIdx+1;i<hand2DGroupVect.size();++i)
		{
			if(curSubjectIdx == hand2DGroupVect[i].front()->m_subjectIdx)
			{
				otherHandIdx = i;
				break;
			}
		}
		if(otherHandIdx>=0)
		{
			checker[otherHandIdx] = true;
			checker[gIdx] = true;
			if(currentGroup.front()->m_whichSide==HAND_LEFT)
				handPairs.push_back(make_pair(gIdx,otherHandIdx));
			else
				handPairs.push_back(make_pair(otherHandIdx,gIdx));
		}
		else
		{
			checker[curSubjectIdx] = true;
			if(currentGroup.front()->m_whichSide==HAND_LEFT)
				handPairs.push_back(make_pair(gIdx,-1));
			else
				handPairs.push_back(make_pair(-1,gIdx));
		}
	}

	for(int pairIdx = 0; pairIdx<handPairs.size(); ++pairIdx)
	{

		if(handPairs[pairIdx].first>=0 && handPairs[pairIdx].second>=0)
		{
			int leftHandIdx = handPairs[pairIdx].first;
			int rightHandIdx = handPairs[pairIdx].second;

			vector<SHand2D*>* pLeftHand2Ds = NULL;
			vector<SHand2D*>* pRightHand2Ds = NULL;
			if(hand2DGroupVect[leftHandIdx].size()>=minimum_vis_thresh)
			{
				int landmarkNum = hand2DGroupVect[leftHandIdx].front()->m_fingerLandmarkVect.size();
				SHand3D& currentHand3D = handReconInit[leftHandIdx];
				currentHand3D.m_whichSide = HAND_LEFT;
				currentHand3D.m_landmark3D.resize(landmarkNum);
				currentHand3D.m_fingerInfo.resize(landmarkNum);
				pLeftHand2Ds = &hand2DGroupVect[leftHandIdx];
			}

			if(hand2DGroupVect[rightHandIdx].size()>=minimum_vis_thresh)
			{
				int landmarkNum = hand2DGroupVect[rightHandIdx].front()->m_fingerLandmarkVect.size();
				SHand3D& currentHand3D = handReconInit[rightHandIdx];
				currentHand3D.m_whichSide = HAND_RIGHT;
				currentHand3D.m_landmark3D.resize(landmarkNum);
				currentHand3D.m_fingerInfo.resize(landmarkNum);
				pRightHand2Ds = &hand2DGroupVect[rightHandIdx];
			}

			if(pLeftHand2Ds!=NULL && pRightHand2Ds!=NULL)
			{
				vector< vector<Point3d> > alreadyReconPt3DVect_index;
				#pragma omp parallel for
				for(int i=0;i<fingerIdxVect.size();++i)
				{	
					vector< vector<Point3d> > alreadyReconPt3DVect;
				//	if(i==5)
					//	alreadyReconPt3DVect = alreadyReconPt3DVect_index;
					bool bDoFingerFlip =false;
					ransac_twohands(*pLeftHand2Ds,*pRightHand2Ds,fingerIdxVect[i],handReconInit[leftHandIdx],handReconInit[rightHandIdx],alreadyReconPt3DVect,bDoFingerFlip);
					//printf("alreadyReconPt3DVect: %d\n",alreadyReconPt3DVect.size());
					//if(i==2)
						//alreadyReconPt3DVect_index = alreadyReconPt3DVect;
				}

				//Rearrangement fingers to build two hands

				Point3d leftWrist = handReconInit[leftHandIdx].m_landmark3D[0];
				Point3d rightWrist = handReconInit[rightHandIdx].m_landmark3D[0];
				for(int i=1;i<fingerIdxVect.size();++i)
				{
					bool bDoSwap = false;
					//check angle consistency
					Point3d leftPt_1 = handReconInit[leftHandIdx].m_landmark3D[fingerIdxVect[i][0]];
					Point3d leftPt_2 = handReconInit[leftHandIdx].m_landmark3D[fingerIdxVect[i][1]];

					Point3d rightPt_1 = handReconInit[rightHandIdx].m_landmark3D[fingerIdxVect[i][0]];
					Point3d rightPt_2 = handReconInit[rightHandIdx].m_landmark3D[fingerIdxVect[i][1]];
					

					//Compute current score
					Point3d leftVect1 = leftPt_1 - leftWrist;
					Point3d leftVect2 = leftPt_2 - leftPt_1;
					Normalize(leftVect1);
					Normalize(leftVect2);
					Point3d rightVect1 = rightPt_1 - rightWrist;
					Point3d rightVect2 = rightPt_2 - rightPt_1;
					Normalize(rightVect1);
					Normalize(rightVect2);

					double currrentScore = leftVect1.dot(leftVect2) + rightVect1.dot(rightVect2);

					//compute swapped scor
					leftVect1 = leftPt_1 - rightWrist;
					Normalize(leftVect1);
					rightVect1 = rightPt_1 - leftWrist;
					Normalize(rightVect1);
					double swappedScore = leftVect1.dot(leftVect2) + rightVect1.dot(rightVect2);
					printf("%d : %f vs %f\n",i, currrentScore, swappedScore);
					if(swappedScore>currrentScore)
						bDoSwap=true;

					if(bDoSwap)
					{
						printf("%d: do swapping\n",i);
						for(int localIdx=0; localIdx<fingerIdxVect[i].size(); ++localIdx)
						{
							int ptIdx = fingerIdxVect[i][localIdx];;
							Point3d buf = handReconInit[rightHandIdx].m_landmark3D[ptIdx];
							handReconInit[rightHandIdx].m_landmark3D[ptIdx] = handReconInit[leftHandIdx].m_landmark3D[ptIdx];
							handReconInit[leftHandIdx].m_landmark3D[ptIdx]=buf;

							SFingerReconInfo bufInfo = handReconInit[rightHandIdx].m_fingerInfo[ptIdx];
							handReconInit[rightHandIdx].m_fingerInfo[ptIdx] = handReconInit[leftHandIdx].m_fingerInfo[ptIdx];
							handReconInit[leftHandIdx].m_fingerInfo[ptIdx]=bufInfo;
						}
					}
				}
			}
			else if(pLeftHand2Ds!=NULL  && pRightHand2Ds==NULL)
			{
				#pragma omp parallel for
				for(int i=0;i<fingerIdxVect.size();++i)
				{	
					ransac(*pLeftHand2Ds,fingerIdxVect[i],handReconInit[leftHandIdx]);
				}
			}
			else if(pLeftHand2Ds==NULL  && pRightHand2Ds!=NULL)
			{
				#pragma omp parallel for
				for(int i=0;i<fingerIdxVect.size();++i)
				{	
					ransac(*pRightHand2Ds,fingerIdxVect[i],handReconInit[rightHandIdx]);
				}
			}
		}
		else
		{
			int gIdx;
			if(handPairs[pairIdx].first)
				gIdx = handPairs[pairIdx].first;
			else
				gIdx = handPairs[pairIdx].second;
			vector<SHand2D*>& currentGroup =  hand2DGroupVect[gIdx];
			int landmarkNum = currentGroup.front()->m_fingerLandmarkVect.size();
			SHand3D& currentHand3D = handReconInit[gIdx];
			currentHand3D.m_landmark3D.resize(landmarkNum);
			currentHand3D.m_fingerInfo.resize(landmarkNum);

			if(currentGroup.size()<minimum_vis_thresh)
				continue;

			#pragma omp parallel for num_threads(10)
			for(int i=0;i<fingerIdxVect.size();++i)
			{	
				ransac(currentGroup,fingerIdxVect[i],currentHand3D);
			}
		}
		
	}	//end of gIdx
	/*
	//g_visData.m_debugPt2.clear();
	for(int gIdx=0;gIdx<hand2DGroupVect.size();++gIdx)
	{
		printf("\n\nGroupIdx %d\n",gIdx);
		vector<SHand2D*>& currentGroup =  hand2DGroupVect[gIdx];

		int landmarkNum = currentGroup.front()->m_fingerLandmarkVect.size();
		SHand3D& currentHand3D = handReconInit[gIdx];
		currentHand3D.m_landmark3D.resize(landmarkNum);
		currentHand3D.m_fingerInfo.resize(landmarkNum);
		//currentHand3D.avgReprError = 1e5;

		if(currentGroup.size()<minimum_vis_thresh)
			continue;

		#pragma omp parallel for
		for(int i=0;i<fingerIdxVect.size();++i)
		{	
			ransac(currentGroup,fingerIdxVect[i],currentHand3D);
		}
	}	//end of gIdx
	*/
	
	for(unsigned int i=0;i<handReconInit.size();++i)
	{
		if(handReconInit[i].m_landmark3D.size()>0)
		{
			handRecon.push_back(handReconInit[i]);
		}
	}
}
//Input: domeImageManager should contains all the information (camLabels, image, calibration)
void CHandRecon::Finger_Landmark_Reconstruction_Voting_hd(const char* handDetectFilePath, CDomeImageManager& domeImageManager,std::vector<SHand3D>& handReconResult)
{
	//Load finger detection result
	vector<vector<SHand2D> > hand2DVect_multiCam;
	bool bLoadingSuccess = LoadFingerDetectResult_MultiCams_Json(handDetectFilePath,domeImageManager.GetFrameIdx(),domeImageManager.m_domeViews,hand2DVect_multiCam,true);
	if(bLoadingSuccess ==false)
	{
		printf("## ERROR: cannot load hand detection results\n");
		return;
	}
	
	//Grouping
	vector<vector<SHand2D*> > hand2DGroupVect;
	vector<vector<SHand2D*> > detectedHandVectRemain;			//outer for each view, inner for each hand in a view
	for(int camIdx=0;camIdx<domeImageManager.GetCameraNum();++camIdx)
	{
		vector<SHand2D*> handVect;
		for(int f=0;f<hand2DVect_multiCam[camIdx].size();++f)
		{
			handVect.push_back(&hand2DVect_multiCam[camIdx][f]);		//used for grouping
		}
		detectedHandVectRemain.push_back(handVect);
	}
	
	bool bUpdated = false; 
	while(true)
	{
		vector<vector<SHand2D*> > remains_updated;			//outer for each view, inner for each hand in a view
		remains_updated.resize(detectedHandVectRemain.size());		//detectedHandVectRemain.size()==number of camera
		vector<SHand2D*> relatedHand2Ds;
		for(int camIdx=0;camIdx<detectedHandVectRemain.size();++camIdx)
		{
			for(int hIdx =0; hIdx < detectedHandVectRemain[camIdx].size();hIdx++)
			{
				if(relatedHand2Ds.size()==0)
					relatedHand2Ds.push_back(detectedHandVectRemain[camIdx][hIdx]);
				else
				{
					if(relatedHand2Ds.front()->m_subjectIdx == detectedHandVectRemain[camIdx][hIdx]->m_subjectIdx
						&& relatedHand2Ds.front()->m_whichSide == detectedHandVectRemain[camIdx][hIdx]->m_whichSide)
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
			detectedHandVectRemain = remains_updated;  //bug fixed
			continue;		//ignore this
		}

		hand2DGroupVect.push_back(relatedHand2Ds);
		detectedHandVectRemain = remains_updated;
	}

	printf("## Hand Triangulation: found %d handGroups\n",hand2DGroupVect.size());
	HandPointsTriangulation(hand2DVect_multiCam, hand2DGroupVect,handReconResult,true);
}

CHandRecon::CHandRecon(void)
{
	m_curImgFrameIdx = m_currentSelectedLocalVectIdx = -1;
	m_vis_avgDetectThresh = 0.0;
	m_vis_reproErrorThresh = 1e5;
	m_vis_visibilityThresh =3;


	//Finger Group
	// 0 
	// Thumb: 1-4
	// Index: 5-8
	// 3rd: 9-12
	// 4th: 13-16
	// pinky: 17-20
	m_fingerGroupIdxVect.resize(6);
	m_fingerGroupIdxVect[0].push_back(0);
	for(int i=1;i<HAND_LANDMARK_NUM;++i)
	{
		int groupIdx = 0;
		if(i<5)
			groupIdx = 1;
		else if(i<9)
			groupIdx = 2;
		else if(i<13)
			groupIdx = 3;
		else if(i<17)
			groupIdx = 4;
		else
			groupIdx = 5;
		m_fingerGroupIdxVect[groupIdx].push_back(i);
	}

	m_fpsType = FPS_HD_30; //default
}

// void CHandRecon::SetCurMemIdxByFrameIdx(int curImgFrameIdx)
// {
// 	m_curImgFrameIdx = curImgFrameIdx;

// 	if(m_handReconMem.size()==0)
// 	{
// 		m_currentSelectedLocalVectIdx = -1;
// 	}
// 	else
// 	{
// 		int initFrameIdx = m_handReconMem.front().frameIdx;
// 		int targetIdx = curImgFrameIdx - initFrameIdx;
// 		m_currentSelectedLocalVectIdx = targetIdx;	//it could be nagative number. But, doenn't matter. Show function will take care of this.
// 	}
// }

// void CHandRecon::ShowHandRecon3DResult(VisualizedData& vis, vector<SHand3D>& handRecon, bool bCompareData)
// {
// 	//float normalLength = cm2world(FACE_NORMAL_LENGTH);

// 	g_drawLock.Lock();
// 	//Draw Face Detection Result
// 	//vis.m_faceLandmarks.clear();
// 	//vis.m_faceNormal.clear();
// 	Point3d red_p3d(1,0,0);
// 	Point3d blue_p3d(0,0,1);
// 	Point3d yellow_p3d(1,1,0);
// 	Point3d white_p3d(1,1,1);
// 	Point3d cyan_p3d(0,1,1);

// 	for(unsigned int i=0;i<handRecon.size();++i)
// 	{

// 		if(handRecon[i].m_identityIdx==g_noVisSubject)
// 		{
// 			printf("Hand: Skip: %d\n",handRecon[i].m_identityIdx);
// 			continue;
// 		}
// 		if(handRecon[i].m_identityIdx==g_noVisSubject2)
// 		{
// 			printf("Hand: Skip: %d\n",handRecon[i].m_identityIdx);
// 			continue;
// 		}

// 		Point3d tempColor;
// 		/*if(faceRecon[i].identityIdx>=0)
// 		{
// 			int colorIdx = faceRecon[i].identityIdx % 7;
// 			tempColor = Point3d(face_color[colorIdx][0],face_color[colorIdx][1],face_color[colorIdx][2]);
// 		}
// 		else*/
// 		if(g_dispOptions.m_bIgnoreOutlierHands && handRecon[i].m_bValid==false)
// 			continue;

// 		if(handRecon[i].m_bValid==false)		//there is any rejected pt
// 			tempColor = white_p3d;
// 		else if(handRecon[i].m_whichSide ==HAND_LEFT)
// 			tempColor = blue_p3d;
// 		else
// 			tempColor = red_p3d;

// 		if(bCompareData)
// 			tempColor = cyan_p3d;

// 		for(unsigned int j=0;j<handRecon[i].m_landmark3D.size();++j)
// 		{
			
// 			if(handRecon[i].m_bValid)
// 				vis.m_handLandmarks.push_back(make_pair(handRecon[i].m_landmark3D[j],tempColor));
// 			else 
// 			{
// 				if(handRecon[i].m_fingerInfo[j].m_bValid)
// 					vis.m_handLandmarks.push_back(make_pair(handRecon[i].m_landmark3D[j],tempColor));
// 				else
// 					vis.m_handLandmarks.push_back(make_pair(handRecon[i].m_landmark3D[j],yellow_p3d));
// 			}

// 			if(abs(handRecon[i].m_fingerInfo[j].m_averageScore) >1.1)
// 				vis.m_handLandmarks.back().first.y =0;

// 			if(handRecon[i].m_fingerInfo[j].m_averageScore < m_vis_avgDetectThresh)
// 				vis.m_handLandmarks.back().first.y =0;

// 			if(handRecon[i].m_fingerInfo[j].m_averageReproError >m_vis_reproErrorThresh)
// 				vis.m_handLandmarks.back().first.y =0;

// 			if(handRecon[i].m_fingerInfo[j].m_visibility.size()<=m_vis_visibilityThresh)
// 				vis.m_handLandmarks.back().first.y =0;
// 		}
// 	/*	vis.m_faceNormal.push_back(make_pair(faceRecon[i].centerPt3D,tempColor));
// 		Point3d normalArrowEnd = faceRecon[i].centerPt3D + normalLength*faceRecon[i].faceNormal;
// 		vis.m_faceNormal.push_back(make_pair(normalArrowEnd,tempColor));*/


// 		if(g_dispOptions.m_showHandNormal)
// 		{
// 			double normalLength = 15;

// 			if(handRecon[i].m_normal.x!=0)
// 			{
// 				Point3d& palmCenter = handRecon[i].m_palmCenter;
// 				vis.m_handNormal.push_back(make_pair(palmCenter,g_cyan_p3d));
// 				Point3d normalArrowEnd = palmCenter + normalLength* handRecon[i].m_normal;
// 				vis.m_handNormal.push_back(make_pair(normalArrowEnd,g_cyan_p3d));
// 			}
// 		}
// 	}


// 	//Hand cam parameter
// 	if(handRecon.size()>0 && handRecon.front().m_normal.x!=0)
// 	{
// 		int subjectIdx = g_visData.m_selectedHandIdx.first;
		
// 		int whichSide;
// 		if(g_visData.m_selectedHandIdx.second==0)
// 			whichSide= HAND_LEFT;
// 		else 
// 			whichSide= HAND_RIGHT;
// 		int handIdx=-1;
// 		for(int i=0;i<handRecon.size();++i)
// 		{
// 			if(handRecon[i].m_identityIdx == subjectIdx && handRecon[i].m_whichSide == whichSide)
// 			{
// 				handIdx= i;
// 				break;
// 			}
// 		}
// 		if(handIdx>=0)
// 		{
// 			Mat_<double> R = Mat_<double>::eye(3,3);
// 			Mat_<double> t = Mat_<double>::zeros(3,1);

// 			R(0,0) = handRecon[handIdx].m_palmX.x;
// 			R(1,0) = handRecon[handIdx].m_palmX.y;
// 			R(2,0) = handRecon[handIdx].m_palmX.z;

// 			R(0,1) = -handRecon[handIdx].m_palmUp.x;
// 			R(1,1) = -handRecon[handIdx].m_palmUp.y;
// 			R(2,1) = -handRecon[handIdx].m_palmUp.z;

// 			R(0,2) = handRecon[handIdx].m_normal.x;
// 			R(1,2) = handRecon[handIdx].m_normal.y;
// 			R(2,2) = handRecon[handIdx].m_normal.z;
// 			R = R.inv();

// 			Mat pt;
// 			Point3d camCenter = handRecon[handIdx].m_palmCenter - handRecon[handIdx].m_normal*50;
// 			Point3dToMat4by1(camCenter,pt);
// 			t = -R*pt.rowRange(0,3);

// 			g_visData.m_hand_modelViewMatGL[0] = R.at<double>(0,0);
// 			g_visData.m_hand_modelViewMatGL[1] = R.at<double>(1,0);
// 			g_visData.m_hand_modelViewMatGL[2] = R.at<double>(2,0);
// 			g_visData.m_hand_modelViewMatGL[3] = 0;
// 			g_visData.m_hand_modelViewMatGL[4] = R.at<double>(0,1);
// 			g_visData.m_hand_modelViewMatGL[5] = R.at<double>(1,1);
// 			g_visData.m_hand_modelViewMatGL[6] = R.at<double>(2,1);
// 			g_visData.m_hand_modelViewMatGL[7] = 0;
// 			g_visData.m_hand_modelViewMatGL[8] = R.at<double>(0,2);
// 			g_visData.m_hand_modelViewMatGL[9] = R.at<double>(1,2);
// 			g_visData.m_hand_modelViewMatGL[10] = R.at<double>(2,2);
// 			g_visData.m_hand_modelViewMatGL[11] = 0;
// 			g_visData.m_hand_modelViewMatGL[12] = t.at<double>(0,0); //4th col
// 			g_visData.m_hand_modelViewMatGL[13] = t.at<double>(1,0);
// 			g_visData.m_hand_modelViewMatGL[14] = t.at<double>(2,0);
// 			g_visData.m_hand_modelViewMatGL[15] = 1;
// 		}
// 	}

// 	g_drawLock.Unlock();
// }

// void CHandRecon::ShowHandReconResult(VisualizedData& vis,bool bCompareData=false)
// {
// 	//visualize visualHull

// 	if(m_currentSelectedLocalVectIdx>=0 && m_currentSelectedLocalVectIdx<m_handReconMem.size())
// 	{
// 		if(m_handReconMem[m_currentSelectedLocalVectIdx].handReconVect.size()>0 && m_handReconMem[m_currentSelectedLocalVectIdx].handReconVect.front().m_landmark3D.size()>0)
// 		{
// 			ShowHandRecon3DResult(vis,m_handReconMem[m_currentSelectedLocalVectIdx].handReconVect,bCompareData);
// 		}
// 	}
// }

//Save Face
void CHandRecon::SaveHandReconResult(const char* folderPath,vector<SHand3D>& handRecon,int frameIdx,bool bIsHD)
{
	/*//Save Face Reconstruction Result
	char folderPath[512];
	sprintf(folderPath,"%s/faceRecon_%d",g_dataMainFolder,g_askedVGACamNum);
	CreateFolder(folderPath);
	*/

	char fullPath[512];
	if(bIsHD ==false)
		sprintf(fullPath,"%s/handRecon3D_%08d.txt",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
	else
		sprintf(fullPath,"%s/handRecon3D_hd%08d.txt",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
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
		if(handRecon[i].m_whichSide==HAND_LEFT)
			fout <<"l ";
		else
			fout <<"r ";
		fout <<handRecon[i].m_landmark3D.size() <<" ";
		for(unsigned int j=0;j<handRecon[i].m_landmark3D.size();++j)
		{
			fout << handRecon[i].m_landmark3D[j].x << " " << handRecon[i].m_landmark3D[j].y << " " << handRecon[i].m_landmark3D[j].z << " ";
		}
		fout <<"\n";
		for(unsigned int j=0;j<handRecon[i].m_fingerInfo.size();++j)
		{
			fout << handRecon[i].m_fingerInfo[j].m_averageScore << " " << handRecon[i].m_fingerInfo[j].m_averageReproError << " ";
			fout << handRecon[i].m_fingerInfo[j].m_visibility.size() <<" ";
			for(unsigned int k=0;k<handRecon[i].m_fingerInfo[j].m_visibility.size();++k)
			{
				fout<<handRecon[i].m_fingerInfo[j].m_visibility[k]<< " ";
			}
			fout<<"\n";
		}
		fout <<"\n";
	}
	fout.close();
}


//Save Face
void SaveHandReconResult_json_singeSubject(ofstream& fout,SHand3D& handRecon)
{
	fout <<"{";
	//fout <<handRecon[i].m_landmark3D.size() <<" ";
	fout <<"\"landmarks\": [";
	for(unsigned int j=0;j<handRecon.m_landmark3D.size();++j)
	{
		//fout << handRecon.m_landmark3D[j].x << " " << handRecon.m_landmark3D[j].y << " " << handRecon.m_landmark3D[j].z << " ";
		fout << handRecon.m_landmark3D[j].x << ", " << handRecon.m_landmark3D[j].y << ", " << handRecon.m_landmark3D[j].z;//
			
		if(j+1!=handRecon.m_landmark3D.size())
			fout<< ", ";
	}
	fout <<"],\n";
	fout <<"\"averageScore\": [";
	for(unsigned int j=0;j<handRecon.m_fingerInfo.size();++j)
	{
		fout << handRecon.m_fingerInfo[j].m_averageScore;// << ", ";
		if(j+1!=handRecon.m_fingerInfo.size())
			fout << ", ";
	}
	fout <<"],\n";

	fout <<"\"averageReproError\": [";
	for(unsigned int j=0;j<handRecon.m_fingerInfo.size();++j)
	{
		fout << handRecon.m_fingerInfo[j].m_averageReproError;//
		if(j+1!=handRecon.m_fingerInfo.size())
			fout << ", ";
	}
	fout <<"],\n";

	fout <<"\"validity\": [";
	for(unsigned int j=0;j<handRecon.m_fingerInfo.size();++j)
	{
		fout << handRecon.m_fingerInfo[j].m_bValid;//
		if(j+1!=handRecon.m_fingerInfo.size())
			fout << ", ";
	}
	fout <<"],\n";


	fout <<"\"visibility\": [";
	for(unsigned int j=0;j<handRecon.m_fingerInfo.size();++j)
	{
		fout <<"[";

	//	fout << handRecon.m_fingerInfo[j].m_visibility.size() <<" ";
		for(unsigned int k=0;k<handRecon.m_fingerInfo[j].m_visibility.size();++k)
		{
			fout<<handRecon.m_fingerInfo[j].m_visibility[k];//<< ", ";
			if(k+1!=handRecon.m_fingerInfo[j].m_visibility.size())
				fout << ", ";
		}
		if(j+1!=handRecon.m_fingerInfo.size())
			fout <<"],";
		else
			fout <<"]";
	}
	fout <<"\n] }";
}

void CHandRecon::SaveHandReconResult_json(const char* folderPath,vector<SHand3D>& handRecon,int frameIdx,bool bIsHD)
{
	//Finding pairs
	vector< pair<SHand3D*,SHand3D*> > peopleVect;
	vector<bool> checker(handRecon.size(),false);

	for(int i=0;i<handRecon.size();++i)
	{
		if(checker[i]==true)
			continue;

		//select current on
		checker[i] = true;	//taken
		peopleVect.resize(peopleVect.size()+1);
		peopleVect.back().first = &handRecon[i];
		peopleVect.back().second = NULL;

		int humanIdx = handRecon[i].m_identityIdx;

		for(int j=i+1;j<handRecon.size();++j)
		{
			if(checker[j]==true)
				continue;

			if(humanIdx == handRecon[j].m_identityIdx)
			{
				peopleVect.back().second  = &handRecon[j];
				checker[j] = true;	//taken
				break;
			}
		}
	}

	char fullPath[512];
	if(bIsHD ==false)
		sprintf(fullPath,"%s/handRecon3D_%08d.json",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
	else
		sprintf(fullPath,"%s/handRecon3D_hd%08d.json",folderPath,frameIdx);//m_domeImageManager.m_currentFrame);
	if(frameIdx%100==0)
		printf("Save to %s\n",fullPath);
	ofstream fout(fullPath,std::ios_base::trunc);

	char logPath[512];
	sprintf(logPath,"%s/lrSwitchedFrames.txt",folderPath);//m_domeImageManager.m_currentFrame);
	ofstream logOut(logPath,std::ios_base::app);
	fout << "{ \"version\": 0.5, \n";		//ver 0.5: added outlier infor
	//fout << handRecon.size() <<"\n";

	fout << "\"people\" :\n";
	fout << "[";

	for(int p=0;p<peopleVect.size();++p)
	{
		fout <<"\n{ \"id\": "<< peopleVect[p].first->m_identityIdx <<",\n";

		SHand3D* pLeft;
		SHand3D* pRight;
		if(peopleVect[p].first->m_whichSide == HAND_LEFT)
		{
			pLeft = peopleVect[p].first;
			pRight = peopleVect[p].second;
		}
		else
		{
			printf("Frame: %d : buggy frame\n",frameIdx);
			logOut << "Frame: " <<frameIdx <<"\n";
			pLeft = peopleVect[p].second;
			pRight = peopleVect[p].first;
		}

		if(pLeft!=NULL)
		{
			fout <<"\"left_hand\": \n";
			SaveHandReconResult_json_singeSubject(fout,*pLeft);
		}
		if(pLeft!=NULL && pRight!=NULL)
			fout <<",\n";
		if(pRight!=NULL)
		{
			fout <<"\"right_hand\": \n";
			SaveHandReconResult_json_singeSubject(fout,*pRight);
		}
		if(p+1!=peopleVect.size())
			fout <<"},\n";
		else
			fout <<"}\n";
	}


	fout <<"\n] \n}";
	fout.close();
	logOut.close();
}

//ver 0.2
void CHandRecon::LoadHand3DByFrame(const char* folderPath,const int firstFrameIdx,const int frameNum,bool bIsHD)
{ 
	m_handReconMem.clear();
	for(int f=0;f<frameNum;++f)
	{
		//Save Face Reconstruction Result
		char fullPath[512];
		if(bIsHD==false)
			sprintf(fullPath,"%s/handRecon3D_%08d.txt",folderPath,firstFrameIdx + f);
		else
			sprintf(fullPath,"%s/handRecon3D_hd%08d.txt",folderPath,firstFrameIdx + f);
		if(f%500==0)
			printf("Load FaceReconResult from %s\n",fullPath);	
		ifstream fin(fullPath, ios::in);
		char dummy[512];
		if(fin)
		{
			m_handReconMem.resize(m_handReconMem.size()+1);
			SHandReconMemory& newMem = m_handReconMem.back();
			newMem.frameIdx = firstFrameIdx + f;

			float ver;
			fin >> dummy >> ver; //version			
			int handNum;
			fin >>handNum;
			//printf("Face Num: %d\n",faceNum);
			newMem.handReconVect.resize(handNum);
			for(int i=0;i<handNum;++i)
			{
				SHand3D currentHand;

				if(ver>0.15)
				{
					int subjectIdx;
					fin >>subjectIdx;
					char whichSide;
					fin >>whichSide;
					newMem.handReconVect[i].m_identityIdx = subjectIdx;
					if(whichSide=='l')
						newMem.handReconVect[i].m_whichSide = HAND_LEFT;
					else
						newMem.handReconVect[i].m_whichSide = HAND_RIGHT;
				}

				int landmarkNum;
				fin >>landmarkNum;
				Point3d pt;
				newMem.handReconVect[i].m_landmark3D.resize(landmarkNum);
				for(int j=0;j<landmarkNum;++j)
				{
					fin >>pt.x >> pt.y >> pt.z;
					newMem.handReconVect[i].m_landmark3D[j] = pt;
				}
				newMem.handReconVect[i].m_fingerInfo.resize(landmarkNum);
				if(ver>0.25)
				{
					for(int j=0;j<landmarkNum;++j)
					{
						fin >> newMem.handReconVect[i].m_fingerInfo[j].m_averageScore >> newMem.handReconVect[i].m_fingerInfo[j].m_averageReproError;
						int visibilityNum;
						fin >> visibilityNum;
						newMem.handReconVect[i].m_fingerInfo[j].m_visibility.resize(visibilityNum);
						int camIdx;
						for(int k=0;k<visibilityNum;++k)
						{
							fin >> camIdx;
							newMem.handReconVect[i].m_fingerInfo[j].m_visibility[k] = camIdx;
						}
					}
				}
			}
			//printf("Load Face has been finished %s\n",fullPath);
		}
		else
			printf("Failure in Loading from %s\n",fullPath);
		fin.close();
	}
}

//domeViews is required to load only the corresponding view's results
bool Load_Undist_HandDetectMultipleCamResult_PoseMachine(
	const char* poseDetectFolder,const char* poseDetectSaveFolder,
	const int currentFrame, CDomeImageManager& domeImMan,bool isHD)
{
	//Export Face Detect Results
	char fileName[512];
	char savefileName[512];
	if(isHD==false)
	{
		sprintf(fileName,"%s/handDetectMC_%08d.txt",poseDetectFolder,currentFrame);
		sprintf(savefileName,"%s/handDetectMC_%08d.txt",poseDetectSaveFolder,currentFrame);
	}
	else
	{
		sprintf(fileName,"%s/handDetectMC_hd%08d.txt",poseDetectFolder,currentFrame);
		sprintf(savefileName,"%s/handDetectMC_hd%08d.txt",poseDetectSaveFolder,currentFrame);
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
	int processedViews;	
	fin >> processedViews;	
	fout << processedViews <<"\n";
	
	//for(int i=0;i<domeViews.size();++i)
	for(int i=0;   ;++i)
	{ 
		int handNum,viewIdx,panelIdx,camIdx;
		fin >> handNum >> viewIdx >> panelIdx >> camIdx;
		if(fin.eof())
			break;
		fout << handNum <<" "<< viewIdx <<" " <<panelIdx <<" "<< camIdx <<"\n";

		CamViewDT* pCamDT = domeImMan.GetViewDTFromPanelCamIdx(panelIdx,camIdx);
		for(int j=0;j<handNum;++j)
		{
			char whichside;
			int subjectIdx, landmarkNum;
			fin >> whichside >>subjectIdx>>landmarkNum;
			fout << whichside <<" " << subjectIdx <<" " <<landmarkNum <<"\n";
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

void CHandRecon::OutlierRejection_length(SHand3D* pHand)
{
	double palmLength_thresh = 13; //cm, 0 - 1,5,9,13,17
	double fingerLength_thresh = 7; //cm, finger bone length except thumb
	double fingerFirstLength_thresh = 9; //cm, finger bone length except thumb

	bool bReject = false;
	for(int gIdx =1;gIdx< m_fingerGroupIdxVect.size(); ++gIdx)			//except thumb
	{
		int ptIdx = m_fingerGroupIdxVect[gIdx].front();
		double dist = Distance(pHand->m_landmark3D[0],pHand->m_landmark3D[ptIdx]);
		//printf("ptIdx: %d: dist:%f\n ",m_fingerGroupIdxVect[gIdx].front(),dist);

		if( dist> palmLength_thresh )
		{
			printf("ptIdx: %d: Rejected by length: %f\n",ptIdx,dist);
			bReject = true;
			//break;
			pHand->m_fingerInfo[ptIdx].m_bValid = false;
		}
		for(int localIdx=1; localIdx< m_fingerGroupIdxVect[gIdx].size(); ++localIdx)
		{
			int ptIdx = m_fingerGroupIdxVect[gIdx][localIdx];
			int ptIdx_parent = m_fingerGroupIdxVect[gIdx][localIdx-1];

			double dist = Distance(pHand->m_landmark3D[ptIdx_parent],pHand->m_landmark3D[ptIdx]);
			//printf("ptIdx: %d: dist:%f\n ",ptIdx,dist);
			double thresh;
			
			if(localIdx==1)
				thresh= fingerFirstLength_thresh;
			else
				thresh= fingerLength_thresh;

			if(dist >thresh )
			{
				printf("ptIdx: %d: Rejected by length: %f\n",ptIdx,dist);
				bReject = true;
				//break;
				pHand->m_fingerInfo[ptIdx].m_bValid = false;
			}
		}
		/*if(bReject)
		{
			break;
		}*/
	}

	for(int gIdx =1;gIdx< m_fingerGroupIdxVect.size(); ++gIdx)			//except thumb
	{
		bool bReject=false;
		for(int localIdx=0; localIdx< m_fingerGroupIdxVect[gIdx].size(); ++localIdx)
		{
			int ptIdx = m_fingerGroupIdxVect[gIdx][localIdx];
			if(bReject)
			{
				pHand->m_fingerInfo[ptIdx].m_bValid = false;
				continue;
			}
			if(pHand->m_fingerInfo[ptIdx].m_bValid==false)
				bReject = true;
		}
	}

	//rejected at lease once
	if(bReject)
	{
		pHand->m_bValid = false;
		return;
	}
}

void CHandRecon::OutlierRejection_overlap(SHand3D* pLHand,SHand3D* pRHand)
{
	double overlap_dist_thresh = 0.5;	//cm
	bool bReject = false;
	for(int gIdx =0;gIdx< m_fingerGroupIdxVect.size(); ++gIdx)
	{
		double avgDist=0;
		for(int localIdx=0; localIdx< m_fingerGroupIdxVect[gIdx].size(); ++localIdx)
		{
			int ptIdx = m_fingerGroupIdxVect[gIdx][localIdx];

			avgDist+= Distance(pLHand->m_landmark3D[ptIdx],pRHand->m_landmark3D[ptIdx]);
		}
		avgDist /= m_fingerGroupIdxVect[gIdx].size();

		if(avgDist <=overlap_dist_thresh)
		{
			printf("OutlierRejection_overlap:: gIdx: %d:: Rejected: %f\n",gIdx,avgDist);
			bReject  =true;
			break;
		}
	}
	if(bReject)
	{
		pLHand->m_bValid = false;
		pRHand->m_bValid = false;
		return;
	}


	//assume flipped finger order
	for(int gIdx =2;gIdx<=5; ++gIdx)
	{
		int otherGroupIdx = 7- gIdx;		//2-5, 3-4, 4-3, 5-2
		double avgDist=0;
		for(int localIdx=0; localIdx< m_fingerGroupIdxVect[gIdx].size(); ++localIdx)
		{
			int ptIdx = m_fingerGroupIdxVect[gIdx][localIdx];
			int ptIdx_other = m_fingerGroupIdxVect[otherGroupIdx][localIdx];

			avgDist+= Distance(pLHand->m_landmark3D[ptIdx],pRHand->m_landmark3D[ptIdx_other]);
		}
		avgDist /= m_fingerGroupIdxVect[gIdx].size();

		if(avgDist <=overlap_dist_thresh)
		{
			//printf("OutlierRejection_overlap:: Rejected: %f\n",avgDist);
			bReject  =true;
			break;
		}
	}
	if(bReject)
	{
		pLHand->m_bValid = false;
		pRHand->m_bValid = false;
		return;
	}


}

void CHandRecon::Scrit_handOutlier_rejection()
{
	//Compute average finger length
	#pragma omp parallel for
	for(int f=0;f<m_handReconMem.size();++f)
	{
		vector<SHand3D>& curHandVect = m_handReconMem[f].handReconVect;
		for(int i=0;i<curHandVect.size();++i)
		{
			SHand3D& curHand = curHandVect[i];

			//Find corresponding
			int corresHandIdx=-1;
			for(int j=i+1;j<curHandVect.size();++j)
			{
				if(curHand.m_identityIdx ==  curHandVect[j].m_identityIdx)
				{
					corresHandIdx = j;
					break;
				}
			}

			SHand3D* pLHand =NULL;
			SHand3D* pRHand =NULL;
			if(corresHandIdx<0)
				continue;
			else
			{
				if(curHand.m_whichSide==HAND_LEFT && curHandVect[corresHandIdx].m_whichSide==HAND_RIGHT)
				{
					pLHand = &curHand;
					pRHand = &curHandVect[corresHandIdx];
				}
				else if(curHand.m_whichSide==HAND_RIGHT && curHandVect[corresHandIdx].m_whichSide==HAND_LEFT)
				{
					pRHand = &curHand;
					pLHand = &curHandVect[corresHandIdx];
				}
				else
				{
					printf("ERROR: something wrong. Two right or left hands are detected. frame: %f, handIdx %d\n",f,i);
					continue;
				}
			}
			//Check whether two hands are overlapped or not in 3D
			OutlierRejection_length(pLHand);
			OutlierRejection_length(pRHand);
			//if(pLHand->m_bValid && pRHand->m_bValid)
				//OutlierRejection_overlap(pLHand,pRHand);
		}
	}
}

void CHandRecon::ComputeHandNormals()
{
	for(int f=0;f<m_handReconMem.size();++f)
	{
		vector<SHand3D>& curHandVect = m_handReconMem[f].handReconVect;
		for(int i=0;i<curHandVect.size();++i)
		{
			SHand3D& curHand = curHandVect[i];
			Point3d vec1 = curHand.m_landmark3D[9]-curHand.m_landmark3D[0];
			Normalize(vec1);
			Point3d vec2;//
			if(curHand.m_whichSide==HAND_LEFT)
				vec2 = curHand.m_landmark3D[5]-curHand.m_landmark3D[17];
			else
				vec2 = curHand.m_landmark3D[17]-curHand.m_landmark3D[5];
			Normalize(vec2);

			curHand.m_normal = vec1.cross(vec2);		
			Normalize(curHand.m_normal);		//handback to palm direction


			Point3d palmUp =curHand.m_landmark3D[9]- curHand.m_landmark3D[0];
			curHand.m_palmCenter = palmUp*0.5 +curHand.m_landmark3D[0];

			curHand.m_palmUp =  palmUp;
			Normalize(curHand.m_palmUp);
			curHand.m_palmX =  curHand.m_palmUp.cross(curHand.m_normal);
			Normalize(curHand.m_palmX);

			curHand.m_palmUp = curHand.m_normal.cross(curHand.m_palmX);
		}
	}
}


}	//end of namespace Module_Hand