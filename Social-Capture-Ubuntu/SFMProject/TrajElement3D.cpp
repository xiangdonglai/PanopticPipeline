// #include "stdafx.h"
#include "TrajElement3D.h"

#include  <opencv2/imgproc/imgproc.hpp>	
#include  <opencv2/calib3d/calib3d.hpp>	

using namespace cv;


//return 3x1 vector 
void TrajElement3D::getArrow1Vect(Mat& arrow)
{
	Mat temp = m_arrow1StepVect4by1 * PATCH3D_GRID_HALFSIZE;
	temp.rowRange(0,3).copyTo(arrow);
}

void TrajElement3D::getArrow2Vect(Mat& arrow)
{
	Mat temp = m_arrow2StepVect4by1 * PATCH3D_GRID_HALFSIZE;//(m_patch.rows/2.0f);
	temp.rowRange(0,3).copyTo(arrow);
}

void TrajElement3D::GetPatch3DCoord(int x, int y,Mat& returnMat4by1)
{
	m_ptMat4by1.copyTo(returnMat4by1);
	returnMat4by1 += m_arrow1StepVect4by1*x;
	returnMat4by1 += m_arrow2StepVect4by1*y;
}

//center is origin.
void TrajElement3D::GetPatchNormzliaed3DCoord(int x, int y,Mat& returnMat4by1)
{
	returnMat4by1 = Mat::zeros(4,1,CV_64F);
	returnMat4by1 += m_arrow1StepVect4by1*x;
	returnMat4by1 += m_arrow2StepVect4by1*y;
}

//Get Trajectory Unit for a time instance using globalMemIdx
TrackUnit* TrajElement3D::GetTrackUnitByMemIdx(int globalMemIdx)	// memIdx is the global memIdx
{
	if(m_actualNextTrackUnit.size()==0 && m_actualPrevTrackUnit.size()==0 )
		return NULL;
	int offset = globalMemIdx - m_initiatedMemoryIdx;

		
	if(offset>0 && offset-1 >=m_actualNextTrackUnit.size())
		return NULL;
	if(offset<0 && -offset-1 >=m_actualPrevTrackUnit.size())
		return NULL;

	if(offset==0)
		return &m_curTrackUnit;
	else if(offset>0)
		return &m_actualNextTrackUnit[offset-1];
	else	//if(offset<0)
		return &m_actualPrevTrackUnit[-offset-1];
}

TrackUnit* TrajElement3D::GetTrackUnitByFrameIdx(int frameIdx)	// memIdx is the global memIdx
{
	int offset = frameIdx - m_initiatedFrameIdx;
		
	if(offset>0 && offset-1 >=m_actualNextTrackUnit.size())
		return NULL;
	if(offset<0 && -offset-1 >=m_actualPrevTrackUnit.size())
		return NULL;

	if(offset==0)
		return &m_curTrackUnit;
	else if(offset>0)
		return &m_actualNextTrackUnit[offset-1];
	else	//if(offset<0)
		return &m_actualPrevTrackUnit[-offset-1];
}


/*bool TrajElement3D::GetTrajPosByFrameIdx(int frameIdx,Point3d& outputPos)
{
	int offset = frameIdx - m_initiatedFrameIdx;
	//if(offset>0 && offset >= m_actualNextTrackUnit.size())
		//return false;

	if(offset<0 && -offset >= m_actualPrevTrackUnit.size())
		return false;

	if(offset==0)
	{
		if(m_actualNextTrackUnit.size()==0 && m_actualPrevTrackUnit.size()==0 ) 
			return false;
		else
		{
			outputPos = m_curTrackUnit.m_pt3D;
			return true;
		}
	}
	else if(offset>0 && offset-1 <m_actualNextTrackUnit.size() )
	{
		outputPos = m_actualNextTrackUnit[offset-1].m_pt3D;
		return true;
	}
	else if(offset<0 && -offset-1 < m_actualPrevTrackUnit.size())
	{
		outputPos = m_actualPrevTrackUnit[-offset-1].m_pt3D;
		return true;
	}

	return false;
}*/


bool TrajElement3D::PatchProjectionBoundaryCheck(const Mat& targetImage, const Mat& P)
{
	Mat temp3DPt;
	GetPatch3DCoord(PATCH3D_GRID_HALFSIZE,PATCH3D_GRID_HALFSIZE,temp3DPt);
	//printMatrix(temp3DPt,"temp");
	Mat imagePt = P *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	if(IsOutofBoundary(targetImage,imagePt.at<double>(0,0),imagePt.at<double>(1,0)))
		return false;

	GetPatch3DCoord(PATCH3D_GRID_HALFSIZE,-PATCH3D_GRID_HALFSIZE,temp3DPt);
	//printMatrix(temp3DPt,"temp");
	imagePt = P *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	if(IsOutofBoundary(targetImage,imagePt.at<double>(0,0),imagePt.at<double>(1,0)))
		return false;

	GetPatch3DCoord(-PATCH3D_GRID_HALFSIZE,-PATCH3D_GRID_HALFSIZE,temp3DPt);
	//printMatrix(temp3DPt,"temp");
	imagePt = P *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	if(IsOutofBoundary(targetImage,imagePt.at<double>(0,0),imagePt.at<double>(1,0)))
		return false;

	GetPatch3DCoord(-PATCH3D_GRID_HALFSIZE,PATCH3D_GRID_HALFSIZE,temp3DPt);
	//printMatrix(temp3DPt,"temp");
	imagePt = P *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	if(IsOutofBoundary(targetImage,imagePt.at<double>(0,0),imagePt.at<double>(1,0)))
		return false;

	return true;
}

//Project Patch as a warped rectangle
void TrajElement3D::Visualize_projectedPatchBoundary(Mat targetImage,const Mat& projMat,Scalar color)
{
	if(PatchProjectionBoundaryCheck(targetImage,projMat)==false)
		return;

	vector<Point2d> rectanglePts;
	GetProjectedPatchBoundary(projMat,rectanglePts);
	/*Point2d rectanglePts[4];
	Mat temp3DPt;
	GetPatch3DCoord(PATCH3D_GRID_HALFSIZE,PATCH3D_GRID_HALFSIZE,temp3DPt);
	Mat imagePt = projMat *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	rectanglePts[0].x = imagePt.at<double>(0,0);
	rectanglePts[0].y = imagePt.at<double>(1,0);
	//rectanglePts[0] = pTargetCam->ProjectionPt(temp3DPt);

	GetPatch3DCoord(PATCH3D_GRID_HALFSIZE,-PATCH3D_GRID_HALFSIZE,temp3DPt);
	imagePt = projMat *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	rectanglePts[1].x = imagePt.at<double>(0,0);
	rectanglePts[1].y = imagePt.at<double>(1,0);
	//rectanglePts[1] = pTargetCam->ProjectionPt(temp3DPt);

	GetPatch3DCoord(-PATCH3D_GRID_HALFSIZE,-PATCH3D_GRID_HALFSIZE,temp3DPt);
	imagePt = projMat *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	rectanglePts[2].x = imagePt.at<double>(0,0);
	rectanglePts[2].y = imagePt.at<double>(1,0);
	//rectanglePts[2] = pTargetCam->ProjectionPt(temp3DPt);

	GetPatch3DCoord(-PATCH3D_GRID_HALFSIZE,PATCH3D_GRID_HALFSIZE,temp3DPt);
	imagePt = projMat *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	rectanglePts[3].x = imagePt.at<double>(0,0);
	rectanglePts[3].y = imagePt.at<double>(1,0);
	//rectanglePts[3] = pTargetCam->ProjectionPt(temp3DPt);*/

	for(int t=0;t<3;++t)
	{
		line(targetImage,rectanglePts[t],rectanglePts[t+1],color,2);
	}
	line(targetImage,rectanglePts[3],rectanglePts[0],color,2);
}

//Project Patch as a warped rectangle
void TrajElement3D::Visualize_projectedMicroStructure(Mat targetImage,const Mat& projMat,Scalar color)
{
	//if(PatchProjectionBoundaryCheck(targetImage,projMat)==false)
	//	return;
	const vector<cv::Point3d>& vertexVect = m_microStructure.GetVertexVectReadOnly();
	int width = sqrt((float)vertexVect.size());
	vector<Point2d> projectedPts(vertexVect.size());
	for(int i=0;i<vertexVect.size();i++)
	{
		Mat_<double> ptMat;
		Point3xToMat4by1(vertexVect[i],ptMat);
		ptMat = projMat*ptMat;
		projectedPts[i].x = ptMat(0,0)/ptMat(2,0);
		projectedPts[i].y = ptMat(1,0)/ptMat(2,0);
	}
	
	for(int i=0;i<vertexVect.size();i+=width)
	{
		for(int j=0;j<width-1;++j)
			line(targetImage,projectedPts[i+j],projectedPts[i+j+1],color,1);

		if(i<vertexVect.size()-width)
			for(int j=0;j<width;++j)
				line(targetImage,projectedPts[i+j],projectedPts[i+j+width],color,1);
	}
}

void TrajElement3D::Visualize_projectedPatchCenter(Mat targetImage,const Mat& P,Scalar color, int crossSize)
{
	Point2d projectedPt2D = GetProjectedPatchCenter(P);

	if(!IsOutofBoundary(targetImage,projectedPt2D.x,projectedPt2D.y))
		DrawCross(targetImage,projectedPt2D.x,projectedPt2D.y,color,3);
}

void TrajElement3D::LocallyRotatePatch(const Mat_<double> R)
{
	//patch rotation
	Mat arrow1;
	getArrow1Vect(arrow1);
	arrow1 = R*arrow1;
	arrow1 += m_ptMat4by1.rowRange(0,3);
	MatToPoint3d(arrow1,m_arrow1stHead);

	Mat arrow2;
	getArrow2Vect(arrow2);
	arrow2 = R*arrow2;
	arrow2 += m_ptMat4by1.rowRange(0,3);
	MatToPoint3d(arrow2,m_arrow2ndHead);

	InitializeArrowVectNormal();
}

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres/rotation.h>

//Degree
//yaw, pitch roll is defined as rotattion w.r.t. 1st,2nd,normal axis
void TrajElement3D::LocallyRotatePatchByYawPitchRoll(double yaw,double pitch,double roll)
{
	//Just rotate in one angle
	Mat arrow1_unit,arrow2_unit;
	normalize(m_arrow1StepVect4by1.rowRange(0,3),arrow1_unit);		//x axis
	normalize(m_arrow2StepVect4by1.rowRange(0,3),arrow2_unit);		//y axis
	Mat_<double> Rot_normToPatch(3,3);		//patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate
	Rot_normToPatch(0,0)= arrow1_unit.at<double>(0,0);
		Rot_normToPatch(1,0)= arrow1_unit.at<double>(1,0);
			Rot_normToPatch(2,0)= arrow1_unit.at<double>(2,0);
	Rot_normToPatch(0,1)= arrow2_unit.at<double>(0,0);
		Rot_normToPatch(1,1)= arrow2_unit.at<double>(1,0);
			Rot_normToPatch(2,1)= arrow2_unit.at<double>(2,0);
	Rot_normToPatch(0,2)= m_normal.at<double>(0,0);
		Rot_normToPatch(1,2)= m_normal.at<double>(1,0);
			Rot_normToPatch(2,2)= m_normal.at<double>(2,0);
	Mat_<double> Rot_patchToNorm= Rot_normToPatch.t();

	//double euler[3] ={yaw,pitch,0};		???
	//double euler[3] ={0,yaw,pitch};		//yaw, roll, pitch??
	double euler[3] ={yaw,pitch,roll};		//yaw, roll, pitch??
	Mat_<double> Rot_byEulerMat(3,3);
	double* Rot_byEuler = (double*)Rot_byEulerMat.ptr();
	ceres::EulerAnglesToRotationMatrix(euler,3,Rot_byEuler);
	//printMatrix("Rot_byEulerMat",Rot_byEulerMat*Rot_patchToNorm);
	LocallyRotatePatch(Rot_normToPatch*Rot_byEulerMat*Rot_patchToNorm);
}
 
void TrajElement3D::BackUpStatus()
{
	m_normal.copyTo(m_origianl_normal);
	m_origianl_pt = m_pt3D;
	m_ptMat4by1.copyTo(m_origianl_ptMat4by1);
	m_origianl_arrow1stHead = m_arrow1stHead;
	m_origianl_arrow2ndHead = m_arrow2ndHead;
}

void TrajElement3D::RollBackPosStatus()
{
	m_origianl_normal.copyTo(m_normal);
	m_pt3D= m_origianl_pt;
	m_origianl_ptMat4by1.copyTo(m_ptMat4by1);
	m_arrow1stHead = m_origianl_arrow1stHead;
	m_arrow2ndHead = m_origianl_arrow2ndHead;
	InitializeArrowVectNormal();
}

void TrajElement3D::InitializeArrowVects()
{
	//printf("PATCH3D_GRID_HALFSIZE: %d\n",PATCH3D_GRID_HALFSIZE);
	m_arrow1StepVect4by1 = Mat::zeros(4,1,CV_64F);
	m_arrow2StepVect4by1 = Mat::zeros(4,1,CV_64F);

	m_arrow1StepVect4by1.at<double>(0,0) = (m_arrow1stHead.x-m_pt3D.x)/PATCH3D_GRID_HALFSIZE;
	m_arrow1StepVect4by1.at<double>(1,0) = (m_arrow1stHead.y-m_pt3D.y)/PATCH3D_GRID_HALFSIZE;
	m_arrow1StepVect4by1.at<double>(2,0) = (m_arrow1stHead.z-m_pt3D.z)/PATCH3D_GRID_HALFSIZE;
	//m_arrow1StepVect4by1.at<double>(3,0) = 0;

	m_arrow2StepVect4by1.at<double>(0,0) = (m_arrow2ndHead.x-m_pt3D.x)/PATCH3D_GRID_HALFSIZE;
	m_arrow2StepVect4by1.at<double>(1,0) = (m_arrow2ndHead.y-m_pt3D.y)/PATCH3D_GRID_HALFSIZE;
	m_arrow2StepVect4by1.at<double>(2,0) = (m_arrow2ndHead.z-m_pt3D.z)/PATCH3D_GRID_HALFSIZE;
	//m_arrow2StepVect4by1.at<double>(3,0) = 0;
}

//Initilize from arrowHeads
void TrajElement3D::InitializeArrowVectNormal()
{
	InitializeArrowVects();

	m_normal = m_arrow1StepVect4by1.rowRange(0,3).cross(m_arrow2StepVect4by1.rowRange(0,3));
	normalize(m_normal,m_normal);
}

//usually called after optimization. 
void TrajElement3D::OrthgonalizePatchAxis(bool bInitializeArrowVectNormal)
{
	Mat vect1_mat,vect2_mat;
	Point3d vect1 = m_arrow1stHead - m_pt3D;
	Point3d vect2 = m_arrow2ndHead - m_pt3D;
	double arrow1Length =norm(vect1);
	double arrow2Length =norm(vect2);
	Point3dToMat(vect1,vect1_mat);
	Point3dToMat(vect2,vect2_mat);
	normalize(vect1_mat,vect1_mat);
	normalize(vect2_mat,vect2_mat);
	Mat patchNormal = vect1_mat.cross(vect2_mat);
	normalize(patchNormal,patchNormal);
	vect2_mat = patchNormal.cross(vect1_mat);
	normalize(vect2_mat,vect2_mat);
	vect1 = MatToPoint3d(vect1_mat);
	vect2 = MatToPoint3d(vect2_mat);
	m_arrow1stHead = m_pt3D + vect1*arrow1Length;
	m_arrow2ndHead = m_pt3D + vect2*arrow2Length;
	m_normal = patchNormal.clone();  //should be the same

	m_scale3D = Distance(m_pt3D,m_arrow1stHead);		//it is just for display
	if(bInitializeArrowVectNormal)
		InitializeArrowVectNormal();
}

void TrajElement3D::PatchResize(double arrow1Length,double arrow2Length, bool bInitializeArrowVectNormal)
{
	Point3d vect1 = m_arrow1stHead - m_pt3D;
	Point3d vect2 = m_arrow2ndHead - m_pt3D;
	Normalize(vect1);
	Normalize(vect2);
	m_arrow1stHead = m_pt3D + vect1*arrow1Length;
	m_arrow2ndHead = m_pt3D + vect2*arrow2Length;

	m_scale3D = Distance(m_pt3D,m_arrow1stHead);		//it is just for display

	if(bInitializeArrowVectNormal)
		InitializeArrowVectNormal();
}


//////////////////////////////////////////////////////////////////////////
// Compute Distance between two trajectories
// 1. Compute average distance on common time window
// 2. Return the max( dist_t - dist_average) for all t
double TrajElement3D::TrajectoryDistance_average_max(TrajElement3D* otherT)
{
	int timeOffSet = m_initiatedMemoryIdx - otherT->m_initiatedMemoryIdx;		//(iter) is corresponding to (iter+timeOffset)

	int searchStartMemIdx = m_initiatedMemoryIdx- m_actualPrevTrackUnit.size();
	int searchEndMemIdx = m_initiatedMemoryIdx+ m_actualNextTrackUnit.size();
	vector<double> distVect;
	double distSum =0;
	for(int i=searchStartMemIdx;i<=searchEndMemIdx;++i)
	{
		TrackUnit* pMyTrackUnit = GetTrackUnitByMemIdx(i);
		TrackUnit* pOtherTrackUnit = otherT->GetTrackUnitByMemIdx(i);
		if(pMyTrackUnit==NULL || pOtherTrackUnit==NULL)
			continue;

		double tempDist = Distance(pMyTrackUnit->m_pt3D,pOtherTrackUnit->m_pt3D);
		distSum += tempDist;
		distVect.push_back(tempDist);
	}

	double dist_average = distSum/distVect.size();
	double maxOff=0;
	for(int i=0;i<distVect.size();++i)
	{
		double tempOff = std::abs(distVect[i] - dist_average);
		if(tempOff> maxOff)
			maxOff = tempOff;
	}

	return maxOff;
}

//////////////////////////////////////////////////////////////////////////
// Compute Distance between two trajectories
// 1. Compute average distance on common time window
// 2. Return the max( dist_t - dist_average) for all t
// 3. the intersection between them should be longer than harf of the reference trajectory
double TrajElement3D::TrajectoryDistance_minMaxCost(TrajElement3D* otherT)
{
	int timeOffSet = m_initiatedMemoryIdx - otherT->m_initiatedMemoryIdx;		//(iter) is corresponding to (iter+timeOffset)

	int searchStartMemIdx = m_initiatedMemoryIdx- m_actualPrevTrackUnit.size();
	int searchEndMemIdx = m_initiatedMemoryIdx+ m_actualNextTrackUnit.size();
	double distMin=1e10;
	double distMax=-1e10;
	int cnt =0;
	for(int i=searchStartMemIdx;i<=searchEndMemIdx;++i)
	{
		TrackUnit* pMyTrackUnit = GetTrackUnitByMemIdx(i);
		TrackUnit* pOtherTrackUnit = otherT->GetTrackUnitByMemIdx(i);
		if(pMyTrackUnit==NULL || pOtherTrackUnit==NULL)
			continue;

		double tempDist = Distance(pMyTrackUnit->m_pt3D,pOtherTrackUnit->m_pt3D);
		if(tempDist<distMin)
			distMin = tempDist ;

		if(tempDist>distMax)
			distMax = tempDist ;

		cnt++;
	}
	//if(cnt<(m_actualPrevTrackUnit.size()+m_actualNextTrackUnit.size()*0.5))
	//	return 1e3;

	return std::abs(distMax-distMin);
}

//////////////////////////////////////////////////////////////////////////
// Compute Distance between two trajectories
void TrajElement3D::TrajectoryDist_minMax(TrajElement3D* otherT,double& minDist,double& maxDist,int& computeCnt)
{
	int timeOffSet = m_initiatedMemoryIdx - otherT->m_initiatedMemoryIdx;		//(iter) is corresponding to (iter+timeOffset)

	int searchStartMemIdx = m_initiatedMemoryIdx- m_actualPrevTrackUnit.size();
	int searchEndMemIdx = m_initiatedMemoryIdx+ m_actualNextTrackUnit.size();
	double distMin=1e10;
	double distMax=-1e10;
	int cnt =0;
	for(int i=searchStartMemIdx;i<=searchEndMemIdx;++i)
	{
		TrackUnit* pMyTrackUnit = GetTrackUnitByMemIdx(i);
		TrackUnit* pOtherTrackUnit = otherT->GetTrackUnitByMemIdx(i);
		if(pMyTrackUnit==NULL || pOtherTrackUnit==NULL)
			continue;

		double tempDist = Distance(pMyTrackUnit->m_pt3D,pOtherTrackUnit->m_pt3D);
		if(tempDist<distMin)
			distMin = tempDist ;

		if(tempDist>distMax)
			distMax = tempDist ;

		cnt++;
	}

	minDist = distMin;
	maxDist = distMax;
	computeCnt = cnt;
}



//Extract valued for m_refPatch from the reference image
//Set up m_refPatch, m_patch3DGridVect
bool TrajElement3D::Extract3DPatchPixelValue(CamViewDT* pRefImage)
{
	m_patch3DGridVect.clear();

	int gridSize = PATCH3D_GRID_SIZE; //should be odd number
	int halfSize = PATCH3D_GRID_HALFSIZE;//int(gridSize/2);

	//Patch 3D estimations
	Mat_<double> centerMat;
	Point3dToMat(m_pt3D,centerMat);

	//calculate RefImage
	m_refPatch = Mat_<double>(gridSize,gridSize);  
	Mat_<double> targetPatch = Mat_<double>(gridSize,gridSize);  

	Point3d arrow1Temp = m_arrow1stHead - m_pt3D;
	Mat_<double> arrow1Mat;
	Point3dToMat(arrow1Temp,arrow1Mat);
	arrow1Mat/= halfSize;
	//printMatrix(arrow1,"arrow1");

	Point3d arrow2Temp = m_arrow2ndHead - m_pt3D;
	Mat_<double> arrow2Mat;
	Point3dToMat(arrow2Temp,arrow2Mat);
	arrow2Mat /= halfSize;
	//printMatrix(arrow2,"arrow2");

	//double xCenter = pPts3D->m_patch.cols/2.0;
	//double yCenter = pPts3D->m_patch.rows/2.0;

	Mat_<double> projPt;
	Point2d imagePt;

	double NCCCost =0;
	int cnt =0;
	double projectionError=0;
	int projectionErrorCnt=0;

	bool bOutofBoundary = false;
	Mat dummy;
	for(int y=-halfSize;y<=halfSize;y++)
	{
		if(bOutofBoundary) 
			return false;
		for(int x=-halfSize;x<=halfSize;x++)
		{
			Mat_<double> temp3D = arrow1Mat*x;
			temp3D += arrow2Mat*y;   //cam coord
			temp3D += centerMat;
			Mat_<double> temp3D4by1 =Mat_<double>::ones(4,1);
			temp3D.copyTo(temp3D4by1.rowRange(0,3));

			projPt = pRefImage->m_P * temp3D4by1;
			imagePt.x = projPt(0,0)/projPt(2,0);
			imagePt.y = projPt(1,0)/projPt(2,0);
			if(IsOutofBoundary(pRefImage->m_inputImage,imagePt.x,imagePt.y))
				return false;

			m_refPatch(y+halfSize,x+halfSize) = BilinearInterpolation((Mat_<uchar>&)pRefImage->m_inputImage,imagePt.x,imagePt.y);

			//m_patch3DGridVect.push_back( make_pair( Point2i(x+halfSize, y+halfSize) ,  dummy) );
			m_patch3DGridVect.push_back( make_pair( Point2i(x+halfSize, y+halfSize) ,  temp3D4by1) );
			//temp3D4by1.copyTo(m_patch3DGridVect.back().second);
		}
	}
	return true;
}

bool TrajElement3D::Extract3DPatchPixelValue(CamViewDT* pCamView,cv::Mat_<double>* patchReturn)
{
	int gridSize = PATCH3D_GRID_SIZE; //should be odd number
	int halfSize = PATCH3D_GRID_HALFSIZE;//int(gridSize/2);

	//Patch 3D estimations
	Mat_<double> centerMat;
	Point3dToMat(m_pt3D,centerMat);

	//calculate RefImage
	(*patchReturn) = Mat_<double>(gridSize,gridSize);  

	Point3d arrow1Temp = m_arrow1stHead - m_pt3D;
	printf("arrow1Temp: %f\n",world2cm(norm(arrow1Temp)));
	Point3MutipliedByScalar(arrow1Temp,1.0/halfSize);
	Mat_<double> arrow1Mat;
	Point3dToMat(arrow1Temp,arrow1Mat);
	//arrow1Mat/= halfSize;
	//printMatrix(arrow1,"arrow1");

	Point3d arrow2Temp = m_arrow2ndHead - m_pt3D;
	printf("arrow1Temp: %f\n",world2cm(norm(arrow2Temp)));

	Point3MutipliedByScalar(arrow2Temp,1.0/halfSize);
	Mat_<double> arrow2Mat;
	Point3dToMat(arrow2Temp,arrow2Mat);
	//arrow2Mat /= halfSize;
	//printMatrix(arrow2,"arrow2");

	//double xCenter = pPts3D->m_patch.cols/2.0;
	//double yCenter = pPts3D->m_patch.rows/2.0;

	Mat_<double> projPt;
	Point2d imagePt;

	double NCCCost =0;
	int cnt =0;
	double projectionError=0;
	int projectionErrorCnt=0;

	bool bOutofBoundary = false;
	Mat dummy;
	for(int y=-halfSize;y<=halfSize;y++)
	{
		if(bOutofBoundary) 
			return false;
		for(int x=-halfSize;x<=halfSize;x++)
		{
			Mat_<double> temp3D = arrow1Mat*x;
			temp3D += arrow2Mat*y;   //cam coord
			temp3D += centerMat;
			Mat_<double> temp3D4by1 =Mat_<double>::ones(4,1);
			temp3D.copyTo(temp3D4by1.rowRange(0,3));

			projPt = pCamView->m_P * temp3D4by1;
			imagePt.x = projPt(0,0)/projPt(2,0);
			imagePt.y = projPt(1,0)/projPt(2,0);
			if(IsOutofBoundary(pCamView->m_inputImage,imagePt.x,imagePt.y))
				return false;

			(*patchReturn)(y+halfSize,x+halfSize) = BilinearInterpolation((Mat_<uchar>&)pCamView->m_inputImage,imagePt.x,imagePt.y);

			//m_patch3DGridVect.push_back( make_pair( Point2i(x+halfSize, y+halfSize) ,  dummy) );
			//m_patch3DGridVect.push_back( make_pair( Point2i(x+halfSize, y+halfSize) ,  temp3D4by1) );
			//temp3D4by1.copyTo(m_patch3DGridVect.back().second);
		}
	}
	cvNamedWindow("test");
	imshow("test",(*patchReturn));
	return true;
}



//Static function for general usage
bool TrajElement3D::Extract3DPatchPixelValue_microStrcuture(CamViewDT* pRefImage,CMicroStructure& microStr
															,vector< pair< cv::Point2i,cv::Mat > >& patch3DGridVect,cv::Mat_<double>& refPatch)	//output
{
	if(pRefImage->m_inputImage.rows==0)
	{
		printf("## Failure: pRefImage->m_inputImage.rows==0\n");
		return false;
	}

	if(microStr.IsValid() ==false)
	{
		printf("Failure: Extract3DPatchPixelValue_microStrcuture: microStr.bValid ==false\n");
		return false;
	}
	patch3DGridVect.clear();

	int gridSize = PATCH3D_GRID_SIZE; //should be odd number
	int halfSize = PATCH3D_GRID_HALFSIZE;//int(gridSize/2);
	//calculate RefImage
	refPatch = Mat_<double>::zeros(gridSize,gridSize);  


	const vector<cv::Point3d>&  vertexVect = microStr.GetVertexVectReadOnly();
	const vector<cv::Point2i>&  indexVect = microStr.GetIndexVectReadOnly();
	Point2d imagePt;
	for(int i=0;i<vertexVect.size();++i)
	{
		Mat_<double> temp3D4by1;//=Mat_<double>::ones(4,1);
		Point3fToMat4by1(vertexVect[i],temp3D4by1);

		Mat_<double> projPt = pRefImage->m_P * temp3D4by1;
		imagePt.x = projPt(0,0)/projPt(2,0);
		imagePt.y = projPt(1,0)/projPt(2,0);
		if(IsOutofBoundary(pRefImage->m_inputImage,imagePt.x,imagePt.y))
			continue;

		refPatch(indexVect[i].y,indexVect[i].x) = BilinearInterpolation((Mat_<uchar>&)pRefImage->m_inputImage,imagePt.x,imagePt.y);
		patch3DGridVect.push_back( make_pair( indexVect[i] ,  temp3D4by1) );
	}

	return true;
}


//Static function for general usage
bool TrajElement3D::Extract3DPatchPixelValue_microStrcuture_rgb(CamViewDT* pRefImage,CMicroStructure& microStr
															,vector< pair< cv::Point2i,cv::Mat > >& patch3DGridVect,cv::Mat_<Vec3d>& refPatch)	//output
{
	if(pRefImage->m_rgbInputImage.rows==0)
	{
		printf("## Failure: pRefImage->m_rgbInputImage.rows==0\n");
		return false;
	}

	if(microStr.IsValid() ==false)
	{
		printf("Failure: Extract3DPatchPixelValue_microStrcuture: microStr.bValid ==false\n");
		return false;
	}
	patch3DGridVect.clear();

	int gridSize = PATCH3D_GRID_SIZE; //should be odd number
	int halfSize = PATCH3D_GRID_HALFSIZE;//int(gridSize/2);
	//calculate RefImage
	refPatch = Mat_<Vec3d>::zeros(gridSize,gridSize);  


	const vector<cv::Point3d>&  vertexVect = microStr.GetVertexVectReadOnly();
	const vector<cv::Point2i>&  indexVect = microStr.GetIndexVectReadOnly();
	Point2d imagePt;
	for(int i=0;i<vertexVect.size();++i)
	{
		Mat_<double> temp3D4by1;//=Mat_<double>::ones(4,1);
		Point3fToMat4by1(vertexVect[i],temp3D4by1);

		Mat_<double> projPt = pRefImage->m_P * temp3D4by1;
		imagePt.x = projPt(0,0)/projPt(2,0);
		imagePt.y = projPt(1,0)/projPt(2,0);
		if(IsOutofBoundary(pRefImage->m_rgbInputImage,imagePt.x,imagePt.y))
			continue;

		refPatch(indexVect[i].y,indexVect[i].x) = BilinearInterpolation((Mat_<Vec3b>&)pRefImage->m_rgbInputImage,imagePt.x,imagePt.y);
		
		patch3DGridVect.push_back( make_pair( indexVect[i] ,  temp3D4by1) );
	}

	return true;
}


//Extract valued for m_refPatch from the reference image
//Set up m_refPatch, m_patch3DGridVect
bool TrajElement3D::Extract3DPatchPixelValue_microStrcuture(CamViewDT* pRefImage)
{
	/*vector< pair< cv::Point2i,cv::Mat > > m_patch3DGridVect;
	cv::Mat_<double> m_refPatch;		//used for appearance cost*/
	return Extract3DPatchPixelValue_microStrcuture(pRefImage,m_microStructure,m_patch3DGridVect,m_refPatch);
}

//Pre-requriement: Extract3DPatchPixelValue should be called first to set m_patch3DGridVect and m_refPatch
//Input: targetCam
//Output: extracted sample pixels from the view (patchReturn) and 0<photometricError<2 (returned)
double TrajElement3D::ComputePhotoConsistCostForOneView(CamViewDT* pTargetImage,Mat_<double>* patchReturn)
{
	//Patch 3D estimations
	Mat_<double> centerMat;
	Point3dToMat(m_pt3D,centerMat);

	//calculate RefImage
	Mat_<double> targetPatch = Mat_<double>(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);  

	Mat_<double> projPt;
	Point2d imagePt;
	bool bOutofBoundary = false;
	for(unsigned int i=0;i<m_patch3DGridVect.size();++i)
	{
		projPt = pTargetImage->m_P * m_patch3DGridVect[i].second;
		imagePt.x = projPt(0,0)/projPt(2,0);
		imagePt.y = projPt(1,0)/projPt(2,0);
		if(IsOutofBoundary(pTargetImage->m_inputImage,imagePt.x,imagePt.y))
		{
			bOutofBoundary = true;
			break;
		}
		targetPatch(m_patch3DGridVect[i].first.y,m_patch3DGridVect[i].first.x) = BilinearInterpolation((Mat_<uchar>&)pTargetImage->m_inputImage,imagePt.x,imagePt.y);
	}

	if(bOutofBoundary) 
		return 100;  //don't care about these at this time. Acutally, it should not happen

	double photoCost = 1 - myNCC(m_refPatch,targetPatch,1);  // in best case, it is 0

	if(patchReturn!=NULL)
	{
		(*patchReturn) = targetPatch;
	}
	return photoCost;
}


void TrajElement3D::GetLocalRotationMatrix(TrajElement3D *pOther,Mat& returnRotMat)
{
	Mat rotAxis= m_arrow1StepVect4by1.rowRange(0,3).cross(pOther->m_arrow1StepVect4by1.rowRange(0,3));
	normalize(rotAxis,rotAxis);

	Mat normalizedArrow1,normalizedOtherArrow1;
	normalize(m_arrow1StepVect4by1.rowRange(0,3),normalizedArrow1);
	normalize(pOther->m_arrow1StepVect4by1.rowRange(0,3),normalizedOtherArrow1);

	double cosAngle = normalizedArrow1.dot(normalizedOtherArrow1);
	double angle = std::acos(cosAngle);
							
	rotAxis*=angle;
	Rodrigues(rotAxis, returnRotMat);
}

void TrajElement3D::SetPos(Point3d& newPos)
{
	m_pt3D = newPos;

	m_ptMat4by1 = Mat::ones(4,1,CV_64F);
	m_ptMat4by1.at<double>(0,0) = m_pt3D.x;
	m_ptMat4by1.at<double>(1,0) = m_pt3D.y;
	m_ptMat4by1.at<double>(2,0) = m_pt3D.z;

	Mat arrow1;
	getArrow1Vect(arrow1);
	Mat arrow1HeadMat = m_ptMat4by1.rowRange(0,3) + arrow1;
	MatToPoint3d(arrow1HeadMat,m_arrow1stHead);

	Mat arrow2;
	getArrow2Vect(arrow2);
	Mat arrow2HeadMat = m_ptMat4by1.rowRange(0,3) + arrow2;
	MatToPoint3d(arrow2HeadMat,m_arrow2ndHead);

	//don't have to do Initialization because arrow vector, normal should be the same
}

void TrajElement3D::SetPosWithoutArrowSet(Point3d& newPos)
{
	m_pt3D = newPos;

	m_ptMat4by1 = Mat::ones(4,1,CV_64F);
	m_ptMat4by1.at<double>(0,0) = m_pt3D.x;
	m_ptMat4by1.at<double>(1,0) = m_pt3D.y;
	m_ptMat4by1.at<double>(2,0) = m_pt3D.z;

}


void TrajElement3D::SetAverageColorUsingSIFTKey(vector<CamViewDT*>& camVect)
{
	Point3d rgb(0,0,0);
	for(unsigned int i =0; i<m_associatedViews.size();++i)
	{
		int frameIdx =m_associatedViews[i].camIdx;
		int keyPtIdx =m_associatedViews[i].keyPtIdx;
		int x = camVect[frameIdx]->m_keypoints[keyPtIdx].pt.x;
		int y = camVect[frameIdx]->m_keypoints[keyPtIdx].pt.y;

		Point2d ptInOriginal = camVect[frameIdx]->ApplyDistort(Point2d(x,y));
		//int x = (int) camVect[frameIdx]->m_originalKeyPt[keyPtIdx].x;
		//int	y = (int) camVect[frameIdx]->m_originalKeyPt[keyPtIdx].y;
		rgb.z += camVect[frameIdx]->m_rgbInputImage.at<Vec3b>(ptInOriginal.y,ptInOriginal.x)[0];  //blue
		rgb.y += camVect[frameIdx]->m_rgbInputImage.at<Vec3b>(ptInOriginal.y,ptInOriginal.x)[1]; //green
		rgb.x += camVect[frameIdx]->m_rgbInputImage.at<Vec3b>(ptInOriginal.y,ptInOriginal.x)[2];	//red
	}
	rgb.x = (rgb.x/255.0)/ m_associatedViews.size();
	rgb.y = (rgb.y/255.0)/ m_associatedViews.size();
	rgb.z = (rgb.z/255.0)/ m_associatedViews.size();
	m_color = rgb;
}

void TrajElement3D::SetAverageColorByProjection(vector<CamViewDT*>& camVect,bool bOnlyHD)
{
	Point3f rgb(0,0,0);
	int cnt=0;
	for(unsigned int i =0; i<m_associatedViews.size();++i)
	{
		if(bOnlyHD && camVect[m_associatedViews[i].camIdx]->m_sensorType ==SENSOR_TYPE_VGA)
			continue;
		cv::Point3f color;
		if(camVect[m_associatedViews[i].camIdx]->GetProjectedPtRGBColor(m_pt3D,color))
		{
			rgb = rgb + color;
			cnt++;
		}
		/*
		rgb.z += camVect[frameIdx]->m_rgbInputImage.at<Vec3b>(ptInOriginal.y,ptInOriginal.x)[0];  //blue
		rgb.y += camVect[frameIdx]->m_rgbInputImage.at<Vec3b>(ptInOriginal.y,ptInOriginal.x)[1]; //green
		rgb.x += camVect[frameIdx]->m_rgbInputImage.at<Vec3b>(ptInOriginal.y,ptInOriginal.x)[2];	//red*/
	}
	
	rgb.x = rgb.x/ cnt;
	rgb.y = rgb.y/ cnt;
	rgb.z = rgb.z/ cnt;
	m_color = rgb;
}




#if 0
//The following is for 2D-3D matching (used for PnP for SfM)
void TrajElement3D::SetColorByReferenceView(vector<CamViewDT*>& camVect)
{
	Point3d rgb(0,0,0);
	int referenceFrame = m_associatedViews[m_refIdxAmongAssoCams].camIdx;
	//int x = camVect[referenceFrame]->m_originalKeyPt[m_refIdxAmongAssoCams].x;
	//int y = camVect[referenceFrame]->m_originalKeyPt[m_refIdxAmongAssoCams].y;
	int x = camVect[referenceFrame]->m_keypoints[m_refIdxAmongAssoCams].m_pt3D.x;
	int y = camVect[referenceFrame]->m_keypoints[m_refIdxAmongAssoCams].m_pt3D.y;

	Point2d ptInOriginal = camVect[referenceFrame]->ApplyDistort(Point2d(x,y));
	//int x = (int) camVect[frameIdx]->m_originalKeyPt[keyPtIdx].x;
	//int	y = (int) camVect[frameIdx]->m_originalKeyPt[keyPtIdx].y;
	rgb.z += camVect[referenceFrame]->m_rgbInputImage.at<Vec3b>(ptInOriginal.y,ptInOriginal.x)[0];  //blue
	rgb.y += camVect[referenceFrame]->m_rgbInputImage.at<Vec3b>(ptInOriginal.y,ptInOriginal.x)[1]; //green
	rgb.x += camVect[referenceFrame]->m_rgbInputImage.at<Vec3b>(ptInOriginal.y,ptInOriginal.x)[2];	//red

	m_color.x = (rgb.x/255.0);
	m_color.y = (rgb.y/255.0);
	m_color.z = (rgb.z/255.0);
}
void TrajElement3D::SetAverageDescriptor(int desciptorSize,vector<CamViewDT*>& camVect)
{
	m_descriptor =  Mat::zeros(1,desciptorSize,CV_32F);
	for(unsigned int i =0; i<m_associatedViews.size();++i)
	{
		int frameIdx =m_associatedViews[i].camIdx;
		int keyPtIdx =m_associatedViews[i].keyPtIdx;
		Mat temp =  camVect[frameIdx]->m_descriptors.rowRange(keyPtIdx,keyPtIdx+1); 
		temp.convertTo(temp,CV_32F);  //I am using all variable with CV_64F type.
		m_descriptor = m_descriptor + temp;  //they are CV_64F type
	}
	m_descriptor /= m_associatedViews.size();
}

void TrajElement3D::SetDescriptorByReferenceView(int desciptorSize,vector<CamViewDT*>& camVect)
{
	m_descriptor =  Mat::zeros(1,desciptorSize,CV_32F);
	int frameIdx = m_associatedViews[m_refIdxAmongAssoCams].camIdx;
	int keyPtIdx = m_associatedViews[m_refIdxAmongAssoCams].keyPtIdx;
	Mat temp =  camVect[frameIdx]->m_descriptors.rowRange(keyPtIdx,keyPtIdx+1); 
	temp.convertTo(m_descriptor ,CV_32F);  //I am using all variable with CV_64F type.
}
#endif

Point2d TrajElement3D::GetProjectedPatchCenter(const Mat& P)
{
	Mat projMat = P*m_ptMat4by1;
	projMat /= projMat.at<double>(2,0);
	Point2d proj;
	proj.x = projMat.at<double>(0,0);
	proj.y = projMat.at<double>(1,0);
	return proj;
}

void TrajElement3D::GetProjectedPatchBoundary(const Mat& projMat,vector<Point2d>& boundaryPts)
{
	boundaryPts.resize(4);
	Mat temp3DPt;
	GetPatch3DCoord(PATCH3D_GRID_HALFSIZE,PATCH3D_GRID_HALFSIZE,temp3DPt);
	Mat imagePt = projMat *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	boundaryPts[0].x = imagePt.at<double>(0,0);
	boundaryPts[0].y = imagePt.at<double>(1,0);
	//boundaryPts[0] = pTargetCam->ProjectionPt(temp3DPt);

	GetPatch3DCoord(PATCH3D_GRID_HALFSIZE,-PATCH3D_GRID_HALFSIZE,temp3DPt);
	imagePt = projMat *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	boundaryPts[1].x = imagePt.at<double>(0,0);
	boundaryPts[1].y = imagePt.at<double>(1,0);
	//boundaryPts[1] = pTargetCam->ProjectionPt(temp3DPt);

	GetPatch3DCoord(-PATCH3D_GRID_HALFSIZE,-PATCH3D_GRID_HALFSIZE,temp3DPt);
	imagePt = projMat *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	boundaryPts[2].x = imagePt.at<double>(0,0);
	boundaryPts[2].y = imagePt.at<double>(1,0);
	//boundaryPts[2] = pTargetCam->ProjectionPt(temp3DPt);

	GetPatch3DCoord(-PATCH3D_GRID_HALFSIZE,PATCH3D_GRID_HALFSIZE,temp3DPt);
	imagePt = projMat *  temp3DPt;
	imagePt = imagePt/imagePt.at<double>(2,0);
	boundaryPts[3].x = imagePt.at<double>(0,0);
	boundaryPts[3].y = imagePt.at<double>(1,0);
	//rectanglePts[3] = pTargetCam->ProjectionPt(temp3DPt);
}
#if 0
void TrajElement3D::CopyTo(TrajElement3D& newOne)
{
	/*//Patch information
	m_ptMat4by1.copyTo(newOne.m_ptMat4by1); 
	newOne.m_pt3D = m_pt3D;
	newOne.m_arrow1stHead = m_arrow1stHead;		
	newOne.m_arrow2ndHead = m_arrow2ndHead;
//	m_patch.copyTo(newOne.m_patch);		
//	m_patchGray.copyTo(newOne.m_patchGray);
	newOne.scale3D = scale3D ; 
	newOne.m_color = m_color;
	m_descriptor.copyTo(newOne.m_descriptor);
	newOne.bValid = bValid;

	newOne.InitializeArrowVectNormal();*/
}
#endif
//R and t should be double type
//R and t are just global world transformation (not local roate)
void TrajElement3D::RotateNTransPatch(Mat_<double>& R,Mat_<double>& t)
{
	Mat newPos;
	m_ptMat4by1.rowRange(0,3).copyTo(newPos);
	newPos = R* newPos + t;
	Mat tempMat = m_ptMat4by1.rowRange(0,3);
	newPos.copyTo(tempMat);
	m_pt3D = MatToPoint3d(newPos);

	Point3dToMat(m_arrow1stHead,newPos);
	newPos = R* newPos + t;
	m_arrow1stHead = MatToPoint3d(newPos);

	Point3dToMat(m_arrow2ndHead,newPos);
	newPos = R* newPos + t;
	m_arrow2ndHead = MatToPoint3d(newPos);

	InitializeArrowVectNormal();
}

//R and t are just global world transformation (not local roate)
void TrajElement3D::InvRotateNTransPatch(Mat_<double>& R,Mat_<double>& t)
{
	Mat invR = R.inv();
	Mat newPos;
	m_ptMat4by1.rowRange(0,3).copyTo(newPos);
	newPos = invR*(newPos-t);
	Mat tempMat = m_ptMat4by1.rowRange(0,3);
	newPos.copyTo(tempMat);
	m_pt3D = MatToPoint3d(newPos);

	Point3dToMat(m_arrow1stHead,newPos);
	newPos = invR*(newPos-t);
	m_arrow1stHead = MatToPoint3d(newPos);

	Point3dToMat(m_arrow2ndHead,newPos);
	newPos = invR*(newPos-t);
	m_arrow2ndHead = MatToPoint3d(newPos);

	InitializeArrowVectNormal();
}

void TrajElement3D::GetPointMatrix(vector<Mat>& pointVector)
{
	Mat tempPt = Mat::zeros(3,1,CV_64F);
	tempPt.at<double>(0,0) = m_arrow1stHead.x;  tempPt.at<double>(1,0)= m_arrow1stHead.y; tempPt.at<double>(2,0)= m_arrow1stHead.z;
	pointVector.push_back(tempPt.clone());

	tempPt.at<double>(0,0) = m_arrow2ndHead.x;  tempPt.at<double>(1,0)= m_arrow2ndHead.y; tempPt.at<double>(2,0)= m_arrow2ndHead.z;
	pointVector.push_back(tempPt.clone());

	tempPt.at<double>(0,0) = m_pt3D.x;  tempPt.at<double>(1,0)= m_pt3D.y; tempPt.at<double>(2,0)= m_pt3D.z;
	pointVector.push_back(tempPt.clone());

	Mat normalHead = m_ptMat4by1.rowRange(0,3) + m_normal;
	pointVector.push_back(normalHead);
}

//http://www.cs.hunter.cuny.edu/~ioannis/registerpts_allen_notes.pdf
void TrajElement3D::GetTransformTo(TrajElement3D* target,Mat& R,Mat& t)
{
	vector<Mat> targetPts;
	target->GetPointMatrix(targetPts);

	Mat targetCenter = Mat::zeros(3,1,CV_64F);
	for(unsigned int i=0;i<targetPts.size();++i)
		targetCenter = targetCenter+targetPts[i];
	targetCenter = targetCenter/targetPts.size();
	vector<Mat> targetPtsNormalized;
	for(unsigned int i=0;i<targetPts.size();++i)
	{
		Mat tempPt;
		targetPtsNormalized.push_back(tempPt);
		targetPtsNormalized.back() = targetPts[i] - targetCenter;
	}

	vector<Mat> soursePts;
	GetPointMatrix(soursePts);
	Mat sourceCenter = Mat::zeros(3,1,CV_64F);
	for(unsigned int i=0;i<soursePts.size();++i)
		sourceCenter  = sourceCenter +soursePts[i];
	sourceCenter = sourceCenter/soursePts.size();
	vector<Mat> sourcePtsNormalized;
	for(unsigned int i=0;i<soursePts.size();++i)
	{
		Mat tempPt;
		sourcePtsNormalized.push_back(tempPt);
		sourcePtsNormalized.back() = soursePts[i] - sourceCenter;
	}

	Mat covMatrix = Mat::zeros(3,3,CV_64F);
	for(unsigned int i=0;i<sourcePtsNormalized.size();++i)
	{
		Mat temp = sourcePtsNormalized[i]*targetPtsNormalized[i].t();
		covMatrix += temp;
	}
	SVD svd(covMatrix);
	R = Mat::eye(3,3,CV_64F);
	R = svd.vt.t()*svd.u.t();
	printf("check detR = %f\n",determinant(R));

	t = Mat::eye(3,1,CV_64F);
	t = targetCenter - R*sourceCenter;
}


void TrajElement3D::VisualizeTrajectory(vector< pair<Point3d,Point3d> >& trajectoryPts,Point3d color,bool bShowOnlyForwardTraj)
{
	trajectoryPts.reserve(trajectoryPts.size() + m_actualNextTrackUnit.size() + m_actualPrevTrackUnit.size() + 1);
	//Point3d blue(0,0,1);
	if(m_actualNextTrackUnit.size()>0)
		for(unsigned int k=0;k<m_actualNextTrackUnit.size();++k)
		{
			if(k==0)
			{
				trajectoryPts.push_back( make_pair(m_curTrackUnit.m_pt3D,color) );
				trajectoryPts.push_back( make_pair(m_actualNextTrackUnit[k].m_pt3D,color) );
			}
			else
			{
				trajectoryPts.push_back( make_pair(m_actualNextTrackUnit[k-1].m_pt3D,color) );
				trajectoryPts.push_back( make_pair(m_actualNextTrackUnit[k].m_pt3D,color) );
			}
		}
		
	if(m_actualPrevTrackUnit.size()>0 && bShowOnlyForwardTraj == false)
		for(unsigned int k=0;k<m_actualPrevTrackUnit.size();++k)
		{
			if(k==0)
			{
				trajectoryPts.push_back( make_pair(m_curTrackUnit.m_pt3D,color) );
				trajectoryPts.push_back( make_pair(m_actualPrevTrackUnit[k].m_pt3D,color) );
			}
			else
			{
				trajectoryPts.push_back( make_pair(m_actualPrevTrackUnit[k-1].m_pt3D,color) );
				trajectoryPts.push_back( make_pair(m_actualPrevTrackUnit[k].m_pt3D,color) );
			}
		}
}

void TrajElement3D::VisualizeTrajectory(vector< pair<Point3d,Point3d> >& trajectoryPts,int curSelectedMemIdx,int curPtMemIdx,Point3d color,bool bShowOnlyForwardTraj)
{
	//Point3d blue(0,0,1);
	trajectoryPts.reserve(trajectoryPts.size() + m_actualNextTrackUnit.size() + m_actualPrevTrackUnit.size() + 1);
	if(m_actualNextTrackUnit.size()>0)
		for(unsigned int k=0;k<m_actualNextTrackUnit.size();++k)
		{
			//int effectiveTime = curPtFrameIdx + 1 +k;		//shooting time of the m_actualNextTrackUnit[k]
			int effectiveTime = curPtMemIdx + 1 +k;		//shooting time of the m_actualNextTrackUnit[k]
			if(effectiveTime>curSelectedMemIdx)
				continue;

			if(k==0)
			{
				trajectoryPts.push_back( make_pair(m_curTrackUnit.m_pt3D,color) );
				trajectoryPts.push_back( make_pair(m_actualNextTrackUnit[k].m_pt3D,color) );
			}
			else
			{
				trajectoryPts.push_back( make_pair(m_actualNextTrackUnit[k-1].m_pt3D,color) );
				trajectoryPts.push_back( make_pair(m_actualNextTrackUnit[k].m_pt3D,color) );
			}
		}

		if(m_actualPrevTrackUnit.size()>0 && bShowOnlyForwardTraj == false)
			for(unsigned int k=0;k<m_actualPrevTrackUnit.size();++k)
			{
				int effectiveTime = curPtMemIdx - 1 -k;		//shooting time of the m_actualNextTrackUnit[k]
				if(effectiveTime>=curSelectedMemIdx)
					continue;

				if(k==0)
				{
					trajectoryPts.push_back( make_pair(m_curTrackUnit.m_pt3D,color) );
					trajectoryPts.push_back( make_pair(m_actualPrevTrackUnit[k].m_pt3D,color) );
				}
				else
				{
					trajectoryPts.push_back( make_pair(m_actualPrevTrackUnit[k-1].m_pt3D,color) );
					trajectoryPts.push_back( make_pair(m_actualPrevTrackUnit[k].m_pt3D,color) );
				}
			}
}




void TrajElement3D::VisualizeTrajectoryLocalColor(vector< pair<Point3d,Point3d> >& trajectoryPts,Point3d color,bool bShowOnlyForwardTraj)
{
	//Point3d blue(0,0,1);
	trajectoryPts.reserve(trajectoryPts.size() + m_actualNextTrackUnit.size() + m_actualPrevTrackUnit.size() + 1);
	if(m_actualNextTrackUnit.size()>0)
		for(unsigned int k=0;k<m_actualNextTrackUnit.size();++k)
		{
			if(k==0)
			{
				trajectoryPts.push_back( make_pair(m_curTrackUnit.m_pt3D,color) );
				trajectoryPts.push_back( make_pair(m_actualNextTrackUnit[k].m_pt3D,color) );
			}
			else
			{
				trajectoryPts.push_back( make_pair(m_actualNextTrackUnit[k-1].m_pt3D,color) );
				trajectoryPts.push_back( make_pair(m_actualNextTrackUnit[k].m_pt3D,color) );
			}
		}

		if(m_actualPrevTrackUnit.size()>0 && bShowOnlyForwardTraj == false)
			for(unsigned int k=0;k<m_actualPrevTrackUnit.size();++k)
			{
				if(k==0)
				{
					trajectoryPts.push_back( make_pair(m_curTrackUnit.m_pt3D,color) );
					trajectoryPts.push_back( make_pair(m_actualPrevTrackUnit[k].m_pt3D,color) );
				}
				else
				{
					trajectoryPts.push_back( make_pair(m_actualPrevTrackUnit[k-1].m_pt3D,color) );
					trajectoryPts.push_back( make_pair(m_actualPrevTrackUnit[k].m_pt3D,color) );
				}
			}
}


/*
void TrajElement3D::VisualizeTrajectory(vector< pair<Point3d,Point3d> >& trajectoryPts,int curTime,int showTrajLength,int curPtFrameIdx,vector< Point3d >& colorMapForNext,vector< Point3d >& colorMapForPrev)
{
	int lowerTimeLimit = max(last - showTrajLength,0);

	if(colorMapForNext.size() < m_actualNextTrackUnit.size() || colorMapForPrev.size()<m_actualPrevTrackUnit.size())
	{
		return;
	}
	//Point3d cyan(0,1,1);

	if(m_actualNextTrackUnit.size()>0)
	{
		int trackUnitNum = m_actualNextTrackUnit.size();
		int last = min(int(curTime-1),trackUnitNum-1);
		for(int k=0;k<=last;++k)
		{
			if(k==0)
			{
				trajectoryPts.push_back(make_pair(m_curTrackUnit.m_pt3D,colorMapForNext[curPtFrameIdx + k]));
				trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k].m_pt3D,colorMapForNext[ curPtFrameIdx+ k]));
			}
			else
			{
				trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k-1].m_pt3D,colorMapForNext[curPtFrameIdx + k]));
				trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k].m_pt3D,colorMapForNext[curPtFrameIdx + k]));
			}
		}
	}
		
	if(m_actualPrevTrackUnit.size()>0)
	{
		int trackUnitNum = m_actualPrevTrackUnit.size();

		int last = max(int(-curTime),0);  // need to think that in for loop k-1 -> k traj is drawn. if curTime ==0 --> should draw all of them. if curTime==-1 draw from nextTrack[0]
		for(int k=last;k<m_actualPrevTrackUnit.size();++k)
		{
			if(k==0)
			{
				trajectoryPts.push_back(make_pair(m_curTrackUnit.m_pt3D,colorMapForPrev[curPtFrameIdx + k]));
				trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k].m_pt3D,colorMapForPrev[curPtFrameIdx + k]));
			}
			else
			{
				trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k-1].m_pt3D,colorMapForPrev[curPtFrameIdx + k]));
				trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k].m_pt3D,colorMapForPrev[curPtFrameIdx + k]));
			}
		}
	}
}*/
#if 0  //old
void TrajElement3D::VisualizeTrajectory(vector< pair<Point3d,Point3d> >& trajectoryPts,int curTime,int curPtFrameIdx,vector< Point3d >& colorMapForNext,vector< Point3d >& colorMapForPrev)
{
	int upperTimeLimit = curTime-curPtFrameIdx;
	int lowerTimeLimit = upperTimeLimit - MAX_TRAJ_SHOW_LENGTH;

	/*if(colorMapForNext.size() < m_actualNextTrackUnit.size() || colorMapForPrev.size()<m_actualPrevTrackUnit.size())
	{
		return;
	}*/
	//Point3d cyan(0,1,1);
	int trackUnitNum = m_actualNextTrackUnit.size();
	int last = min(upperTimeLimit-1,trackUnitNum-1);
	int first = max(lowerTimeLimit,0);
	for(int k=first;k<=last;++k)
	{
		if(k==0)
		{
			trajectoryPts.push_back(make_pair(m_curTrackUnit.m_pt3D,colorMapForNext[curPtFrameIdx + k]));
			trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k].m_pt3D,colorMapForNext[ curPtFrameIdx+ k]));
		}
		else
		{
			trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k-1].m_pt3D,colorMapForNext[curPtFrameIdx + k]));
			trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k].m_pt3D,colorMapForNext[curPtFrameIdx + k]));
		}
	}

	trackUnitNum = m_actualPrevTrackUnit.size();
	last = min(int(-lowerTimeLimit),trackUnitNum-1);  // need to think that in for loop k-1 -> k traj is drawn. if curTime ==0 --> should draw all of them. if curTime==-1 draw from nextTrack[0]
	first = max(int(-upperTimeLimit),0); 
	for(int k=first;k<=last;++k)
	{
		if(k==0)
		{
			trajectoryPts.push_back(make_pair(m_curTrackUnit.m_pt3D,colorMapForPrev[curPtFrameIdx + k]));
			trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k].m_pt3D,colorMapForPrev[curPtFrameIdx + k]));
		}
		else
		{
			trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k-1].m_pt3D,colorMapForPrev[curPtFrameIdx + k]));
			trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k].m_pt3D,colorMapForPrev[curPtFrameIdx + k]));
		}
	}
}
#elif 1

void TrajElement3D::VisualizeTrajectory(vector< pair<Point3d,Point3d> >& trajectoryPts,vector< float>& trajectoryPts_alpha,int curSelectedMemIdx
	,vector< Point3d >& colorMapForNext,vector< Point3d >& colorMapForPrev,int colorMode,vector< Point3f >& colorMapGeneral,bool showOnlyFoward,int firstTimeMemIdx,int lastTimeMemIdx,int prevCutLeng,int backCutLeng)
{
	float fulltimeRange = curSelectedMemIdx - firstTimeMemIdx+1;
	int upperTimeLimit = curSelectedMemIdx;
	int lowerTimeLimit = upperTimeLimit - MAX_TRAJ_SHOW_LENGTH;

	Point3f colorByLength;
	int length = int(m_actualNextTrackUnit.size() + m_actualPrevTrackUnit.size());
	//int colorIdx= (length / 100.0) * 255.0;
	int colorIdx= (length / 100.0) * 255.0;
	colorIdx= max(0,colorIdx);
	colorIdx = min(255,colorIdx);
	colorByLength= colorMapGeneral[colorIdx] ;

	Point3f ptColor = m_color;
	if(g_bColorEnhanced)
	{
		ptColor = ptColor*2;
		ptColor.x = min(ptColor.x,1.0f);
		ptColor.y = min(ptColor.y,1.0f);
		ptColor.z = min(ptColor.z,1.0f);
	}
	
	//if(showOnlyFoward  == true)
		//return;

	//printf("%f %f %f\n",m_randomColor.x,m_randomColor.y,m_randomColor.z);
	int tempTrjLength = int(m_actualNextTrackUnit.size())-backCutLeng;
	for(int  k=0;k<tempTrjLength;++k)
	{
		int effectiveTime = m_initiatedMemoryIdx+ 1 +k;		//shooting time of the m_actualNextTrackUnit[k]
		Point3d tempColor;
		if(colorMode ==0) //time coding
		{
			if(effectiveTime>=1)
				tempColor = colorMapForNext[effectiveTime-1];
			else
				tempColor = colorMapForPrev[-effectiveTime];
		}
		else if(colorMode ==1) //Local Color
		{
			int colorIdx = double(effectiveTime-lowerTimeLimit)/(upperTimeLimit-lowerTimeLimit)*255;
			//int colorIdx = double(k)/m_actualNextTrackUnit.size()*255;
			colorIdx= max(0,colorIdx);
			colorIdx = min(255,colorIdx);
			tempColor  = colorMapGeneral[colorIdx];
			//tempColor = m_randomColor;
		}
		else //by trajectory length
			//tempColor = colorByLength;
			tempColor = ptColor;		//point color
		float tempAlpha = (effectiveTime-firstTimeMemIdx)/fulltimeRange;	//0~1;
		if(tempAlpha<0.3) tempAlpha= 0.3;
		if(tempAlpha>1) tempAlpha= 1;

		if(lowerTimeLimit  <= effectiveTime  && effectiveTime <=upperTimeLimit)
		{
			if(k==0)
			{
				trajectoryPts.push_back(make_pair(m_curTrackUnit.m_pt3D,tempColor));
				trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k].m_pt3D,tempColor));
			}
			else
			{
				trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k-1].m_pt3D,tempColor));
				trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k].m_pt3D,tempColor));
			}
			trajectoryPts_alpha.push_back(tempAlpha);

		}
	}

	tempTrjLength = int(m_actualPrevTrackUnit.size())-prevCutLeng;
	//for(int  k=0;k<m_actualPrevTrackUnit.size()-prevCutLeng;++k)
	for(int  k=0;k<tempTrjLength;++k)
	{
		int effectiveTime = m_initiatedMemoryIdx - 1  - k;	 //shooting time of the m_actualPrevTrackUnit[k]
		Point3d tempColor;
		if(colorMode ==0)  //time coding
		{
			if(effectiveTime<=-1)
				tempColor = colorMapForPrev[-effectiveTime-1];
			else
				tempColor = colorMapForNext[effectiveTime];
			/*//if(effectiveTime>=0)
				tempColor = colorMapForNext[effectiveTime-1];
			else
				tempColor = colorMapForPrev[-effectiveTime-1];*/
		}
		else if(colorMode ==1)   //Local Color
		{
			int colorIdx = double(effectiveTime-lowerTimeLimit)/(upperTimeLimit-lowerTimeLimit)*255;
			//int colorIdx = double(k)/m_actualNextTrackUnit.size()*255;
			colorIdx= max(0,colorIdx);
			colorIdx = min(255,colorIdx);
			tempColor  = colorMapGeneral[colorIdx];
		}
		else  //by trajectory length
			tempColor = ptColor;		//point color
			//tempColor = colorByLength;

		float tempAlpha = (effectiveTime-firstTimeMemIdx)/fulltimeRange;	//0~1;
		if(tempAlpha<0.6) tempAlpha= 0.3;
		if(tempAlpha>1) tempAlpha= 1;

		if(lowerTimeLimit  <= effectiveTime  && effectiveTime <upperTimeLimit)
		{
			if(k==0)
			{
				trajectoryPts.push_back(make_pair(m_curTrackUnit.m_pt3D,tempColor));
				trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k].m_pt3D,tempColor));
			}
			else
			{
				trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k-1].m_pt3D,tempColor));
				trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k].m_pt3D,tempColor));
			}
			trajectoryPts_alpha.push_back(tempAlpha);
		}
	}
}
#endif


//Similar to above except returnFrameIdx. returnFrameIdx contains frameIdx corresponding to trajectoryPts
//trajectoryPts contains even element (src,dst,src,dst), and returnFrameIdx contains (srcFrame,dstFrame,srcFrame,dstFrame)..
void TrajElement3D::VisualizeTrajectoryReturnWithFrame(vector< pair<Point3d,Point3d> >& trajectoryPts,vector<int>& returnFrameIdx,vector< float>& trajectoryPts_alpha,int curSelectedMemIdx
	,vector< Point3d >& colorMapForNext,vector< Point3d >& colorMapForPrev,int colorMode,vector< Point3f >& colorMapGeneral,bool showOnlyFoward,int firstTimeMemIdx,int lastTimeMemIdx,int prevCutLeng,int backCutLeng)
{
	float fulltimeRange = curSelectedMemIdx - firstTimeMemIdx+1;
	int upperTimeLimit = curSelectedMemIdx;
	int lowerTimeLimit = upperTimeLimit - MAX_TRAJ_SHOW_LENGTH;

	Point3f colorByLength;
	int length = int(m_actualNextTrackUnit.size() + m_actualPrevTrackUnit.size());
	//int colorIdx= (length / 100.0) * 255.0;
	int colorIdx= (length / 100.0) * 255.0;
	colorIdx= max(0,colorIdx);
	colorIdx = min(255,colorIdx);
	colorByLength= colorMapGeneral[colorIdx] ;
	
	//if(showOnlyFoward  == true)
		//return;

	//printf("%f %f %f\n",m_randomColor.x,m_randomColor.y,m_randomColor.z);
	int tempTrjLength = int(m_actualNextTrackUnit.size())-backCutLeng;
	for(int  k=0;k<tempTrjLength;++k)
	{
		int effectiveTime = m_initiatedMemoryIdx+ 1 +k;		//shooting time of the m_actualNextTrackUnit[k]
		Point3d tempColor;
		if(colorMode ==0) //time coding
		{
			if(effectiveTime>=1)
				tempColor = colorMapForNext[effectiveTime-1];
			else
				tempColor = colorMapForPrev[-effectiveTime];
		}
		else if(colorMode ==1) //Local Color
		{
			int colorIdx = double(effectiveTime-lowerTimeLimit)/(upperTimeLimit-lowerTimeLimit)*255;
			//int colorIdx = double(k)/m_actualNextTrackUnit.size()*255;
			colorIdx= max(0,colorIdx);
			colorIdx = min(255,colorIdx);
			tempColor  = colorMapGeneral[colorIdx];
			//tempColor = m_randomColor;
		}
		else //by trajectory length
			tempColor = colorByLength;
		float tempAlpha = (effectiveTime-firstTimeMemIdx)/fulltimeRange;	//0~1;
		if(tempAlpha<0.3) tempAlpha= 0.3;
		if(tempAlpha>1) tempAlpha= 1;

		if(lowerTimeLimit  <= effectiveTime  && effectiveTime <=upperTimeLimit)
		{
			if(k==0)
			{
				trajectoryPts.push_back(make_pair(m_curTrackUnit.m_pt3D,tempColor));
				trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k].m_pt3D,tempColor));

				returnFrameIdx.push_back(m_initiatedFrameIdx);
				returnFrameIdx.push_back(m_initiatedFrameIdx+k);
			}
			else
			{
				trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k-1].m_pt3D,tempColor));
				trajectoryPts.push_back(make_pair(m_actualNextTrackUnit[k].m_pt3D,tempColor));
				
				returnFrameIdx.push_back(m_initiatedFrameIdx + k);	//bug fixed. should be K not K-1
				returnFrameIdx.push_back(m_initiatedFrameIdx+ k+1);
			}
			trajectoryPts_alpha.push_back(tempAlpha);

		}
	}


	tempTrjLength = int(m_actualPrevTrackUnit.size())-prevCutLeng;
	//for(int  k=0;k<m_actualPrevTrackUnit.size()-prevCutLeng;++k)
	for(int  k=0;k<tempTrjLength;++k)
	{
		int effectiveTime = m_initiatedMemoryIdx - 1  - k;	 //shooting time of the m_actualPrevTrackUnit[k]
		Point3d tempColor;
		if(colorMode ==0)  //time coding
		{
			if(effectiveTime<=-1)
				tempColor = colorMapForPrev[-effectiveTime-1];
			else
				tempColor = colorMapForNext[effectiveTime];
			/*//if(effectiveTime>=0)
				tempColor = colorMapForNext[effectiveTime-1];
			else
				tempColor = colorMapForPrev[-effectiveTime-1];*/
		}
		else if(colorMode ==1)   //Local Color
		{
			int colorIdx = double(effectiveTime-lowerTimeLimit)/(upperTimeLimit-lowerTimeLimit)*255;
			//int colorIdx = double(k)/m_actualNextTrackUnit.size()*255;
			colorIdx= max(0,colorIdx);
			colorIdx = min(255,colorIdx);
			tempColor  = colorMapGeneral[colorIdx];
		}
		else  //by trajectory length
			tempColor = colorByLength;

		float tempAlpha = (effectiveTime-firstTimeMemIdx)/fulltimeRange;	//0~1;
		if(tempAlpha<0.6) tempAlpha= 0.3;
		if(tempAlpha>1) tempAlpha= 1;

		if(lowerTimeLimit  <= effectiveTime  && effectiveTime <upperTimeLimit)
		{
			if(k==0)
			{
				trajectoryPts.push_back(make_pair(m_curTrackUnit.m_pt3D,tempColor));
				trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k].m_pt3D,tempColor));

				returnFrameIdx.push_back(m_initiatedFrameIdx);
				returnFrameIdx.push_back(m_initiatedFrameIdx-k);
			}
			else
			{
				trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k-1].m_pt3D,tempColor));
				trajectoryPts.push_back(make_pair(m_actualPrevTrackUnit[k].m_pt3D,tempColor));

				returnFrameIdx.push_back(m_initiatedFrameIdx - k);		//bug fixed should be -k not -(k-1)
				returnFrameIdx.push_back(m_initiatedFrameIdx- (k+1) );
			}
			trajectoryPts_alpha.push_back(tempAlpha);
		}
	}
}

void TrajElement3D::VisualizeTripletTrajectory(vector<Point3d>& trajectoryPts)
{

	for(unsigned int k=0;k<m_actualNextTrackUnit.size();++k)
	{
		if(k==0)
		{
			trajectoryPts.push_back(m_curTrackUnit.m_arrowHead3D[0]);
			trajectoryPts.push_back(m_actualNextTrackUnit[k].m_arrowHead3D[0]);

			trajectoryPts.push_back(m_curTrackUnit.m_arrowHead3D[1]);
			trajectoryPts.push_back(m_actualNextTrackUnit[k].m_arrowHead3D[1]);
		}
		else
		{
			trajectoryPts.push_back(m_actualNextTrackUnit[k-1].m_arrowHead3D[0]);
			trajectoryPts.push_back(m_actualNextTrackUnit[k].m_arrowHead3D[0]);

			trajectoryPts.push_back(m_actualNextTrackUnit[k-1].m_arrowHead3D[1]);
			trajectoryPts.push_back(m_actualNextTrackUnit[k].m_arrowHead3D[1]);
		}
	}
		
	for(unsigned int k=0;k<m_actualPrevTrackUnit.size();++k)
	{
		if(k==0)
		{
			trajectoryPts.push_back(m_curTrackUnit.m_arrowHead3D[0]);
			trajectoryPts.push_back(m_actualPrevTrackUnit[k].m_arrowHead3D[0]);

			trajectoryPts.push_back(m_curTrackUnit.m_arrowHead3D[1]);
			trajectoryPts.push_back(m_actualPrevTrackUnit[k].m_arrowHead3D[1]);
		}
		else
		{
			trajectoryPts.push_back(m_actualPrevTrackUnit[k-1].m_arrowHead3D[0]);
			trajectoryPts.push_back(m_actualPrevTrackUnit[k].m_arrowHead3D[0]);

			trajectoryPts.push_back(m_actualPrevTrackUnit[k-1].m_arrowHead3D[1]);
			trajectoryPts.push_back(m_actualPrevTrackUnit[k].m_arrowHead3D[1]);
		}
	}
}

//Input refCam
//Normal: parallel to image plane
//Patch size: defeind by projectedPatchScale such that the projected 3D patch on the refCam has projectedPatchScale width
void TrajElement3D::InitializePatchByRefCam(CamViewDT* refCam,double projectedPatchScale)
{
	//setting  ref patch information
	Point2d imagePt = Project3DPt(m_ptMat4by1,refCam->m_P);
	int patchHalfSize = projectedPatchScale/2;
	Rect bbox(imagePt.x - patchHalfSize,imagePt.y - patchHalfSize,patchHalfSize*2+1,patchHalfSize*2+1);
	//if(IsOutofBoundary(refCam->m_inputImage,bbox))
		//return false;

	Mat camCoordPt3D = refCam->m_R * m_ptMat4by1.rowRange(0,3) + refCam->m_t;
	double depth = camCoordPt3D.at<double>(2,0);

	Point2d arrow_1 = imagePt;//(*m_pMajorCamViewVect)[frameIdx]->m_keypoint1stArrow[keyPtIdx];
	arrow_1.x = arrow_1.x + patchHalfSize;
	Get3DPtfromDist(refCam->m_invK,refCam->m_invR,refCam->m_t,arrow_1,depth ,m_arrow1stHead);

	Point2d arrow_2 = imagePt;//(*m_pMajorCamViewVect)[frameIdx]->m_keypoint2ndArrow[keyPtIdx];
	arrow_2.y = arrow_2.y - patchHalfSize;
	Get3DPtfromDist(refCam->m_invK,refCam->m_invR,refCam->m_t,arrow_2,depth ,m_arrow2ndHead);

	//determine the patch dimension. this "pPts3D->m_patch" does have only dimention info. No actual texture. texture is generated in optimization loop
	//pPts3D->m_patch = Mat::zeros(refPatchBBox.height,refPatchBBox.width,CV_8UC3); 
	//pPts3D->m_patch = Mat::zeros(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE,CV_8UC3);    //just manully determine
	//pPts3D->m_patchGray = Mat_<double>(pPts3D->m_patch.rows,pPts3D->m_patch.cols); 
	m_scale3D = Distance(m_pt3D,m_arrow1stHead);		//it is just for display

	//InitializeArrowVects();  
	InitializeArrowVectNormal();  
}

//Input normal direction
//Normal: define by the given normal direction
//Patch size: defeind by patch3DWidthInCM (cm unit)
//Patch axis: arrow_2 is defined such that it is on the Y-normal plane
void TrajElement3D::InitializePatchByNormal(Point3d normal,double patch3DHalfWidthInCM)
{
	Normalize(normal);

	double halfPathcSize_w  = cm2world(patch3DHalfWidthInCM);
	Point3dToMat(normal,m_normal);	
    Point3d arrow2_direct = Point3d(0,-1,0);		//note that -y direction is up direction

	Point3d arrow1_direct = arrow2_direct.cross(normal);  //Note that it is NOT normal.dot(arrow2_direct);
	Normalize(arrow1_direct);
	m_arrow1stHead = m_pt3D + arrow1_direct*halfPathcSize_w ;

	arrow2_direct = normal.cross(arrow1_direct);
	m_arrow2ndHead = m_pt3D + arrow2_direct*halfPathcSize_w ;

	m_scale3D = halfPathcSize_w;

	InitializeArrowVects();  
}

void TrajElement3D::InitializeArrowsByNormal(Mat& normal,CamViewDT* refImage,double halfPatchSize)
{
	if(norm(normal)==0)
	{
		m_arrow1stHead = m_pt3D;
		m_arrow2ndHead = m_pt3D;
		InitializeArrowVectNormal();
		return;
	}

	Point2d imagePt_1 = Project3DPt(m_ptMat4by1,refImage->m_P);
	/*if(IsOutofBoundary(refImage->m_inputImage,Rect(imagePt_1.x-25,imagePt_1.y-25,50,50)))
	{
		(*m_pPts3D)[ptIdx]->m_arrow1stHead = (*m_pPts3D)[ptIdx]->m_pt3D;
		(*m_pPts3D)[ptIdx]->m_arrow2ndHead = (*m_pPts3D)[ptIdx]->m_pt3D;
		(*m_pPts3D)[ptIdx]->InitializeArrowVectNormal();
		continue;
	}*/

	Mat pt3D;
	Point3dToMat(m_pt3D,pt3D);
	Mat normalHead = pt3D + normal;

	pt3D = refImage->m_R * pt3D  + refImage->m_t; //local coord
	normalHead = refImage->m_R * normalHead+ refImage->m_t; //local normalHead coord
	normal = normalHead - pt3D; //local normal Coord
	normalize(normal,normal);  //we may don't have to do this
	//normalize(pt3D,pt3D);

	for(int t=0;t<2;++t)
	{
		//Get ray from imagePt(arrow1) on refFrameIdx
		Mat ptMat = Mat::ones(3,1,CV_64F);
		if(t==0)
		{
			ptMat.at<double>(0,0) = imagePt_1.x + halfPatchSize;
			ptMat.at<double>(1,0) = imagePt_1.y ;
		}
		else
		{
			ptMat.at<double>(0,0) = imagePt_1.x ;
			ptMat.at<double>(1,0) = imagePt_1.y - halfPatchSize;
		}
		Mat ray = refImage->m_invK * ptMat;  
		normalize(ray,ray);

		double d = normal.dot(pt3D) / normal.dot(ray);
		//calclute arrow1
		ray = ray * d;

		ptMat = refImage->m_invR * (ray - refImage->m_t);//from cam coord to world coord
		if(t==0)
			MatToPoint3d(ptMat,m_arrow1stHead);
		else
			MatToPoint3d(ptMat,m_arrow2ndHead);

		/*
		//test normal
		Point3d arrowTemp = (*m_pPts3D)[ptIdx]->m_arrow1stHead - (*m_pPts3D)[ptIdx]->m_pt3D;
		Point3dToMat(arrowTemp,ptMat);
		normalize(ptMat,ptMat);
		Point3dToMat((*m_pPts3D)[ptIdx]->m_curTrackUnit.m_normal,normal);
			
		printf("%f\n",ptMat.dot(normal));*/
			
	}
	InitializeArrowVectNormal();
}


//Set status variables (m_ptMat4by1, m_pt3D, m_arrow1stHead, m_arrow2ndHead)
bool TrajElement3D::SetTrajStatusByFrameIdx(int frameIdx)
{
	TrackUnit* pTrackUnit = GetTrackUnitByFrameIdx(frameIdx);
	if(pTrackUnit==NULL)
		return false;

	this->m_pt3D = pTrackUnit->m_pt3D;
	Point3xToMat4by1(pTrackUnit->m_pt3D,this->m_ptMat4by1);
	this->m_arrow1stHead = pTrackUnit->m_arrowHead3D[0];
	this->m_arrow2ndHead = pTrackUnit->m_arrowHead3D[1];

	Point3ToMatDouble(pTrackUnit->m_normal,this->m_normal);

	InitializeArrowVects();

	this->m_pCamVisibilityForVis = &pTrackUnit->m_visibleCamIdxVector;
	return true;
}
//propagatedPtsUptoNow is propgated history data 
//frameIdxforLastElmt is the time for propagatedPtsUptoNow.back()
//bForwardTracking==false means backward tracking 
double TrajElement3D::BoneToTrajLongestDist(vector< vector<Point3f> >& propagatedPtsUptoNow,int frameIdxforLastElmt,bool bForwardTracking)
{
	float longestDist = 0;
	int l = int(propagatedPtsUptoNow.size()) -1;
	int iter=0;
	for(int t=l;t>=0;--t)
	{
		int tempFrameIdx;
		if(bForwardTracking)
			tempFrameIdx = frameIdxforLastElmt - iter;
		else
			tempFrameIdx = frameIdxforLastElmt + iter;

		Point3d tempTrajPt;
		bool bSuccess = GetTrajPoseBySelectedFrameIdx(tempFrameIdx,tempTrajPt);
		if(bSuccess==false)
			break;

		assert(propagatedPtsUptoNow[t].size()==2);
		Point3f& childJointdPos = propagatedPtsUptoNow[t][0];
		Point3f& parentJointdPos = propagatedPtsUptoNow[t][1];
		Point3f boneDirectionalVect = parentJointdPos - childJointdPos;
		Normalize(boneDirectionalVect);

		//////////////////////////////////////////////////////////////////////////
		//Compute orthogonal dist
		Point3f childToPt = Point3f(tempTrajPt.x,tempTrajPt.y,tempTrajPt.z) - childJointdPos;
		float distAlongBoneDirection = childToPt.dot(boneDirectionalVect);
		Point3f orthoVect = childToPt- boneDirectionalVect*distAlongBoneDirection;
		double orthoDist = norm(orthoVect);

		if(orthoDist>longestDist)
			longestDist = orthoDist;

		iter++;
	}
	return longestDist;
}

void TrajElement3D::GenerateMicroStruct(CMicroStructure& str,double patchArrowSizeCm)
{
	vector<cv::Point3d>& vertexVect = str.GetVertexVectForModify();
	vector<cv::Point2i>& idxVect = str.GetIndexVectForModify();

	vertexVect.clear();
	idxVect.clear();

	str.SetValid(true);
	int gridSize = PATCH3D_GRID_SIZE; //should be odd number
	int halfSize = PATCH3D_GRID_HALFSIZE;//int(gridSize/2);
	//double scaleFactor = cm2world(patchArrowSizeCm)/Distance(m_arrow1stHead,m_pt3D);

	vertexVect.reserve(gridSize*gridSize);
	idxVect.reserve(gridSize*gridSize);
	for(int y=-halfSize;y<=halfSize;y++)
	{
		for(int x=-halfSize;x<=halfSize;x++)
		{
			Mat_<double> temp3D = m_arrow1StepVect4by1.rowRange(0,3)*x;//*scaleFactor;
			//printf("%d,%d: length %f\n",x,y,world2cm(norm(temp3D)));
			temp3D += m_arrow2StepVect4by1.rowRange(0,3)*y;//*scaleFactor;   //cam coord
			temp3D += m_ptMat4by1.rowRange(0,3);
			cv::Point3d pt= MatToPoint3d(temp3D);
			vertexVect.push_back(pt);
			idxVect.push_back(cv::Point2i(x+halfSize,y+halfSize));
		}
	}
	str.ParamSetting();
	//printf("half size: %d\n",str.GetHalfSize());
}
/*

int SelectRefImageByNormalOnlyFromHD(TrajElement3D* pPts3D,vector<CamViewDT*>& camVect,vector<int>& visibleCamIdxVector,bool bBoundaryCheck)
{
	int refCamIdx = -1;
	Mat pt3DMat;
	Point3dToMat(pPts3D->m_pt3D,pt3DMat);
	double maxDotValue =  -1;
	for(int i=0;i<visibleCamIdxVector.size();++i)
	{
		int camIdx= visibleCamIdxVector[i];

		//Only caring HD or kinect
		if(!( camVect[camIdx]->m_actualPanelIdx == PANEL_HD || camVect[camIdx]->m_actualPanelIdx == PANEL_KINECT) )
			continue;

		if(bBoundaryCheck)
			if(pPts3D->PatchProjectionBoundaryCheck(camVect[camIdx]->m_inputImage,camVect[camIdx]->m_P) ==false)
				continue;
		//if(camIdx == pPts3D->m_prevRefImageIdx)    //131108 debug for better display
			//return camIdx;

		Mat ray = camVect[camIdx]->m_CamCenter - pt3DMat;   //pt To Center
		normalize(ray,ray);
		double dotValue = pPts3D->m_normal.dot(ray);
		if(maxDotValue <dotValue)
		{
			maxDotValue = dotValue;
			refCamIdx = camIdx;
		}
	}
	return refCamIdx;
}
*/
