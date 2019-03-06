// #include "stdafx.h"
#include "PatchOptimization.h"
#include "Constants.h"
#include <fstream>

#include <opencv2/calib3d/calib3d.hpp>	//RQDecomp3x3

using namespace cv;
using namespace ceres;


//RQ decomposition Method
//K can be changed
static void CamParamSetting(CamViewDT& element,Mat& P,Mat& K,Mat& R,Mat& t)
{
	P.copyTo(element.m_P);
	//R.copyTo(element.m_R);
	//K.copyTo(element.m_K);
	//t.copyTo(element.m_t);

	Mat invR = R.inv();
	invR.copyTo(element.m_invR);
	Mat camCenter = - invR*t;
	camCenter.copyTo(element.m_CamCenter);

	Mat invK = K.inv();
	invK.copyTo(element.m_invK);

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

	//quaternion
	Mat quat(4,1,CV_64F);
//	Rotation2Quaternion(&(CvMat)R,&(CvMat)quat);
	Rotation2Quaternion(R,quat);

	//printMatrix(quat,"Quat");
	quat.copyTo(element.m_R_Quat);

	//opticalAxis
	Mat imageCenter = Mat::zeros(3,1,CV_64F);
	imageCenter.at<double>(2,0) = 1;
	imageCenter = invR*(imageCenter-t);
	element.m_opticalAxis = imageCenter - camCenter;
	normalize(element.m_opticalAxis,element.m_opticalAxis);
}

#ifdef BUNDLE_EVERYTHING

double BundleAdjustment_ceres(vector<CamViewDT*>& sequences,vector<TrajElement3D*>& Pts3D)
{
	for(unsigned int i=0;i<sequences.size();++i)
	{
		Mat rotationVector = Mat::zeros(3,1,CV_64F);
		Rodrigues(sequences[i]->m_R, rotationVector );
		sequences[i]->m_camParamForBundle[0] = rotationVector.at<double>(0,0);
		sequences[i]->m_camParamForBundle[1] = rotationVector.at<double>(1,0);
		sequences[i]->m_camParamForBundle[2] = rotationVector.at<double>(2,0);
		//printMatrix("R",rotationVector);

		sequences[i]->m_camParamForBundle[3] = sequences[i]->m_t.at<double>(0,0);
		sequences[i]->m_camParamForBundle[4] = sequences[i]->m_t.at<double>(1,0);
		sequences[i]->m_camParamForBundle[5] = sequences[i]->m_t.at<double>(2,0);

		sequences[i]->m_camParamForBundle[6] = sequences[i]->m_K.at<double>(0,0);
		//sequences[i]->m_camParamForBundle[7] = 0;//sequences[i]->m_K.at<double>(0,0);
		//sequences[i]->m_camParamForBundle[8] = 0;//sequences[i]->m_K.at<double>(0,0);
		//printMatrix("t",sequences[i]->m_t);

	}
	
	double** pt = new double*[Pts3D.size()];
	for (unsigned int i = 0; i < Pts3D.size(); ++i) 
	{
		pt[i] = new double[3];
		pt[i][0] = Pts3D[i]->m_pt3D.x;
		pt[i][1] = Pts3D[i]->m_pt3D.y;
		pt[i][2] = Pts3D[i]->m_pt3D.z;
	}


	//double bundleMaxNum =8000;
	//int searchEnd = Pts3D.size()-bundleMaxNum;
	//if(searchEnd<0)  searchEnd=0;
	// Create residuals for each observation in the bundle adjustment problem. The
	// parameters for cameras and points are added automatically.
	ceres::Problem problem;
	for (unsigned int i = 0; i < Pts3D.size(); ++i) 
	//for (int i = (int)Pts3D.size()-1; i>=searchEnd  ; --i) 
	{
		for(unsigned int k =0;k<Pts3D[i]->m_associatedViews.size();++k)
		{
			int imageIdx = Pts3D[i]->m_associatedViews[k].frameIdx;
			int keyIdx = Pts3D[i]->m_associatedViews[k].keyPtIdx;
			
			bool firstCam = false;
			if(imageIdx ==0)
				firstCam = true;

			if(imageIdx ==1 && sequences.size()>2)		//if sequences.size()==2, the apply bundle for secondOne
				firstCam = true;


			// Each Residual block takes a point and a camera as input and outputs a 2
			// dimensional residual. Internally, the cost function stores the observed
			// image location and compares the reprojection against the observation.
			ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 8, 3>(
						new SnavelyReprojectionError(
							//sequences[imageIdx]->m_keypoints[keyIdx].pt.x-sequences[imageIdx]->m_K.at<double>(0,2),
							//sequences[imageIdx]->m_keypoints[keyIdx].pt.y-sequences[imageIdx]->m_K.at<double>(1,2),
							sequences[imageIdx]->m_originalKeyPt[keyIdx].x-sequences[imageIdx]->m_K.at<double>(0,2),
							sequences[imageIdx]->m_originalKeyPt[keyIdx].y-sequences[imageIdx]->m_K.at<double>(1,2),
							sequences[imageIdx]->m_K.at<double>(0,0),firstCam,sequences[imageIdx]->m_camParamForBundle)
							);

			problem.AddResidualBlock(cost_function,
					NULL /* squared loss */ ,
					sequences[imageIdx]->m_camParamForBundle,
					pt[i]
					);
		}
	}
	
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.gradient_tolerance = 1e-18;
	options.function_tolerance = 1e-18;
	options.parameter_tolerance = 1e-18;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if(summary.initial_cost >summary.final_cost)
		std::cout << summary.FullReport() << "\n";

	for(unsigned int i=0;i<sequences.size();++i)
	{
		Mat rotationVector = Mat::zeros(3,1,CV_64F);
		rotationVector.at<double>(0,0) = sequences[i]->m_camParamForBundle[0];
		rotationVector.at<double>(1,0) = sequences[i]->m_camParamForBundle[1];
		rotationVector.at<double>(2,0) = sequences[i]->m_camParamForBundle[2];
		Rodrigues(rotationVector,sequences[i]->m_R);
		//printMatrix("R",sequences[i]->m_R);

		sequences[i]->m_t.at<double>(0,0) = sequences[i]->m_camParamForBundle[3];
		sequences[i]->m_t.at<double>(1,0) = sequences[i]->m_camParamForBundle[4];
		sequences[i]->m_t.at<double>(2,0) = sequences[i]->m_camParamForBundle[5];
		//printMatrix("t",sequences[i]->m_t);

		sequences[i]->m_K.at<double>(0,0) = sequences[i]->m_camParamForBundle[6];
		sequences[i]->m_K.at<double>(1,1) = sequences[i]->m_camParamForBundle[6];

		Mat M(3,4,CV_64F);
		sequences[i]->m_R.copyTo(M.colRange(0,3));
		sequences[i]->m_t.copyTo(M.colRange(3,4));
		Mat P = sequences[i]->m_K*M;
		CamParamSetting(*sequences[i],P,sequences[i]->m_K,sequences[i]->m_R,sequences[i]->m_t);
		//printMatrix("newK",sequences[i]->m_K);

		/*
		//save undistorted keypoint using updated parameters
		for(unsigned int t=0;t<sequences[i]->m_keypoints.size();++t)
		{
			sequences[i]->m_keypoints[t].pt = sequences[i]->ApplyUnditort(sequences[i]->m_originalKeyPt[t]);
		}*/

	}
	for (unsigned int i = 0; i < Pts3D.size(); ++i) 
	{
		Pts3D[i]->m_pt3D.x = pt[i][0];
		Pts3D[i]->m_pt3D.y = pt[i][1];
		Pts3D[i]->m_pt3D.z = pt[i][2];

		Pts3D[i]->m_ptMat4by1.at<double>(0,0) = Pts3D[i]->m_pt3D.x;
		Pts3D[i]->m_ptMat4by1.at<double>(1,0) = Pts3D[i]->m_pt3D.y;
		Pts3D[i]->m_ptMat4by1.at<double>(2,0) = Pts3D[i]->m_pt3D.z;
	}

	for (unsigned int i = 0; i < Pts3D.size(); ++i) 
		delete[] pt[i];
	delete[] pt;

	return (summary.initial_cost - summary.final_cost);
}

#else

double BundleAdjustment_ceres(vector<CamViewDT*>& sequences,vector<TrajElement3D*>& Pts3D)
{
	for(unsigned int i=0;i<sequences.size();++i)
	{
		Mat rotationVector = Mat::zeros(3,1,CV_64F);
		Rodrigues(sequences[i]->m_R, rotationVector );
		sequences[i]->m_camParamForBundle[0] = rotationVector.at<double>(0,0);
		sequences[i]->m_camParamForBundle[1] = rotationVector.at<double>(1,0);
		sequences[i]->m_camParamForBundle[2] = rotationVector.at<double>(2,0);
		//printMatrix("R",rotationVector);

		sequences[i]->m_camParamForBundle[3] = sequences[i]->m_t.at<double>(0,0);
		sequences[i]->m_camParamForBundle[4] = sequences[i]->m_t.at<double>(1,0);
		sequences[i]->m_camParamForBundle[5] = sequences[i]->m_t.at<double>(2,0);

		sequences[i]->m_camParamForBundle[6] = sequences[i]->m_K.at<double>(0,0);
		//sequences[i]->m_camParamForBundle[7] = 0;//sequences[i]->m_K.at<double>(0,0);
		//sequences[i]->m_camParamForBundle[8] = 0;//sequences[i]->m_K.at<double>(0,0);
		//printMatrix("t",sequences[i]->m_t);
	}
	
	double** pt = new double*[Pts3D.size()];
	for (unsigned int i = 0; i < Pts3D.size(); ++i) 
	{
		pt[i] = new double[3];
		pt[i][0] = Pts3D[i]->m_pt3D.x;
		pt[i][1] = Pts3D[i]->m_pt3D.y;
		pt[i][2] = Pts3D[i]->m_pt3D.z;
	}

	//double bundleMaxNum =8000;
	//int searchEnd = Pts3D.size()-bundleMaxNum;
	//if(searchEnd<0)  searchEnd=0;
	// Create residuals for each observation in the bundle adjustment problem. The
	// parameters for cameras and points are added automatically.
	ceres::Problem problem;
	for (unsigned int i = 0; i < Pts3D.size(); ++i) 
	//for (int i = (int)Pts3D.size()-1; i>=searchEnd  ; --i) 
	{
		for(unsigned int k =0;k<Pts3D[i]->m_associatedViews.size();++k)
		{
			int imageIdx = Pts3D[i]->m_associatedViews[k].camIdx;
			int keyIdx = Pts3D[i]->m_associatedViews[k].keyPtIdx;
			Point2f originalPt = Pts3D[i]->m_associatedViews[k].originalImagePt;
			
			bool firstCam = false;
			if(imageIdx ==0)
				firstCam = true;

			//if(imageIdx ==1 && sequences.size()>2)		//if sequences.size()==2, the apply bundle for secondOne
			if(imageIdx ==1)		//if sequences.size()==2, the apply bundle for secondOne
				firstCam = true;

			// Each Residual block takes a point and a camera as input and outputs a 2
			// dimensional residual. Internally, the cost function stores the observed
			// image location and compares the reprojection against the observation.
			ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 8, 3>(
						/*new SnavelyReprojectionError(
							sequences[imageIdx]->m_keypoints[keyIdx].pt.x-sequences[imageIdx]->m_K.at<double>(0,2),
							sequences[imageIdx]->m_keypoints[keyIdx].pt.y-sequences[imageIdx]->m_K.at<double>(1,2),
							firstCam,sequences[imageIdx]->m_camParamForBundle)
							);*/

						new SnavelyReprojectionError(
							originalPt.x-sequences[imageIdx]->m_K.at<double>(0,2),
							originalPt.y-sequences[imageIdx]->m_K.at<double>(1,2),
							firstCam,sequences[imageIdx]->m_camParamForBundle)
							);

			problem.AddResidualBlock(cost_function,
					NULL /* squared loss */ ,
					sequences[imageIdx]->m_camParamForBundle,
					pt[i]
					);
		}
	}
	
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;	
	options.minimizer_progress_to_stdout = false;//true;
	options.gradient_tolerance = 1e-16;
	options.function_tolerance = 1e-16;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if(summary.initial_cost >summary.final_cost)
		std::cout << summary.FullReport() << "\n";

	for(unsigned int i=0;i<sequences.size();++i)
	{
		Mat rotationVector = Mat::zeros(3,1,CV_64F);
		rotationVector.at<double>(0,0) = sequences[i]->m_camParamForBundle[0];
		rotationVector.at<double>(1,0) = sequences[i]->m_camParamForBundle[1];
		rotationVector.at<double>(2,0) = sequences[i]->m_camParamForBundle[2];
		Rodrigues(rotationVector,sequences[i]->m_R);
		//printMatrix("R",sequences[i]->m_R);

		sequences[i]->m_t.at<double>(0,0) = sequences[i]->m_camParamForBundle[3];
		sequences[i]->m_t.at<double>(1,0) = sequences[i]->m_camParamForBundle[4];
		sequences[i]->m_t.at<double>(2,0) = sequences[i]->m_camParamForBundle[5];
		//printMatrix("t",sequences[i]->m_t);

		sequences[i]->m_K.at<double>(0,0) = sequences[i]->m_camParamForBundle[6];
		sequences[i]->m_K.at<double>(1,1) = sequences[i]->m_camParamForBundle[6];

		Mat M(3,4,CV_64F);
		sequences[i]->m_R.copyTo(M.colRange(0,3));
		sequences[i]->m_t.copyTo(M.colRange(3,4));
		Mat P = sequences[i]->m_K*M;
		CamParamSetting(*sequences[i],P,sequences[i]->m_K,sequences[i]->m_R,sequences[i]->m_t);
		//printMatrix("newK",sequences[i]->m_K);
	}
	for (unsigned int i = 0; i < Pts3D.size(); ++i) 
	{
		Pts3D[i]->m_pt3D.x = pt[i][0];
		Pts3D[i]->m_pt3D.y = pt[i][1];
		Pts3D[i]->m_pt3D.z = pt[i][2];

		Pts3D[i]->m_ptMat4by1.at<double>(0,0) = Pts3D[i]->m_pt3D.x;
		Pts3D[i]->m_ptMat4by1.at<double>(1,0) = Pts3D[i]->m_pt3D.y;
		Pts3D[i]->m_ptMat4by1.at<double>(2,0) = Pts3D[i]->m_pt3D.z;
	}

	for (unsigned int i = 0; i < Pts3D.size(); ++i) 
		delete[] pt[i];
	delete[] pt;

	return (summary.initial_cost - summary.final_cost);
}

#endif



template <typename T>
bool SingleReconPatchPosEstCostFunc::operator()(const T* const camRot,
					const T* const camTrans,
					T* residuals) const 
{
	// camera[0,1,2] are the angle-axis rotation.
	T p[3];
	p[0] = T(observed_3D[0]); p[1] = T(observed_3D[1]); p[2] = T(observed_3D[2]);
	ceres::AngleAxisRotatePoint(camRot, p, p);
	//p[0] = T(observed_3D[0]); p[1] = T(observed_3D[1]); p[2] = T(observed_3D[2]);
	// camTrans[0,1,2] are the translation.
	p[0] += camTrans[0]; p[1] += camTrans[1]; p[2] += camTrans[2];

	// Compute the center of distortion. The sign change comes from
	// the camera model that Noah Snavely¡¯s Bundler assumes, whereby
	// the camera coordinate system has a negative z axis.
	T xp = p[0] / p[2];
	T yp = p[1] / p[2];

	//projection
	xp = KMatrix[0]*xp + KMatrix[2];
	yp = KMatrix[1]*yp + KMatrix[3];
/*
	// Apply second and fourth order radial distortion.
	const T& l1 = camera[7];
	const T& l2 = camera[8];
	T r2 = xp*xp + yp*yp;
	T distortion = T(1.0) + r2 * (l1 + l2 * r2);
	// Compute final projected point position.
	const T& focal = camera[6];
	T predicted_x = focal * distortion * xp;
	T predicted_y = focal * distortion * yp;
	*/

	// The error is the difference between the predicted and observed position.
	residuals[0] = xp - T(observed_ImX);
	residuals[1] = yp - T(observed_ImY);

	return 	true;
}


//Given 3D and 2D correspondences, and initial camera - patch relative position, find local optimal new cam position (that is, new R and t)
//Initial values are transmitted thorugh R and T
//Also, return values are saved in R and t
void SingleReconPatchPosEst(vector<Point3d>& corresPt3Ds,vector<Point2d>& corresPt2Ds,Mat& K,Mat& R,Mat& t)
{
	double K11 = K.at<double>(0,0);
	double K22 = K.at<double>(1,1);
	double K13 = K.at<double>(0,2);
	double K23 = K.at<double>(1,2);

	double paramRAngleAxis[3];
	Mat angleAxis = Mat::zeros(3,1,CV_64F);
	Rodrigues(R,angleAxis);
	memcpy(paramRAngleAxis,angleAxis.data,sizeof(double)*3);
	/*if(paramRAngleAxis[0] + paramRAngleAxis[1] + paramRAngleAxis[2]  <1e-5)
	{
		paramRAngleAxis[2] = 2*PI;  //??
	}*/

	double paramTrans[3];
	paramTrans[0] = t.at<double>(0,0);
	paramTrans[1] = t.at<double>(1,0);
	paramTrans[2] = t.at<double>(2,0);
	
	ceres::Problem problem;
	for (unsigned int i = 0; i < corresPt3Ds.size(); ++i)
	{
		ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<SingleReconPatchPosEstCostFunc, 2, 3, 3>(
			new SingleReconPatchPosEstCostFunc(K11,K22,K13,K23,corresPt2Ds[i].x,corresPt2Ds[i].y,corresPt3Ds[i].x,corresPt3Ds[i].y,corresPt3Ds[i].z));
		problem.AddResidualBlock(cost_function,
								NULL, //squared loss 
								paramRAngleAxis,
								paramTrans);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	//options.minimizer_progress_to_stdout = true;
	options.parameter_tolerance = 1e-20;
	options.function_tolerance = 1e-20;
	options.gradient_tolerance = 1e-20;
	options.max_num_iterations = 1000;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	//if(summary.initial_cost >summary.final_cost)
	//	std::cout << summary.FullReport() << "\n";

	t.at<double>(0,0) = paramTrans[0] ;
	t.at<double>(1,0) = paramTrans[1] ;
	t.at<double>(2,0) = paramTrans[2] ;

	memcpy(angleAxis.data,paramRAngleAxis,sizeof(double)*3);
	Rodrigues(angleAxis,R);
}
/*
//for normal search
double FindBestRatioForNormal(Mat& pt_1,Mat& pt_2,Point2d& centerPt,Mat& KMatrix)
{
	double bestD = 1;
	double dummy =0;

	ceres::Problem problem;
	ceres::CostFunction* cost_function =
		new ceres::AutoDiffCostFunction<DiagonalAngleDetector, 1, 1, 1>(
		new DiagonalAngleDetector(pt_1,pt_2,KMatrix,centerPt));
	problem.AddResidualBlock(cost_function,
							NULL, //squared loss 
							&bestD,
							&dummy);

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	//options.minimizer_progress_to_stdout = true;
	options.parameter_tolerance = 1e-15;
	//options.function_tolerance = 1e-40;
	options.gradient_tolerance = 1e-20;
	//options.max_num_iterations = 100;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	//std::cout << summary.FullReport() << "\n";

	return bestD;
}*/
/*
template <typename T>
bool DiagonalAngleDetector::operator()(const T* const d,
					const T* const dummy,
					T* residuals) const 
{
	T  ratio = d[0];
	T temp_pt2_x = pt2_x*ratio;
	T temp_pt2_y = pt2_y*ratio;
	T temp_pt2_z = pt2_z*ratio;

	T centerPt3D[3];
	centerPt3D[0] = (pt1_x+temp_pt2_x)/2.0;
	centerPt3D[1] = (pt1_y+temp_pt2_y)/2.0;
	centerPt3D[2] = (pt1_z+temp_pt2_z)/2.0;

	// the camera coordinate system has a negative z axis.
	T xp = centerPt3D[0] / centerPt3D[2];
	T yp = centerPt3D[1] / centerPt3D[2];

	//projection
	xp = KMatrix[0]*xp + KMatrix[2];
	yp = KMatrix[1]*yp + KMatrix[3];

	xp = T(observed_centerX)-xp;
	yp = T(observed_centerY)-yp;
	residuals[0] = xp*xp +yp*yp;
	return true;
}
*/

#if 0

//patch optimization
void PatchOptimization_ceres(vector<TrajElement3D*>& pts3D, vector<CamViewDT*>& sequences,bool debug)
{
	//google::LogToStderr();
	//nonlinear optimization
	for(unsigned int ptIdx =0;ptIdx<pts3D.size();++ptIdx)
		pts3D[ptIdx]->m_normal.copyTo(pts3D[ptIdx]->m_oldNormal);

	for(unsigned int ptIdx =0;ptIdx<pts3D.size();++ptIdx)
	{
//		if(!(ptIdx == 1508 || ptIdx == 2103 || ptIdx == 1507 || ptIdx == 2101 || ptIdx == 2102 ||ptIdx == 487 || ptIdx == 1503 || ptIdx == 1723 ||ptIdx == 2095 ||ptIdx == 1487 ||ptIdx == 1489 ||ptIdx == 1488))
#if SHOW_PATCH_OPTIMIZATION
		if(!(ptIdx == TARGET_PATCH_NUM))
			continue;
#endif

#if SHOW_PATCH_OPTIMIZATION
		printf("%d \n",ptIdx);
#endif
		if(pts3D[ptIdx]->m_patch.rows==0)
			continue;

		if(pts3D[ptIdx]->m_patch.rows<50)
			continue;

		double param[8];
		//parametar setting for each patch
		//3: x,y,z   
		//6: sx,sy,r,nx,ny,nz 
	
		param[0] = pts3D[ptIdx]->m_pt3D.x;
		param[1] = pts3D[ptIdx]->m_pt3D.y;
		param[2] = pts3D[ptIdx]->m_pt3D.z;
	
		Mat normal = pts3D[ptIdx]->m_arrow1StepVect4by1.rowRange(0,3).cross(pts3D[ptIdx]->m_arrow2StepVect4by1.rowRange(0,3));
		normalize(normal,normal);

		if(debug)
		{
			param[3] = (rand()%30)/100.0f;//0.2;//quat; 
			param[4] = (rand()%30)/100.0f;//normal.at<double>(0,0);
			param[5] = (rand()%30)/100.0f;//normal.at<double>(1,0);
			//param[5] = normal.at<double>(2,0);
		}
		else
		{
			param[3] = 0;
			param[4] = 0;
			param[5] = 0;
		}

		double sx = norm(pts3D[ptIdx]->m_arrow1StepVect4by1);
		double sy = norm(pts3D[ptIdx]->m_arrow2StepVect4by1);
		
		param[6] = sx;
		param[7] = sy;

		//debug
		if(debug)
		{
			Mat rotMatrix;
			Mat rotationVector = Mat::zeros(3,1,CV_64F);
			rotationVector.at<double>(0,0) = param[3];
			rotationVector.at<double>(1,0) = param[4];
			rotationVector.at<double>(2,0) = param[5];
			Rodrigues(rotationVector, rotMatrix);

			Mat arrow1st;
			pts3D[ptIdx]->m_arrow1StepVect4by1.rowRange(0,3).copyTo(arrow1st);
			arrow1st = rotMatrix *arrow1st;

			Mat arrow2nd;
			pts3D[ptIdx]->m_arrow2StepVect4by1.rowRange(0,3).copyTo(arrow2nd);
			arrow2nd = rotMatrix *arrow2nd;

			arrow1st.copyTo(pts3D[ptIdx]->m_arrow1StepVect4by1.rowRange(0,3));
			arrow2nd.copyTo(pts3D[ptIdx]->m_arrow2StepVect4by1.rowRange(0,3));
			continue;;
		}

	/*	Mat xAxis = Mat::zeros(3,1,CV_64F);
		xAxis.at<double>(0,0) = 1;
		Mat arrow2nd =  normal.cross(xAxis);
		normalize(arrow2nd,arrow2nd);
		Mat arrow1st =  arrow2nd.cross(normal);
		normalize(arrow1st,arrow1st);

		Mat orignalArrow1UnitAxis;
		pts3D[ptIdx]->m_arrow1StepVect4by1.rowRange(0,3).copyTo(orignalArrow1UnitAxis);
		normalize(orignalArrow1UnitAxis,orignalArrow1UnitAxis);
		float cosAngle = orignalArrow1UnitAxis.dot(arrow1st);
		double rotationAngle = acos(cosAngle);// *180/PI;
		if(normal.dot(arrow1st.cross(orignalArrow1UnitAxis))<0)
			rotationAngle = -rotationAngle;
		param[7]= rotationAngle;//+0.05; //temporary

		Mat orignalArrow2UnitAxis;
		pts3D[ptIdx]->m_arrow2StepVect4by1.rowRange(0,3).copyTo(orignalArrow2UnitAxis);
		normalize(orignalArrow2UnitAxis,orignalArrow2UnitAxis);
		float cosAngle2 = orignalArrow2UnitAxis.dot(arrow1st);
		double rotationAngle2 = acos(cosAngle2);// *180/PI;
		if(normal.dot(arrow1st.cross(orignalArrow2UnitAxis))<0)
			rotationAngle2 = -rotationAngle2;
		param[8]= rotationAngle2-rotationAngle;// *180/PI;			//this is fixed
		*/

#if SHOW_PATCH_OPTIMIZATION
		std::cout << "Original Parameters: ";
		for(int i=0;i<8;++i)
			std::cout << param[i] << " ";	
		std::cout << "\n";

		cout <<"original\n";
		std::cout << pts3D[ptIdx]->m_arrow1StepVect4by1.at<double>(0,0) <<" " << pts3D[ptIdx]->m_arrow1StepVect4by1.at<double>(1,0)<<" " << pts3D[ptIdx]->m_arrow1StepVect4by1.at<double>(2,0)<<"\n";
		std::cout << pts3D[ptIdx]->m_arrow2StepVect4by1.at<double>(0,0) <<" " << pts3D[ptIdx]->m_arrow2StepVect4by1.at<double>(1,0)<<" " << pts3D[ptIdx]->m_arrow2StepVect4by1.at<double>(2,0)<<"\n";
#endif		
		double originalParam[8];
		for(int i=0;i<8;++i)
			originalParam[i] = param[i];
		
		Problem problem;
		for(unsigned int i =0;i<pts3D[ptIdx]->m_associatedViews.size();++i)
		//for(unsigned int i =0;i<pts3D[ptIdx]->m_associatedViews.size()-1;++i)
		{
			int imageIdx = pts3D[ptIdx]->m_associatedViews[i].frameIdx;

			CostFunction* cost_function =
						new NumericDiffCostFunction<PatchOptCostFunction, CENTRAL, 1, 6> (
								new PatchOptCostFunction(sequences[imageIdx],pts3D[ptIdx],param,8,5), TAKE_OWNERSHIP);

			 // Build the problem.
			problem.AddResidualBlock(cost_function, NULL, param);
		}
		/*
		Problem problem;
		//int imageIdx = pts3D[ptIdx]->m_associatedViews[i].frameIdx;

		CostFunction* cost_function =
					new NumericDiffCostFunction<PatchOptCostFunction2, CENTRAL, 1, 6> (
							new PatchOptCostFunction2(sequences,pts3D[ptIdx],normal,param,8), TAKE_OWNERSHIP);
		// Build the problem.
		problem.AddResidualBlock(cost_function, NULL, param);*/

#define solver
#ifdef solver
		// Run the solver!
		Solver::Options options;
		options.max_num_iterations = 40;
		options.linear_solver_type = DENSE_QR;
#if SHOW_PATCH_OPTIMIZATION
		options.minimizer_progress_to_stdout = true;
#endif
		options.parameter_tolerance = 1e-12;
		//options.numeric_derivative_relative_step_size =1;//e-12;
		Solver::Summary summary;
		Solve(options, &problem, &summary);
		//std::cout << summary.BriefReport() << "\n";
		//std::cout << "sX : " << initial_x 	<< " -> " << x << "\n";
		//printf("ceres Start::done\n");
#if SHOW_PATCH_OPTIMIZATION
		std::cout << summary.FullReport() << "\n";
#endif
#endif

	//double newParam[8];

#if SHOW_PATCH_OPTIMIZATION
		std::cout << "original\n";
		for(int i=0;i<8;++i)
			std::cout << originalParam[i] << " ";
		std::cout << "\n";

		std::cout << "Final\n";
		/*for(int i=0;i<9;++i)
			std::cout << param[i] << " ";
		std::cout << "\n";*/

		for(int i=0;i<8;++i)
		{
			newParam[i] = originalParam[i] + (param[i]-originalParam[i])*paramScale[i];
			std::cout << newParam[i] << " ";	
		}
		std::cout << "\n";
		std::cout << "diff\n";
		for(int i=0;i<8;++i)
			std::cout << (param[i]-originalParam[i])*paramScale[i]<< " ";	
		std::cout << "\n";
#endif

		//result treatment
		pts3D[ptIdx]->m_pt3D.x = param[0];
		pts3D[ptIdx]->m_pt3D.y = param[1];
		pts3D[ptIdx]->m_pt3D.z = param[2];

		pts3D[ptIdx]->m_ptMat4by1.at<double>(0,0) = param[0];
		pts3D[ptIdx]->m_ptMat4by1.at<double>(1,0) = param[1];
		pts3D[ptIdx]->m_ptMat4by1.at<double>(2,0) = param[2];

		Mat rotMatrix;
		Mat rotationVector = Mat::zeros(3,1,CV_64F);
		rotationVector.at<double>(0,0) = param[3];
		rotationVector.at<double>(1,0) = param[4];
		rotationVector.at<double>(2,0) = param[5];
		Rodrigues(rotationVector, rotMatrix);

		Mat arrow1st;
		pts3D[ptIdx]->m_arrow1StepVect4by1.rowRange(0,3).copyTo(arrow1st);
		arrow1st = rotMatrix *arrow1st;

		Mat arrow2nd;
		pts3D[ptIdx]->m_arrow2StepVect4by1.rowRange(0,3).copyTo(arrow2nd);
		arrow2nd = rotMatrix *arrow2nd;

		//cout <<"After rotation: recalculated from paramter\n";
		//std::cout << arrow1st.at<double>(0,0) <<" " << arrow1st.at<double>(1,0)<<" " << arrow1st.at<double>(2,0)<<"\n";
		//std::cout << arrow2nd.at<double>(0,0) <<" " << arrow2nd.at<double>(1,0)<<" " << arrow2nd.at<double>(2,0)<<"\n";

		arrow1st.copyTo(pts3D[ptIdx]->m_arrow1StepVect4by1.rowRange(0,3));
		arrow2nd.copyTo(pts3D[ptIdx]->m_arrow2StepVect4by1.rowRange(0,3));

		Mat temp = pts3D[ptIdx]->m_ptMat4by1 + (pts3D[ptIdx]->m_arrow1StepVect4by1*pts3D[ptIdx]->m_patch.cols/2.0f);
		MatToPoint3d(temp,pts3D[ptIdx]->m_arrow1stHead);
		temp = pts3D[ptIdx]->m_ptMat4by1 + (pts3D[ptIdx]->m_arrow2StepVect4by1*pts3D[ptIdx]->m_patch.cols/2.0f);
		MatToPoint3d(temp,pts3D[ptIdx]->m_arrow2ndHead);

		pts3D[ptIdx]->Initialize();
	}
}
#endif


#if 0
//pts3D should be located on camera coordinate
void PatchLocalOptimization_ceres_ForTesting(TrajElement3D* pts3D, CamViewDT* pTargeImage,Point3d& centerOfMass)
{
	//google::LogToStderr();
	//nonlinear optimization

	double param[6];
	//parametar setting for each patch
	//3: x,y,z   
	//6: sx,sy,r,nx,ny,nz 
	
	param[0] = pts3D->m_pt3D.x;
	param[1] = pts3D->m_pt3D.y;
	param[2] = pts3D->m_pt3D.z;
	
	//angle axis
	param[3] = 0;
	param[4] = 0;
	param[5] = 0;

	//double sx = norm(pts3D->m_arrow1StepVect4by1);
	//double sy = norm(pts3D->m_arrow2StepVect4by1);
	double originalParam[8];
	for(int i=0;i<6;++i)
		originalParam[i] = param[i];

	//for(int iter=1;iter<=5;++iter)
	for(int iter=4;iter<=4;++iter)
	{
		int gridStep =1;
		switch(iter)
		{
			case 1:
				gridStep = 10;
			break;
			case 2:
				gridStep = 5;
			break;
			case 3:
				gridStep = 3;
			break;
			case 4:
				gridStep = 2;
			break;
			case 5:
				gridStep = 1;
			break;
		}
		Problem problem;
		CostFunction* cost_function =
		new NumericDiffCostFunction<PatchLocalOptCostFunction, CENTRAL, 1, 6> (
						new PatchLocalOptCostFunction(pTargeImage,pts3D,centerOfMass,param,6,gridStep), TAKE_OWNERSHIP);
						
		// Build the problem.
		problem.AddResidualBlock(cost_function, NULL, param);

		// Run the solver!
		Solver::Options options;
		options.max_num_iterations = 200;

		if(iter==5)
			options.max_num_iterations = 1000;

		options.linear_solver_type = DENSE_QR;
		//options.minimizer_progress_to_stdout = true;
		/*options.parameter_tolerance = 1e-40;
		options.function_tolerance= 1e-40;
		options.gradient_tolerance = 1e-40;*/
		//options.numeric_derivative_relative_step_size =1e-12;
		Solver::Summary summary;
		Solve(options, &problem, &summary);
		//std::cout << summary.FullReport() << "\n";

		/*std::cout << "\n";
		std::cout << "diff\n";
		for(int i=0;i<8;++i)
			std::cout << (param[i]-originalParam[i])<< " ";	
		std::cout << "\n";*/
	}

	//result treatment
	pts3D->m_pt3D.x = param[0];
	pts3D->m_pt3D.y = param[1];
	pts3D->m_pt3D.z = param[2];

	pts3D->m_ptMat4by1.at<double>(0,0) = param[0];
	pts3D->m_ptMat4by1.at<double>(1,0) = param[1];
	pts3D->m_ptMat4by1.at<double>(2,0) = param[2];

	Mat rotMatrix;
	Mat rotationVector = Mat::zeros(3,1,CV_64F);
	rotationVector.at<double>(0,0) = param[3];
	rotationVector.at<double>(1,0) = param[4];
	rotationVector.at<double>(2,0) = param[5];
	Rodrigues(rotationVector, rotMatrix);
	printMatrix(rotMatrix,"rotation result");

	Mat arrow1st;
	pts3D->m_arrow1StepVect4by1.rowRange(0,3).copyTo(arrow1st);
	//printMatrix(arrow1st,"before arrow1st");

	arrow1st = rotMatrix *arrow1st;
	//printMatrix(arrow1st,"after arrow1st");

	Mat arrow2nd;
	pts3D->m_arrow2StepVect4by1.rowRange(0,3).copyTo(arrow2nd);
	arrow2nd = rotMatrix *arrow2nd;

	//arrow1st.copyTo(pts3D->m_arrow1StepVect4by1.rowRange(0,3));
	//arrow2nd.copyTo(pts3D->m_arrow2StepVect4by1.rowRange(0,3));

	Mat temp = pts3D->m_ptMat4by1 + (arrow1st*pts3D->m_patch.cols/2.0f);
	MatToPoint3d(temp,pts3D->m_arrow1stHead);
	temp = pts3D->m_ptMat4by1 + (arrow2nd*pts3D->m_patch.cols/2.0f);
	MatToPoint3d(temp,pts3D->m_arrow2ndHead);

	pts3D->InitializeArrowVectNormal();
}
#endif

/*
bool PatchLocalOptCostFunction::Evaluate(double const* const* parameters,
						double* residuals,
						double** jacobians)  const
{
	//Patch projection
	double xCenter = p3dPt->m_patchGray.cols/2.0;
	double yCenter = p3dPt->m_patchGray.rows/2.0;

	double projectionError=0;
	int projectionErrorCnt=0;
		
	double initP[3];
	initP[0] = parameters[0][0];//parameters[0][0];
	initP[1] = parameters[0][1];//parameters[0][1];
	initP[2] = parameters[0][2];//parameters[0][2];

	Mat rotMatrix = Mat::eye(3,3,CV_64F);
	Mat rotationVector = Mat::zeros(3,1,CV_64F);
	rotationVector.at<double>(0,0) = parameters[0][3];
	rotationVector.at<double>(1,0) = parameters[0][4];
	rotationVector.at<double>(2,0) = parameters[0][5];
	Rodrigues(rotationVector, rotMatrix);

	Mat arrow1st;
	p3dPt->m_arrow1StepVect4by1.rowRange(0,3).copyTo(arrow1st);
	arrow1st = rotMatrix *arrow1st;

	Mat arrow2nd;
	p3dPt->m_arrow2StepVect4by1.rowRange(0,3).copyTo(arrow2nd);
	arrow2nd = rotMatrix *arrow2nd;

	//arrow2nd = arrow1st*m_innerParam[6];// newParam[6];//parameters[0][7];
	//arrow1st = arrow1st*m_innerParam[5];//*newParam[5];//parameters[0][6];
	Mat_<double>& targetPatch = (Mat_<double>&)m_targetPatch;  //to avoid repeated mem allocation
	for(int y=0;y<p3dPt->m_patchGray.rows;y+=gridStep)
	{
		for(int x=0;x<p3dPt->m_patchGray.cols;x+=gridStep)
		{
			double dx = x - xCenter;
			double dy = y - yCenter;

			Mat offset = arrow1st*dx;
			offset += arrow2nd*dy;
			double p[3];
			p[0] = initP[0]+offset.at<double>(0,0);
			p[1] = initP[1]+offset.at<double>(1,0);
			p[2] = initP[2]+offset.at<double>(2,0);

		//	/printMatrix("test",p3dPt->m_arrow2StepVect4by1);
		//	printMatrix("test",temp3DPt);

			//printMatrix(temp3DPt,"temp");
			double imagePt[3];
			imagePt[0] = pSequence->m_K.at<double>(0,0)*p[0]+ pSequence->m_K.at<double>(0,1)*p[1] + pSequence->m_K.at<double>(0,2)*p[2];
			imagePt[1] = pSequence->m_K.at<double>(1,0)*p[0]+ pSequence->m_K.at<double>(1,1)*p[1] + pSequence->m_K.at<double>(1,2)*p[2];
			imagePt[2] = pSequence->m_K.at<double>(2,0)*p[0]+ pSequence->m_K.at<double>(2,1)*p[1] + pSequence->m_K.at<double>(2,2)*p[2];
				
			imagePt[0] =imagePt[0]/imagePt[2];
			imagePt[1] =imagePt[1]/imagePt[2];

			if(IsOutofBoundary(pSequence->m_inputImage,imagePt[0],imagePt[1]))
			{
				residuals[0] =2;
				return true;
			}
			targetPatch(y,x) = BilinearInterpolation((Mat_<uchar>&)pSequence->m_inputImage,imagePt[0],imagePt[1]);
		}
	}
	double newLength = pow(initP[0] - m_centerOfMass.x,2) + pow(initP[1] - m_centerOfMass.y,2) + pow(initP[2] - m_centerOfMass.z,2);
	newLength = ceres::sqrt(newLength);
	double rigidity = ceres::abs(newLength- m_initDistFromCenter)/m_initDistFromCenter;  //0~1

	residuals[0] =  1 - myNCC(p3dPt->m_patchGray,targetPatch,gridStep);  // in best case, it is 0
	residuals[0] = residuals[0] +  2*rigidity;

	return true;
}*/



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Global Patch pose opimization (Cam pose of testCamera optimization)   ///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*

//Want to optimize global translation using projection
//Pts3Ds should be World coordinate
//pTargeImage contains initial m_P,m_R,m_t
void PatchGlobalOptimization_ceres(vector<TrajElement3D*>& pts3Ds, CamViewDT* pTargeImage)
{
	double param[6];
	//parametar setting for each patch
	
	param[0] = pTargeImage->m_t.at<double>(0,0);
	param[1] = pTargeImage->m_t.at<double>(1,0);
	param[2] = pTargeImage->m_t.at<double>(2,0);
	
	Mat rotationVector = Mat::zeros(3,1,CV_64F);
	Rodrigues(pTargeImage->m_R, rotationVector );
	param[3] = rotationVector.at<double>(0,0);
	param[4] = rotationVector.at<double>(1,0);
	param[5] = rotationVector.at<double>(2,0);

	double originalParam[8];
	for(int i=0;i<6;++i)
		originalParam[i] = param[i];

	//for(int iter=1;iter<=5;++iter)
	for(int iter=4;iter<=4;++iter)
	{
		int gridStep =1;
		switch(iter)
		{
			case 1:
				gridStep = 10;
			break;
			case 2:
				gridStep = 5;
			break;
			case 3:
				gridStep = 3;
			break;
			case 4:
				gridStep = 2;
			break;
			case 5:
				gridStep = 1;
			break;
		}
		Problem problem;
		CostFunction* cost_function =
		new NumericDiffCostFunction<PatchGlobalOptCostFunction, CENTRAL, 1, 6> (
						new PatchGlobalOptCostFunction(pTargeImage,&pts3Ds,param,6,gridStep), TAKE_OWNERSHIP);
						
		// Build the problem.
		problem.AddResidualBlock(cost_function, NULL, param);

		// Run the solver!
		Solver::Options options;
		options.max_num_iterations = 200;

		if(iter==5)
			options.max_num_iterations = 1000;

		options.linear_solver_type = DENSE_QR;
		options.minimizer_progress_to_stdout = true;
		//options.numeric_derivative_relative_step_size =1e-12;
		Solver::Summary summary;
		Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << "\n";
	}
	
	std::cout << "\n";
	std::cout << "diff\n";
	for(int i=0;i<8;++i)
		std::cout << (param[i]-originalParam[i])<<" ";	
	std::cout << "\n";

	//result treatment
	pTargeImage->m_t.at<double>(0,0) =param[0];
	pTargeImage->m_t.at<double>(1,0) =param[1];
	pTargeImage->m_t.at<double>(2,0) =param[2];

	rotationVector.at<double>(0,0) = param[3];
	rotationVector.at<double>(1,0) = param[4];
	rotationVector.at<double>(2,0) = param[5];
	Rodrigues(rotationVector,pTargeImage->m_R);

	Mat M(3,4,CV_64F);
	pTargeImage->m_R.copyTo(M.colRange(0,3));
	pTargeImage->m_t.copyTo(M.colRange(3,4));
	pTargeImage->m_P= pTargeImage->m_K*M;
	//CamParamSetting(*pTargeImage,P,sequences[i]->m_K,sequences[i]->m_R,sequences[i]->m_t);
}*/
/*
bool PatchGlobalOptCostFunction::Evaluate(double const* const* parameters,
						double* residuals,
						double** jacobians)  const
{
	Mat& trans= (Mat&)m_trans;  //to avoid repeated mem allocation
	trans.at<double>(0,0) = parameters[0][0];
	trans.at<double>(1,0) = parameters[0][1];
	trans.at<double>(2,0) = parameters[0][2];

	Mat& rotVector= (Mat&)m_rotVector;  //to avoid repeated mem allocation
	Mat& rotMatrix= (Mat&)m_rotMatrix;  //to avoid repeated mem allocation
	rotVector.at<double>(0,0) = parameters[0][3];
	rotVector.at<double>(1,0) = parameters[0][4];
	rotVector.at<double>(2,0) = parameters[0][5];
	Rodrigues(rotVector, rotMatrix);
	
	Mat arrow1st;
	Mat arrow2nd;
	Mat center;
	double errrorMean = 0;
	for(unsigned int patchIdx = 0;patchIdx<p3DPts->size();++patchIdx)
	{
		double xCenter = (*p3DPts)[patchIdx]->m_patchGray.cols/2.0;
		double yCenter = (*p3DPts)[patchIdx]->m_patchGray.rows/2.0;

		Point3dToMat((*p3DPts)[patchIdx]->m_arrow1stHead,arrow1st);
		arrow1st = rotMatrix*arrow1st + trans;

		Point3dToMat((*p3DPts)[patchIdx]->m_arrow2ndHead,arrow2nd);
		arrow2nd = rotMatrix*arrow2nd + trans;

		Point3dToMat((*p3DPts)[patchIdx]->m_pt3D,center);
		center = rotMatrix*center + trans;

		arrow1st =arrow1st-center;
		arrow1st = arrow1st / xCenter;

		arrow2nd =arrow2nd-center;
		arrow2nd = arrow2nd/yCenter;

		//arrow2nd = arrow1st*m_innerParam[6];// newParam[6];//parameters[0][7];
		//arrow1st = arrow1st*m_innerParam[5];//*newParam[5];//parameters[0][6];
		Mat_<double>& targetPatch = (Mat_<double>&)m_targetPatches[patchIdx];  //to avoid repeated mem allocation
		for(int y=0;y<(*p3DPts)[patchIdx]->m_patchGray.rows;y+=gridStep)
		{
			for(int x=0;x<(*p3DPts)[patchIdx]->m_patchGray.cols;x+=gridStep)
			{
				double dx = x - xCenter;
				double dy = y - yCenter;

				Mat offset = arrow1st*dx;
				offset += arrow2nd*dy;
				double p[3];
				p[0] = center.at<double>(0,0)+offset.at<double>(0,0);
				p[1] = center.at<double>(1,0)+offset.at<double>(1,0);
				p[2] = center.at<double>(2,0)+offset.at<double>(2,0);

			//	/printMatrix("test",p3dPt->m_arrow2StepVect4by1);
			//	printMatrix("test",temp3DPt);

				//printMatrix(temp3DPt,"temp");
				double imagePt[3];
				imagePt[0] = pSequence->m_K.at<double>(0,0)*p[0]+ pSequence->m_K.at<double>(0,1)*p[1] + pSequence->m_K.at<double>(0,2)*p[2];
				imagePt[1] = pSequence->m_K.at<double>(1,0)*p[0]+ pSequence->m_K.at<double>(1,1)*p[1] + pSequence->m_K.at<double>(1,2)*p[2];
				imagePt[2] = pSequence->m_K.at<double>(2,0)*p[0]+ pSequence->m_K.at<double>(2,1)*p[1] + pSequence->m_K.at<double>(2,2)*p[2];
				
				imagePt[0] =imagePt[0]/imagePt[2];
				imagePt[1] =imagePt[1]/imagePt[2];

				if(IsOutofBoundary(pSequence->m_inputImage,imagePt[0],imagePt[1]))
				{
					residuals[0] =2;
					return true;
				}
				targetPatch(y,x) = BilinearInterpolation((Mat_<uchar>&)pSequence->m_inputImage,imagePt[0],imagePt[1]);
			}
		}
		 
		errrorMean += 1 - myNCC((*p3DPts)[patchIdx]->m_patchGray,targetPatch,gridStep);  // in best case, it is 0
	}
	
	residuals[0] =  errrorMean /p3DPts->size();
	
	return true;
}
*/

//revisited 13.0809  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Normal Estimation general Version
void PatchOptimziationForTracking_ceres(TrajElement3D* pPts3D, int refImageIdx,vector<int>& visbilityIdxVector,vector<CamViewDT*>& sequences,int stepSIze,bool bDoFastOpt)
{
	if(visbilityIdxVector.size() <2)
		return;

	CamViewDT* pRefImage = sequences[refImageIdx];

	double param[3];  //depth for center,  depth for arrow1, depth for arrow2
	//calculate initial depth
	Mat_<double> ptCenter;
	Point3dToMat(pPts3D->m_pt3D,ptCenter);
	ptCenter = pRefImage->m_R * ptCenter + pRefImage->m_t;  //cameraCordinate
	param[0] = ptCenter(2,0);
	ptCenter = ptCenter/ptCenter(2,0);  //ray

	Mat_<double> ptArrow1;
	Point3dToMat(pPts3D->m_arrow1stHead,ptArrow1);
	ptArrow1 = pRefImage->m_R * ptArrow1 + pRefImage->m_t;  //cameraCordinate
	param[1] = ptArrow1(2,0); //depth
	ptArrow1 = ptArrow1/ptArrow1(2,0); //ray, ray*[depth] = 3D local point

	Mat_<double> ptArrow2;
	Point3dToMat(pPts3D->m_arrow2ndHead,ptArrow2);
	ptArrow2 = pRefImage->m_R * ptArrow2 + pRefImage->m_t;  //cameraCordinate
	param[2] = ptArrow2(2,0); //depth
	ptArrow2 = ptArrow2/ptArrow2(2,0); //ray

//#if SHOW_PATCH_OPTIMIZATION
	double originalParam[3];
	memcpy(originalParam,param,sizeof(double)*3);
	//printfLog("Start: Original Parameters: %f, %f, %f\n",param[0],param[1],param[2]);
//#endif		

	Problem problem;
	for(unsigned int i =0;i<visbilityIdxVector.size();++i)
	{
		int imageIdx = visbilityIdxVector[i];
		if(imageIdx == refImageIdx)
			continue;

		CostFunction* cost_function =
					new NumericDiffCostFunction<PatchNormalOptCostFunctionUsingTripleRays, CENTRAL, 1, 3> (
							new PatchNormalOptCostFunctionUsingTripleRays(pRefImage,sequences[imageIdx],pPts3D,ptCenter,ptArrow1,ptArrow2,stepSIze), TAKE_OWNERSHIP);
							
		// Build the problem.
		problem.AddResidualBlock(cost_function, NULL, param);
	}

	//int imageIdx = pts3D[ptIdx]->m_associatedViews[i].frameIdx;

#define solver
#ifdef solver
	// Run the solver!
	Solver::Options options;
	options.linear_solver_type = DENSE_QR;

	if(bDoFastOpt)
	{
		options.max_num_iterations = 50;
		//options.parameter_tolerance = 1e-14;
		//options.function_tolerance = 1e-14;
	}
	else
	{
		options.max_num_iterations = 200;
		options.parameter_tolerance = 1e-14;
		options.function_tolerance = 1e-14;
	}

	

	//options.numeric_derivative_relative_step_size =1;//e-12;
	Solver::Summary summary;
	Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	//std::cout << "sX : " << initial_x 	<< " -> " << x << "\n";
	//printf("ceres Start::done\n");
//#if SHOW_PATCH_OPTIMIZATION
	//std::cout << summary.FullReport() << "\n";
//#endif
#endif
	
	Mat_<double> center = ptCenter*param[0];  //cam coord
	center = pRefImage->m_invR * (center - pRefImage->m_t);  //world coord
	MatToPoint3d(center,pPts3D->m_pt3D);
	pPts3D->m_ptMat4by1.at<double>(0,0) = pPts3D->m_pt3D.x;
	pPts3D->m_ptMat4by1.at<double>(1,0) = pPts3D->m_pt3D.y;
	pPts3D->m_ptMat4by1.at<double>(2,0) = pPts3D->m_pt3D.z;
	pPts3D->m_ptMat4by1.at<double>(3,0) = 1;

	Mat_<double> arrow1 = ptArrow1*param[1];
	arrow1 = pRefImage->m_invR * (arrow1 - pRefImage->m_t);
	MatToPoint3d(arrow1,pPts3D->m_arrow1stHead);

	Mat_<double> arrow2 = ptArrow2*param[2];
	arrow2 = pRefImage->m_invR * (arrow2 - pRefImage->m_t);
	MatToPoint3d(arrow2,pPts3D->m_arrow2ndHead);

	pPts3D->InitializeArrowVectNormal();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//13.0809  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool PatchOptimizeForTrackingCostFunction::Evaluate(double const* const* param,
						double* residuals,
						double** jacobians)  const
{
	//Patch 3D estimations
	Mat_<double> center = rayCenter*param[0][0];
	Mat_<double> arrow1 = rayArrow1*param[0][1];   //camCenter to arrow1Head
	Mat_<double> arrow2 = rayArrow2*param[0][2];
	Mat_<double> temp3D;
	Mat_<double> temp3D4by1= Mat_<double>::ones(4,1);
	Mat_<double> projPt;
	Point2d imagePt;
	//calculate axis direction from patch center
	arrow1  = arrow1 - center;
	arrow1 = arrow1 /xCenter;  //actually, xCenter == yCenter
	arrow2  = arrow2 - center;
	arrow2 = arrow2 /yCenter;  
	
	Mat_<double>& refPatch = (Mat_<double>&)m_refPatch;  //to avoid repeated mem allocation
	Mat_<double>& targetPatch = (Mat_<double>&)m_targetPatch;  //to avoid repeated mem allocation

	double projectionError=0;
	int projectionErrorCnt=0;
		
	for(int y=0;y<PATCH3D_GRID_SIZE;y+=gridStep)
	{
		for(int x=0;x<PATCH3D_GRID_SIZE;x+=gridStep)
		{
			double dx = x - xCenter;
			double dy = y - yCenter;

			temp3D = center + arrow1*dx;
			temp3D += arrow2*dy;   //cam coord

			projPt = pRefImage->m_K *temp3D;
			imagePt.x = projPt(0,0)/projPt(2,0);
			imagePt.y = projPt(1,0)/projPt(2,0);
			if(IsOutofBoundary(pRefImage->m_inputImage,imagePt.x,imagePt.y))
			{
				residuals[0] =2;
				return true;
			}
			refPatch(y,x) = BilinearInterpolation((Mat_<uchar>&)pRefImage->m_inputImage,imagePt.x,imagePt.y);

			temp3D = pRefImage->m_invR*(temp3D - pRefImage->m_t); // world coord
			temp3D.copyTo(temp3D4by1.rowRange(0,3));
			projPt = pTargetImage->m_P * temp3D4by1;
			imagePt.x = projPt(0,0)/projPt(2,0);
			imagePt.y = projPt(1,0)/projPt(2,0);
			if(IsOutofBoundary(pTargetImage->m_inputImage,imagePt.x,imagePt.y))
			{
				residuals[0] =2;
				return true;
			}
			targetPatch(y,x) = BilinearInterpolation((Mat_<uchar>&)pTargetImage->m_inputImage,imagePt.x,imagePt.y);
		}
	}
	residuals[0] =  1 - myNCC(refPatch,targetPatch,gridStep);  // in best case, it is 0

	return true;
}

double TriangulationOptimization(vector<Mat*>& M,vector<Point2d>& p,Mat& X)
{
    //printf("TriagulateionOptimization: ptNum %d\n",p.size());
	double paramX[3];
	paramX[0] = X.at<double>(0,0);
	paramX[1] = X.at<double>(1,0);
	paramX[2] = X.at<double>(2,0);
	//double dummy=0;
	ceres::Problem problem;
	for (unsigned int i = 0; i < M.size(); ++i)
	{
		double camParam[12];
		memcpy(camParam,M[i]->data,sizeof(double)*12);
		// Each Residual block takes a point and a camera as input and outputs a 2
		// dimensional residual. Internally, the cost function stores the observed
		// image location and compares the reprojection against the observation.
		/*ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<ReprojectionErrorForTriangulation, 2, 3, 1>(
			new ReprojectionErrorForTriangulation(p[i].x,p[i].y,camParam));
		problem.AddResidualBlock(cost_function,
			NULL, //squared loss 
			paramX,
			&dummy);*/
		ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<ReprojectionErrorForTriangulation, 2, 3>(
			new ReprojectionErrorForTriangulation(p[i].x,p[i].y,camParam));
		problem.AddResidualBlock(cost_function,
			//NULL, //squared loss 
			new HuberLoss(2.0),
			paramX);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	//options.minimizer_progress_to_stdout = true;
	//options.parameter_tolerance = 1e-20;
	//options.function_tolerance= 1e-20;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//if(summary.initial_cost >summary.final_cost)
	//std::cout << summary.FullReport() << "\n";

	X.at<double>(0,0) = paramX[0];
	X.at<double>(1,0) = paramX[1];
	X.at<double>(2,0) = paramX[2];
	X.at<double>(3,0) = 1;

	return sqrt(float(summary.initial_cost - summary.final_cost))/M.size();
}

double TriangulationOptimizationWithWeight(vector<cv::Mat*>& M,vector<cv::Point2d>& p,vector<double>& weights,cv::Mat& X)
{
	//printf("TriagulateionOptimization: ptNum %d\n",p.size());
	double paramX[3];
	paramX[0] = X.at<double>(0,0);
	paramX[1] = X.at<double>(1,0);
	paramX[2] = X.at<double>(2,0);
	//double dummy=0;

	ceres::Problem problem;
	for (unsigned int i = 0; i < M.size(); ++i)
	{
		double camParam[12];
		memcpy(camParam,M[i]->data,sizeof(double)*12);
		// Each Residual block takes a point and a camera as input and outputs a 2
		// dimensional residual. Internally, the cost function stores the observed
		// image location and compares the reprojection against the observation.
		/*ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<ReprojectionErrorForTriangulation, 2, 3, 1>(
			new ReprojectionErrorForTriangulation(p[i].x,p[i].y,camParam));
		problem.AddResidualBlock(cost_function,
			NULL, //squared loss 
			paramX,
			&dummy);*/
		ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<ReprojectionErrorForTriangulation_weighted, 2, 3>(
			new ReprojectionErrorForTriangulation_weighted(p[i].x,p[i].y,weights[i],camParam));
		problem.AddResidualBlock(cost_function,
			NULL, //squared loss 
			paramX);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	//options.minimizer_progress_to_stdout = true;
	//options.parameter_tolerance = 1e-20;
	//options.function_tolerance= 1e-20;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//if(summary.initial_cost >summary.final_cost)
	//std::cout << summary.FullReport() << "\n";

	X.at<double>(0,0) = paramX[0];
	X.at<double>(1,0) = paramX[1];
	X.at<double>(2,0) = paramX[2];
	X.at<double>(3,0) = 1;

	return sqrt(float(summary.initial_cost - summary.final_cost))/M.size();
}


//Mat should be double
double TriangulationOptimizationF(vector<Mat*>& M,vector<Point2f>& p,Mat& X)
{
	double paramX[3];
	paramX[0] = X.at<double>(0,0);
	paramX[1] = X.at<double>(1,0);
	paramX[2] = X.at<double>(2,0);
	//double dummy=0;
	
	ceres::Problem problem;
	for (unsigned int i = 0; i < M.size(); ++i)
	{
		double camParam[12];
		memcpy(camParam,M[i]->data,sizeof(double)*12);
		// Each Residual block takes a point and a camera as input and outputs a 2
		// dimensional residual. Internally, the cost function stores the observed
		// image location and compares the reprojection against the observation.
		ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<ReprojectionErrorForTriangulation, 2, 3>(
			new ReprojectionErrorForTriangulation(p[i].x,p[i].y,camParam));

		/*
		ceres::CostFunction* cost_function =
			new ceres::NumericDiffCostFunction<ReprojectionErrorForTriangulation, ceres::CENTRAL, 1, 3>(
			new ReprojectionErrorForTriangulation(p[i].x,p[i].y,camParam),TAKE_OWNERSHIP);
			*/
		problem.AddResidualBlock(cost_function,
			NULL, //squared loss 
			paramX);
	}

	ceres::Solver::Options options;
	options.logging_type = SILENT;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	//options.parameter_tolerance = 1e-20;
	//options.numeric_derivative_relative_step_size;
	//options.function_tolerance= 1e-20;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//if(summary.initial_cost >summary.final_cost)
	//	std::cout << summary.FullReport() << "\n";
	
	X.at<double>(0,0) = paramX[0];
	X.at<double>(1,0) = paramX[1];
	X.at<double>(2,0) = paramX[2];
	X.at<double>(3,0) = 1;

	return (summary.initial_cost - summary.final_cost);
	return 0;
}

/*template <typename T>
bool ReprojectionErrorForTriangulation::operator()(const T* const pt,
					const T* const dummy,
					T* residuals) const */

template <typename T>
bool ReprojectionErrorForTriangulation::operator()(const T* const pt,
					T* residuals) const 
{
		T predicted[3];
		predicted[0] = T(camParam[0])*pt[0]+ T(camParam[1])*pt[1] + T(camParam[2])*pt[2] + T(camParam[3]);
		predicted[1] = T(camParam[4])*pt[0]+ T(camParam[5])*pt[1] + T(camParam[6])*pt[2] + T(camParam[7]);
		predicted[2] = T(camParam[8])*pt[0]+ T(camParam[9])*pt[1] + T(camParam[10])*pt[2] + T(camParam[11]);

		predicted[0] =predicted[0]/predicted[2];
		predicted[1] =predicted[1]/predicted[2];

		residuals[0]= T(observed_x) - predicted[0];
		residuals[1]= T(observed_y) - predicted[1];

		//residuals[0]= T(pow(predicted[0] - observed_x,2) + pow(predicted[1] - observed_y,2));
		/*residuals[0]= -pow(predicted[0] - T(observed_x),2);
		residuals[1]= -pow(predicted[1] - T(observed_y),2);*/

		return 	true;
}

bool ReprojectionErrorForTriangulation::Evaluate(double const* const* pt,
	double* residuals,
	double** jacobians) const
{
	
	/*for(int i=0;i<12;++i)
	{
		printf("%f ",camParam[i]);
		if((i+1)%4==0)
			printf("\n");
	}
	printf("pt: %f, %f, %f\n",pt[0][0],pt[0][1],pt[0][2]);*/

		double predicted[3];
		predicted[0] =camParam[0]*pt[0][0]+ camParam[1]*pt[0][1] + camParam[2]*pt[0][2] + camParam[3];
		predicted[1] = camParam[4]*pt[0][0]+ camParam[5]*pt[0][1] + camParam[6]*pt[0][2] + camParam[7];
		predicted[2] = camParam[8]*pt[0][0]+ camParam[9]*pt[0][1] + camParam[10]*pt[0][2] + camParam[11];

		predicted[0] =predicted[0]/predicted[2];
		predicted[1] =predicted[1]/predicted[2];

		//residuals[0]= predicted[0] - observed_x;
		//residuals[1]= predicted[1] - observed_y;

		residuals[0]= sqrtf(pow(predicted[0] - observed_x,2) + pow(predicted[1] - observed_y,2));
		//printf("%f\n",residuals[0]);
		//residuals[0]= pow(predicted[0] - (observed_x),2);
		//residuals[1]= pow(predicted[1] - (observed_y),2);

		return 	true;
}

template <typename T>
bool ReprojectionErrorForTriangulation_weighted::operator()(const T* const pt,
					T* residuals) const 
{
		T predicted[3];
		predicted[0] = T(camParam[0])*pt[0]+ T(camParam[1])*pt[1] + T(camParam[2])*pt[2] + T(camParam[3]);
		predicted[1] = T(camParam[4])*pt[0]+ T(camParam[5])*pt[1] + T(camParam[6])*pt[2] + T(camParam[7]);
		predicted[2] = T(camParam[8])*pt[0]+ T(camParam[9])*pt[1] + T(camParam[10])*pt[2] + T(camParam[11]);

		predicted[0] =predicted[0]/predicted[2];
		predicted[1] =predicted[1]/predicted[2];

		residuals[0]= T(observed_weight)*(T(observed_x) - predicted[0]);
		residuals[1]= T(observed_weight)*(T(observed_y) - predicted[1]);

		return 	true;
}

#if 0
bool ReprojectionErrorForTriangulation_weighted::Evaluate(double const* const* pt,
	double* residuals,
	double** jacobians) const
{
	
	/*for(int i=0;i<12;++i)
	{
		printf("%f ",camParam[i]);
		if((i+1)%4==0)
			printf("\n");
	}
	printf("pt: %f, %f, %f\n",pt[0][0],pt[0][1],pt[0][2]);*/

		double predicted[3];
		predicted[0] =camParam[0]*pt[0][0]+ camParam[1]*pt[0][1] + camParam[2]*pt[0][2] + camParam[3];
		predicted[1] = camParam[4]*pt[0][0]+ camParam[5]*pt[0][1] + camParam[6]*pt[0][2] + camParam[7];
		predicted[2] = camParam[8]*pt[0][0]+ camParam[9]*pt[0][1] + camParam[10]*pt[0][2] + camParam[11];

		predicted[0] =predicted[0]/predicted[2];
		predicted[1] =predicted[1]/predicted[2];

		//residuals[0]= predicted[0] - observed_x;
		//residuals[1]= predicted[1] - observed_y;

		residuals[0]= sqrtf(pow(predicted[0] - observed_x,2) + pow(predicted[1] - observed_y,2));
		//printf("%f\n",residuals[0]);
		//residuals[0]= pow(predicted[0] - (observed_x),2);
		//residuals[1]= pow(predicted[1] - (observed_y),2);

		return 	true;
}
#endif


#include "PatchOptimization.h"
void PoseReconNodeTrajOptimization_ceres(Point3f startPt,vector< vector<Mat_<double> > >& transformVector,vector< vector<pair<Point3d,double> > >& relatedPeaks,Point3f& optimizedStartPt)
{
	double pStartPt[3];
	pStartPt[0] = startPt.x;
	pStartPt[1] = startPt.y;
	pStartPt[2] = startPt.z;


	ceres::Problem problem;
	for(int b=0;b<transformVector.size();++b)
	{
		for(int f=0;f<transformVector[b].size();++f)
		{
			if(relatedPeaks.size()<=f)
				break;

			double* pTransf= transformVector[b][f].ptr<double>(0);

			for(int c=0;c<relatedPeaks[f].size();++c)
			{
				ceres::CostFunction* cost_function =
					new ceres::AutoDiffCostFunction<PoseReconJointTrajError, 1,3>(new PoseReconJointTrajError(pTransf,relatedPeaks[f]));

				problem.AddResidualBlock(cost_function,
					NULL /* squared loss */ ,
					pStartPt);
			}
		}
	}
	
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = false;//true;
	options.gradient_tolerance = 1e-16;
	options.function_tolerance = 1e-16;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	/*if(summary.initial_cost >summary.final_cost)
		std::cout << summary.FullReport() << "\n";*/


	optimizedStartPt.x = pStartPt[0];
	optimizedStartPt.y = pStartPt[1];
	optimizedStartPt.z = pStartPt[2];

}

#if 0
//patch optimizationRot_normToPatch
void NewPatchOptimization_ceres(TrajElement3D* pPts3D, int refImageIdx,vector<CamViewDT*>& sequences,int stepSIze,bool debug)
{
	CamViewDT* pRefImage = sequences[refImageIdx];

	double param[3];  //depth for center,  depth for arrow1, depth for arrow2
	//calculate initial depth
	Mat_<double> ptCenter;
	Point3dToMat(pPts3D->m_pt3D,ptCenter);
	ptCenter = pRefImage->m_R * ptCenter +pRefImage->m_t;  //cameraCordinate
	param[0] = ptCenter(2,0);
	ptCenter = ptCenter/ptCenter(2,0);  //ray

	Mat_<double> ptArrow1;
	Point3dToMat(pPts3D->m_arrow1stHead,ptArrow1);
	ptArrow1 = pRefImage->m_R * ptArrow1 + pRefImage->m_t;  //cameraCordinate
	param[1] = ptArrow1(2,0);
	ptArrow1 = ptArrow1/ptArrow1(2,0); //ray


	Mat_<double> ptArrow2;
	Point3dToMat(pPts3D->m_arrow2ndHead,ptArrow2);
	ptArrow2 = pRefImage->m_R * ptArrow2 + pRefImage->m_t;  //cameraCordinate
	param[2] = ptArrow2(2,0);
	ptArrow2 = ptArrow2/ptArrow2(2,0); //ray


//#if SHOW_PATCH_OPTIMIZATION
	double originalParam[3];
	memcpy(originalParam,param,sizeof(double)*3);
	//printfLog("Start: Original Parameters: %f, %f, %f\n",param[0],param[1],param[2]);
//#endif		

	Problem problem;
	for(unsigned int i =0;i<pPts3D->warpSequenceIdx.size();++i)
	{
		int imageIdx = pPts3D->warpSequenceIdx[i];
		if(imageIdx == refImageIdx)
			continue;

		CostFunction* cost_function =
					new NumericDiffCostFunction<PatchNormalOptCostFunctionUsingTripleRays, CENTRAL, 1, 3> (
							new PatchNormalOptCostFunctionUsingTripleRays(pRefImage,sequences[imageIdx],pPts3D,ptCenter,ptArrow1,ptArrow2,stepSIze), TAKE_OWNERSHIP);
							
		// Build the problem.
		problem.AddResidualBlock(cost_function, NULL, param);
	}

	//int imageIdx = pts3D[ptIdx]->m_associatedViews[i].frameIdx;

#define solver
#ifdef solver
	// Run the solver!
	Solver::Options options;
	options.max_num_iterations = 200;
	options.linear_solver_type = DENSE_QR;
//#if SHOW_PATCH_OPTIMIZATION
	//options.minimizer_progress_to_stdout = true;
//#endif
	options.parameter_tolerance = 1e-14;
	options.function_tolerance = 1e-14;

	//options.numeric_derivative_relative_step_size =1;//e-12;
	Solver::Summary summary;
	Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	//std::cout << "sX : " << initial_x 	<< " -> " << x << "\n";
	//printf("ceres Start::done\n");
//#if SHOW_PATCH_OPTIMIZATION
	//std::cout << summary.FullReport() << "\n";
//#endif
#endif
	/*
	{
		Problem problem;
		for(unsigned int i =0;i<pPts3D->warpSequenceIdx.size();++i)
		{
			int imageIdx = pPts3D->warpSequenceIdx[i];
			if(imageIdx == refImageIdx)
				continue;

			CostFunction* cost_function =
						new NumericDiffCostFunction<PatchNormalOptCostFunctionUsingTripleRays, CENTRAL, 1, 3> (
								new PatchNormalOptCostFunctionUsingTripleRays(pRefImage,sequences[imageIdx],pPts3D,ptCenter,ptArrow1,ptArrow2,1), TAKE_OWNERSHIP);
							
			// Build the problem.
			problem.AddResidualBlock(cost_function, NULL, param);
		}

		//int imageIdx = pts3D[ptIdx]->m_associatedViews[i].frameIdx;

	#define solver
	#ifdef solver
		// Run the solver!
		Solver::Options options;
		options.max_num_iterations = 200;
		options.linear_solver_type = DENSE_QR;
	//#if SHOW_PATCH_OPTIMIZATION
		//options.minimizer_progress_to_stdout = true;
	//#endif
		options.parameter_tolerance = 1e-14;
		options.function_tolerance = 1e-14;

		//options.numeric_derivative_relative_step_size =1;//e-12;
		Solver::Summary summary;
		Solve(options, &problem, &summary);
		//std::cout << summary.BriefReport() << "\n";
		//std::cout << "sX : " << initial_x 	<< " -> " << x << "\n";
		//printf("ceres Start::done\n");
	//#if SHOW_PATCH_OPTIMIZATION
		//std::cout << summary.FullReport() << "\n";
	//#endif
	#endif
	}
	*/
//#if SHOW_PATCH_OPTIMIZATION
	//printfLog("End: Original Parameters: %f, %f, %f\n",originalParam[0],originalParam[1],originalParam[2]);
	//printfLog("End: After optimization: %f, %f, %f\n",param[0],param[1],param[2]);
	//printfLog("End: Diff: %f, %f, %f\n",param[0]-originalParam[0],param[1]-originalParam[1],param[2]-originalParam[2]);
//#endif

	Mat_<double> center = ptCenter*param[0];  //cam coord
	center = pRefImage->m_invR * (center - pRefImage->m_t);  //world coord
	MatToPoint3d(center,pPts3D->m_pt3D);
	pPts3D->m_ptMat4by1.at<double>(0,0) = pPts3D->m_pt3D.x;
	pPts3D->m_ptMat4by1.at<double>(1,0) = pPts3D->m_pt3D.y;
	pPts3D->m_ptMat4by1.at<double>(2,0) = pPts3D->m_pt3D.z;
	pPts3D->m_ptMat4by1.at<double>(3,0) = 1;

	Mat_<double> arrow1 = ptArrow1*param[1];
	arrow1 = pRefImage->m_invR * (arrow1 - pRefImage->m_t);
	MatToPoint3d(arrow1,pPts3D->m_arrow1stHead);

	Mat_<double> arrow2 = ptArrow2*param[2];
	arrow2 = pRefImage->m_invR * (arrow2 - pRefImage->m_t);
	MatToPoint3d(arrow2,pPts3D->m_arrow2ndHead);

	pPts3D->InitializeArrowVectNormal();
}
#endif

bool PatchNormalOptCostFunctionUsingTripleRays::Evaluate(double const* const* param,
						double* residuals,
						double** jacobians)  const
{
	//Patch 3D estimations
	Mat_<double> center = rayCenter*param[0][0];
	Mat_<double> arrow1 = rayArrow1*param[0][1];
	Mat_<double> arrow2 = rayArrow2*param[0][2];
	Mat_<double> temp3D;
	Mat_<double> temp3D4by1= Mat_<double>::ones(4,1);
	Mat_<double> projPt;
	Point2d imagePt;
	//calculate axis direction
	arrow1  = arrow1 - center;
	arrow1 = arrow1 /xCenter;  //actually, xCenter == yCenter
	arrow2  = arrow2 - center;
	arrow2 = arrow2 /yCenter;  
	
	//Mat_<double>& refPatch = (Mat_<double>&)m_refPatch;  //to avoid repeated mem allocation
	//Mat_<double>& targetPatch = (Mat_<double>&)m_targetPatch;  //to avoid repeated mem allocation
	Mat_<double> refPatch(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);// = (Mat_<double>&)m_refPatch;  //to avoid repeated mem allocation
	Mat_<double> targetPatch(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);// = (Mat_<double>&)m_targetPatch;  //to avoid repeated mem allocation


	double projectionError=0;
	int projectionErrorCnt=0;
		
	for(int y=0;y<PATCH3D_GRID_SIZE;y+=gridStep)
	{
		for(int x=0;x<PATCH3D_GRID_SIZE;x+=gridStep)
		{
			double dx = x - xCenter;
			double dy = y - yCenter;

			temp3D = center + arrow1*dx;
			temp3D += arrow2*dy;   //cam coord

			projPt = pRefImage->m_K *temp3D;
			imagePt.x = projPt(0,0)/projPt(2,0);
			imagePt.y = projPt(1,0)/projPt(2,0);
			if(IsOutofBoundary(pRefImage->m_inputImage,imagePt.x,imagePt.y))
			{
				residuals[0] =2;
				return true;
			}
			refPatch(y,x) = BilinearInterpolation((Mat_<uchar>&)pRefImage->m_inputImage,imagePt.x,imagePt.y);

			temp3D = pRefImage->m_invR*(temp3D - pRefImage->m_t); // world coord
			temp3D.copyTo(temp3D4by1.rowRange(0,3));
			projPt = pTargetImage->m_P * temp3D4by1;
			imagePt.x = projPt(0,0)/projPt(2,0);
			imagePt.y = projPt(1,0)/projPt(2,0);
			if(IsOutofBoundary(pTargetImage->m_inputImage,imagePt.x,imagePt.y))
			{
				residuals[0] =2;
				return true;
			}
			targetPatch(y,x) = BilinearInterpolation((Mat_<uchar>&)pTargetImage->m_inputImage,imagePt.x,imagePt.y);
		}
	}
	residuals[0] =  1 - myNCC(refPatch,targetPatch,gridStep);  // in best case, it is 0

	return true;
}

//Assume that patch axises and normal is orthogonal
void PatchOptimziationUsingRayYawPitch_ceres(TrajElement3D* pPts3D, int refCamIdx,vector<int>& visbilityIdxVector,vector<CamViewDT*>& camVector,int stepSIze,bool bDoFastOpt)
{
	/* Test
	//Just rotate in one angle
	Mat arrow1_unit,arrow2_unit;
	normalize(pPts3D->m_arrow1StepVect4by1.rowRange(0,3),arrow1_unit);		//x axis
	normalize(pPts3D->m_arrow2StepVect4by1.rowRange(0,3),arrow2_unit);		//y axis
	Mat_<double> Rot_normToPatch(3,3);		//patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate
	Rot_normToPatch(0,0)= arrow1_unit.at<double>(0,0);
		Rot_normToPatch(1,0)= arrow1_unit.at<double>(1,0);
			Rot_normToPatch(2,0)= arrow1_unit.at<double>(2,0);
	Rot_normToPatch(0,1)= arrow2_unit.at<double>(0,0);
		Rot_normToPatch(1,1)= arrow2_unit.at<double>(1,0);
			Rot_normToPatch(2,1)= arrow2_unit.at<double>(2,0);
	Rot_normToPatch(0,2)= pPts3D->m_normal.at<double>(0,0);
		Rot_normToPatch(1,2)= pPts3D->m_normal.at<double>(1,0);
			Rot_normToPatch(2,2)= pPts3D->m_normal.at<double>(2,0);

	Mat_<double> Rot_patchToNorm= Rot_normToPatch.t();

	double euler[3] ={10,0,0};
	Mat_<double> Rot_byEulerMat(3,3);
	double* Rot_byEuler = (double*)Rot_byEulerMat.ptr();
	ceres::EulerAnglesToRotationMatrix(euler,3,Rot_byEuler);
	//printMatrix("Rot_byEulerMat",Rot_byEulerMat*Rot_patchToNorm);
	pPts3D->LocallyRotatePatch(Rot_normToPatch*Rot_byEulerMat*Rot_patchToNorm);*/

	if(visbilityIdxVector.size() <2)
		return;

	CamViewDT* pRefCam = camVector[refCamIdx];

	double param[3];  //distance of patchCenterRay,  yaw angle and pitch angle of rotation matrix 
	param[0] = norm(pPts3D->m_ptMat4by1.rowRange(0,3) - pRefCam->m_CamCenter);
	param[1] = param[2] = 0.0;
	
	Problem problem;
	for(unsigned int i =0;i<visbilityIdxVector.size();++i)
	{
		int imageIdx = visbilityIdxVector[i];
		if(imageIdx == refCamIdx)
			continue;

		CostFunction* cost_function =
					new NumericDiffCostFunction<PatchNormalOptCostFunctionUsingYawPitch, CENTRAL, 1, 3> (
					new PatchNormalOptCostFunctionUsingYawPitch(pRefCam,camVector[imageIdx],pPts3D,stepSIze), TAKE_OWNERSHIP);
							
		// Build the problem.
		problem.AddResidualBlock(cost_function, NULL, param);
	}

	//int imageIdx = pts3D[ptIdx]->m_associatedViews[i].frameIdx;

#define solver
#ifdef solver
	// Run the solver!
	Solver::Options options;
	options.linear_solver_type = DENSE_QR;

	if(bDoFastOpt)
	{
		options.max_num_iterations = 50;
		//options.parameter_tolerance = 1e-14;
		//options.function_tolerance = 1e-14;
	}
	else
	{
		options.max_num_iterations = 200;
		options.parameter_tolerance = 1e-14;
		options.function_tolerance = 1e-14;
	}

	//options.numeric_derivative_relative_step_size =1;//e-12;
	Solver::Summary summary;
	Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	//std::cout << "sX : " << initial_x 	<< " -> " << x << "\n";
	//printf("ceres Start::done\n");
//#if SHOW_PATCH_OPTIMIZATION
	//std::cout << summary.FullReport() << "\n";
//#endif
#endif
	
	PatchNormalOptCostFunctionUsingYawPitch::ApplyParametersToPatch(pPts3D,pRefCam->m_CamCenter,param);
}

bool PatchNormalOptCostFunctionUsingYawPitch::Evaluate(double const* const* param,
						double* residuals,
						double** jacobians)  const
{
	//Patch 3D estimations
	Mat_<double> patchCenter = m_rayDirect*param[0][0] +  m_pRefImage->m_CamCenter;			//Origin to patch center
	
	double angleAxis[3] ={param[0][1],param[0][2],0};		//yaw, pitch
	//double euler[3] ={0,param[0][2],0};		//yaw, pitch
	////double euler[3] ={0,param[0][1],param[0][2]};		//roll, yaw, pitch
	//double euler[3] ={0,param[0][1],0};		//roll, yaw, pitch
	
	Mat_<double> Rot_byEulerMat(3,3);
	double* Rot_byEuler = (double*)Rot_byEulerMat.ptr();			//TODO: consider!! ceres is column major where opencv is row major..
	ceres::AngleAxisToRotationMatrix(angleAxis,Rot_byEuler);
	Rot_byEulerMat = Rot_byEulerMat.t();		//opencv is row-major vs ceres is column major

	Mat_<double> R_toPatchCoord = m_Rot_normToPatch*Rot_byEulerMat;		//after rotation by params, rotate to the original patch coordinate
	Mat_<double> arrow1Step = R_toPatchCoord*m_arrow1Step_normCoord;		//Length is not unit vector. The length is half of the width of the patch (local x-axis)
	Mat_<double> arrow2Step = R_toPatchCoord*m_arrow2Step_normCoord;		

	Mat_<double> temp3D;
	Mat_<double> temp3D4by1= Mat_<double>::ones(4,1);
	Mat_<double> projPt;
	Point2d imagePt;
	//calculate axis direction
	//Mat_<double>& refPatch = (Mat_<double>&)m_refPatch;  //to avoid repeated mem allocation
	//Mat_<double>& targetPatch = (Mat_<double>&)m_targetPatch;  //to avoid repeated mem allocation
	Mat_<double> refPatch(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);// = (Mat_<double>&)m_refPatch;  //to avoid repeated mem allocation
	Mat_<double> targetPatch(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);// = (Mat_<double>&)m_targetPatch;  //to avoid repeated mem allocation

	double projectionError=0;
	int projectionErrorCnt=0;

	for(int y=0;y<PATCH3D_GRID_SIZE;y+=gridStep)
	{
		for(int x=0;x<PATCH3D_GRID_SIZE;x+=gridStep)
		{
			double dx = x - m_xCenter;
			double dy = y - m_yCenter;

			temp3D = patchCenter + arrow1Step*dx;
			temp3D += arrow2Step*dy;   //world coordinate
			temp3D.copyTo(temp3D4by1.rowRange(0,3));
			//Reference Cam
			projPt = m_pRefImage->m_P * temp3D4by1;
			imagePt.x = projPt(0,0)/projPt(2,0);
			imagePt.y = projPt(1,0)/projPt(2,0);
			if(IsOutofBoundary(m_pRefImage->m_inputImage,imagePt.x,imagePt.y))
			{
				residuals[0] =2;
				return true;
			}
			refPatch(y,x) = BilinearInterpolation((Mat_<uchar>&)m_pRefImage->m_inputImage,imagePt.x,imagePt.y);

			//Target Cam
			projPt = m_pTargetImage->m_P *temp3D4by1;
			imagePt.x = projPt(0,0)/projPt(2,0);
			imagePt.y = projPt(1,0)/projPt(2,0);
			if(IsOutofBoundary(m_pTargetImage->m_inputImage,imagePt.x,imagePt.y))
			{
				residuals[0] =2;
				return true;
			}
			targetPatch(y,x) = BilinearInterpolation((Mat_<uchar>&)m_pTargetImage->m_inputImage,imagePt.x,imagePt.y);
		}
	}
	residuals[0] =  1 - myNCC(refPatch,targetPatch,gridStep);  // in best case, it is 0
	return true;
}


void PatchOptimziationUsingPhotometricShapeRefine_ceres(TrajElement3D* pPts3D, int refImageIdx,vector<int>& visbilityIdxVector,vector<CamViewDT*>& sequences,int stepSIze,bool bDoFastOpt)
{
	if(visbilityIdxVector.size() <2)
		return;

	CamViewDT* pRefImage = sequences[refImageIdx];

	int gridSize = PATCH3D_GRID_SIZE; //should be odd number
	int halfSize = PATCH3D_GRID_HALFSIZE;//int(gridSize/2);
	//double scaleFactor = cm2world(patchArrowSizeCm)/Distance(m_arrow1stHead,m_pt3D);

	vector<double> params(PATCH3D_GRID_SIZE*PATCH3D_GRID_SIZE);
	vector< Mat_<double> > rays(PATCH3D_GRID_SIZE*PATCH3D_GRID_SIZE);
	//vector<cv::Point3f> vertexVectDebug;
	/*vector<cv::Point2i> idxVect;
	vertexVect.reserve(gridSize*gridSize);*/
	//idxVect.reserve(gridSize*gridSize);
	int cnt=0;
	for(int y=-halfSize;y<=halfSize;y++)
	{
		for(int x=-halfSize;x<=halfSize;x++)
		{
			Mat_<double> temp3D = pPts3D->m_arrow1StepVect4by1.rowRange(0,3)*x;//*scaleFactor;
			//printf("%d,%d: length %f\n",x,y,world2cm(norm(temp3D)));
			temp3D += pPts3D->m_arrow2StepVect4by1.rowRange(0,3)*y;//*scaleFactor;   //cam coord
			temp3D += pPts3D->m_ptMat4by1.rowRange(0,3);
		//	cv::Point3d pt= MatToPoint3d(temp3D);
			//vertexVectDebug.push_back(pt);

			temp3D = pRefImage->m_R * temp3D + pRefImage->m_t;  //cameraCordinate
			params[cnt] = temp3D(2,0);		//depth to be optimized
			rays[cnt] = temp3D/temp3D(2,0);		//ray, ray*[depth] = 3D local point
			cnt++;
		}
	}

	if(PATCH_PHOTOMET_GRID_SQ!=gridSize*gridSize)
	{
		printf("ERROR: PATCH_PHOTOMET_GRID_SQ!=gridSize*gridSize\n");
		return;
	}

	Problem problem;
	for(unsigned int i =0;i<visbilityIdxVector.size();++i)
	{
		int imageIdx = visbilityIdxVector[i];
		if(imageIdx == refImageIdx)
			continue;

		CostFunction* cost_function =
			new NumericDiffCostFunction<MicroStrcutureOptCostFunctionUsingRays, CENTRAL, 1, PATCH_PHOTOMET_GRID_SQ> (
			new MicroStrcutureOptCostFunctionUsingRays(pRefImage,sequences[imageIdx],pPts3D,rays,stepSIze), TAKE_OWNERSHIP);

		// Build the problem.
		problem.AddResidualBlock(cost_function, NULL, &params[0]);
	}

	/*//Smoothness Term
	for(int i=0;i<rays.size()-gridSize;i+=gridSize)
	{
		for(int j=0;j<gridSize;++j)
		{
			CostFunction* cost_function =
				new NumericDiffCostFunction<MicroStrcutureOptCostFunctionUsingRays_smooth, CENTRAL, 1,PATCH_PHOTOMET_GRID_SQ> (
				new MicroStrcutureOptCostFunctionUsingRays_smooth(rays), DO_NOT_TAKE_OWNERSHIP);
			problem.AddResidualBlock(cost_function, NULL, &params[i+j],&params[i+j+gridSize]);
		}
	}*/
	CostFunction* cost_function =
		new NumericDiffCostFunction<MicroStrcutureOptCostFunctionUsingRays_smooth, CENTRAL, 1,PATCH_PHOTOMET_GRID_SQ> (
		new MicroStrcutureOptCostFunctionUsingRays_smooth(rays), TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_function, NULL, &params[0]);
	
	//int imageIdx = pts3D[ptIdx]->m_associatedViews[i].frameIdx;

#define solver
#ifdef solver
	// Run the solver!
	Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.linear_solver_type = DENSE_QR;

	if(bDoFastOpt)
	{
		options.max_num_iterations = 20;
		//options.parameter_tolerance = 1e-14;
		//options.function_tolerance = 1e-14;
	}
	else
	{
		options.max_num_iterations = 200;
		options.parameter_tolerance = 1e-14;
		options.function_tolerance = 1e-14;
	}

	//options.numeric_derivative_relative_step_size =1;//e-12;
	Solver::Summary summary;
	Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	//std::cout << "sX : " << initial_x 	<< " -> " << x << "\n";
	//printf("ceres Start::done\n");
	//#if SHOW_PATCH_OPTIMIZATION
	std::cout << summary.FullReport() << "\n";
	//#endif
#endif

	vector<cv::Point3d>& vertexVect = pPts3D->m_microStructure.GetVertexVectForModify();
	vertexVect.clear();
	vertexVect.resize(rays.size());
	for(int i=0;i<rays.size();++i)
	{
		Mat_<double> tempPT = rays[i]*params[i];
		//printf("%d: %f,%f,%f\n",i,tempPT(0,0),tempPT(1,0),tempPT(2,0));
		tempPT = pRefImage->m_invR * (tempPT - pRefImage->m_t);
		Point3d pt;
		MatToPoint3d(tempPT,pt);

		vertexVect[i] =pt;
	}
	//vertexVect = vertexVectDebug;
	/*Mat_<double> center = ptCenter*param[0];  //cam coord
	center = pRefImage->m_invR * (center - pRefImage->m_t);  //world coord
	MatToPoint3d(center,pPts3D->m_pt3D);
	pPts3D->m_ptMat4by1.at<double>(0,0) = pPts3D->m_pt3D.x;
	pPts3D->m_ptMat4by1.at<double>(1,0) = pPts3D->m_pt3D.y;
	pPts3D->m_ptMat4by1.at<double>(2,0) = pPts3D->m_pt3D.z;
	pPts3D->m_ptMat4by1.at<double>(3,0) = 1;

	Mat_<double> arrow1 = ptArrow1*param[1];
	arrow1 = pRefImage->m_invR * (arrow1 - pRefImage->m_t);
	MatToPoint3d(arrow1,pPts3D->m_arrow1stHead);

	Mat_<double> arrow2 = ptArrow2*param[2];
	arrow2 = pRefImage->m_invR * (arrow2 - pRefImage->m_t);
	MatToPoint3d(arrow2,pPts3D->m_arrow2ndHead);
*/
	pPts3D->InitializeArrowVectNormal();
}


bool MicroStrcutureOptCostFunctionUsingRays::Evaluate(double const* const* param,
														 double* residuals,
														 double** jacobians)  const
{
	//Patch 3D estimations
	Mat_<double> temp3D;
	Mat_<double> temp3D4by1= Mat_<double>::ones(4,1);
	Mat_<double> projPt;
	Point2d imagePt;

	//Mat_<double>& refPatch = (Mat_<double>&)m_refPatch;  //to avoid repeated mem allocation
	//Mat_<double>& targetPatch = (Mat_<double>&)m_targetPatch;  //to avoid repeated mem allocation
	Mat_<double> refPatch(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);// = (Mat_<double>&)m_refPatch;  //to avoid repeated mem allocation
	Mat_<double> targetPatch(PATCH3D_GRID_SIZE,PATCH3D_GRID_SIZE);// = (Mat_<double>&)m_targetPatch;  //to avoid repeated mem allocation

	double projectionError=0;
	int projectionErrorCnt=0;

	//for(int i=0;i<m_rays.size();++i)
	vector < Mat_<double> > ptVect(PATCH_PHOTOMET_GRID_SQ);
	for(int i=0;i<PATCH_PHOTOMET_GRID_SQ;++i)			//121 is static value....for an inital test
	{
		ptVect[i] = m_rays[i]*param[0][i];
		Mat_<double> temp3D = ptVect[i].clone();

		projPt = pRefImage->m_K *temp3D;
		imagePt.x = projPt(0,0)/projPt(2,0);
		imagePt.y = projPt(1,0)/projPt(2,0);
		if(IsOutofBoundary(pRefImage->m_inputImage,imagePt.x,imagePt.y))
		{
			residuals[0] =2;
			return true;
		}
		refPatch(i/PATCH_PHOTOMET_GRID,i%PATCH_PHOTOMET_GRID) = BilinearInterpolation((Mat_<uchar>&)pRefImage->m_inputImage,imagePt.x,imagePt.y);


		temp3D = pRefImage->m_invR*(temp3D - pRefImage->m_t); // world coord
		temp3D.copyTo(temp3D4by1.rowRange(0,3));
		projPt = pTargetImage->m_P * temp3D4by1;
		imagePt.x = projPt(0,0)/projPt(2,0);
		imagePt.y = projPt(1,0)/projPt(2,0);
		if(IsOutofBoundary(pTargetImage->m_inputImage,imagePt.x,imagePt.y))
		{
			residuals[0] =2;
			return true;
		}
		targetPatch(i/PATCH_PHOTOMET_GRID,i%PATCH_PHOTOMET_GRID) = BilinearInterpolation((Mat_<uchar>&)pTargetImage->m_inputImage,imagePt.x,imagePt.y);
	}
	residuals[0] =  1 - myNCC(refPatch,targetPatch,gridStep);  // in best case, it is 0

	return true;
}

bool MicroStrcutureOptCostFunctionUsingRays_smooth_single::Evaluate(double const* const* param,
													  double* residuals,
													  double** jacobians)  const
{
	Mat_<double> pt1 = m_ray1*param[0][0];
	Mat_<double> pt2 = m_ray2*param[1][0];
	
	residuals[0] = norm(pt1-pt2);

	return true;
}

bool MicroStrcutureOptCostFunctionUsingRays_smooth::Evaluate(double const* const* param,
																	double* residuals,
																	double** jacobians)  const
{
	//Smoothness Term
	double smoothnessTerm=0;
	for(int i=0;i<m_rays.size()-PATCH_PHOTOMET_GRID;i+=PATCH_PHOTOMET_GRID)
	{
		for(int j=0;j<PATCH_PHOTOMET_GRID;++j)
		{
			Mat_<double> pt1 = m_rays[i+j]*param[0][i+j];
			Mat_<double> pt2 = m_rays[i+j+PATCH_PHOTOMET_GRID]*param[0][i+j+PATCH_PHOTOMET_GRID];
			smoothnessTerm += norm(pt1-pt2);
		}
	}
	residuals[0] = smoothnessTerm;

	return true;
}

//////////////////////////////////////////////////////////////////////////
// Patch R,t optimization minimizing reprojection error
//Input
//	- targetPatch
//	- sampled 3D points
//	- projected 2d pts of sampled3DPts
//Output
// - Internally compute R (local patch rotation), newCenter to minimize reprojection error
// - Update pPts3D, and pPts3D->m_microStructure
void PatchOptimziationMinimizingReproError_ceres(TrajElement3D* pPts3D
												 ,vector<Point3d>& sample3DPts,vector< vector<Mat*> >& ProjVect, vector< vector<Point2d> >& target2dPts
												 ,int stepSize,bool bVerbose)
{
	if(ProjVect.size() <3)			//at least 3 points needed
		return;

	double param[6];  //patchCenter, yaw,pitch,roll
	param[0] = pPts3D->m_pt3D.x;
	param[1] = pPts3D->m_pt3D.y;
	param[2] = pPts3D->m_pt3D.z;
	param[3] = param[4] = param[5] = 0.0;			//yaw and pitch

	Mat_<double> Rot_normToPatch,Rot_patchToNorm;
	PatchTrackByReproError::ComputeRotNormToPatch(pPts3D,Rot_normToPatch);
	Rot_patchToNorm = Rot_normToPatch.t();
	Problem problem;
	for(unsigned int s =0;s<ProjVect.size();++s)			//for each sample pt
	{
		if(ProjVect[s].size()==0)
			continue;

		Point3d& originalPt = sample3DPts[s];
		Mat normSamplePt;	//3x1 double
		PatchTrackByReproError::ComputeNormSampleLocation(pPts3D,originalPt,Rot_patchToNorm,normSamplePt);		//by applying rotToPatch

		for(unsigned int v =0;v<ProjVect[s].size();++v)			//for each view
		{
			Mat& P= *ProjVect[s][v];
			Point2d& targetImgPt = target2dPts[s][v];

			// Each Residual block takes patch parameters (yaw,pitch,t) as input and outputs a 2
			// dimensional residual. Internally, the cost function stores the observed
			// image location and compares the reprojection against the observation.
			ceres::CostFunction* cost_function =
				new ceres::AutoDiffCostFunction<PatchTrackByReproError, 2, 6>(
				new PatchTrackByReproError(P,Rot_normToPatch,
										normSamplePt,
										targetImgPt.x,targetImgPt.y)
										);

			problem.AddResidualBlock(cost_function,
				new HuberLoss(1.0),
				//NULL /* squared loss */ ,
				param
			);
		}
	}

	// Run the solver!
	Solver::Options options;
	options.linear_solver_type = DENSE_QR;

	/*if(bDoFastOpt)
	{
		options.max_num_iterations = 50;
		//options.parameter_tolerance = 1e-14;
		//options.function_tolerance = 1e-14;
	}
	else*/
	{
		//options.max_num_iterations = 100;
		//options.parameter_tolerance = 1e-14;
		//options.function_tolerance = 1e-14;
	}

	//options.numeric_derivative_relative_step_size =1;//e-12;
	Solver::Summary summary;
	Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	//std::cout << "sX : " << initial_x 	<< " -> " << x << "\n";
	//printf("ceres Start::done\n");
	if(bVerbose)
	{
		std::cout << summary.FullReport() << "\n";
	}

	//Apply patch transformation
	//Update microstructure
	PatchTrackByReproError::ApplyParametersToPatch(pPts3D,Rot_patchToNorm,Rot_normToPatch,param);
	pPts3D->OrthgonalizePatchAxis(true);		//to avoid numerial error accumulation
}


template <typename T>
bool PatchTrackByReproError::operator()(const T* const param,
				T* residuals) const
{
	T normPt[3];
	normPt[0]=T(m_normSamplePt_mat.at<double>(0,0));
	normPt[1]=T(m_normSamplePt_mat.at<double>(1,0));
	normPt[2]=T(m_normSamplePt_mat.at<double>(2,0));
	T p[3];
	ceres::AngleAxisRotatePoint(&param[3],normPt,p);	//rotate in normalized coordinate. IMPORTANT: param is radians
	T p_final[3];
	p_final[0] = T(m_rotNormToPatch.at<double>(0,0))*p[0] + T(m_rotNormToPatch.at<double>(0,1))*p[1] +T(m_rotNormToPatch.at<double>(0,2))*p[2]  +param[0];
	p_final[1] = T(m_rotNormToPatch.at<double>(1,0))*p[0] + T(m_rotNormToPatch.at<double>(1,1))*p[1] +T(m_rotNormToPatch.at<double>(1,2))*p[2]  +param[1];
	p_final[2] = T(m_rotNormToPatch.at<double>(2,0))*p[0] + T(m_rotNormToPatch.at<double>(2,1))*p[1] +T(m_rotNormToPatch.at<double>(2,2))*p[2]  +param[2];

	T projPt[3];
	projPt[0] = T(m_projMat.at<double>(0,0))*p_final[0] + T(m_projMat.at<double>(0,1))*p_final[1] +T(m_projMat.at<double>(0,2))*p_final[2] +T(m_projMat.at<double>(0,3));
	projPt[1] = T(m_projMat.at<double>(1,0))*p_final[0] + T(m_projMat.at<double>(1,1))*p_final[1] +T(m_projMat.at<double>(1,2))*p_final[2] +T(m_projMat.at<double>(1,3));
	projPt[2] = T(m_projMat.at<double>(2,0))*p_final[0] + T(m_projMat.at<double>(2,1))*p_final[1] +T(m_projMat.at<double>(2,2))*p_final[2] +T(m_projMat.at<double>(2,3));
	if(projPt[2]<1e-5)
		return false;
	T imagePt_x = projPt[0]/projPt[2];
	T imagePt_y = projPt[1]/projPt[2];

	residuals[0] = imagePt_x - T(m_observed_ImX);
	residuals[1] = imagePt_y - T(m_observed_ImY);
	return true;
}

//normalizedPt_out: 3x1 double
void PatchTrackByReproError::ComputeNormSampleLocation(TrajElement3D* pPatch3D,const Point3d& originalPt,const Mat& rotPatchToNorm,Mat& normalizedPt_out)		//by applying rotToPatch
{
	//Compute sample pt location in the normalized coordinate
	Point3d normalizedPt = originalPt - pPatch3D->m_pt3D;
	
	Point3dToMat(normalizedPt,normalizedPt_out);
	normalizedPt_out = rotPatchToNorm*normalizedPt_out;
}


//output: patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate
void PatchTrackByReproError::ComputeRotNormToPatch(TrajElement3D* pPatch3D,Mat_<double>& Rot_normToPatch)	
{
	//Compute rotation of Patch to normalizedCoordinate (tp align x,y,z to arrow1,arrow2,normal)
	cv::Mat arrow1_unit,arrow2_unit;
	normalize(pPatch3D->m_arrow1StepVect4by1.rowRange(0,3),arrow1_unit);		//x axis
	normalize(pPatch3D->m_arrow2StepVect4by1.rowRange(0,3),arrow2_unit);		//y axis
	Rot_normToPatch = cv::Mat_<double>(3,3);		//patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate
	Rot_normToPatch(0,0)= arrow1_unit.at<double>(0,0);
	Rot_normToPatch(1,0)= arrow1_unit.at<double>(1,0);
	Rot_normToPatch(2,0)= arrow1_unit.at<double>(2,0);
	Rot_normToPatch(0,1)= arrow2_unit.at<double>(0,0);
	Rot_normToPatch(1,1)= arrow2_unit.at<double>(1,0);
	Rot_normToPatch(2,1)= arrow2_unit.at<double>(2,0);
	Rot_normToPatch(0,2)= pPatch3D->m_normal.at<double>(0,0);
	Rot_normToPatch(1,2)= pPatch3D->m_normal.at<double>(1,0);
	Rot_normToPatch(2,2)= pPatch3D->m_normal.at<double>(2,0);

	//printf("Debug: det(Rot_normToPatch) ==%f (%f,%f,%f)\n",determinant(Rot_normToPatch),norm(arrow1_unit),norm(arrow2_unit),norm(pPatch3D->m_normal));
	//printf("Debug: right angle check: %f vs %f %f\n",arrow1_unit.dot(arrow2_unit),arrow1_unit.dot(pPatch3D->m_normal),arrow2_unit.dot(pPatch3D->m_normal));
}


void PatchTrackByReproError::ApplyParametersToPatch(TrajElement3D* pPatch3D,
														   cv::Mat_<double> Rot_patchToNorm,  //patch patch original (arrow1,arrow2,normal) coordinate to normalize (x,y,z) coordinate
														   cv::Mat_<double> Rot_normToPatch, //patch normalize (x,y,z) coordinate to patch original (arrow1,arrow2,normal) coordinate		
														   double params[6])
{
	//printf("newCenter: center: %f %f %f, rot: %f %f %f (%f %f %f)\n",params[0],params[1],params[2],params[3],params[4],params[5],params[3]*180.0/PI,params[4]*180.0/PI,params[5]*180.0/PI);
	
	//double euler[3] ={params[3]*180.0/PI,params[4]*180.0/PI,params[5]*180.0/PI};
	double euler[3] ={params[3],params[4],params[5]};
	cv::Mat_<double> Rot_byEulerMat(3,3);
	double* Rot_byEuler = (double*)Rot_byEulerMat.ptr();
	ceres::AngleAxisToRotationMatrix(euler,Rot_byEuler);		//values in euler are degrees
	Rot_byEulerMat = Rot_byEulerMat.t();		//opencv is row-major vs ceres is column major

	Point3d newPatchCenter(params[0],params[1],params[2]);
	pPatch3D->SetPos(newPatchCenter);
	pPatch3D->LocallyRotatePatch(Rot_normToPatch*Rot_byEulerMat*Rot_patchToNorm);

	//Debug
	Mat checkerMat = Rot_normToPatch*Rot_byEulerMat*Rot_patchToNorm;
	//printf("## debug: %f, %f, %f => %f\n",determinant(Rot_normToPatch),determinant(Rot_byEulerMat),determinant(Rot_patchToNorm),determinant(checkerMat));
	//Transform microstructure
	pPatch3D->GenerateMicroStruct(pPatch3D->m_microStructure,PATCH_3D_ARROW_SIZE_CM);		//At this moment, just regenerate instead of transforming
}
