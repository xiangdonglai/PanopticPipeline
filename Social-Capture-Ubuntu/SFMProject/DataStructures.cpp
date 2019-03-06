#include "DataStructures.h"
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

//for skeleton generation
//int CURRENT_JOINT_NUM = 13;
int CURRENT_JOINT_NUM = 7;		//upper body
pair<int,int> g_selectedJoint(0,0);		//first: subject, second: joint
SRenderPacket g_depthRenderPacket;

bool operator<(const CEdgeElement& a, const CEdgeElement& b)
{
	return a.centerDistance < b.centerDistance;
}

bool CamViewDT::LoadPhysicalImage(EnumLoadMode loadingMode,bool bIsHD,bool bUseSavedPath)
{
	return LoadPhysicalImage(loadingMode,m_actualImageFrameIdx,bIsHD,bUseSavedPath);
}

bool CamViewDT::LoadPhysicalImage(EnumLoadMode loadingMode,int frameIdx,bool bIsHD,bool bUseSavedPath)
{
	char filePath[512];
	if(bUseSavedPath)
		strcmp(filePath,m_fullPath.c_str());
	else
	{
		if(bIsHD)
			sprintf(filePath,"%s/%08d/hd%08d_%02d_%02d.png",g_dataImageFolder,frameIdx,frameIdx,m_actualPanelIdx,m_actualCamIdx);
		else
			sprintf(filePath,"%s/%08d/%08d_%02d_%02d.png",g_dataImageFolder,frameIdx,frameIdx,m_actualPanelIdx,m_actualCamIdx);

		if(IsFileExist(filePath)==false)
		{
			if(bIsHD)
				sprintf(filePath,"%s/hd_30/%03dXX/%08d/hd%08d_%02d_%02d.png",g_dataImageFolder,int(frameIdx/100),frameIdx,frameIdx,m_actualPanelIdx,m_actualCamIdx);
			else
				sprintf(filePath,"%s/vga_25/%03dXX/%08d/%08d_%02d_%02d.png",g_dataImageFolder,int(frameIdx/100),frameIdx,frameIdx,m_actualPanelIdx,m_actualCamIdx);
		}
		if(IsFileExist(filePath)==false)
		{
			if(bIsHD)
				sprintf(filePath,"%s/%03dXX/%08d/hd%08d_%02d_%02d.png",g_dataImageFolder,int(frameIdx/100),frameIdx,frameIdx,m_actualPanelIdx,m_actualCamIdx);
			else
				sprintf(filePath,"%s/%03dXX/%08d/%08d_%02d_%02d.png",g_dataImageFolder,int(frameIdx/100),frameIdx,frameIdx,m_actualPanelIdx,m_actualCamIdx);
		}

	}

	printf("LoadPhysicalImage:: Loading from %s\n",filePath);

	if(loadingMode== LOAD_MODE_GRAY_RGB)
	{
		m_rgbInputImage = imread(filePath);
		if(m_rgbInputImage.rows==0)
			return false;

		/*if(DO_HISTOGRAM_EQUALIZE)
		{
			Mat grayImage;
			cvtColor(m_rgbInputImage,grayImage,CV_RGB2GRAY);
			equalizeHist(grayImage,m_inputImage);
		}
		else*/
			cvtColor(m_rgbInputImage,m_inputImage,CV_RGB2GRAY);
	}
	else if(loadingMode==LOAD_MODE_GRAY)
	{
		/*if(DO_HISTOGRAM_EQUALIZE)
		{
			Mat grayImage= imread(filePath,CV_LOAD_IMAGE_GRAYSCALE);
			equalizeHist(grayImage,m_inputImage);
		}
		else*/
			m_inputImage = imread(filePath,CV_LOAD_IMAGE_GRAYSCALE);
			
		if(m_inputImage.rows==0)
			return false;
	}
	else if(loadingMode==LOAD_MODE_RGB)
	{
		m_rgbInputImage = imread(filePath);
		if(m_rgbInputImage.rows==0)
			return false;
	}
	printf("LoadPhysicalImage:: Success in loading from %s\n",filePath);
	return true;
}

bool CamViewDT::LoadIntrisicParameter(const char* calibFolderPath,int panelIdx,int camIdx)
{
	//read K matrix
	char calibParamPath[256];
	sprintf(calibParamPath,"%s/%02d_%02d.txt",calibFolderPath,panelIdx,camIdx); //load same K for all cam
	ifstream fin(calibParamPath);
	if(fin.is_open()==false)
	{
		printfLog("## Warning: Failed in loading:: No intrinsic parameter files %s\n",calibParamPath);
		return false;
	}

	if(fin)
	{
		m_K  = Mat(3,3,CV_64F);
		double* pt = m_K.ptr<double>(0);
		double value;
		for(int i=0;i<9;++i)
		{
			fin >> value;
			pt[i] = value;
		}
		m_invK = m_K.inv();
		m_camParamForBundle[6] = (m_K.at<double>(0,0) + m_K.at<double>(1,1))/2.0;  //why do I need this?
		//fin >> m_camParamForBundle[7];
		//fin >> m_camParamForBundle[8];  //distortion paramter, kc(2)			
	}
	//printMatrix("K",m_K);

	//If it has 2 more lense distortion parameters
	double isMoreInput;
	fin >> isMoreInput;		//because iostream never know it is at the eof before trying to read
	if(!fin.eof())
	{
		m_distortionParams.resize(2);
		m_distortionParams[0] = isMoreInput;
		fin >> m_distortionParams[1];
		//fin >>targetImage.m_camParamForBundle[7];  //distortion paramter, kc(1)
		//m_camParamForBundle[7] = isMoreInput;
		//fin >> m_camParamForBundle[8];  //distortion paramter, kc(2)
	}

	//If it has 3 more lense distortion parameters (5 in total)
	fin >> isMoreInput;		//because iostream never know it is at the eof before trying to read
	if(!fin.eof())
	{
		m_distortionParams.resize(5);
		m_distortionParams[2] = isMoreInput;
		fin >> m_distortionParams[3];
		fin >> m_distortionParams[4];
		//m_camParamForBundle[9] = isMoreInput;
		//fin >> m_camParamForBundle[10];  //distortion paramter, kc(2)
		//fin >> m_camParamForBundle[11];  //distortion paramter, kc(2)
	}
	fin.close();
	return true;
}
void CamViewDT::UndistortImage(Mat& originalImg,Mat& idealImg)  //undistortion using parameter of this camera
{
	idealImg = Mat::zeros(originalImg.rows,originalImg.cols,originalImg.type());
	#pragma omp parallel for
	for(int r=0;r<idealImg.rows;++r)
	{
		for(int c=0;c<idealImg.cols;++c)
		{
			if(idealImg.type()==CV_8UC3)
			{
				Point2d ptInOrigianl = ApplyDistort(Point2d(c,r));
				if(::IsOutofBoundary(originalImg,ptInOrigianl.x,ptInOrigianl.y))
					continue;

				Mat_<Vec3b> temp = originalImg;
				idealImg.at<Vec3b>(r,c) = BilinearInterpolation( temp,ptInOrigianl.x,ptInOrigianl.y);
			}
			else if(idealImg.type()==CV_8U)
			{
				Point2d ptInOrigianl = ApplyDistort(Point2d(c,r));
				if(::IsOutofBoundary(originalImg,ptInOrigianl.x,ptInOrigianl.y))
					continue;
				Mat_<uchar> temp = originalImg;
				idealImg.at<uchar>(r,c) = BilinearInterpolation( temp,ptInOrigianl.x,ptInOrigianl.y);
			}
			else
			{
			//	printf("ERROR: UndistortImage:: unknown type\n");
				//break;
			}
		}
	}
}

Point_<double> CamViewDT::ApplyUndistort(const Point_<double>& pt_d) //input: image coord, output: ideal_image coord
{
	if(m_distortionParams.size()==0 || m_distortionParams[0] ==0)
		return pt_d;
	else if(m_distortionParams[0]>0)//true)//m_distortionParams.size() <=2)
		return ApplyUndistort_visSfM(pt_d);
	else //if(m_distortionParams.size() ==2)
		return ApplyUndistort_openCV(pt_d);
}

Point_<double> CamViewDT::ApplyDistort(const Point_<double>& pt_d)   //input: ideal_image coord, output: image coord
{
	if(m_distortionParams.size()==0 || m_distortionParams[0] ==0)
		return pt_d;
	else if(m_distortionParams[0]>0)//(m_distortionParams.size() <=2)
		return ApplyDistort_visSfM(pt_d);
	else //if(m_distortionParams.size() ==2)
		return ApplyDistort_openCV(pt_d);
}

Point_<double> CamViewDT::ApplyUndistort_openCV(const Point_<double>& pt_d)
{
	//printf("## WARNING: ApplyUndistort_openCV is not implemented yet\n");
	//return pt_d;
	/*
	Mat_<float> dist(1,5); dist <<0,0,0,0,0;//m_distortionParams[0],m_distortionParams[1],m_distortionParams[2],m_distortionParams[4],m_distortionParams[4];


	Mat_<Point2d> input(1,1);
	input(0) = pt_d;
	Mat_<Point2d> output(1,1);
	undistortPoints(input,output,m_K,dist);
	Point2d pt_u = output(0);
	return pt_u;
	*/
	
	Mat_<float> dist(1,5); dist <<m_distortionParams[0],m_distortionParams[1],m_distortionParams[4],m_distortionParams[3],m_distortionParams[2];		//3,4 order should be swtiched Gakus' vs Opencv
//	(x',y') = undistort(x",y",dist_coeffs)
/*	Mat pt_d_Mat = Mat::ones(3,1,CV_64F);
	Mat pt_d_Mat = Mat::ones(3,1,CV_64F);

	pt_d_Mat.at<double>(0,0) = pt_d.x;
	pt_d_Mat.at<double>(1,0) = pt_d.y;
	/*pt_d_Mat = m_invK*pt_d_Mat;
	pt_d_Mat /= pt_d_Mat.at<double>(2,0);
	*/
	//Mat_<float> cam(3,3); cam << m_K.at<double>(0,0),m_K.at<double>(0,1),m_K.at<double>(0,2),m_K.at<double>(1,0),m_K.at<double>(1,1),m_K.at<double>(1,2),m_K.at<double>(2,0),m_K.at<double>(2,1),m_K.at<double>(2,2);//1531.49,0,1267.78,0,1521.439,952.078,0,0,1;
	Mat_<Point2d> input(1,1);
	input(0) = Point2d(pt_d.x,pt_d.y);
	Mat_<Point2d> output(1,1);
	undistortPoints(input,output,m_K,dist);
	//cout <<m_K<<endl;
	//cout <<output<<endl;

	Mat pt_u_Mat = Mat::ones(3,1,CV_64F);
	pt_u_Mat.at<double>(0,0) = output(0).x;
	pt_u_Mat.at<double>(1,0) = output(0).y;
	pt_u_Mat = m_K*pt_u_Mat;
	pt_u_Mat /= pt_u_Mat.at<double>(2,0);
	//cout << pt_u_Mat<<endl;
	Point2d pt_u;
	pt_u.x = pt_u_Mat.at<double>(0,0);
	pt_u.y = pt_u_Mat.at<double>(1,0);
	//printf("%f,%f -> %f,%f\n",input(0).x,input(0).y,pt_u.x,pt_u.y);
	return pt_u;
}

//Parameter order is from Gaku's 
//Note that the order is different from OpenCV
Point_<double> CamViewDT::ApplyDistort_openCV(const Point_<double>& pt_d)    //from ideal_image coord to image coord
{
	Mat pt_d_Mat = Mat::ones(3,1,CV_64F);
	pt_d_Mat.at<double>(0,0) = pt_d.x;
	pt_d_Mat.at<double>(1,0) = pt_d.y;
	pt_d_Mat = m_invK*pt_d_Mat;
	pt_d_Mat /= pt_d_Mat.at<double>(2,0);
	
	double xn = pt_d_Mat.at<double>(0,0);
	double yn = pt_d_Mat.at<double>(1,0);
	double r2 = xn*xn + yn*yn;
	double r4 = r2*r2;
	double r6 = r2*r4;
	double X2 = xn*xn;
	double Y2 = yn*yn;
	double XY = xn*yn;

	double a0 = m_distortionParams[0], a1 = m_distortionParams[1], a2 = m_distortionParams[2];
	double p0 = m_distortionParams[3],  p1 = m_distortionParams[4];			//p0, p1 order is switched compared to opencv parameters

	double radial       = 1.0 + a0*r2 + a1*r4 + a2*r6;
	double tangential_x = p0*(r2 + 2.0*X2) + 2.0*p1*XY;
	double tangential_y = p1*(r2 + 2.0*Y2) + 2.0*p0*XY;
	
	Point2d pt_u;
	pt_d_Mat.at<double>(0,0)= radial*xn + tangential_x;
	pt_d_Mat.at<double>(1,0)= radial*yn + tangential_y;
	pt_d_Mat =m_K*pt_d_Mat;
	pt_d_Mat /= pt_d_Mat.at<double>(2,0);

	pt_u.x = pt_d_Mat.at<double>(0,0);
	pt_u.y = pt_d_Mat.at<double>(1,0);

	return pt_u;
}

Point_<double> CamViewDT::ApplyUndistort_visSfM(const Point_<double>& pt_d)
{
	Mat pt_d_Mat = Mat::ones(3,1,CV_64F);
	pt_d_Mat.at<double>(0,0) = pt_d.x;
	pt_d_Mat.at<double>(1,0) = pt_d.y;
	pt_d_Mat = m_invK*pt_d_Mat;
	pt_d_Mat /= pt_d_Mat.at<double>(2,0);
		
	double r2 = pt_d_Mat.at<double>(0,0)*pt_d_Mat.at<double>(0,0) + pt_d_Mat.at<double>(1,0)*pt_d_Mat.at<double>(1,0);
	double distortion = 1.0 + r2 * (m_distortionParams[0] + m_distortionParams[1] * r2);

	Point2d pt_u;
	pt_d_Mat.at<double>(0,0)= distortion * pt_d_Mat.at<double>(0,0);
	pt_d_Mat.at<double>(1,0)= distortion * pt_d_Mat.at<double>(1,0);
	pt_d_Mat =m_K*pt_d_Mat;
	pt_d_Mat /= pt_d_Mat.at<double>(2,0);

	pt_u.x = pt_d_Mat.at<double>(0,0);
	pt_u.y = pt_d_Mat.at<double>(1,0);

	return pt_u;
}

Point_<double> CamViewDT::ApplyDistort_visSfM(const Point_<double>& pt_d)    //from ideal_image coord to image coord
{
	Mat pt_d_Mat = Mat::ones(3,1,CV_64F);
	pt_d_Mat.at<double>(0,0) = pt_d.x;
	pt_d_Mat.at<double>(1,0) = pt_d.y;
	pt_d_Mat = m_invK*pt_d_Mat;
	pt_d_Mat /= pt_d_Mat.at<double>(2,0);
	Point2d pt(pt_d_Mat.at<double>(0,0),pt_d_Mat.at<double>(1,0));
	if(pt.y ==0)
		pt.y = 1e-3;
	double k1= m_distortionParams[0];
	if (k1 == 0) 
		return pt_d; 
	const double t2 = pt.y*pt.y; 
	const double t3 = t2*t2*t2; 
	const double t4 = pt.x*pt.x; 
	const double t7 = k1*(t2+t4);

	Point2d returnPt;
		
	if (k1 > 0) { 
		const double t8 = 1.0/t7; 
		const double t10 = t3/(t7*t7); 
		const double t14 = sqrt(t10*(0.25+t8/27.0)); 
		const double t15 = t2*t8*pt.y*0.5; 
		const double t17 = pow(t14+t15,1.0/3.0); 
		const double t18 = t17-t2*t8/(t17*3.0); 
		returnPt =  Point2d(t18*pt.x/pt.y, t18); 
	} else { 
		const double t9 = t3/(t7*t7*4.0); 
		const double t11 = t3/(t7*t7*t7*27.0); 
		const std::complex<double> t12 = t9+t11; 
		const std::complex<double> t13 = sqrt(t12); 
		const double t14 = t2/t7; 
		const double t15 = t14*pt.y*0.5; 
		const std::complex<double> t16 = t13+t15; 
		const std::complex<double> t17 = pow(t16,1.0/3.0); 
		const std::complex<double> t18 = (t17+t14/ 
			(t17*3.0))*std::complex<double>(0.0,sqrt(3.0)); 
		const std::complex<double> t19 = -0.5*(t17+t18)+t14/(t17*6.0); 
		returnPt  =  Point2d(t19.real()*pt.x/pt.y, t19.real()); 
	} 

	pt_d_Mat = Mat::ones(3,1,CV_64F);
	pt_d_Mat.at<double>(0,0) = returnPt.x;
	pt_d_Mat.at<double>(1,0) = returnPt.y;
	pt_d_Mat =m_K*pt_d_Mat;
	pt_d_Mat /= pt_d_Mat.at<double>(2,0);

	returnPt.x = pt_d_Mat.at<double>(0,0);
	returnPt.y = pt_d_Mat.at<double>(1,0);
	return returnPt;
}

/*
//To draw related trajecoty. 
///There is similar version for vector in SFM.cpp
//Why here, not SFM.cpp? to be used at BodyPoseRecon
//bShowOnlyForwardTraj is from g_visalize.m_showOnlyForwardTraj
void GetPointPosFromTrajSet(int currrentFrameIdx, int veryFirstImgIdxOfTrajMemory,bool bShowOnlyForwardTraj, set<TrajElement3D*>& trajectorySet, vector<Point3f>& outputPtCloud, vector<TrajElement3D*>& outputTrajPointers,bool bEnforceAllOutput)	
{
	int selectedTrajMemTime = currrentFrameIdx - veryFirstImgIdxOfTrajMemory;

	outputPtCloud.clear();
	outputTrajPointers.clear();
	outputPtCloud.reserve(trajectorySet.size());
	outputTrajPointers.reserve(trajectorySet.size());

	set<TrajElement3D*>::iterator iter = trajectorySet.begin();
	while(iter++!=trajectorySet.end())
	{
		TrajElement3D* pTraj3D = *iter;
		int memIdx = pTraj3D->m_initiatedMemoryIdx;
		int memOffset = selectedTrajMemTime - memIdx;  //this is a kind local frame number for this memory data. memOffset=0 means init position
		//memOffset= 5 means m_actualNextTrackUnit[4]. memOffset=-4  -> m_actualPrevTrackUnit[3]
		outputPtCloud.push_back(Point3f(0,0,0));
		outputTrajPointers.push_back(NULL);
		Point3d tempColor;

		//filtering by Range
		int veryFirstTrackedFrame = memIdx - pTraj3D->m_actualPrevTrackUnit.size();
		int veryLastTrackedFrame = memIdx  + pTraj3D->m_actualNextTrackUnit.size();

		if(memOffset==0)
		{
			if(pTraj3D->m_actualNextTrackUnit.size()==0 && pTraj3D->m_actualPrevTrackUnit.size()==0 ) 
			{
				outputPtCloud.pop_back();
				outputTrajPointers.pop_back();		//these were missig. huge bug....
				continue;
			}

			outputPtCloud.back() = pTraj3D->m_curTrackUnit.m_pt3D;
			outputTrajPointers.back() = pTraj3D;
		}
		else if(memOffset>0)
		{
			if(memOffset-1 <pTraj3D->m_actualNextTrackUnit.size())
			{
				outputPtCloud.back() = pTraj3D->m_actualNextTrackUnit[memOffset-1].m_pt3D;
				outputTrajPointers.back() = pTraj3D;
			}
		}
		//else if(memOffset<0 && g_visData.m_showOnlyForwardTraj == false)
		else if(memOffset<0 && bShowOnlyForwardTraj == false)
		{
			if(-memOffset-1 < pTraj3D->m_actualPrevTrackUnit.size())
			{
				outputPtCloud.back() = pTraj3D->m_actualPrevTrackUnit[-memOffset-1].m_pt3D;
				outputTrajPointers.back() = pTraj3D;
			}
		}
		if(bEnforceAllOutput==false && outputTrajPointers.back()==NULL)
		{
			outputPtCloud.pop_back();
			outputTrajPointers.pop_back();
		}

		//if(outputTrajPointers.back()==NULL)
		//	printf("what\n");
	} 

	if(bEnforceAllOutput && outputPtCloud.size() !=outputTrajPointers.size())
	{
		printf("error:: prevTrajVect.size() !=nextPtCloud.size() \n");
	}
}*/


void CamViewDT::SettingRMatrixGL()		//to visualize in opengl 
{
	//colwise
	m_RMatrixGL[0] = m_invR.at<double>(0,0);
	m_RMatrixGL[1] = m_invR.at<double>(1,0);
	m_RMatrixGL[2] = m_invR.at<double>(2,0);
	m_RMatrixGL[3] = 0;
	m_RMatrixGL[4] = m_invR.at<double>(0,1);
	m_RMatrixGL[5] = m_invR.at<double>(1,1);
	m_RMatrixGL[6] = m_invR.at<double>(2,1);
	m_RMatrixGL[7] = 0;
	m_RMatrixGL[8] = m_invR.at<double>(0,2);
	m_RMatrixGL[9] = m_invR.at<double>(1,2);
	m_RMatrixGL[10] = m_invR.at<double>(2,2);
	m_RMatrixGL[11] = 0;
	m_RMatrixGL[12] = 0; //4th col
	m_RMatrixGL[13] = 0;
	m_RMatrixGL[14] = 0;
	m_RMatrixGL[15] = 1;
}



/**
 @brief basic function to produce an OpenGL projection matrix and associated viewport parameters
 which match a given set of camera intrinsics. This is currently written for the Eigen linear
 algebra library, however it should be straightforward to port to any 4x4 matrix class.
 @param[out] frustum Eigen::Matrix4d projection matrix.  Eigen stores these matrices in column-major (i.e. OpenGL) order.
 @param[out] viewport 4-component OpenGL viewport values, as might be retrieved by glGetIntegerv( GL_VIEWPORT, &viewport[0] )
 @param[in]  alpha x-axis focal length, from camera intrinsic matrix
 @param[in]  alpha y-axis focal length, from camera intrinsic matrix
 @param[in]  skew  x and y axis skew, from camera intrinsic matrix
 @param[in]  u0 image origin x-coordinate, from camera intrinsic matrix
 @param[in]  v0 image origin y-coordinate, from camera intrinsic matrix
 @param[in]  img_width image width, in pixels
 @param[in]  img_height image height, in pixels
 @param[in]  near_clip near clipping plane z-location, can be set arbitrarily > 0, controls the mapping of z-coordinates for OpenGL
 @param[in]  far_clip  far clipping plane z-location, can be set arbitrarily > near_clip, controls the mapping of z-coordinate for OpenGL


 K = [a skew u0 ; 0 beta v0 ; 0 0 1]
*/
#if 0
void build_opengl_projection_for_intrinsics( GLfloat*frustum, int *viewport,
					double alpha, double beta, double skew, double u0, double v0, int img_width, int img_height, double near_clip, double far_clip )
{
    // These parameters define the final viewport that is rendered into by
    // the camera.
    double L = 0;
    double R = img_width;
    double B = 0;
    double T = img_height;
     
    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    double N = near_clip;
    double F = far_clip;
     
    // set the viewport parameters
    viewport[0] = L;
    viewport[1] = B;
    viewport[2] = R-L;
    viewport[3] = T-B;
     
    // construct an orthographic matrix which maps from projected
    // coordinates to normalized device coordinates in the range
    // [-1, 1].  OpenGL then maps coordinates in NDC to the current
    // viewport
	Mat_<double> ortho = Mat_<double>::zeros(4,4);
    //Eigen::Matrix4d ortho = Eigen::Matrix4d::Zero();
    ortho(0,0) =  2.0/(R-L); ortho(0,3) = -(R+L)/(R-L);
    ortho(1,1) =  2.0/(T-B); ortho(1,3) = -(T+B)/(T-B);
    ortho(2,2) = -2.0/(F-N); ortho(2,3) = -(F+N)/(F-N);
    ortho(3,3) =  1.0;
     
    // construct a projection matrix, this is identical to the 
    // projection matrix computed for the intrinsicx, except an
    // additional row is inserted to map the z-coordinate to
    // OpenGL. 
	Mat_<double> tproj = Mat_<double>::zeros(4,4);
    //Eigen::Matrix4d tproj = Eigen::Matrix4d::Zero();
    /*tproj(0,0) = alpha; tproj(0,1) = skew; tproj(0,2) = u0;
                        tproj(1,1) = beta; tproj(1,2) = v0;
                                           tproj(2,2) = -(N+F); tproj(2,3) = -N*F;
                                           tproj(3,2) = 1.0;*/
    tproj(0,0) = alpha; tproj(0,1) = skew; tproj(0,2) = u0 - img_width/2;
                        tproj(1,1) = beta; tproj(1,2) = v0;
                                           tproj(2,2) = -(N+F); tproj(2,3) = -N*F;
                                           tproj(3,2) = 1.0; 
    // resulting OpenGL frustum is the product of the orthographic
    // mapping to normalized device coordinates and the augmented
    // camera intrinsic matrix
	//printMatrix("ortho",ortho);
	//printMatrix("tproj",tproj);
	//Mat_<double> frustumMat= ortho*tproj;
	Mat_<double> frustumMat= tproj;

	frustumMat = frustumMat.t();		//openCV(row-major) -> opengl(col-major)
	//printMatrix("frustumMat",frustumMat);
	double* data = (double*) frustumMat.data;
	for(int i=0;i<16;++i)
	{
		frustum[i] = data[i];
	//	printf("%f ",frustum[i]);
	}
}
#else

void build_opengl_projection_for_intrinsics(GLfloat* frustum,GLfloat* perspective,double fx, double fy, double skew, double u0, double v0, int width, int height, double near_clip, double far_clip )
{
	//double l = 0.0, r = 1.0*width, b = 0.0, t = 1.0*height;
	double l = 0.0, r = 1.0*width, b = 1.0*height, t = 0.0;
	double tx = -(r+l)/(r-l), ty = -(t+b)/(t-b), tz = -(far_clip+near_clip)/(far_clip-near_clip);

	double ortho[16] = {2.0/(r-l), 0.0, 0.0, tx, 
		0.0, 2.0/(t-b), 0.0, ty,
		0.0, 0.0, -2.0/(far_clip-near_clip), tz, 
		0.0, 0.0, 0.0, 1.0};

	double Intrinsic[16] = {fx, skew, u0, 0.0,
		0.0, fy, v0, 0.0, 
		0.0, 0.0, -(near_clip+far_clip), +near_clip*far_clip, 
		//0.0, 0.0, -(near_clip+far_clip)/2.0 , 0.0, 
		0.0, 0.0, 1.0 , 0.0};

	Mat orthoMat = Mat(4, 4, CV_64F, ortho);
	Mat IntrinsicMat = Mat(4, 4, CV_64F, Intrinsic);
	Mat Frustrum = orthoMat*IntrinsicMat;
	//Mat Frustrum = IntrinsicMat;
	Frustrum = Frustrum.t();

	double* data = (double*) Frustrum.data;
	for(int i=0;i<16;++i)
	{
		frustum[i] = data[i];
	}

	IntrinsicMat.t();
	data = (double*) IntrinsicMat.data;
	for(int i=0;i<16;++i)
	{
		perspective[i] = data[i];
	}
}
#endif

void CamViewDT::SettingModelViewMatrixGL() //to visualize in opengl 
{
	//ModelViewMat
	//colwise
	m_modelViewMatGL[0] = m_R.at<double>(0,0);
	m_modelViewMatGL[1] = m_R.at<double>(1,0);
	m_modelViewMatGL[2] = m_R.at<double>(2,0);
	m_modelViewMatGL[3] = 0;
	m_modelViewMatGL[4] = m_R.at<double>(0,1);
	m_modelViewMatGL[5] = m_R.at<double>(1,1);
	m_modelViewMatGL[6] = m_R.at<double>(2,1);
	m_modelViewMatGL[7] = 0;
	m_modelViewMatGL[8] = m_R.at<double>(0,2);
	m_modelViewMatGL[9] = m_R.at<double>(1,2);
	m_modelViewMatGL[10] = m_R.at<double>(2,2);
	m_modelViewMatGL[11] = 0;
	m_modelViewMatGL[12] = m_t.at<double>(0,0); //4th col
	m_modelViewMatGL[13] = m_t.at<double>(1,0);
	m_modelViewMatGL[14] = m_t.at<double>(2,0);
	m_modelViewMatGL[15] = 1;

	/*
	Mat_<double> test(4,4);
	double*  testPtr = (double*) test.data;
	for(int i=0;i<16;++i)
		testPtr[i]	= m_modelViewMatGL[i];
	test.t();
	for(int i=0;i<16;++i)
		m_modelViewMatGL[i] = testPtr[i];*/

	if(m_heightExpected==0)
	{
		printf("## ERROR: SettingModelViewMatrixGL:: sensortype is not defined\n");
		return;
	}
	int imageWidth = m_widthExpected;
	int imageHeight = m_heightExpected;
	/*if(m_actualPanelIdx==0 || m_actualPanelIdx==50)
	{
		imageWidth =1920;
		imageHeight=1080;
	}
	else
	{
		imageWidth =640;
		imageHeight= 480;
	}*/
	//printMatrix("test",m_K);
	/*build_opengl_projection_for_intrinsics(m_frustumGL,m_viewPortGL
		 ,m_K.at<double>(0,0),m_K.at<double>(1,1),m_K.at<double>(0,1),m_K.at<double>(0,2),m_K.at<double>(1,2)
		,imageWidth,imageHeight,10,100);*/
	build_opengl_projection_for_intrinsics(m_projMatGL,m_perspectGL
		 ,m_K.at<double>(0,0),m_K.at<double>(1,1),m_K.at<double>(0,1),m_K.at<double>(0,2),m_K.at<double>(1,2)
		 ,imageWidth,imageHeight,g_nearPlaneForDepthRender,g_farPlaneForDepthRender);
	
	Mat_<double> ProjMat(4,4); 
	Mat_<double> viewModelMat(4,4);
	double*  ProjMatPtr = (double*) ProjMat.data;
	double*  viewModelMatPtr = (double*) viewModelMat.data;
	for(int i=0;i<16;++i)
	{
		ProjMatPtr[i] = m_projMatGL[i];
		viewModelMatPtr[i]	= m_modelViewMatGL[i];
	}
	Mat_<double> mvp = ProjMat.t()*viewModelMat.t();
	mvp = mvp.t();

	/* Debug 
	if(m_actualPanelIdx==0 && m_actualCamIdx==15)
	{
		cout << viewModelMat<<endl;
		cout << ProjMat<<endl;
		cout << mvp <<endl;
	}
	*/
	/*
	//Simple test
	Mat_<double> test(4,1);
	test(0,0) = 0;
	test(1,0) = 0;
	test(2,0) = g_farPlaneForDepthRender;
	test(3,0) = 1;
	test = ProjMat.t() * test;
	test = test / test(3,0);
	printf("tset: %f, %f, %f\n",test(0,0),test(1,0),test(2,0));*/

	double* mvpPtr = (double*) mvp.data;
	for(int i=0;i<16;++i)
	{
		m_mvpMatGL[i] = mvpPtr[i];
	}
}

//Return false, if any image loading has been failed.
//Return true, otherwise
//DEPRECIATED, it cannot handle HD and VGA automatically
bool LoadPhysicalImageForAllCams(vector<CamViewDT*>& camVector,EnumLoadMode loadingMode,bool bIsHD,bool bUseSavedPath)
{
	for(int i=0;i<camVector.size();++i)
	{
		if(camVector[i]->LoadPhysicalImage(loadingMode,bIsHD,bUseSavedPath) ==false)
		{
			printf(" FAILURE: LoadPhysicalImageForAllCams\n");
			continue;
		}
	}
	return true;
}


//Return false, if any image loading has been failed.
//Return true, otherwise
//DEPRECIATED, it cannot handle HD and VGA automatically
bool LoadPhysicalImageForAllCams(vector<CamViewDT*>& camVector,EnumLoadMode loadingMode,int frameIdx,bool bIsHD,bool bUseSavedPath)
{
	for(int i=0;i<camVector.size();++i)
	{
		if(camVector[i]->LoadPhysicalImage(loadingMode,frameIdx,bIsHD,bUseSavedPath) ==false)
		{
			printf(" FAILURE: LoadPhysicalImageForAllCams\n");
			continue;
		}
	}
	return true;
}



void CMicroStructure::SetCenterPos(Point3d& newPt)
{
	if(m_vertexVect.size()==0)
	{
		printf("## ERROR: m_vertexVect.size()==0\n");
		return;
	}

	int centerIdx = m_vertexVect.size()/2.0;
	printf("check %d,%d \n",m_vertexIdxInPatch[centerIdx].x,m_vertexIdxInPatch[centerIdx].y);
	Point3d diff = newPt -  m_vertexVect[centerIdx];
	for(int i=0;i<m_vertexVect.size();++i)
		m_vertexVect[i] = m_vertexVect[i] + diff;
}