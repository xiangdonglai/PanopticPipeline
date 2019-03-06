// #include "stdafx.h"
#include <iostream>
#include "Utility.h"
#include "PatchOptimization.h"
#include <sys/stat.h>
#include "Constants.h"
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>		//applyColorMap
#include <opencv2/calib3d/calib3d.hpp>	//Rodrigues

using namespace std;
using namespace cv;

void CreateFolder(const char* folderName)
{
#ifndef LINUX_COMPILE
	CreateDirectory(folderName,NULL);
#else
	mkdir(folderName,0755);
#endif
}
int COLOR_CLAMP(int color)
{
	if(color<0) color =0;
	if(color>255) color =255;
	return color;
}

//In Windows, folderName is changed, if there exist folder with same name already.
void CreateFolderAvoidOverwrite(char* folderName)
{
#ifndef LINUX_COMPILE
	int maxTrial=5;
	char newFolderName[512];
	srand(time(0));
	strcpy(newFolderName,folderName);
	while(!CreateDirectory(newFolderName,NULL))		//if failed to make
	{
		if(maxTrial--==0)
			break;	//give up to make a folder
		//sprintf(newFolderName,"%s/updated_%d",g_dataSpecificFolder,rand()%100);
		sprintf(newFolderName,"%s_%d",folderName,rand()%100);
	}
	strcpy(folderName,newFolderName);
#else
	mkdir(folderName,0755);
#endif
}

/*
double Distance(Point2d& pt_1,Point2f& pt_2)
{
	double dist = pow(pt_1.x -pt_2.x,2) +pow(pt_1.y - pt_2.y,2);
	return sqrt(dist);
}
double Distance(Point2f& pt_1,Point2f& pt_2)
{
	double dist = pow(pt_1.x -pt_2.x,2) +pow(pt_1.y - pt_2.y,2);
	return sqrt(dist);
}

double Distance(Point2d& pt_1,Point2d& pt_2)
{
	double dist = pow(pt_1.x -pt_2.x,2) +pow(pt_1.y - pt_2.y,2);
	return sqrt(dist);
}


double Distance(Point3d& pt_1,Point3d& pt_2)
{
	double dist = pow(pt_1.x -pt_2.x,2) +pow(pt_1.y - pt_2.y,2)+pow(pt_1.z - pt_2.z,2);
	return sqrt(dist);
}

double Distance(Point3f& pt_1,Point3f& pt_2)
{
	double dist = pow(pt_1.x -pt_2.x,2) +pow(pt_1.y - pt_2.y,2)+pow(pt_1.z - pt_2.z,2);
	return sqrt(dist);
}

*/
double Distance(Point3d& pt_1,Mat& pt_2)
{
	double* pt_2Addr = pt_2.ptr<double>(0);
	double dist = pow(pt_1.x -pt_2Addr[0],2) +pow(pt_1.y - pt_2Addr[1],2)+pow(pt_1.z - pt_2Addr[2],2);
	return sqrt(dist);
}

double Distance(Mat& pt_2,Point3d& pt_1)
{
	return Distance(pt_1,pt_2);
}

//input should be 3x1 matrices
double Distance(Mat& pt_1,Mat& pt_2)
{
	double* pt_1Addr = pt_1.ptr<double>(0);
	double* pt_2Addr = pt_2.ptr<double>(0);
	double dist = pow(pt_1Addr[0] -pt_2Addr[0],2) +pow(pt_1Addr[1] - pt_2Addr[1],2)+pow(pt_1Addr[2] - pt_2Addr[2],2);
	return sqrt(dist);
}
void writeMatrix(char* outputFileName,Mat &M)
{
	writeMatrix(M,outputFileName);
}

void writeMatrix(Mat &M, char* outputFileName)
{
	ofstream fout;
	fout.open(outputFileName);
	//printf("Matrix \"%s\" is %i x %i\n", matrix.c_str(), M.rows, M.cols);
	for( int r=0;r<M.rows;++r)
	{
		for( int c=0;c<M.cols;++c)
		{
			//printf("%.20lf ",M.at<double>(r,c));
			char buf[256];
			sprintf(buf,"%.6lf\t",M.at<double>(r,c));
			fout << buf;
		}
		fout << "\n";
	}
	fout << "\n";
	fout.close();
}

void printMatrix(Mat &M, std::string matrix)
{
    printf("Matrix \"%s\" is %i x %i\n", matrix.c_str(), M.rows, M.cols);
	for( int r=0;r<M.rows;++r)
	{
		for( int c=0;c<M.cols;++c)
		{
			//printf("%.20lf ",M.at<double>(r,c));
			printf("%.6lf ",M.at<double>(r,c));
		}
		cout <<endl;
	}
	cout <<endl;
}

/*
void printMatrix(std::string matrix,Mat &M)
{
	printfLog("Matrix \"%s\" is %i x %i\n", matrix.c_str(), M.rows, M.cols);
    //printf("Matrix \"%s\" is %i x %i\n", matrix.c_str(), M.rows, M.cols);
	for( int r=0;r<M.rows;++r)
	{
		for( int c=0;c<M.cols;++c)
		{
			//printf("%.20lf ",M.at<double>(r,c));
			//printf("%.6lf ",M.at<double>(r,c));
			printfLog("%.6lf ",M.at<double>(r,c));
		}
		//cout <<endl;
		printfLog("\n");
	}
	//cout <<endl;
	printfLog("\n");
}*/

void printMatrix(std::string matrix,Mat &M)
{
    printf("Matrix \"%s\" is %i x %i\n", matrix.c_str(), M.rows, M.cols);
	for( int r=0;r<M.rows;++r)
	{
		for( int c=0;c<M.cols;++c)
		{
			//printf("%.20lf ",M.at<double>(r,c));
			printf("%.6lf ",M.at<double>(r,c));
		}
		cout <<endl;
		//printfLog("\n");
	}
	cout <<endl;
	//printfLog("\n");
}

void printMatrixFloat(Mat &M, std::string matrix)
{
    printf("Matrix \"%s\" is %i x %i\n", matrix.c_str(), M.rows, M.cols);
	for( int r=0;r<M.rows;++r)
	{
		for( int c=0;c<M.cols;++c)
		{
			printf("%.6f ",M.at<float>(r,c));
		}
		cout <<endl;
	}
	cout <<endl;
}

/*
//both 3x1 and 4x1 matrix are fine with this function
void Point3dToMat(Point3d& pt_1,Mat& returnMat)
{
	returnMat = Mat::zeros(3,1,CV_64F);
	returnMat.at<double>(0,0) = pt_1.x;
	returnMat.at<double>(1,0) = pt_1.y;
	returnMat.at<double>(2,0) = pt_1.z;
}

//both 3x1 and 4x1 matrix are fine with this function
void Point3fToMat(Point3f& pt_1,Mat& returnMat)
{
returnMat = Mat::zeros(3,1,CV_64F);
returnMat.at<double>(0,0) = pt_1.x;
returnMat.at<double>(1,0) = pt_1.y;
returnMat.at<double>(2,0) = pt_1.z;
}
*/

//both 3x1 and 4x1 matrix are fine with this function
void Point3fTo3d(const Point3f& pt,Point3d& returnPt)
{
	returnPt = Point3d(pt.x,pt.y,pt.z);
}

void Point3dTo3f(const Point3d& pt,Point3f& returnPt)
{
	returnPt = Point3f(pt.x,pt.y,pt.z);
}
Point3d Point3fTo3d(const Point3f& pt)
{
	return Point3d(pt.x,pt.y,pt.z);
}

Point3f Point3dTo3f(const Point3d& pt)
{
	return Point3f(pt.x,pt.y,pt.z);
}

void Point3fToMat(const cv::Point3f& pt,cv::Mat_<double>& returnMat)
{
	Point3ToMatDouble(pt,returnMat); 
}
void Point3fToMat(const Point3f& pt,Mat& returnMat)
{ 
	Point3ToMatDouble(pt,returnMat); 
}

void Point3dToMat(const Point3d& pt,Mat& returnMat)
{
	Point3ToMatDouble(pt,returnMat); 
}

void Point3dToMat(const Point3d& pt,Mat_<double>& returnMat)
{
	Point3ToMatDouble(pt,returnMat); 
}
/*
void Point3dToMat(Point3d& pt_1,Mat_<double>& returnMat)
{
	returnMat = Mat_<double>(3,1);
	returnMat(0,0) = pt_1.x;
	returnMat(1,0) = pt_1.y;
	returnMat(2,0) = pt_1.z;F
}


void Point3fToMat(Point3f& pt_1,Mat_<double>& returnMat)
{
	returnMat = Mat_<double>(3,1);
	returnMat(0,0) = pt_1.x;
	returnMat(1,0) = pt_1.y;
	returnMat(2,0) = pt_1.z;
}
*/

//both 3x1 and 4x1 matrix are fine with this function
void Point3dToMat4by1(const Point3d& pt_1,Mat& returnMat)
{
	returnMat = Mat::ones(4,1,CV_64F);
	returnMat.at<double>(0,0) = pt_1.x;
	returnMat.at<double>(1,0) = pt_1.y;
	returnMat.at<double>(2,0) = pt_1.z;
}

void Point3dToMat4by1(const Point3d& pt_1,Mat_<double>& returnMat)
{
	returnMat = Mat_<double>(4,1);
	returnMat(0,0) = pt_1.x;
	returnMat(1,0) = pt_1.y;
	returnMat(2,0) = pt_1.z;
	returnMat(3,0) = 1.0;
}

//both 3x1 and 4x1 matrix are fine with this function
void Point3fToMat4by1(const Point3f& pt_1,Mat& returnMat)
{
	returnMat = Mat::ones(4,1,CV_64F);
	returnMat.at<double>(0,0) = pt_1.x;
	returnMat.at<double>(1,0) = pt_1.y;
	returnMat.at<double>(2,0) = pt_1.z;
}

void Point3fToMat4by1(const Point3f& pt_1,Mat_<double>& returnMat)
{
	returnMat = Mat_<double>(4,1);
	returnMat(0,0) = pt_1.x;
	returnMat(1,0) = pt_1.y;
	returnMat(2,0) = pt_1.z;
	returnMat(3,0) = 1.0;
}

void MatToPoint3d(Mat& mat,Point3d& returnPt)
{
	if(mat.rows==4 && abs(mat.at<double>(3,0))>1e-10 )
		mat = mat/ mat.at<double>(3,0) ;
	
	returnPt.x = mat.at<double>(0,0) ;
	returnPt.y = mat.at<double>(1,0) ;
	returnPt.z = mat.at<double>(2,0) ;
}


Point3d MatToPoint3d(Mat& mat)
{
	if(mat.rows==4 && abs(mat.at<double>(3,0))>1e-10 )
		mat = mat/ mat.at<double>(3,0) ;
	
	Point3d  returnPt;
	returnPt.x = mat.at<double>(0,0) ;
	returnPt.y = mat.at<double>(1,0) ;
	returnPt.z = mat.at<double>(2,0) ;

	return returnPt;
}


Point3f MatToPoint3f(Mat& mat)
{
	if(mat.rows==4 && abs(mat.at<double>(3,0))>1e-10 )
		mat = mat/ mat.at<double>(3,0) ;

	Point3f  returnPt;
	returnPt.x = mat.at<double>(0,0) ;
	returnPt.y = mat.at<double>(1,0) ;
	returnPt.z = mat.at<double>(2,0) ;

	return returnPt;
}


bool LineInterSection(Point2d p1,Point2d p2,Point2d q1,Point2d q2,Point2d& interPt)
{
	interPt = Point2d(0,0);
	if(p1.x == p2.x)
	{
		if(q1.x ==q2.x)
		{
			//assert(false);
			return false;
		}
		else
		{
			double a2 = (q1.y-q2.y)/(q1.x-q2.x);
			double b2 = q1.y-a2*q1.x;

			interPt.x = p1.x;
			interPt.y = a2*interPt.x+b2;
		}
	}
	else
	{
		if(q1.x ==q2.x)
		{
			double a1 = (p1.y-p2.y)/(p1.x-p2.x);
			double b1 = p1.y-a1*p1.x;

			interPt.x = q1.x;
			interPt.y = a1*interPt.x+b1;
		}
		else
		{
			double a1 = (p1.y-p2.y)/(p1.x-p2.x);
			double b1 = p1.y-a1*p1.x;

			double a2 = (q1.y-q2.y)/(q1.x-q2.x);
			double b2 = q1.y-a2*q1.x;

			if(a1==a2 && b1!=b2)
			{
				return false;
				//assert(false);
			}
			else if(a1==a2 && b1==b2)
				return false;


			interPt.x = (b2-b1)/(a1-a2);
			interPt.y = a1*interPt.x+b1;
		}
	}
	return true;
}

Point3d GetCenter(Point3d p1,Point3d p2)
{
	Point3d center = p1+p2;
	center.x /=2.0;
	center.y /=2.0;
	center.z /=2.0;
	return center;
}


void GetRotMatrix(Mat &axis1,Mat& axis2,Mat& returnRotMat)
{
	Mat rotAxis= axis1.cross(axis2);
	normalize(rotAxis,rotAxis);

	Mat normalizedArrow1,normalizedOtherArrow1;
	normalize(axis1,normalizedArrow1);
	normalize(axis2,normalizedOtherArrow1);

	double cosAngle = normalizedArrow1.dot(normalizedOtherArrow1);
	cosAngle = ((cosAngle>1) ? 1 : cosAngle);
	cosAngle = ((cosAngle<-1) ? -1 : cosAngle);
	double angle = acos(cosAngle);
							
	rotAxis*=angle;
	Rodrigues(rotAxis, returnRotMat);
}



void DrawCrossGray(Mat& image,int x,int y,int pixelGrayValue,int size)
{
	int halfSize = size /2;
	for(int dy = -halfSize;dy<=halfSize;++dy)
	{
		if(IsOutofBoundary(image,x,y+dy))
			continue;
		image.at<uchar>(y+dy,x) = (int) pixelGrayValue;
	}

	for(int dx = -halfSize;dx<=halfSize;++dx)
	{
		if(IsOutofBoundary(image,x+dx,y))
			continue;

		image.at<uchar>(y,x+dx) = (int) pixelGrayValue;
	}
}


void DrawCross(Mat& image,int x,int y,Scalar color,int size)
{
	if(image.channels()==1)
	{
		DrawCrossGray(image,x,y,color(0),size);
		return;
	}
	int halfSize = size /2;
	for(int dy = -halfSize;dy<=halfSize;++dy)
	{
		if(IsOutofBoundary(image,x,y+dy))
			continue;
		image.at<Vec3b>(y+dy,x)(0) = (int) color(0);
		image.at<Vec3b>(y+dy,x)(1) = (int) color(1);//*2.0/3.0;;
		image.at<Vec3b>(y+dy,x)(2) = (int) color(2);
	}

	for(int dx = -halfSize;dx<=halfSize;++dx)
	{
		if(IsOutofBoundary(image,x+dx,y))
			continue;

		image.at<Vec3b>(y,x+dx)(0) = (int) color(0);
		image.at<Vec3b>(y,x+dx)(1) = (int) color(1);//*2.0/3.0;;
		image.at<Vec3b>(y,x+dx)(2) = (int) color(2);
	}
}

void DrawDotGray(Mat& image,int x,int y,int grayValue,int size)
{
	if(IsOutofBoundary(image,x,y))
		return;

	int halfSize = size/2;
	for(int dy =-halfSize;dy<=halfSize;++dy)
		for(int dx =-halfSize;dx<=halfSize;++dx)
		{
			if(dx<0|| dy<0 || dy>=image.rows || dx>=image.cols)
				continue;

			image.at<uchar>(y+dy,x+dx) = COLOR_CLAMP(grayValue);
		}
}
void DrawDotRGB(Mat& image,int x,int y,cv::Scalar& color,int size)
{
	if(image.channels()==1)
	{
		DrawDotGray(image,x,y,color(0),size);
		return;
	}
	if(IsOutofBoundary(image,x,y))
		return;
	int halfSize = size/2;
	for(int dy =-halfSize;dy<=halfSize;++dy)
		for(int dx =-halfSize;dx<=halfSize;++dx)
		{
			int yy =y+dy;
			int xx =x+dx;
			if(yy<0|| xx<0 || yy>=image.rows || xx>=image.cols)
				continue;
			image.at<Vec3b>(yy,xx)(0) = COLOR_CLAMP(color(0));
			image.at<Vec3b>(yy,xx)(1) = COLOR_CLAMP(color(1));//*2.0/3.0;;
			image.at<Vec3b>(yy,xx)(2) = COLOR_CLAMP(color(2));
		}
	
}
void DrawDotRGB(Mat& image,int x,int y,Point3f color,int size)
{
	if(image.channels()==1)
	{
		DrawDotGray(image,x,y,color.x,size);
		return;
	}
	if(IsOutofBoundary(image,x,y))
		return;
	int halfSize = size/2;
	for(int dy =-halfSize;dy<=halfSize;++dy)
		for(int dx =-halfSize;dx<=halfSize;++dx)
		{
			if(dx<0|| dy<0 || dy>=image.rows || dx>=image.cols)
				continue;
			image.at<Vec3b>(y+dy,x+dx)(0) = COLOR_CLAMP(color.x*255);
			image.at<Vec3b>(y+dy,x+dx)(1) = COLOR_CLAMP(color.y*255);//*2.0/3.0;;
			image.at<Vec3b>(y+dy,x+dx)(2) = COLOR_CLAMP(color.z*255);
		}
	
}

void DrawDotBGR(Mat& image,int x,int y,Point3f color,int size)
{
	if(image.channels()==1)
	{
		DrawDotGray(image,x,y,color.x,size);
		return;
	}
	if(IsOutofBoundary(image,x,y))
		return;

	int halfSize = size/2;
	for(int dy =-halfSize;dy<=halfSize;++dy)
		for(int dx =-halfSize;dx<=halfSize;++dx)
		{
			if(dx<0|| dy<0 || dy>=image.rows || dx>=image.cols)
				continue;
			image.at<Vec3b>(y+dy,x+dx)(0) = COLOR_CLAMP(color.z*255);
			image.at<Vec3b>(y+dy,x+dx)(1) = COLOR_CLAMP(color.y*255);//*2.0/3.0;;
			image.at<Vec3b>(y+dy,x+dx)(2) = COLOR_CLAMP(color.x*255);
		}
}

Point2d Plus(Point2d& pt1,Point2d& pt2)
{
	Point2d returnPt;
	returnPt.x = pt1.x + pt2.x;
	returnPt.y = pt1.y + pt2.y;

	return returnPt;
}

Point2d Plus(Point2d& pt1,Point2f& pt2)
{
	Point2d returnPt;
	returnPt.x = pt1.x + pt2.x;
	returnPt.y = pt1.y + pt2.y;

	return returnPt;
}

Point2d Plus(Point2f& pt1,Point2d& pt2)
{
	Point2d returnPt;
	returnPt.x = pt1.x + pt2.x;
	returnPt.y = pt1.y + pt2.y;

	return returnPt;
}

Point2d Minus(Point2d& pt1,Point2d& pt2)
{
	Point2d returnPt;
	returnPt.x = pt1.x - pt2.x;
	returnPt.y = pt1.y - pt2.y;

	return returnPt;
}


Point2d Minus(Point2d& pt1,Point2f& pt2)
{
	Point2d returnPt;
	returnPt.x = pt1.x - pt2.x;
	returnPt.y = pt1.y - pt2.y;

	return returnPt;
}

Point2d Minus(Point2f& pt1,Point2d& pt2)
{
	Point2d returnPt;
	returnPt.x = pt1.x - pt2.x;
	returnPt.y = pt1.y - pt2.y;

	return returnPt;
}

bool IsOutofBoundary(const Mat& image,double x, double y)
{
	if(image.rows==0 || image.cols==0)
		return true;
	if(x<1 || y<1 || x>=image.cols-1 ||y>=image.rows-1)
	//if(x<0 || y<0 || x>=image.cols ||y>=image.rows)
		return true;
	else
		return false;
}

bool IsOutofBoundary(const Mat& image,Rect& bbox)
{
	if(bbox.x<0 || bbox.y<0 || bbox.x+bbox.width >=image.cols ||bbox.y+bbox.height >=image.rows )
		return true;
	else
		return false;
}

bool IsOutofBoundary(int width,int height,double x, double y)
{
	if(x<0 || y<0 || x>=width ||y>=height)
		return true;
	else
		return false;
}

#define DEBUGLOGFILE "c:/tempResult/DebugLog"
char LOG_FILE_NAME[512];

void printfLog(const char *strLog,...)
{

//#ifdef LINUX_COMPILE
#if 1
	
	char finalLog[1024];
	//char strLog2[1024];
	va_list marker;
 
	// 가변 인수를 조립한다.
	va_start( marker, strLog );
	vsprintf(finalLog,strLog,marker);
	printf("%s",finalLog);
#else
	static int count=0;

	//HANDLE hLog;
	ofstream fout;
	char finalLog[1024];
	//char strLog2[1024];
	va_list marker;
	SYSTEMTIME st;
 
	va_start( marker, strLog );
	vsprintf(finalLog,strLog,marker);
	printf("%s",finalLog);
	
	
	if (count == 0) 
	{
		GetLocalTime(&st);
		sprintf(LOG_FILE_NAME,"%s_%d_%d_%d.txt",DEBUGLOGFILE,st.wDay,st.wHour,st.wMinute);
		fout.open(LOG_FILE_NAME,std::ios_base::trunc);

		char startLog[1024];
		sprintf(startLog,"Log Start(%d/%d %d:%d) \n", st.wMonth,st.wDay,st.wHour,st.wMinute,st.wSecond,st.wMilliseconds);
        fout << startLog;
		//hLog=CreateFile(DEBUGLOGFILE,GENERIC_WRITE,0,NULL,
		//CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);
		//WriteFile(hLog,startLog,strlen(startLog),&dwWritten,NULL);
		count++;
	} 
	else 
	{
		fout.open(LOG_FILE_NAME,std::ios_base::app);
		//hLog=CreateFile(DEBUGLOGFILE,GENERIC_WRITE,0,NULL,
		//OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);
	}
 
     //GetLocalTime(&st);
     //wsprintf(strLog2,"카운터=%06d(%d:%d:%d:%d) %s\r\n",count++,
          //st.wHour,st.wMinute,st.wSecond,st.wMilliseconds,szLog);
     //SetFilePointer(hLog,0,NULL,FILE_END);
    // WriteFile(hLog,strLog2,strlen(strLog2),&dwWritten,NULL);
	 //WriteFile(hLog,szLog,strlen(szLog),&dwWritten,NULL);
     //CloseHandle(hLog);
	 fout <<finalLog;
	 fout.close();
#endif
}

//assume that both image size are same
double myNCC(Mat_<double>& im_1,Mat_<double>& im_2)
{
	assert(im_1.cols == im_2.cols);
	assert(im_1.rows == im_2.rows);
	int width = im_1.cols;
	int height = im_1.rows;

	int templateCnt = width * height;
	double mean_1 =0;
	double mean_2 =0;
	for(int y= 0;y<height;++y)
	{
		for(int x= 0;x<width;++x)
		{
			mean_1 += im_1(y,x);
			mean_2 += im_2(y,x);
		}
	}
	mean_1 /=templateCnt;
	mean_2 /=templateCnt;

	double numerator =0;
	double denom_1 =0;
	double denom_2 =0;

	///int cnt =0;
	for(int y= 0;y<height;++y)
	{
		for(int x= 0;x<width;++x)
		{
			double normTemIm_1 = im_1(y,x)- mean_1;
			double normTemIm_2 = im_2(y,x)- mean_2;

			numerator += normTemIm_1* normTemIm_2;
			denom_1 += normTemIm_1*normTemIm_1;
			denom_2 += normTemIm_2*normTemIm_2;
			//cnt ++;
		}
	}
	double denom = sqrt(denom_1 *denom_2);
	if(denom==0)
	{
		return 0;  //there is no correlation
	}

	return numerator / denom;
}

double myNCC(Mat_<double>& im_1,Mat_<double>& im_2,int step)
{
	assert(im_1.cols == im_2.cols);
	assert(im_1.rows == im_2.rows);
	int width = im_1.cols;
	int height = im_1.rows;

	int templateCnt=0;
	double mean_1 =0;
	double mean_2 =0;
	for(int y= 0;y<height;y+=step)
	{
		for(int x= 0;x<width;x+=step)
		{
			mean_1 += im_1(y,x);
			mean_2 += im_2(y,x);

			templateCnt++;
		}
	}
	mean_1 /=templateCnt;
	mean_2 /=templateCnt;

	double numerator =0;
	double denom_1 =0;
	double denom_2 =0;

	for(int y= 0;y<height;y+=step)
	{
		for(int x= 0;x<width;x+=step)
		{
			double normTemIm_1 = im_1(y,x)- mean_1;
			double normTemIm_2 = im_2(y,x)- mean_2;

			numerator += normTemIm_1* normTemIm_2;
			denom_1 += normTemIm_1*normTemIm_1;
			denom_2 += normTemIm_2*normTemIm_2;
		}
	}
	double denom = sqrt(denom_1 *denom_2);
	if(denom==0)
	{
		return 0;  //there is no correlation
	}
	return numerator / denom;
}


double BilinearInterpolation(Mat_<uchar>& grayImage,double x,double y)
{
	//interpolation
	int floor_y = (int)floor(y);
	int ceil_y = (int)ceil(y);
	int floor_x = (int)floor(x);
	int ceil_x = (int)ceil(x);

	double value_lb = grayImage(floor_y,floor_x);   //Vec3d: double vector
	double value_lu = grayImage(ceil_y,floor_x);
	double value_rb = grayImage(floor_y,ceil_x);
	double value_ru = grayImage(ceil_y,ceil_x);

	double alpha = y - floor_y;
	double value_l= (1-alpha) *value_lb + alpha * value_lu;   //left
	double value_r= (1-alpha) *value_rb + alpha * value_ru;
	double beta = x - floor_x;
	double finalValue = (1-beta) *value_l + beta * value_r;

	return finalValue;
}



//For RGB images
//output: Vec3d which is a double vector
Vec3d BilinearInterpolation(Mat_<Vec3b>& rgbImage,double x,double y)
{
	int floor_y = (int)floor(y);
	int ceil_y = (int)ceil(y);
	int floor_x = (int)floor(x);
	int ceil_x = (int)ceil(x);

	if( floor_y>=rgbImage.rows || ceil_y<0  || floor_x>=rgbImage.cols || ceil_x<0 )
		return Vec3d(0,0,0);
	
	if(ceil_y>= rgbImage.rows )	//no need to interplation
		ceil_y = floor_y;
	else if(floor_y <0 )	//no need to interplation
		floor_y = ceil_y;
	
	if(ceil_x>= rgbImage.cols)
		ceil_x = floor_x;
	else if(floor_x < 0)
		floor_x = ceil_x;

	//interpolation
	Vec3d value_lb = rgbImage(floor_y,floor_x);   //Vec3d: double vector
	Vec3d value_lu = rgbImage(ceil_y,floor_x);
	Vec3d value_rb = rgbImage(floor_y,ceil_x);
	Vec3d value_ru = rgbImage(ceil_y,ceil_x);

	double alpha = y - floor_y;
	Vec3d value_l= (1-alpha) *value_lb + alpha * value_lu;
	Vec3d value_r= (1-alpha) *value_rb + alpha * value_ru;
	double beta = x - floor_x;
	Vec3d finalValue = (1-beta) *value_l + beta * value_r;

	return finalValue;
}

void Vec3dToVec3b(Vec3d& source,Vec3b& target)
{
	target(0) = (source(0) >255) ? 255:uchar(source(0)+0.5);
	target(1) = (source(1) >255) ? 255:uchar(source(1)+0.5);
	target(2) = (source(2) >255) ? 255:uchar(source(2)+0.5);
}



#include <cstdlib>
#include <ctime>
//return random number between [min,max]
void randGenerator(int min,int max,int selectionNum, vector<int>& output)
{
	output.clear(); // it should clear the output vector before adding something
	//srand(time(NULL));
	while(output.size() <selectionNum)
	{
		int diff = max - min;
		int tempRandNum= rand() % (diff+1);  //0~ diff
		tempRandNum += min;

		//check
		if(find(output.begin(),output.end(),tempRandNum) != output.end())
			continue;
		
		output.push_back(tempRandNum);
	}
}


void randGenerator2(int min,int max,int selectionNum, vector<int>& output)
{
	Mat_<int> numArray(max-min+1,1);

	int r =0;
	for(int i=min;i<=max;++i)
	{
		numArray(r,0) = i;
		r++;
	}
	
	randShuffle(numArray);

	//printfLog("rand ");
	for(int i=0;i<selectionNum;++i)
	{
		//printf("%d ",numArray(i,0));
		output.push_back(numArray(i,0));
	}
	//printf("\n");

	/*
	//srand(time(NULL));
	while(output.size() <selectionNum)
	{
		int diff = max - min;
		int tempRandNum= rand() % (diff+1);  //0~ diff
		tempRandNum += min;

		//check
		if(find(output.begin(),output.end(),tempRandNum) != output.end())
			continue;
		
		output.push_back(tempRandNum);
	}*/
}

Point2f Project3DPtF(Point3f& pt,Mat_<float>& PMat)
{
	Mat pt4by1;
	//Point3fToMat4by1(pt,pt4by1);
	pt4by1 = Mat::ones(4,1,CV_32F);
	pt4by1.at<float>(0,0) = pt.x;
	pt4by1.at<float>(1,0) = pt.y;
	pt4by1.at<float>(2,0) = pt.z;

	Mat imagePt = PMat* pt4by1;
	imagePt = imagePt/imagePt.at<float>(2,0);
	Point2f returnPt;
	returnPt.x = imagePt.at<float>(0,0);
	returnPt.y = imagePt.at<float>(1,0);
	return returnPt;
}

Point2d Project3DPt(Point3f& pt,Mat& PMat)
{
	Mat pt4by1;
	Point3fToMat4by1(pt,pt4by1);
	Mat imagePt = PMat* pt4by1;
	imagePt = imagePt/imagePt.at<double>(2,0);
	Point2d returnPt;
	returnPt.x = imagePt.at<double>(0,0);
	returnPt.y = imagePt.at<double>(1,0);
	return returnPt;
}

Point2d Project3DPt(Point3d& pt,Mat& PMat)
{
	Mat pt4by1;
	Point3dToMat4by1(pt,pt4by1);
	Mat imagePt = PMat* pt4by1;
	imagePt = imagePt/imagePt.at<double>(2,0);
	Point2d returnPt;
	returnPt.x = imagePt.at<double>(0,0);
	returnPt.y = imagePt.at<double>(1,0);
	return returnPt;
}

Point2d Project3DPt(Mat& pt4by1,Mat& PMat)
{
	Mat imagePt = PMat* pt4by1;
	imagePt = imagePt/imagePt.at<double>(2,0);
	Point2d returnPt;
	returnPt.x = imagePt.at<double>(0,0);
	returnPt.y = imagePt.at<double>(1,0);
	return returnPt;
}

void ParameterSetting(char* filePath)
{
	printf("loading from: %s\n",filePath);
	char buf[512];
	ifstream fin(filePath);
	
	if(fin.is_open()==false)
	{
		printfLog("there is no inputParameterFile: %s\n",filePath);

		return;
	}

	//Initialization
	sprintf(g_manualRenderImgSaveFolder,"");
	//sprintf(g_panopticDataFolder,"");

	while(!fin.eof())
	{
		fin >> buf;
		if(buf[0] =='#')
		{
			fin >> buf; //comment
			continue;
		}
		else if(strcmp(buf,"") ==0)
			continue;

		cout << "Set: " <<buf  <<"=>";

		if(strcmp(buf,"exit") ==0 || strcmp(buf,"exit;") ==0)
		{
			cout<< "Stopped by exit command\n";
			break;
		}
		if(strcmp(buf,"g_dataImageFolder") ==0)
		{
			fin >> g_dataImageFolder;
			cout<<g_dataImageFolder<<"\n";
		}
		else if(strcmp(buf,"g_dataFrameStartIdx") ==0)
		{
			fin >> g_dataFrameStartIdx;
			cout<<g_dataFrameStartIdx<<"\n";
		}
		else if(strcmp(buf,"g_memoryLoadingDataFirstFrameIdx") ==0)
		{
			fin >>g_memoryLoadingDataFirstFrameIdx;
			cout<<g_memoryLoadingDataFirstFrameIdx<<"\n";
		}
		else if(strcmp(buf,"g_dataSpecificFolder") ==0)
		{
			fin >>g_dataSpecificFolder;
			cout<<g_dataSpecificFolder<<"\n";
		}
		else if(strcmp(buf,"g_GTposeEstLoadingDataFolderPath") ==0)
		{
			fin >>g_GTposeEstLoadingDataFolderPath;
			cout<<g_GTposeEstLoadingDataFolderPath<<"\n";
		}
		else if(strcmp(buf,"g_GTSubjectNum") ==0)
		{
			fin >>g_GTSubjectNum;
			cout<<g_GTSubjectNum<<"\n";
		}
		else if(strcmp(buf,"g_memoryLoadingDataNum") ==0)
		{
			fin >> g_memoryLoadingDataNum;
			cout<<g_memoryLoadingDataNum<<"\n";
		}
		else if(strcmp(buf,"g_testDataDirPath") ==0)
		{
			fin>>g_testDataDirPath;
			cout<<g_testDataDirPath<<"\n";
		}
		else if(strcmp(buf,"g_testDataStartFrame") ==0)
		{
			fin>>g_testDataStartFrame;
			cout<<g_testDataStartFrame<<"\n";
		}
		else if(strcmp(buf,"g_testDataImageNum") ==0)
		{
			fin>>g_testDataImageNum;
			cout<<g_testDataImageNum<<"\n";
		}
		else if(strcmp(buf,"g_calibrationFolder") ==0)
		{
			fin >>g_calibrationFolder;
			cout <<g_calibrationFolder<<"\n";
		}
		else if(strcmp(buf,"g_testCaliDataDirPath") ==0)
		{
			fin >>g_testCaliDataDirPath;
			cout <<g_testCaliDataDirPath<<"\n";
		}
		else if(strcmp(buf,"g_dataMainFolder") ==0)
		{
			fin>> g_dataMainFolder;
			cout <<g_dataMainFolder<<"\n";
		}
		else if(strcmp(buf,"g_sequenceName") ==0)
		{
			fin>> g_sequenceName;
			cout <<g_sequenceName<<"\n";
		}
		else if(strcmp(buf,"g_panopticDataFolder") ==0)
		{
			fin>> g_panopticDataFolder;
			cout <<g_panopticDataFolder<<"\n";
		}
		else if(strcmp(buf,"g_trackingFrameLimitNum") ==0)
		{
			fin >> g_trackingFrameLimitNum;
			cout <<g_trackingFrameLimitNum<<"\n";
		}
		else if(strcmp(buf,"g_dataFrameNum") ==0)
		{
			fin >> g_dataFrameNum;
			cout <<g_dataFrameNum<<"\n";
		}
		else if(strcmp(buf,"g_askedVGACamNum") ==0)
		{
			fin >> g_askedVGACamNum;
			cout <<g_askedVGACamNum<<"\n";
		}
		else if(strcmp(buf,"MAX_TRAJ_SHOW_LENGTH") ==0)
		{
			fin >> MAX_TRAJ_SHOW_LENGTH;
			cout <<MAX_TRAJ_SHOW_LENGTH<<"\n";
		}
		else if(strcmp(buf,"g_dataterm_ratio_motion") ==0)
		{
			fin >> g_dataterm_ratio_motion;
			cout <<g_dataterm_ratio_motion<<"\n";
		}
		else if(strcmp(buf,"g_dataterm_ratio_appearance") ==0)
		{
			fin >> g_dataterm_ratio_appearance;
			cout <<g_dataterm_ratio_appearance<<"\n";
		}
		else if(strcmp(buf,"g_MRF_smoothRatio") ==0)
		{
			fin >> g_MRF_smoothRatio;
			cout <<g_MRF_smoothRatio<<"\n";
		}
		else if(strcmp(buf,"g_dataterm_ratio_normal") ==0)
		{
			fin >> g_dataterm_ratio_normal;
			cout <<g_dataterm_ratio_normal<<"\n";
		}
		else if(strcmp(buf,"MAGIC_VISIBILITY_COST_THRESHOLD") ==0)
		{
			fin >> MAGIC_VISIBILITY_COST_THRESHOLD;
			cout <<MAGIC_VISIBILITY_COST_THRESHOLD<<"\n";
		}
		else if(strcmp(buf,"g_bDataterm_ratio_auto") ==0)
		{
			fin >> g_bDataterm_ratio_auto;
			cout <<g_bDataterm_ratio_auto<<"\n";
		}
		else if(strcmp(buf,"g_enableRANSAC") ==0)
		{
			fin >> g_enableRANSAC;
			cout <<g_enableRANSAC<<"\n";
		}
		else if(strcmp(buf,"g_enableMRF") ==0)
		{
			fin >> g_enableMRF;
			cout <<g_enableMRF<<"\n";
		}
		else if(strcmp(buf,"g_pointSelectionByInputParam") ==0)
		{
			fin >> g_pointSelectionByInputParam;
			cout <<g_pointSelectionByInputParam<<"\n";
		}
		else if(strcmp(buf,"g_ShowLongTermThreshDuration") ==0)
		{
			fin >> g_ShowLongTermThreshDuration;
			cout <<g_ShowLongTermThreshDuration<<"\n";
		}
		else if(strcmp(buf,"g_memoryLoadingDataInterval") ==0)
		{
			fin >> g_memoryLoadingDataInterval;
			cout <<g_memoryLoadingDataInterval<<"\n";
		}
		else if(strcmp(buf,"g_backgroundImagePath") ==0)
		{
			fin >> g_backgroundImagePath;
			cout <<g_backgroundImagePath<<"\n";
		}
		else if(strcmp(buf,"g_maskFolderPathForVisualHull") ==0)
		{
			fin >> g_maskFolderPathForVisualHull;
			cout <<g_maskFolderPathForVisualHull<<"\n";
		}
		else if(strcmp(buf,"g_backgroundImageStartIdx") ==0)
		{
			fin >> g_backgroundImageStartIdx;
			cout <<g_backgroundImageStartIdx<<"\n";
		}
		else if(strcmp(buf,"g_backgroundImageNum") ==0)
		{
			fin >> g_backgroundImageNum;
			cout <<g_backgroundImageNum<<"\n";
		}
		else if(strcmp(buf,"DO_HISTOGRAM_EQUALIZE") ==0)
		{
			fin >> DO_HISTOGRAM_EQUALIZE;
			cout <<DO_HISTOGRAM_EQUALIZE<<"\n";
		}
		else if(strcmp(buf,"DONT_USE_SAVED_FEATURE") ==0)
		{
			fin >> DONT_USE_SAVED_FEATURE;
			cout <<DONT_USE_SAVED_FEATURE<<"\n";
		}
		else if(strcmp(buf,"USE_PNG_IMAGE") ==0)
		{
			fin >> USE_PNG_IMAGE;
			cout <<USE_PNG_IMAGE<<"\n";
		}
		else if(strcmp(buf,"g_bDoPhotometOptimization") ==0)
		{
			fin >> g_bDoPhotometOptimization;
			cout <<g_bDoPhotometOptimization<<"\n";
		}
		else if(strcmp(buf,"g_dataFrameInterval") ==0)
		{
			fin >> g_dataFrameInterval;
			cout <<g_dataFrameInterval<<"\n";
		}
		else if(strcmp(buf,"g_bDoRejection_PhotoCons") ==0)
		{
			fin >> g_bDoRejection_PhotoCons;
			cout <<g_bDoRejection_PhotoCons<<"\n";
		}
		else if(strcmp(buf,"g_bDoRejection_PhotoCons_onlySmallMotion") ==0)
		{
			fin >> g_bDoRejection_PhotoCons_onlySmallMotion;
			cout <<g_bDoRejection_PhotoCons_onlySmallMotion<<"\n";
		}
		else if(strcmp(buf,"g_bDoRejection_MotionMagnitude") ==0)
		{
			fin >> g_bDoRejection_MotionMagnitude;
			cout <<g_bDoRejection_MotionMagnitude<<"\n";
		}
		else if(strcmp(buf,"g_bDoRejection_VisCamNum") ==0)
		{
			fin >> g_bDoRejection_VisCamNum;
			cout <<g_bDoRejection_VisCamNum<<"\n";
		}
		else if(strcmp(buf,"g_bDoRejection_PatchSize") ==0)
		{
			fin >> g_bDoRejection_PatchSize;
			cout <<g_bDoRejection_PatchSize<<"\n";
		}
		else if(strcmp(buf,"g_bDoRejection_RANSAC_ERROR") ==0)
		{
			fin >> g_bDoRejection_RANSAC_ERROR;
			cout <<g_bDoRejection_RANSAC_ERROR<<"\n";
		}
		else if(strcmp(buf,"g_faceLoadingDataFirstFrameIdx") ==0)
		{
			fin >> g_faceLoadingDataFirstFrameIdx;
			cout <<g_faceLoadingDataFirstFrameIdx<<"\n";
		}
		else if(strcmp(buf,"g_faceLoadingDataNum") ==0)
		{
			fin >> g_faceLoadingDataNum;
			cout <<g_faceLoadingDataNum<<"\n";
		}
		else if(strcmp(buf,"g_faceMemoryLoadingDataFolderPath") ==0)
		{
			fin >> g_faceMemoryLoadingDataFolderPath;
			cout <<g_faceMemoryLoadingDataFolderPath<<"\n";
		}
		else if(strcmp(buf,"g_poseEstLoadingDataFolderComparePath") ==0)
		{
			fin >> g_poseEstLoadingDataFolderComparePath;
			cout <<g_poseEstLoadingDataFolderComparePath<<"\n";
		}
		else if(strcmp(buf,"g_poseEstLoadingDataFirstFrameIdx") ==0)
		{
			fin >> g_poseEstLoadingDataFirstFrameIdx;
			cout <<g_poseEstLoadingDataFirstFrameIdx<<"\n";
		}
		else if(strcmp(buf,"g_poseEstLoadingDataNum") ==0)
		{
			fin >> g_poseEstLoadingDataNum;
			cout <<g_poseEstLoadingDataNum<<"\n";
		}
		else if(strcmp(buf,"g_poseEstLoadingDataInterval") ==0)
		{
			fin >> g_poseEstLoadingDataInterval;
			cout <<g_poseEstLoadingDataInterval<<"\n";
		}

		else if(strcmp(buf,"g_visualHullLoadingDataFolderPath") ==0)
		{
			fin >> g_visualHullLoadingDataFolderPath;
			cout <<g_visualHullLoadingDataFolderPath<<"\n";
		}
		else if(strcmp(buf,"g_visualHullLoadingDataFirstFrameIdx") ==0)
		{
			fin >> g_visualHullLoadingDataFirstFrameIdx;
			cout <<g_visualHullLoadingDataFirstFrameIdx<<"\n";
		}
		else if(strcmp(buf,"g_visualHullLoadingDataNum") ==0)
		{
			fin >> g_visualHullLoadingDataNum;
			cout <<g_visualHullLoadingDataNum<<"\n";
		}
		else if(strcmp(buf,"PATCH3D_GRID_SIZE") ==0)
		{
			fin >> PATCH3D_GRID_SIZE;
			cout <<PATCH3D_GRID_SIZE<<"\n";

			PATCH3D_GRID_HALFSIZE = PATCH3D_GRID_SIZE/2;
			//PATCH3D_GRID_SIZE = INPUT_PATCH3D_GRID_SIZE;
			printfLog("PATCH3D_GRID_SIZE : %f\n ",PATCH3D_GRID_SIZE);
		}
		else if(strcmp(buf,"PATCH_3D_ARROW_SIZE_CM") ==0)
		{
			fin >> PATCH_3D_ARROW_SIZE_CM;
			cout <<PATCH_3D_ARROW_SIZE_CM<<"\n";

			PATCH_3D_ARROW_SIZE_WORLD_UNIT = cm2world(PATCH_3D_ARROW_SIZE_CM);///WORLD_TO_CM_RATIO;
			printfLog("PATCH_3D_ARROW_SIZE_WORLD_UNIT : %f\n ",PATCH_3D_ARROW_SIZE_WORLD_UNIT);
		}
		else if(strcmp(buf,"g_PoseReconTargetJoint") ==0)
		{
			fin >>g_PoseReconTargetJoint;
			cout<<g_PoseReconTargetJoint<<"\n";
		}
		else if(strcmp(buf,"g_PoseReconTargetSubject") ==0)
		{
			fin >>g_PoseReconTargetSubject;
			cout<<g_PoseReconTargetSubject<<"\n";
		}
		else if(strcmp(buf,"g_PoseReconTargetRequestedNum") ==0)
		{
			fin >>g_PoseReconTargetRequestedNum;
			cout<<g_PoseReconTargetRequestedNum<<"\n";
		}
		else if(strcmp(buf,"g_gpu_device_id") ==0)
		{
			fin >>g_gpu_device_id;
			cout<<g_gpu_device_id<<"\n";
		}
		else if(strcmp(buf,"g_noVisSubject") ==0)
		{
			fin >> g_noVisSubject;
			cout<< g_noVisSubject<<"\n";
		}
		else if(strcmp(buf,"g_noVisSubject2") ==0)
		{
			fin >> g_noVisSubject2;
			cout<< g_noVisSubject2<<"\n";
		}
		else if(strcmp(buf,"g_fpsType") ==0)
		{
			char str[512];
			fin >>str;
			cout<<str<<"\n";
			if(strcmp(str,"FPS_HD_30")==0)
			{
				g_fpsType = FPS_HD_30;
			}
			else if(strcmp(str,"FPS_VGA_25")==0)
				g_fpsType = FPS_VGA_25;
			else
			{
				printf("## WARNING: unknown fps type Just set to the default mode: FPS_VGA_25");
				g_fpsType = FPS_VGA_25;
			}
		}
		else if(strcmp(buf,"g_smpl_model_path") ==0)
		{
			fin >> g_smpl_model_path;
			cout<< g_smpl_model_path<<"\n";
		}
		else if(strcmp(buf,"g_face_model_path") ==0)
		{
			fin >> g_face_model_path;
			cout<< g_face_model_path<<"\n";
		}
		else if(strcmp(buf,"g_handr_model_path") ==0)
		{
			fin >> g_handr_model_path;
			cout<< g_handr_model_path<<"\n";
		}
		else if(strcmp(buf,"g_handl_model_path") ==0)
		{
			fin >> g_handl_model_path;
			cout<< g_handl_model_path<<"\n";
		}
		else if(strcmp(buf,"") ==0)
		{
		}
	}
	fin.close();

	//Other parameter setting
	GetFinalFolderNameInFullPath(g_dataMainFolder,g_sequenceName);
	cout << "AutoSet: g_sequenceName: " << g_sequenceName <<"\n";

	printfLog("\nLoading parameter has been finished\n\n");

}

#if 0
int ExtractFrameIdxFromPath(CString filePath)
{
#ifdef DORM_DATA
	int idx1 = filePath.ReverseFind('/')+1;
	int idx1_otherChoise = filePath.ReverseFind('\\')+1 ;
	if(idx1_otherChoise>idx1)
		idx1 =idx1_otherChoise;
	int idx2 = filePath.ReverseFind('_');
	CString strIdx = filePath.Mid(idx1,idx2-idx1);//dlg.GetFolderPath();
	int numberInt = atoi(strIdx);
	return numberInt;
#else
	int idx1 = filePath.ReverseFind('/')+1;
	int idx1_otherChoise = filePath.ReverseFind('\\')+1 ;
	if(idx1_otherChoise>idx1)
		idx1 =idx1_otherChoise;
	int idx2 = filePath.ReverseFind('.');
	CString strIdx = filePath.Mid(idx1,idx2-idx1);//dlg.GetFolderPath();
	int numberInt = atoi(strIdx);
	return numberInt;
#endif
}

//only for Dorm systme
int ExtractCamIdxFromPath(CString filePath)
{
	int idx1 = filePath.ReverseFind('_')+1;
	int idx2 = filePath.ReverseFind('.');
	CString strIdx = filePath.Mid(idx1,idx2-idx1);//dlg.GetFolderPath();
	int numberInt = atoi(strIdx);
	return numberInt;
}

int ExtractPanelIdxFromPath(CString filePath)
{
	int idx1 = filePath.ReverseFind('_');
	CString strIdx = filePath.Mid(idx1-2,idx1-1);//dlg.GetFolderPath();
	int numberInt = atoi(strIdx);
	return numberInt;
}
#else
int ExtractFrameIdxFromPath(const char* fullPath)  //00000100_01_01.bmp
{
	int leng = strlen(fullPath);
	int folderLeng=0;
	for(int i=leng-1;i>=0;--i)
	{
		if(fullPath[i]=='/' || fullPath[i]=='\\')	
		{
			folderLeng = i;
			break;
		}
	}

	char frameStr[64];
	//memcpy(folderPath,fullPath,sizeof(char)*folderLeng);  //copying from 0~ folderLeng
	memcpy(frameStr,fullPath + folderLeng+1,sizeof(char)*8);  //copying from 0~ folderLeng
	frameStr[8] =0;  //folderLeng < actual folderPath array size

	int numberInt = atoi(frameStr);
	return numberInt;
}



//only for Dorm systme
int ExtractCamIdxFromPath(const char* fullPath)		//00000100_01_01.bmp
{
	int leng = strlen(fullPath);
	int folderLeng=0;
	for(int i=leng-1;i>=0;--i)
	{
		if(fullPath[i]=='/' || fullPath[i]=='\\')	
		{
			folderLeng = i;
			break;
		}
	}
	
	char camStr[64];
	//memcpy(folderPath,fullPath,sizeof(char)*folderLeng);  //copying from 0~ folderLeng
	memcpy(camStr,fullPath + folderLeng+1+12,sizeof(char)*2);  //copying from 0~ folderLeng
	camStr[2] =0;  //folderLeng < actual folderPath array size

	int numberInt = atoi(camStr);
	return numberInt;
}

int ExtractPanelIdxFromPath(const char* fullPath)
{
	int leng = strlen(fullPath);
	int folderLeng;
	for(int i=leng-1;i>=0;--i)
	{
		if(fullPath[i]=='/' || fullPath[i]=='\\')	
		{
			folderLeng = i;
			break;
		}
	}
	
	char panelStr[64];
	//memcpy(folderPath,fullPath,sizeof(char)*folderLeng);  //copying from 0~ folderLeng
	memcpy(panelStr,fullPath + folderLeng+1+9,sizeof(char)*2);  //copying from 0~ folderLeng
	panelStr[2] =0;  //folderLeng < actual folderPath array size

	int numberInt = atoi(panelStr);
	return numberInt;
}

#endif

int ExtractFrameIdxFromPath_last8digit(const char* fullPath)  //00000100_01_01.bmp
{
	int leng = strlen(fullPath);
	char frameStr[64];
	//memcpy(folderPath,fullPath,sizeof(char)*folderLeng);  //copying from 0~ folderLeng
	memcpy(frameStr,fullPath + leng - 4 - 8,sizeof(char)*8);  //copying from 0~ folderLeng
	frameStr[8] =0;  //folderLeng < actual folderPath array size
	int numberInt = atoi(frameStr);
	printf("str: %s-> int : %d\n",frameStr,numberInt);
	return numberInt;
}



void GetFolderPathFromFullPath(const char* fullPath,char folderPath[])
{
	int leng = strlen(fullPath);
	int folderLeng;
	for(int i=leng-1;i>=0;--i)
	{
		if(fullPath[i]=='/' || fullPath[i]=='\\')	
		{
			folderLeng = i;
			break;
		}
	}
	//memcpy(folderPath,fullPath,sizeof(char)*folderLeng);  //copying from 0~ folderLeng
	memcpy(folderPath,fullPath,leng);  //copying from 0~ folderLeng
	folderPath[folderLeng] =0;  //folderLeng < actual folderPath array size
}


void GetFinalFolderNameInFullPath(const char* fullPath,char* folderName)
{
	int leng = strlen(fullPath);
	int folderBeginPt = -1;
	for(int i=leng-1;i>=0;--i)
	{
		if(fullPath[i]=='/' || fullPath[i]=='\\')	
		{
			folderBeginPt = i;
			break;
		}
	}
	if(folderBeginPt<0 )
		return ;

	int lengSub =  leng - folderBeginPt;
	//memcpy(folderPath,fullPath,sizeof(char)*folderLeng);  //copying from 0~ folderLeng
	memcpy(folderName,&fullPath[folderBeginPt+1],lengSub-1);  //copying from 0~ folderLeng
	folderName[lengSub-1] =0;  //folderLeng < actual folderPath array size
}

void GetFileNameFromFullPath(const char* fullPath,char* fileName)
{
	int leng = strlen(fullPath);
	int slashIdx;
	for(int i=leng-1;i>=0;--i)
	{
		if(fullPath[i]=='/' || fullPath[i]=='\\')
		{
			slashIdx = i;
			break;
		}
	}
	int fileNameLeng = leng-slashIdx-1;
	memcpy(fileName,fullPath + (slashIdx+1),sizeof(char)*fileNameLeng);  //copying from 0~ folderLeng
	fileName[fileNameLeng] =0;  //folderLeng < actual folderPath array size
}


#define QW_ZERO 1e-6

void Rotation2Quaternion(Mat& R, Mat& q)
{
	/*double r11 = cvGetReal2D(R,0,0);	double r12 = cvGetReal2D(R,0,1);	double r13 = cvGetReal2D(R,0,2);
	double r21 = cvGetReal2D(R,1,0);	double r22 = cvGetReal2D(R,1,1);	double r23 = cvGetReal2D(R,1,2);
	double r31 = cvGetReal2D(R,2,0);	double r32 = cvGetReal2D(R,2,1);	double r33 = cvGetReal2D(R,2,2);*/

	double r11 = R.at<double>(0,0);	double r12 = R.at<double>(0,1); double r13 = R.at<double>(0,2);
	double r21 = R.at<double>(1,0);	double r22 = R.at<double>(1,1);	double r23 = R.at<double>(1,2);
	double r31 = R.at<double>(2,0);	double r32 = R.at<double>(2,1); double r33 = R.at<double>(2,2);
	
	
	double qw = sqrt(abs(1.0+r11+r22+r33))/2;
	double qx, qy, qz;
	if (qw > QW_ZERO)
	{
		qx = (r32-r23)/4/qw;
		qy = (r13-r31)/4/qw;
		qz = (r21-r12)/4/qw;
	}
	else
	{
		double d = sqrt((r12*r12*r13*r13+r12*r12*r23*r23+r13*r13*r23*r23));
		qx = r12*r13/d;
		qy = r12*r23/d;
		qz = r13*r23/d;
	}

	q.at<double>(0,0) = qw;  //042913 bug fix
	q.at<double>(1,0) = qx;
	q.at<double>(2,0) = qy;
	q.at<double>(3,0) = qz;

	/*cvSetReal2D(q, 0,0,qw);
	cvSetReal2D(q, 1,0,qx);
	cvSetReal2D(q, 2,0,qy);
	cvSetReal2D(q, 3,0,qz);*/

	//QuaternionNormalization(q);
	normalize(q,q);
}

void Quaternion2Rotation(Mat& q, Mat& R)
{
	normalize(q,q);
	//QuaternionNormalization(q);
	/*double qw = cvGetReal2D(q, 0, 0);
	double qx = cvGetReal2D(q, 1, 0);
	double qy = cvGetReal2D(q, 2, 0);
	double qz = cvGetReal2D(q, 3, 0);*/
	double qw = q.at<double>(0,0);
	double qx = q.at<double>(1,0);
	double qy = q.at<double>(2,0);
	double qz = q.at<double>(3,0);

	if(R.rows==0)
		R = Mat::zeros(3,3,CV_64F);

	R.at<double>(0,0) = 1.0-2*qy*qy-2*qz*qz;
	R.at<double>(0,1) = 2*qx*qy-2*qz*qw;
	R.at<double>(0,2) = 2*qx*qz+2*qy*qw;

	R.at<double>(1,0) = 2*qx*qy+2*qz*qw;
	R.at<double>(1,1) = 1.0-2*qx*qx-2*qz*qz;
	R.at<double>(1,2) = 2*qz*qy-2*qx*qw;

	R.at<double>(2,0) = 2*qx*qz-2*qy*qw;
	R.at<double>(2,1) = 2*qy*qz+2*qx*qw;
	R.at<double>(2,2) = 1.0-2*qx*qx-2*qy*qy;
}


#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void ImageSC(Mat& target)
{
	Mat resultMat;
	target.copyTo(resultMat);

	//imwrite("c:/errorPatch.bmp",errorPatch);
	double min, max;
	int minInd, maxInd;
	minMaxIdx(resultMat, &min, &max, &minInd, &maxInd, Mat());
	resultMat=resultMat/max*255;
	resultMat.convertTo(resultMat,CV_8U);

	Mat displayImage;
	applyColorMap(resultMat,displayImage,COLORMAP_JET);

	imshow("colorMap_result",displayImage);
	cvWaitKey();
}

/*
void ImageSC(Mat_<double>& target)
{
	Mat resultMat;
	target.copyTo(resultMat);

	//imwrite("c:/errorPatch.bmp",errorPatch);
	double min, max;
	int minInd, maxInd;
	minMaxIdx(resultMat, &min, &max, &minInd, &maxInd, Mat());
	resultMat=resultMat/max*255;
	resultMat.convertTo(resultMat,CV_8U);

	Mat displayImage;
	applyColorMap(resultMat,displayImage,COLORMAP_JET);

	imshow("colorMap_result",displayImage);
}

void ImageSC(Mat_<float>& target)
{
	Mat resultMat;
	target.copyTo(resultMat);

	//imwrite("c:/errorPatch.bmp",errorPatch);
	double min, max;
	int minInd, maxInd;
	minMaxIdx(resultMat, &min, &max, &minInd, &maxInd, Mat());
	resultMat=resultMat/max*255;
	resultMat.convertTo(resultMat,CV_8U);

	Mat displayImage;
	applyColorMap(resultMat,displayImage,COLORMAP_JET);

	imshow("colorMap_result",displayImage);
}
*/
void ImageSC(Mat_<float>& target,Mat& resultMat,bool noPopup)
{
	//Mat_<float> resultMat;
	target.copyTo(resultMat);

	//imwrite("c:/errorPatch.bmp",errorPatch);
	double min, max;
	int minInd, maxInd;
	minMaxIdx(resultMat, &min, &max, &minInd, &maxInd, Mat());
	resultMat=resultMat/max*255;
	resultMat.convertTo(resultMat,CV_8U);

	Mat displayImage;
	applyColorMap(resultMat,displayImage,cv::COLORMAP_JET);

	if(noPopup==false)
		imshow("colorMap_result",displayImage);
}


void ImageSC(string name,Mat& target)
{
	Mat resultMat;
	target.copyTo(resultMat);

	//imwrite("c:/errorPatch.bmp",errorPatch);
	double min, max;
	int minInd, maxInd;
	minMaxIdx(resultMat, &min, &max, &minInd, &maxInd, Mat());
	resultMat=resultMat/max*255;
	resultMat.convertTo(resultMat,CV_8U);

	Mat displayImage;
	applyColorMap(resultMat,displayImage,COLORMAP_JET);

	imshow(name.c_str(),displayImage);
}



void ImageSC(string name,Mat_<double>& target,Mat& resultMat,bool noPopup)
{
	//Mat_<float> resultMat;
	target.copyTo(resultMat);

	//imwrite("c:/errorPatch.bmp",errorPatch);
	double min, max;
	int minInd, maxInd;
	minMaxIdx(resultMat, &min, &max, &minInd, &maxInd, Mat());
	resultMat=resultMat/max*255;
	resultMat.convertTo(resultMat,CV_8U);

	Mat displayImage;
	applyColorMap(resultMat,displayImage,COLORMAP_JET);
	resultMat = displayImage.clone();

	
	if(noPopup==false)
		imshow(name.c_str(),displayImage);
}

void ImageSC(Mat_<double>& target,Mat& resultMat,bool noPopup)
{
	//Mat_<float> resultMat;
	target.copyTo(resultMat);

	//imwrite("c:/errorPatch.bmp",errorPatch);
	double min, max;
	int minInd, maxInd;
	minMaxIdx(resultMat, &min, &max, &minInd, &maxInd, Mat());
	resultMat=resultMat/max*255;
	resultMat.convertTo(resultMat,CV_8U);

	Mat displayImage;
	applyColorMap(resultMat,displayImage,COLORMAP_JET);
	resultMat = displayImage.clone();

	
	if(noPopup==false)
		imshow("colorMap_result",displayImage);
}
//distorting point using first distortion parameter, in any direction
Point2f DistortPointR1(Point2f& pt, double k1) 
{ 
        if (k1 == 0) 
                return pt; 

		if(pt.y==0)
			pt.y=1e-5;

        const double t2 = pt.y*pt.y; 
        const double t3 = t2*t2*t2; 
        const double t4 = pt.x*pt.x; 
        const double t7 = k1*(t2+t4); 
        if (k1 > 0) { 
                const double t8 = 1.0/t7; 
                const double t10 = t3/(t7*t7); 
                const double t14 = sqrt(t10*(0.25+t8/27.0)); 
                const double t15 = t2*t8*pt.y*0.5; 
                const double t17 = pow(t14+t15,1.0/3.0); 
                const double t18 = t17-t2*t8/(t17*3.0); 
                return Point2f(t18*pt.x/pt.y, t18); 
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
                return Point2f(t19.real()*pt.x/pt.y, t19.real()); 
        } 
}


//set default value to the non max pixels
//by checking windowSize x windowSize 
void NonMaxSuppression(Mat_<float>& input,int windowSize,float defaultValue)
{
	Mat_<float> original;
	input.copyTo(original);
	input = defaultValue;

	int colEnd = input.cols- windowSize;
	int rowEnd = input.rows- windowSize;
	double minVal; double maxVal; Point minLoc; Point maxLoc;

	int offset = int(windowSize/2.0);

	for(int c =0; c<colEnd;c++)
	{
		for(int r =0;r<rowEnd;r++)
		{
			Mat subMatOriginal = original(Rect(c,r,windowSize,windowSize));
			minMaxLoc( subMatOriginal, &minVal, &maxVal, &minLoc, &maxLoc);	
			if(subMatOriginal.at<float>(offset,offset) ==maxVal)
				input(r+offset,c+offset) = maxVal;
		}
	}
}

//set default value to the non max pixels
//by checking windowSize x windowSize 
void NonMinSuppression(Mat_<float>& input,int windowSize,float defaultValue)
{
	Mat_<float> original;
	input.copyTo(original);
	input = defaultValue;

	int colEnd = input.cols- windowSize;
	int rowEnd = input.rows- windowSize;
	double minVal; //floatmaxVal; Point minLoc; Point maxLoc;

	int offset = int(windowSize/2.0);

	for(int c =0; c<colEnd;c++)
	{
		for(int r =0;r<rowEnd;r++)
		{
			Mat_<float> subMatOriginal = original(Rect(c,r,windowSize,windowSize));
			minMaxLoc( subMatOriginal, &minVal);//, &maxVal, &minLoc, &maxLoc);	
			if(subMatOriginal.at<float>(offset,offset) == minVal)
			{
				input(r+offset,c+offset) = minVal;
			}
		}
	}
}

void CalVariance(Mat& input,Mat_<float>& output,int windowSize)
{
	output = Mat_<float>(input.size());

	int colEnd = input.cols - windowSize;
	int rowEnd = input.rows - windowSize;


	int offset = int(windowSize/2.0);

	#pragma omp parallel for
	for(int c =0; c<colEnd;c++)
	{
		for(int r =0;r<rowEnd;r++)
		{
			Scalar mean; Scalar stddev; 
			Mat subMat = input(Rect(c,r,windowSize,windowSize));
			meanStdDev(subMat,mean,stddev);
			output(r+offset,c+offset) = stddev[0];
		}
	}
}


Rect GetRect(double x,double y,double halfWidth,double halfHeight)
{
	int topLeftX =  int(x - halfWidth);
	int topLeftY =  int(y - halfHeight);
	return Rect(topLeftX,topLeftY,halfWidth*2,halfHeight*2);
}

Rect GetRect(KeyPoint& k)
{
	double radius = k.size/2;
	return Rect(int(k.pt.x - radius),int(k.pt.y - radius),int(k.size),int(k.size));
}


void GetWarppedPoints(Mat& homography,vector<Point2f>& sourcePoints,vector<Point2f>& warppedPoints)
{
	for(int i=0;i<sourcePoints.size();++i)
	{
		Mat_<double> sourceMat(3,1);
		sourceMat(0,0) = sourcePoints[i].x;
		sourceMat(1,0) = sourcePoints[i].y;
		sourceMat(2,0) = 1;

		Mat_<double> warppedMat;
		warppedMat = homography * sourceMat;
		warppedMat = warppedMat/warppedMat(2,0);

		Point2f resultPt;
		resultPt.x = warppedMat(0,0);
		resultPt.y = warppedMat(1,0);

		warppedPoints.push_back(resultPt);
	}
}

//return integer rect
//for image crop
void GetBboxCoveringTriplet(vector<Point2f>& triplet,double scaleFactor,Rect& returnBbox)
{
	assert(triplet.size()==3);

	float scaleMax = -1;
	Point2f targetPt = triplet[0];
	for(unsigned int i=1;i<triplet.size();++i)
	{
		Point2f tempDist = triplet[i] - targetPt;

		scaleMax = max(scaleMax,abs(tempDist.x));
		scaleMax = max(scaleMax,abs(tempDist.y));
	}
	int finalScale = ceil( scaleMax * scaleFactor );

	if(finalScale<1)
		finalScale = 1;

	returnBbox = Rect( int(targetPt.x) - finalScale,int(targetPt.y) - finalScale,2*finalScale+1,2*finalScale+1);
}


int g_saveImageCnt =0;
//template<typename _Tp>
void SaveImageWithPoint(Mat& image,Point_<double> pt,int radius,Scalar color)
{
	Mat RGBImage;
	if(image.type() != CV_8UC3)
	{
		cvtColor(image,RGBImage,CV_GRAY2RGB);
	}
	else
		image.copyTo(RGBImage);

	circle(RGBImage,pt,radius,color);
	char savePath[512];
	sprintf(savePath,"c:/tempResult/SavedImage/debug_%04d.bmp",g_saveImageCnt++);
	imwrite(savePath,RGBImage);
}


#ifdef LINUX_COMPILE
#include <dirent.h>
#endif

void GetFilesInDirectory(const string &directory,std::vector<string> &out)
{

#ifdef LINUX_COMPILE
	/*DIR *dir;
	class dirent *ent;
	class stat st;

	dir = opendir(directory);
	while ((ent = readdir(dir)) != NULL) {
		const string file_name = ent->d_name;
		const string full_file_name = directory + "/" + file_name;

		if (file_name[0] == '.')
			continue;

		if (stat(full_file_name.c_str(), &st) == -1)
			continue;

		const bool is_directory = (st.st_mode & S_IFDIR) != 0;

		if (is_directory)
			continue;

		out.push_back(full_file_name);
	}
	closedir(dir);*/

#else    
    HANDLE dir;
    WIN32_FIND_DATA file_data;

    if ((dir = FindFirstFile((directory + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
        return; /* No files found */

    do {
        const string file_name = file_data.cFileName;
        const string full_file_name = directory + "/" + file_name;
        const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

        if (file_name[0] == '.')
            continue;

        if (is_directory)
            continue;

        out.push_back(full_file_name);
    } while (FindNextFile(dir, &file_data));

    FindClose(dir);
#endif
} // GetFilesInDirectory


Mat PlaneFitting(vector<Point3f>& neighborVoxMat)
{
	Mat A = Mat::zeros(neighborVoxMat.size(),4,CV_64F);
	

	for(int i=0;i<neighborVoxMat.size();++i)
	{
		A.at<double>(i,0) = neighborVoxMat[i].x;
		A.at<double>(i,1) = neighborVoxMat[i].y;
		A.at<double>(i,2) = neighborVoxMat[i].z;
		A.at<double>(i,3) = 1;
	}
	Mat X;
	SVD svd(A);
	svd.solveZ(A,X);

	return X;
}

/*
#ifndef LINUX_COMPILE

void CalVarianceGPU(cuda::GpuMat& input,Mat_<float>& output,int windowSize)
{
	output = Mat_<float>(input.size());

	int colEnd = input.cols - windowSize;
	int rowEnd = input.rows - windowSize;
	Scalar mean; Scalar stddev; 

	int offset = int(windowSize/2.0);

	for(int c =0; c<colEnd;c++)
	{
		for(int r =0;r<rowEnd;r++)
		{
			cuda::GpuMat subMat = input(Rect(c,r,windowSize,windowSize));
			meanStdDev(subMat,mean,stddev);
			output(r+offset,c+offset) = stddev[0];
		}
	}
}
#else
void CalVarianceGPU(cuda::GpuMat& input,Mat_<float>& output,int windowSize)
{
}
#endif
*/
#include "boost/filesystem/operations.hpp"
bool IsFileExist(const char* tempFramePath)
{
	return boost::filesystem::exists(tempFramePath);  //works for both files and folders
	/*
	std::ifstream test(tempFramePath);
	if(test.is_open())
	{
		test.close();
		return true;
	}
	else 
		return false;*/
}

//////////////////////////////////////////////////////////////////////////
/// maxValue should be 0~1
void PutGaussianKernel(Mat_<float>& costMap,Point2d centerPt,int bandwidth,double maxValue,float sigmaRatio)
{
	//maxValue *=2;
	//maxValue =1;
	//printf("score %f\n",maxValue);
	double sigma = bandwidth*sigmaRatio;
	for(int dy=-bandwidth;dy<=bandwidth;dy++)
	{
		for(int dx=-bandwidth;dx<=bandwidth;dx++)
		{
			int newX = dx+centerPt.x;
			int newY = dy+centerPt.y;
			if(newX<0 || newY<0 || newX>=costMap.cols || newY>=costMap.rows  )
				continue;

			double d = sqrt(double(dx*dx + dy*dy));
			if(d>bandwidth)
				continue;
			float prob = maxValue* exp(- d*d/(2*sigma*sigma))*255.0f;
			float originalValue = costMap(newY,newX);			//write only if current value is bigger then already exist value
			costMap(newY,newX) = min(max(prob,originalValue),255.0f);
		}
	}
}

//Need to compare cost map. 
//So, I merged both PutGaussian and PutIndex function
//Whenever it put value, it also put index 
void PutGaussianKernelWithIndexLog(Mat_<float>& costMap,Point2d centerPt,int bandwidth,double maxValue,Mat_<int>& indexMap,int index)
{
	//maxValue *=2;
	//maxValue =1;
	//printf("score %f\n",maxValue);
	double sigma = bandwidth*0.6;
	for(int dy=-bandwidth;dy<=bandwidth;dy++)
	{
		for(int dx=-bandwidth;dx<=bandwidth;dx++)
		{
			int newX = dx+centerPt.x;
			int newY = dy+centerPt.y;
			if(newX<0 || newY<0 || newX>=costMap.cols || newY>=costMap.rows  )
				continue;

			double d = sqrt(double(dx*dx + dy*dy));
			if(d>bandwidth)
				continue;
			float prob = maxValue* exp(- d*d/(2*sigma*sigma))*255.0f;
			float originalValue = costMap(newY,newX);			//write only if current value is bigger then already exist value
			costMap(newY,newX) = min(max(prob,originalValue),255.0f);
			indexMap(newY,newX) = index;		//added
		}
	}
}


//put a fixed scalar value on the image. 
//Used to write down associated pt index for the above gaussian
void PutValueKernel(Mat_<int>& indexMap,Point2d centerPt,int bandwidth,int index)
{
	for(int dy=-bandwidth;dy<=bandwidth;dy++)
	{
		for(int dx=-bandwidth;dx<=bandwidth;dx++)
		{
			int newX = dx+centerPt.x;
			int newY = dy+centerPt.y;
			if(newX<0 || newY<0 || newX>=indexMap.cols || newY>=indexMap.rows  )
				continue;

			double d = sqrt(double(dx*dx + dy*dy));
			if(d>bandwidth)
				continue;
			indexMap(newY,newX) = index;
		}
	}
}

void PutGaussianKernel(Mat_<uchar>& costMap,Point2d centerPt,int bandwidth,double maxValue,float sigmaRatio)
{
	//maxValue *=2;
	//maxValue =1;
	//printf("score %f\n",maxValue);
	double sigma = bandwidth*sigmaRatio;
	for(int dy=-bandwidth;dy<=bandwidth;dy++)
	{
		for(int dx=-bandwidth;dx<=bandwidth;dx++)
		{
			int newX = dx+centerPt.x;
			int newY = dy+centerPt.y;
			if(newX<0 || newY<0 || newX>=costMap.cols || newY>=costMap.rows  )
				continue;

			double d = sqrt(double(dx*dx + dy*dy));
			if(d>bandwidth)
				continue;
			int prob = maxValue* exp(- d*d/(2*sigma*sigma))*255;
			int originalValue = costMap(newY,newX);			//write only if current value is bigger then already exist value
			costMap(newY,newX) = min(max(prob,originalValue),255);
		}
	}
}





//////////////////////////////////////////////////////////////////////////
///// Traingaulation Tools
//////////////////////////////////////////////////////////////////////////
#include "PatchOptimization.h"


double CalcReprojectionError_weighted(vector<cv::Mat*>& M,vector<cv::Point2d>& pt2D,vector<double>& weights,cv::Mat& X)
{
	double averageError=0;
	for(unsigned int i=0;i<M.size();++i)
	{
		//printMatrix("M",*M[i]);
		//printMatrix("X",X);
		Mat imageX = (*M[i])* X;
		imageX /= imageX.at<double>(2,0);
		double error = weights[i]*weights[i]*(pow(imageX.at<double>(0,0) -  pt2D[i].x,2) + pow(imageX.at<double>(1,0)-  pt2D[i].y,2));
		//printfLog("error %f",error);
		averageError +=error;
	}
	return averageError /M.size();
}

//for One 3Dpt
double CalcReprojectionError(vector<Mat*>& M,vector<Point2f>& pt2D,Mat& X)
{
	double averageError=0;
	for(unsigned int i=0;i<M.size();++i)
	{
		//printMatrix("M",*M[i]);
		//printMatrix("X",X);
		Mat imageX = (*M[i])* X;
		imageX /= imageX.at<double>(2,0);
		double error = std::sqrt(pow(imageX.at<double>(0,0) -  pt2D[i].x,2) + pow(imageX.at<double>(1,0)-  pt2D[i].y,2));
		//printfLog("error %f",error);
		averageError +=error;
	}
	return averageError /M.size();
}

//for One 3Dpt
double CalcReprojectionError(vector<Mat*>& M,vector<Point2d>& pt2D,Mat& X)
{
	double averageError=0;
	for(unsigned int i=0;i<M.size();++i)
	{
		//printMatrix("M",*M[i]);
		//printMatrix("X",X);
		Mat imageX = (*M[i])* X;
		imageX /= imageX.at<double>(2,0);
		double error = std::sqrt(pow(imageX.at<double>(0,0) -  pt2D[i].x,2) + pow(imageX.at<double>(1,0)-  pt2D[i].y,2));
		//printfLog("error %f",error);
		averageError +=error;
	}
	return averageError /M.size();
}

//for One 3Dpt
double CalcReprojectionError_returnErroVect(vector<Mat*>& M,vector<Point2d>& pt2D,Mat& X,vector<double>& errorReturn)
{
	double averageError=0;
	errorReturn.clear();
	errorReturn.resize(M.size());
	for(unsigned int i=0;i<M.size();++i)
	{
		//printMatrix("M",*M[i]);
		//printMatrix("X",X);
		Mat imageX = (*M[i])* X;
		imageX /= imageX.at<double>(2,0);
		double error = std::sqrt(pow(imageX.at<double>(0,0) -  pt2D[i].x,2) + pow(imageX.at<double>(1,0)-  pt2D[i].y,2));
		errorReturn[i] = error;
		//printfLog("error %f",error);
		averageError +=error;
	}
	return averageError /M.size();
}

double CalcReprojectionError(Mat& ptMat4by1,Mat& projMat,Point2f& pt2D)
{
	Mat x = projMat* ptMat4by1;   //bug fixed 2013-03-27
	x /= x.at<double>(2,0);
	double error = std::sqrt(pow(x.at<double>(0,0) -  pt2D.x,2) + pow(x.at<double>(1,0)-  pt2D.y,2));
	return error;	
}

//DLT
void triangulate(Mat& M1,Point2d& p1, Mat& M2,Point2d& p2,Mat& X)
{
	Mat A = Mat::zeros(4,4,CV_64F);
	Mat temp = p1.x*M1.rowRange(2,3)- M1.rowRange(0,1);
	temp.copyTo(A.rowRange(0,1));
	temp = p1.y*M1.rowRange(2,3)- M1.rowRange(1,2);
	temp.copyTo(A.rowRange(1,2));
	temp = p2.x*M2.rowRange(2,3)- M2.rowRange(0,1);
	temp.copyTo(A.rowRange(2,3));
	temp = p2.y*M2.rowRange(2,3)- M2.rowRange(1,2);
	temp.copyTo(A.rowRange(3,4));

	SVD svd(A);
	svd.solveZ(A,X);
	X /= X.at<double>(3,0);
}

//DLT
void triangulate(vector<Mat*>& M,vector<Point2d*>& p,Mat& X)
{
	int ptNum = (int)M.size();
	Mat A = Mat::zeros(ptNum*2,4,CV_64F);

	for(int i=0;i<ptNum;++i)
	{
		Mat temp = p[i]->x*M[i]->rowRange(2,3)- M[i]->rowRange(0,1);
		temp.copyTo(A.rowRange(i*2,i*2+1));
		temp = p[i]->y*M[i]->rowRange(2,3)- M[i]->rowRange(1,2);
		temp.copyTo(A.rowRange(i*2+1,i*2+2));
	}
	
	SVD svd(A);
	svd.solveZ(A,X);

	X /= X.at<double>(3,0);
}

void triangulate(vector<Mat*>& M,vector<Point2d>& p,Mat& X)
{
	int ptNum = (int)M.size();
	Mat A = Mat::zeros(ptNum*2,4,CV_64F);

	for(int i=0;i<ptNum;++i)
	{
		Mat temp = p[i].x*M[i]->rowRange(2,3)- M[i]->rowRange(0,1);
		temp.copyTo(A.rowRange(i*2,i*2+1));
		temp = p[i].y*M[i]->rowRange(2,3)- M[i]->rowRange(1,2);
		temp.copyTo(A.rowRange(i*2+1,i*2+2));
	}
	
	SVD svd(A); 
	svd.solveZ(A,X);

	X /= X.at<double>(3,0);
}

double triangulateWithOptimization(vector<Mat*>& M,vector<Point2d>& p,Mat& X)
{
	int ptNum = (int)M.size();
	Mat A = Mat::zeros(ptNum*2,4,CV_64F);

	for(int i=0;i<ptNum;++i)
	{
		Mat temp = p[i].x*M[i]->rowRange(2,3)- M[i]->rowRange(0,1);
		temp.copyTo(A.rowRange(i*2,i*2+1));
		temp = p[i].y*M[i]->rowRange(2,3)- M[i]->rowRange(1,2);
		temp.copyTo(A.rowRange(i*2+1,i*2+2));
	}
	
	SVD svd(A);
	svd.solveZ(A,X);
	X /= X.at<double>(3,0);

	//return 0;
	//if(M.size()>=3)
	//double beforeError = CalcReprojectionError(M,p,X);
	double change = TriangulationOptimization(M,p,X);
	//double afterError = CalcReprojectionError(M,p,X);
	//printfLog("!!Mine %.8f , inFunc %.8f \n",beforeError-afterError,change);
	return change;
}


//Mat should be double type always.
double triangulateWithOptimizationF(vector<Mat*>& M,vector<Point2f>& p,Mat& X)
{
	int ptNum = (int)M.size();
	Mat A = Mat::zeros(ptNum*2,4,CV_64F);

	for(int i=0;i<ptNum;++i)
	{
		Mat temp = p[i].x*M[i]->rowRange(2,3)- M[i]->rowRange(0,1);
		temp.copyTo(A.rowRange(i*2,i*2+1));
		temp = p[i].y*M[i]->rowRange(2,3)- M[i]->rowRange(1,2);
		temp.copyTo(A.rowRange(i*2+1,i*2+2));
	}
	
	SVD svd(A);
	svd.solveZ(A,X);
	X /= X.at<double>(3,0);

//   if(M.size()<2)
//  printf("M.size() : %d\n",M.size());
	//double beforeError = CalcReprojectionError(M,p,X);
//	if(M.size()>=2)
	double change = TriangulationOptimizationF(M,p,X);
	//printfLog("Optimization change: %f\n",change);
	//double afterError = CalcReprojectionError(M,p,X);
	//printfLog("??Mine %.8f , inFunc %.8f \n",beforeError-afterError,change);

	return change;
}

double triangulateWithRANSAC(vector<Mat*>& M,vector<Point2d>& pt2D,int iterNum,Mat& X,double thresholdForInlier,vector<unsigned int>& inliers)
{
	if(M.size()<=3)
	{
		for(unsigned int k=0;k<M.size();++k)
			inliers.push_back(k);
		triangulateWithOptimization(M,pt2D,X);

		double error = CalcReprojectionError(M,pt2D,X);
		return error;
	}

	int iter =0;
	Point3d dummy(0,0,0);
	double prevBestArror =1e5;
	while(iter<iterNum)
	{
		vector<Mat*> projecMatVect;
		vector<Point2d> imagePtVect;
		vector<int> selectedIdx; //a huge bug... it should be here, or clear before calling randGenerator
		randGenerator(0,(int)M.size()-1,3,selectedIdx);

		for(unsigned int k=0;k<selectedIdx.size();++k)
		{
			int idx = selectedIdx[k];
			imagePtVect.push_back(pt2D[idx]);  //scale in the image
			projecMatVect.push_back(M[idx]);  //scale in the image
		}
		/*if(imagePtVect[0] ==imagePtVect[1] || imagePtVect[1] ==imagePtVect[2]  || imagePtVect[0] ==imagePtVect[2] )
		{
			iter++;
			continue;
		}*/

		Mat estX(4,1,CV_64F);
		//triangulateWithOptimization(projecMatVect,imagePtVect,estX);
		triangulate(projecMatVect,imagePtVect,estX);

		//find inliers
		vector<unsigned int> tempInliers;
		double averageError=0;
		for(unsigned int i=0;i<M.size();++i)
		{
			Mat imageX = (*M[i])* estX;
			imageX /= imageX.at<double>(2,0);
			double error = sqrt(pow(imageX.at<double>(0,0) -  pt2D[i].x,2) + pow(imageX.at<double>(1,0)-  pt2D[i].y,2));
			if(error<thresholdForInlier)
			{
				averageError +=error;
				tempInliers.push_back(i);
			}
		}
		if(tempInliers.size()>0)
			averageError /= tempInliers.size();
		else
			averageError = 1e5;
		if(tempInliers.size()>inliers.size())
		//if(prevBestArror >averageError)
		{
			inliers = tempInliers;
			//averageError /= tempInliers.size();
			prevBestArror = averageError;
			if(inliers.size()>=4 && averageError<0.2)
				break;
		}
		iter++;
	}
	
	//printfLog("RANSAC: iterNum %d // InlierNum %d // reporoError %.4f \n",iter,inliers.size(),prevBestArror);

	//if(inliers.size()<2)
	if(inliers.size()<=2)
	{
	//	printfLog("too small inliers\n");
		return 1e5;
	}

	vector<Mat*> projecMatVect;
	vector<Point2d> imagePtVect;
	for(unsigned int k=0;k<inliers.size();++k)
	{
		int idx = inliers[k];
		imagePtVect.push_back(pt2D[idx]);  //scale in the image
		projecMatVect.push_back(M[idx]);  //scale in the image
	}

	triangulateWithOptimization(projecMatVect,imagePtVect,X);
	double finalError = CalcReprojectionError(projecMatVect,imagePtVect,X);
	//printfLog("Final Error %.4f  //",finalError);
	//if(beforeError<finalError)
		//printfLog("changes %.4f \n",(beforeError-finalError));
	return finalError;
}


double triangulateWithRANSACExhaustive(vector<Mat*>& M,vector<Point2d>& pt2D,Mat& X,double thresholdForInlier,vector<unsigned int>& inliers)
{
	if(M.size()<=3)
	{
		for(unsigned int k=0;k<M.size();++k)
			inliers.push_back(k);
		triangulateWithOptimization(M,pt2D,X);

		double error = CalcReprojectionError(M,pt2D,X);
		return error;
	}

	int iter =0;
	Point3d dummy(0,0,0);
	double prevBestArror =1e5;

	for (int ii=0;ii<pt2D.size();++ii)
	{
		for (int jj=ii+1;jj<pt2D.size();++jj)
		{
			vector<Mat*> projecMatVect;
			vector<Point2d> imagePtVect;
			vector<int> selectedIdx; //huge bug... it should be here, or clear before calling randGenerator
			//randGenerator(0,(int)M.size()-1,3,selectedIdx);
			selectedIdx.push_back(ii);
			selectedIdx.push_back(jj);

			if(M[ii]==M[jj])
				continue;

			for(unsigned int k=0;k<selectedIdx.size();++k)
			{
				int idx = selectedIdx[k];
				imagePtVect.push_back(pt2D[idx]);  //scale in the image
				projecMatVect.push_back(M[idx]);  //scale in the image
			}
			/*if(imagePtVect[0] ==imagePtVect[1] || imagePtVect[1] ==imagePtVect[2]  || imagePtVect[0] ==imagePtVect[2] )
			{
				iter++;
				continue;
			}*/

			Mat estX(4,1,CV_64F);
			//triangulateWithOptimization(projecMatVect,imagePtVect,estX);
			triangulate(projecMatVect,imagePtVect,estX);

			//find inliers
			vector<unsigned int> tempInliers;
			double averageError=0;
			for(unsigned int i=0;i<M.size();++i)
			{
				Mat imageX = (*M[i])* estX;
				imageX /= imageX.at<double>(2,0);
				double error = std::sqrt(std::pow(imageX.at<double>(0,0) -  pt2D[i].x,2) + std::pow(imageX.at<double>(1,0)-  pt2D[i].y,2));
				if(error<thresholdForInlier)
				{
					averageError +=error;
					tempInliers.push_back(i);
				}
			}
			if(tempInliers.size()>0)
				averageError /= tempInliers.size();
			else
				averageError = 1e5;
			if(tempInliers.size()>inliers.size())
			//if(prevBestArror >averageError)
			{
				inliers = tempInliers;
				//averageError /= tempInliers.size();
				prevBestArror = averageError;
				if(inliers.size()>=3 && averageError<0.2)
					break;
			}
			iter++;
		}
	}
	
	//printfLog("RANSAC: iterNum %d // InlierNum %d // reporoError %.4f \n",iter,inliers.size(),prevBestArror);

	//if(inliers.size()<2)
	if(inliers.size()<=2)
	{
	//	printfLog("too small inliers\n");
		return 1e5;
	}

	vector<Mat*> projecMatVect;
	vector<Point2d> imagePtVect;
	for(unsigned int k=0;k<inliers.size();++k)
	{
		int idx = inliers[k];
		imagePtVect.push_back(pt2D[idx]);  //scale in the image
		projecMatVect.push_back(M[idx]);  //scale in the image
	}

	triangulateWithOptimization(projecMatVect,imagePtVect,X);
	double finalError = CalcReprojectionError(projecMatVect,imagePtVect,X);
	//printfLog("Final Error %.4f  //",finalError);
	//if(beforeError<finalError)
		//printfLog("changes %.4f \n",(beforeError-finalError));
	return finalError;
}

double triangulateWithRANSAC(vector<Mat*>& M,vector<Point2f>& pt2D,int iterNum,Mat& X,double thresholdForInlier,vector<unsigned int>& inliers)
{
	if(M.size()<=3)
	{
		for(unsigned int k=0;k<M.size();++k)
			inliers.push_back(k);
		triangulateWithOptimizationF(M,pt2D,X);

		double error = CalcReprojectionError(M,pt2D,X);
		return error;
	}

	int iter =0;
	Point3d dummy(0,0,0);
	double prevBestArror =1e5;
	while(iter<iterNum)
	{
		vector<Mat*> projecMatVect;
		vector<Point2d> imagePtVect;
		vector<int> selectedIdx; //huge bug... it should be here, or clear before calling randGenerator
		randGenerator(0,(int)M.size()-1,3,selectedIdx);

		for(unsigned int k=0;k<selectedIdx.size();++k)
		{
			int idx = selectedIdx[k];
			imagePtVect.push_back(pt2D[idx]);  //scale in the image
			projecMatVect.push_back(M[idx]);  //scale in the image
		}
		/*if(imagePtVect[0] ==imagePtVect[1] || imagePtVect[1] ==imagePtVect[2]  || imagePtVect[0] ==imagePtVect[2] )
		{
			iter++;
			continue;
		}*/

		Mat estX(4,1,CV_64F);
		//triangulateWithOptimization(projecMatVect,imagePtVect,estX);
		triangulate(projecMatVect,imagePtVect,estX);

		//find inliers
		vector<unsigned int> tempInliers;
		double averageError=0;
		for(unsigned int i=0;i<M.size();++i)
		{
			Mat imageX = (*M[i])* estX;
			imageX /= imageX.at<double>(2,0);
			double error = std::sqrt(std::pow(imageX.at<double>(0,0) -  pt2D[i].x,2) + std::pow(imageX.at<double>(1,0)-  pt2D[i].y,2));
			if(error<thresholdForInlier)
			{
				averageError +=error;
				tempInliers.push_back(i);
			}
		}
		if(tempInliers.size()>0)
			averageError /= tempInliers.size();
		else
			averageError = 1e5;
		if(tempInliers.size()>inliers.size())
		//if(prevBestArror >averageError)
		{
			inliers = tempInliers;
			//averageError /= tempInliers.size();
			prevBestArror = averageError;
			if(inliers.size()>=3 && averageError<0.2)
				break;
		}
		iter++;
	}
	
	//printfLog("RANSAC: iterNum %d // InlierNum %d // reporoError %.4f \n",iter,inliers.size(),prevBestArror);

	//if(inliers.size()<2)
	if(inliers.size()<=2)
	{
	//	printfLog("too small inliers\n");
		return 1e5;
	}

	vector<Mat*> projecMatVect;
	vector<Point2f> imagePtVect;
	for(unsigned int k=0;k<inliers.size();++k)
	{
		int idx = inliers[k];
		imagePtVect.push_back(pt2D[idx]);  //scale in the image
		projecMatVect.push_back(M[idx]);  //scale in the image
	}

	triangulateWithOptimizationF(projecMatVect,imagePtVect,X);
	double finalError = CalcReprojectionError(projecMatVect,imagePtVect,X);
	//printfLog("Final Error %.4f  //",finalError);
	//if(beforeError<finalError)
		//printfLog("changes %.4f \n",(beforeError-finalError));
	return finalError;
}


//return 3x4 matrix
Mat_<double> getRigidTransform(vector<Point3f>& src,vector<Point3f>& dst)
{
	if(src.size() != dst.size())
	{
		Mat_<double> dummy;
		return dummy;
	}

	Point3f s_mean(0,0,0), d_mean(0,0,0);
	Mat_<double> s_bar, d_bar;
	s_bar= Mat_<double>::zeros(3,src.size());
	d_bar= Mat_<double>::zeros(3,dst.size());

	for(int i=0;i<src.size();++i)
		s_mean = s_mean + src[i];
	s_mean = s_mean* (1.0/src.size());
	for(int i=0;i<src.size();++i)
	{
		Point3f temp = src[i]-s_mean;
		s_bar(0,i) = temp.x;
		s_bar(1,i) = temp.y;
		s_bar(2,i) = temp.z;
	}

	for(int i=0;i<dst.size();++i)
		d_mean = d_mean + dst[i];
	d_mean = d_mean* (1.0/dst.size());
	for(int i=0;i<dst.size();++i)
	{
		Point3f temp = dst[i]-d_mean;
		d_bar(0,i) = temp.x;
		d_bar(1,i) = temp.y;
		d_bar(2,i) = temp.z;
	}
	Mat S = s_bar*d_bar.t();
	cv::SVD svd(S);

	Mat R = svd.vt.t() * svd.u.t();
	//R = Mat::eye(3,3,CV_64F);
	//printf("determinine %f \n",determinant(R));
	Mat s_meanMat,d_meanMat;
	Point3ToMatDouble(s_mean,s_meanMat);
	Point3ToMatDouble(d_mean,d_meanMat);
	Mat t = d_meanMat - R*s_meanMat;

	/*
	//Verification
	double errors=0;
	for(int i=0;i<src.size();++i)
	{
		Mat srcMat,dstMat;
		Point3fToMat(src[i],srcMat);
		Point3fToMat(dst[i],dstMat);
		Mat expected = R*srcMat +t;
		errors += Distance(dstMat,expected);
	}
	printf("Rigid TransCompute errors: %f\n",world2cm(errors/src.size()));
	*/

	Mat_<double> rigidT = Mat_<double>(3,4);
	R.copyTo(rigidT.colRange(0,3));
	t.copyTo(rigidT.colRange(3,4));

	return rigidT;
};




//return 3x4 matrix
Mat_<double> getRigidTransform_RANSAC(vector<Point3f>& src,vector<Point3f>& dst)
{
	float ransac_inlier_thresh = 0.1;
	vector< Mat> srcMat;		//to make easy to compute transfomred pt 
	srcMat.resize(src.size());

	for(int t=0;t<src.size();++t)
	{
		Point3fToMat4by1(src[t],srcMat[t]);
	}
	int ransac_iter_num = 10;
	vector<int> bestInlierVector;
	int ranIter=0;
	for(ranIter =0;ranIter<ransac_iter_num;++ranIter)
	{
		//sect point
		vector<int> selectedPts;
		randGenerator(0,src.size()-1,3,selectedPts);
		vector< Point3f> src_selected;
		vector< Point3f> dst_selected;
		for(int selectedIter =0; selectedIter<selectedPts.size(); ++selectedIter)
		{
			int idx = selectedPts[selectedIter];
			src_selected.push_back( src[idx]);
			dst_selected.push_back( dst[idx]);
		}
		Mat_<double> rigidT;
		rigidT = getRigidTransform(src_selected,dst_selected);
		Mat rigidT4by4= Mat_<double>::eye(4,4);
		rigidT.copyTo(rigidT4by4.rowRange(0,3));
		//Determinant check
		if(abs(1-abs(determinant(rigidT4by4)))>0.01)	
			continue;;

		int numInliers=0;
		vector<int> inlierIdxVect;
		inlierIdxVect.reserve(srcMat.size());
		for(int i=0;i<srcMat.size();++i)
		{
			Mat transformedPt = rigidT4by4 * srcMat[i];
			float dist = Distance( MatToPoint3f(transformedPt), dst[i]);
			if(dist<ransac_inlier_thresh)
			{
				inlierIdxVect.push_back(i);
			}
		}

		if(inlierIdxVect.size()> bestInlierVector.size())
		{
			bestInlierVector = inlierIdxVect;
		}
	}
	if(bestInlierVector.size()< 5)
	{
		Mat_<double> dummy;
		return dummy;
	}


	printf("RANSAC done: iterNum %d: inliers %d / %d \n",ranIter, bestInlierVector.size(),src.size());
	vector<Point3f> src_best(bestInlierVector.size());
	vector<Point3f> dst_best(bestInlierVector.size());
	for(int i=0;i<bestInlierVector.size();++i)
	{
		src_best[i] = src[bestInlierVector[i]];
		dst_best[i] = dst[bestInlierVector[i]];
	}
	return getRigidTransform(src_best,dst_best);


}

//return 3x4 matrix
Mat_<double> getRigidTransformWithVerification(vector<Point3f>& src,vector<Point3f>& dst)
{
	if(src.size() != dst.size())
	{
		Mat_<double> dummy;
		return dummy;
	}

	Point3f s_mean(0,0,0), d_mean(0,0,0);
	Mat_<double> s_bar, d_bar;
	s_bar= Mat_<double>::zeros(3,src.size());
	d_bar= Mat_<double>::zeros(3,dst.size());

	for(int i=0;i<src.size();++i)
		s_mean = s_mean + src[i];
	s_mean = s_mean* (1.0/src.size());
	for(int i=0;i<src.size();++i)
	{
		Point3f temp = src[i]-s_mean;
		s_bar(0,i) = temp.x;
		s_bar(1,i) = temp.y;
		s_bar(2,i) = temp.z;
	}

	for(int i=0;i<dst.size();++i)
		d_mean = d_mean + dst[i];
	d_mean = d_mean* (1.0/dst.size());
	for(int i=0;i<dst.size();++i)
	{
		Point3f temp = dst[i]-d_mean;
		d_bar(0,i) = temp.x;
		d_bar(1,i) = temp.y;
		d_bar(2,i) = temp.z;
	}
	Mat S = s_bar*d_bar.t();
	cv::SVD svd(S);

	Mat R = svd.vt.t() * svd.u.t();
	//R = Mat::eye(3,3,CV_64F);
	//printf("determinine %f \n",determinant(R));
	Mat s_meanMat,d_meanMat;
	Point3ToMatDouble(s_mean,s_meanMat);
	Point3ToMatDouble(d_mean,d_meanMat);
	Mat t = d_meanMat - R*s_meanMat;

	
	//Verification
	if(src.size()<30)
	{
		double errors=0;
		for(int i=0;i<src.size();++i)
		{
			Mat srcMat,dstMat;
			Point3fToMat(src[i],srcMat);
			Point3fToMat(dst[i],dstMat);
			Mat expected = R*srcMat +t;
			errors += Distance(dstMat,expected);
		}
		//printf("Rigid TransCompute errors: %f\n",world2cm(errors/src.size()));
		if(errors>1.0)
		{
			Mat_<double> garbage;
			return garbage;
		}
	}
	
	Mat_<double> rigidT = Mat_<double>(3,4);
	
	R.copyTo(rigidT.colRange(0,3));
	t.copyTo(rigidT.colRange(3,4));

	return rigidT;
};



//Return 4x4 transform matrtix
Mat_<double> getRigidTransformWithIterationWithReject(vector<Point3f>& originalSrc,vector<Point3f>& originalDst,int iterationNum)
{
	Mat rigidTConcat= Mat_<double>::eye(4,4);


	Mat_<double> rigidT = getRigidTransform(originalSrc,originalDst);
	Mat rigitT4by4= Mat_<double>::eye(4,4);
	rigidT.copyTo(rigitT4by4.rowRange(0,3));
	rigidTConcat = rigitT4by4*rigidTConcat;

	for(int iter=1;iter<iterationNum;++iter)
	{
		vector<Point3f> src,dst;

		//Find inlier set
		double errorSum=0;
		for(int i=0;i<originalSrc.size();++i)
		{
			float initDist = Distance(originalSrc[i],originalDst[i]);
			Mat pt;
			Point3fToMat4by1(originalSrc[i],pt);
			pt = rigidTConcat * pt;		//transformed pt
			Point3f transformedPt = MatToPoint3f(pt);
			float afterDist = Distance(transformedPt,originalDst[i]);
			errorSum+=afterDist;

			//if(initDist*0.5>=afterDist)		//This is an inlier, since rigidT is working for it. 
			{
				src.push_back(transformedPt);
				dst.push_back(originalDst[i]);
			}

		}
		printf("error before next computation %f\n",errorSum/=src.size());

		if(src.size()<5)// || src.size()<sizeThreshold)
			break;
		rigidT = getRigidTransform(src,dst);
		//printMatrix("temp",rigidT);
			
		Mat rigitT4by4= Mat_<double>::eye(4,4);
		rigidT.copyTo(rigitT4by4.rowRange(0,3));
		rigidTConcat = rigitT4by4*rigidTConcat;
	}

	return rigidTConcat;
}



//Return 4x4 transform matrtix
Mat_<double> getRigidTransformWithIteration(vector<Point3f>& originalSrc,vector<Point3f>& originalDst,int iterationNum)
{
	if(originalSrc.size()<10)// || src.size()<sizeThreshold)
	{
		Mat rigidTConcat= Mat_<double>::zeros(4,4);
		return rigidTConcat;
	}


	Mat rigidTConcat= Mat_<double>::eye(4,4);

	Mat_<double> rigidT = getRigidTransform(originalSrc,originalDst);
	Mat rigitT4by4= Mat_<double>::eye(4,4);
	rigidT.copyTo(rigidTConcat.rowRange(0,3));
	for(int iter=1;iter<iterationNum;++iter)
	{
		vector<Point3f> src,dst;
		src.reserve(originalSrc.size());

		//Find inlier set
		//double beforeErrorSum=0;
		//double afterErrorrSum=0;
		for(int i=0;i<originalSrc.size();++i)
		{
			//float initDist = Distance(originalSrc[i],originalDst[i]);
			Mat pt;
			Point3fToMat4by1(originalSrc[i],pt);
			pt = rigidTConcat * pt;		//transformed pt
			Point3f transformedPt = MatToPoint3f(pt);
			//float afterDist = Distance(transformedPt,originalDst[i]);
			//beforeErrorSum +=afterDist;

			//if(initDist*0.5>=afterDist)		//This is an inlier, since rigidT is working for it. 
			{
				src.push_back(transformedPt);
				//dst.push_back(originalDst[i]);
			}

		}
		//printf("error before next computation %f\n",beforeErrorSum/=src.size());
		
		rigidT = getRigidTransform(src,originalDst);
		Mat rigitT4by4= Mat_<double>::eye(4,4);
		rigidT.copyTo(rigitT4by4.rowRange(0,3));
		rigidTConcat = rigitT4by4*rigidTConcat;
	}


	return rigidTConcat;
}







Scalar g_black(0,0,0);
Scalar g_blue(255,0,0);		
Scalar g_green(0,255,0);
Scalar g_red(0,0,255);
Scalar g_white(255,255,255);
Scalar g_cyan(255,255,0);
Scalar g_yellow(0,255,255);

Point3f g_black_p3f(0,0,0);
Point3f g_red_p3f(1,0,0);
Point3f g_yellow_p3f(1,1,0);
Point3f g_gray_p3f(0.5,0.5,0.5);
Point3f g_blue_p3f(0,0,1);
Point3f g_green_p3f(0,1,0);
Point3f g_cyan_p3f(0,1,1);
Point3f g_magenta_p3f(1,0,1);
Point3f g_orange_p3f(1,0.5,0);

Point3d g_black_p3d(0,0,0);
Point3d g_red_p3d(1,0,0);
Point3d g_yellow_p3d(1,1,0);
Point3d g_gray_p3d(0.5,0.5,0.5);
Point3d g_blue_p3d(0,0,1);
Point3d g_green_p3d(0,1,0);
Point3d g_cyan_p3d(0,1,1);
Point3d g_magenta_p3d(1,0,1);
Point3d g_orange_p3d(1,0.5,0);





void VisualizePointReprojection(Mat& targetImage,Point3d pt3D,Mat& PMatrix,Scalar& tempColor)
{
	Mat ptMat4by1 = Mat::ones(4,1,CV_64F);
	Point3dToMat4by1(pt3D,ptMat4by1);

	Mat imagePt = PMatrix *  ptMat4by1;
	imagePt = imagePt/imagePt.at<double>(2,0);
	double imagePtX = imagePt.at<double>(0,0);
	double imagePtY = imagePt.at<double>(1,0);

	if(!IsOutofBoundary(targetImage,imagePtX,imagePtY))
		DrawCross(targetImage,imagePtX,imagePtY,tempColor,10);
		//circle(targetImage,Point2d(imagePtX,imagePtY),10,tempColor,3);
}


void PathChangeFromPatchCloudToTrajStream(const char* patchCloudFile,char* trajStreamFile)
{
	if(patchCloudFile==NULL)
		return;

	int filePathLength = strlen(patchCloudFile);
	strcpy(trajStreamFile,patchCloudFile);
	if(patchCloudFile[filePathLength-1] == 'm')	//.mem file: old-fashioned naming style
	{
		trajStreamFile[filePathLength-4] =0;			//.mem
		sprintf(trajStreamFile,"%s.track",trajStreamFile);  
	}
	else if(patchCloudFile[filePathLength-1] == 't')
	{
		trajStreamFile[filePathLength-4] =0;			//.pat
		sprintf(trajStreamFile,"%s.traj",trajStreamFile);
	}
	else
	{
		printf("Unknown naming format: %s\n",patchCloudFile);
	}
	
}


bool WorkingVolumeCheck(Mat X)
{
	Point3d tempX = MatToPoint3d(X);
	double dist = Distance( DOME_VOLUMECUT_CENTER_PT , tempX);

	if( dist > DOME_VOLUMECUT_RADIOUS)
		return false;

	//Eliminate bottom
	double* pt = X.ptr<double>(0);
	if(pt[1] >DOME_VOLUMECUT_Y_MAX)
		return false;

	/*
	double* pt = X.ptr<double>(0);
	if(pt[0] <DOME_VOLUMECUT_X_MIN || pt[0] >DOME_VOLUMECUT_X_MAX)
		return false;
	if(pt[1] <DOME_VOLUMECUT_Y_MIN || pt[1] >DOME_VOLUMECUT_Y_MAX)
		return false;
	if(pt[2] <DOME_VOLUMECUT_Z_MIN || pt[2] >DOME_VOLUMECUT_Z_MAX)
		return false;
	*/

	return true;
}

bool WorkingVolumeCheck(Point3d X)
{
	double dist = Distance( DOME_VOLUMECUT_CENTER_PT , X);

	if( dist > DOME_VOLUMECUT_RADIOUS)
		return false;
	/*
	double* pt = X.ptr<double>(0);
	if(pt[0] <DOME_VOLUMECUT_X_MIN || pt[0] >DOME_VOLUMECUT_X_MAX)
		return false;
	if(pt[1] <DOME_VOLUMECUT_Y_MIN || pt[1] >DOME_VOLUMECUT_Y_MAX)
		return false;
	if(pt[2] <DOME_VOLUMECUT_Z_MIN || pt[2] >DOME_VOLUMECUT_Z_MAX)
		return false;
	*/
	return true;
}

void Get3DPtfromDist(Mat& Kinv,Mat& Rinv,Mat& t,Point2d& pt,double depthDist,Point3d& pt3D)
{
	Mat ptMat = Mat::ones(3,1,CV_64F);
	ptMat.at<double>(0,0) = pt.x;
	ptMat.at<double>(1,0) = pt.y;

	ptMat = Kinv *ptMat;  
	ptMat = ptMat/ptMat.at<double>(2,0);		//depth coordinate is 1. 
	ptMat = ptMat * depthDist;				//depth coordinate is depthDist
	ptMat = Rinv * (ptMat - t);//from cam coord to world coord

	pt3D.x = ptMat.at<double>(0,0);
	pt3D.y = ptMat.at<double>(1,0);
	pt3D.z = ptMat.at<double>(2,0);
}

#if 0
Mat ReadImageFromDisk(const char* folderName,int frameIdx,int panelIdx,int camIdx,bool bLoadAsGray)
{
	Mat img;
	char fileName[256];
	stringstream ssError;
	char buf[512];

	//temporary
	if(panelIdx==0 )
		sprintf(fileName,"%s/cam%d/frame%05d.png",folderName,camIdx+1,frameIdx);
	if(bLoadAsGray==false)
		img = imread(fileName);
	else
		img = imread(fileName,CV_LOAD_IMAGE_GRAYSCALE);
	if(img.rows>0)
	{
		printf("## Success:: load image: %s\n",fileName);
		return img;
	}
	sprintf(buf,"## WARNING:: Try1 : failure to load image: %s\n",fileName);
	ssError << buf;



	if(panelIdx==0 )
		sprintf(fileName,"%s/hd_30/%08d/hd%08d_%02d_%02d.png",folderName,frameIdx,frameIdx,panelIdx,camIdx);
	else
		sprintf(fileName,"%s/vga_25/%08d/%08d_%02d_%02d.png",folderName,frameIdx,frameIdx,panelIdx,camIdx);
	if(bLoadAsGray==false)
		img = imread(fileName);
	else
		img = imread(fileName,CV_LOAD_IMAGE_GRAYSCALE);
	if(img.rows>0)
	{
		printf("## Success:: load image: %s\n",fileName);
		return img;
	}
	sprintf(buf,"## WARNING:: Try2 : failure to load image: %s\n",fileName);
	ssError << buf;

	//Retry
	if(panelIdx==0 )
		sprintf(fileName,"%s/hd_30/%03dXX/%08d/hd%08d_%02d_%02d.png",folderName,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
	else
		sprintf(fileName,"%s/vga_25/%03dXX/%08d/%08d_%02d_%02d.png",folderName,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
	if(bLoadAsGray==false)
		img = imread(fileName);
	else
		img = imread(fileName,CV_LOAD_IMAGE_GRAYSCALE);
	if(img.rows>0)
	{
		printf("## Success:: load image: %s\n",fileName);
		return img;
	}
	sprintf(buf,"## WARNING:: Try3 : failure to load image: %s\n",fileName);
	ssError << buf;


	//Retry
	if(panelIdx==0 )
		sprintf(fileName,"%s/%03dXX/%08d/hd%08d_%02d_%02d.png",folderName,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
	else
		sprintf(fileName,"%s/%03dXX/%08d/%08d_%02d_%02d.png",folderName,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
	if(bLoadAsGray==false)
		img = imread(fileName);
	else
		img = imread(fileName,CV_LOAD_IMAGE_GRAYSCALE);
	if(img.rows>0)
	{
		printf("## Success:: load image: %s\n",fileName);
		return img;
	}

	//Failed to load
	sprintf(buf,"## WARNING:: Try4 : failure to load image: %s\n",fileName);
	ssError << buf;

	cout <<ssError.str();
	return img;
}

#else

Mat ReadImageFromDisk(const char* folderName,int frameIdx,int panelIdx,int camIdx,bool bLoadAsGray)
{
	Mat img;
	char fileName[256];
	stringstream ssError;
	char buf[512];

	//Check existing folders
	vector<string> folderCandidate;
	if(panelIdx==0)  //HD
	{
		sprintf(fileName,"%s/hd_30/%03dXX/%08d/hd%08d_%02d_%02d.png",folderName,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
		folderCandidate.push_back(string(fileName));

		if(g_calibrationFolder[strlen(g_calibrationFolder)-1]=='s')		//calibFiles
		{
			//from panoptic data folder
			sprintf(fileName,"%s/%s/hdImgs/00_%02d/%02d_%02d_%08d.jpg",g_panopticDataFolder,g_sequenceName,camIdx,panelIdx,camIdx,frameIdx);
			folderCandidate.push_back(string(fileName));
		}
		else   //calibFiles_withoutDistortion
		{
			sprintf(fileName,"%s/idealImgs/hd_30/%03dXX/%08d/hd%08d_%02d_%02d.png",g_dataMainFolder,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
			folderCandidate.push_back(string(fileName));
		}
		
		//old version
		sprintf(fileName,"%s/hd_30/%08d/hd%08d_%02d_%02d.png",folderName,frameIdx,frameIdx,panelIdx,camIdx);
		folderCandidate.push_back(string(fileName));

		sprintf(fileName,"%s/%03dXX/%08d/%08d_%02d_%02d.png",folderName,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
		folderCandidate.push_back(string(fileName));

		sprintf(fileName,"%s/cam%d/frame%05d.png",folderName,camIdx+1,frameIdx);
		folderCandidate.push_back(string(fileName));
	}
	else   //VGA
	{
		sprintf(fileName,"%s/vga_25/%03dXX/%08d/%08d_%02d_%02d.png",folderName,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
		folderCandidate.push_back(string(fileName));

		if(g_calibrationFolder[strlen(g_calibrationFolder)-1]=='s')		//calibFiles
		{
			//from panoptic data folder
			sprintf(fileName,"%s/%s/vgaImgs/%02d_%02d/%02d_%02d_%08d.jpg",g_panopticDataFolder,g_sequenceName,panelIdx,camIdx,panelIdx,camIdx,frameIdx);
			folderCandidate.push_back(string(fileName));
		}
		else
		{
			sprintf(fileName,"%s/idealImgs/vga_25/%03dXX/%08d/%08d_%02d_%02d.png",g_dataMainFolder,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
			folderCandidate.push_back(string(fileName));
		}

		//old version
		sprintf(fileName,"%s/vga_25/%08d/%08d_%02d_%02d.png",folderName,frameIdx,frameIdx,panelIdx,camIdx);
		folderCandidate.push_back(string(fileName));

		sprintf(fileName,"%s/%03dXX/%08d/%08d_%02d_%02d.png",folderName,int(frameIdx/100),frameIdx,frameIdx,panelIdx,camIdx);
		folderCandidate.push_back(string(fileName));

		sprintf(fileName,"%s/cam%d/frame%05d.png",folderName,camIdx+1,frameIdx);
		folderCandidate.push_back(string(fileName));
	}

	printf("ReadImageFromDisk\n");
	for (int i=0;i<folderCandidate.size();++i)
	{
		if(IsFileExist(folderCandidate[i].c_str())==false)
		{
			sprintf(buf,"## WARNING:: Failure to load image: %s\n",folderCandidate[i].c_str());
			ssError << buf;
			continue;
		}
		printf("## Load from: %s\n",folderCandidate[i].c_str());
		if(bLoadAsGray==false)
			img = imread(folderCandidate[i].c_str());
		else
			img = imread(folderCandidate[i].c_str(),CV_LOAD_IMAGE_GRAYSCALE);

		if(img.rows>0)
		{
			break;
		}
	}
	if(img.rows==0)
	{
		sprintf(buf,"## WARNING:: Finally Failed in loading\n");
		cout <<ssError.str();
	}

	return img;
}
#endif

int GetCurGlobalImgFrame()
{
	if(g_fpsType==FPS_VGA_25)
		return g_imgFrameIdxSlider_vga;
	else
		return g_imgFrameIdxSlider_hd;
}