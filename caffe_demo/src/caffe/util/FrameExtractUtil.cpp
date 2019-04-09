#include "FrameExtractUtil.h"
#include <fstream>
#include <omp.h>
#include <sys/stat.h>       //for mkdir
#ifndef _fseeki64
#define _fseeki64 fseeko64
#endif

#ifndef __int64
#define __int64 int64
#endif

#include "pfcmu_config.h"
//#include "SyncManager.h"


#ifndef SYNCMANAGER_H
const int VGA_IMGWIDTH=640;
const int VGA_IMGHEIGHT=480;
const __int64 VGA_PIXNUM = VGA_IMGWIDTH * VGA_IMGHEIGHT;
const int VGA_CAM_NUM_IN_A_PANEL = 24;
const __int64 VGA_BLOCKSIZE = VGA_PIXNUM*VGA_CAM_NUM_IN_A_PANEL;
const int VGA_TIMECODE_SIZE = 4;

//HD Parameters
const int HD_IMGWIDTH = 1920;
const int HD_IMGHEIGHT = 1080;
const int HD_STRBLOCKSIZE = 256;
const __int64 HD_BLOCKSIZE =  HD_IMGWIDTH*HD_IMGHEIGHT*2;
const __int64 HD_BLOCKSIZE_WITHHEAD =  HD_BLOCKSIZE + HD_STRBLOCKSIZE;
const double HDfps = 30.0;

//KINECT Parameters
const int KINECT_IMGWIDTH = 1920;
const int KINECT_IMGHEIGHT = 1080;
const __int64 KINECT_BLOCKSIZE = KINECT_IMGWIDTH*KINECT_IMGHEIGHT*2;
#endif

using namespace std;
using namespace cv;

template <typename T>
inline unsigned char clamp(T x) {
    return (x>255 ? 255 : (x<0 ? 0:x));
}

void ycrcb2rgb_yuv422_kinect(int Y, int Cr, int Cb, unsigned char* R, unsigned char* G, unsigned char* B)
{
    //From Tomas's code
    *R = clamp(Y + 1.402*(Cr-128));
    *G = clamp(Y - 0.344 * (Cb-128) - 0.714 * (Cr-128));
    *B = clamp(Y + 1.722 * (Cb-128));
}

//Found somewhere... Blackmagic
void ycrcb2rgb_HD(int Y, int Cr, int Cb, unsigned char* R, unsigned char* G, unsigned char* B)
{
#if 0           //HD camera's original code. (actuall for SD type, so need to check)
    *R = clamp(1.164*(Y-16) + 1.596*(Cr-128));
    *G = clamp(1.164*(Y-16) - 0.813*(Cr-128)- 0.391*(Cb-128));
    *B = clamp(1.164*(Y-16) + 2.018*(Cb-128));
#else           //new from internat
    // for HD. https://forum.blackmagicdesign.com/viewtopic.php?f=12&t=29413
    *R = clamp(1.164*(Y - 16) + 1.793*(Cr - 128));
    *G = clamp(1.164*(Y - 16) - 0.534*(Cr - 128) - 0.213*(Cb - 128));
    *B = clamp(1.164*(Y - 16) + 2.115*(Cb - 128));
#endif
}


//VGA
int ConvFrameToTC_vga(FILE* fp_vga,int vgaFrameIdx)     //FrameIdx starts from 0
{
    __int64 memPos = __int64(vgaFrameIdx) * VGA_BLOCKSIZE;
    if(_fseeki64(fp_vga, memPos, SEEK_SET)!=0)
	return -1;
    char im_vga[VGA_BLOCKSIZE];
    //m_im_vga = new char[VGA_BLOCKSIZE];
    int re= fread(im_vga, VGA_TIMECODE_SIZE,1,fp_vga);
    PFCMU::timestamp_t ts = PFCMU::get_timestamp(im_vga);
    return int(ts);
}


//use firstTC
bool imageExtraction_vga_withDropChecker(VGAPanelInfo& panelInfo
                        ,int refFrameIdx,int firstTC
                        ,bool bUndist,vector<int>& camIdxVect,vector<cv::Mat>& outImgVect)
{

	FILE* fp = panelInfo.m_fp;
	if(fp ==NULL)
	{
		printf("File Pointer is NULL\n");
        	return false;
	}

	int tc = ConvFrameToTC_vga(fp,refFrameIdx);
	if(tc<0)
	{
		printf("reaches end of file\n");
		return 0;
	}
	int computedFrameIdx = (tc - firstTC)/4;
	if(refFrameIdx == computedFrameIdx)
	{
		//do nothing
	}
	else if(refFrameIdx > computedFrameIdx)
	{
		printf("Frame drop detected\n");
		int memLocation =  refFrameIdx;
		int history = computedFrameIdx;
		while(refFrameIdx > computedFrameIdx)		//refFrameIdx > computedFrameIdx
		{
			memLocation++;
			int tc = ConvFrameToTC_vga(fp,memLocation);
			if(tc<0)
			{
				printf("reached end of file in the while loop\n");
				return false;
			}
			computedFrameIdx = (tc - firstTC)/4;
			if(history == computedFrameIdx)
			{

				printf("debug: %d -> %d\n",history,computedFrameIdx);
				printf("reached end of file in the while loop\n");
				return false;
			}
						
		}
		if(refFrameIdx != computedFrameIdx)
			return false;

		printf("Frame drop detected, but handled the problem: refFrameIdx %d vs localFrameIdx %d\n",refFrameIdx,computedFrameIdx);
		refFrameIdx = memLocation;

	}
	else	//refFrameIdx < computeFrameIdx
	{
		printf("Frame drop detected (This VGA Raw has more frames than others??) \n");
		int memLocation =  refFrameIdx;
		while(refFrameIdx < computedFrameIdx)		//refFrameIdx > computedFrameIdx
		{
			memLocation--;
			int tc = ConvFrameToTC_vga(fp,memLocation);
			computedFrameIdx = (tc - firstTC)/4;
		}
		if(refFrameIdx != computedFrameIdx)
			return false;
		printf("Frame drop detected, but handled the problem: refFrameIdx %d vs localFrameIdx %d\n",refFrameIdx,computedFrameIdx);
		refFrameIdx = memLocation;
	}


	//Mat bufImage(KINECT_IMGHEIGHT,KINECT_IMGWIDTH*2,CV_8UC1);
	IplImage *img = cvCreateImage(cvSize(VGA_IMGWIDTH,VGA_IMGHEIGHT),IPL_DEPTH_8U,1);
	//IplImage *rgbimg = cvCreateImage(cvSize(VGA_IMGWIDTH,VGA_IMGHEIGHT),IPL_DEPTH_8U,3);

	outImgVect.resize(camIdxVect.size());
	
	for(int cc=0;cc<camIdxVect.size();++cc)
	{

		int camIdx = camIdxVect[cc];
		__int64 memPos;
		memPos = __int64(refFrameIdx)*VGA_BLOCKSIZE + __int64(camIdx-1)*VGA_PIXNUM;
		_fseeki64(fp, memPos, SEEK_SET);

		fread(img->imageData, VGA_PIXNUM, 1,fp) ;
    		outImgVect[cc] = Mat(VGA_IMGHEIGHT,VGA_IMGWIDTH,CV_8UC3);
		IplImage rgbIm = outImgVect[cc];	
		IplImage* prgbIm = &rgbIm;	
		cvCvtColor(img,prgbIm,CV_BayerGR2BGR);
	}

	cvReleaseImage(&img);
	//cvReleaseImage(&rgbimg);

//	printf("%d: vs ts: %d (%d)\n",refFrameIdx,(ts-6000)/4,ts);
    return true;
}


void imageExtraction_vga(VGAPanelInfo& panelInfo,const char* outputFolder, int panelIdx,
                          int startFrameIdx,int endFrameIdx, int frameInterval
                         ,bool bUndist,bool bJpgOut)

{
    FILE* fp = panelInfo.m_fp;
     if(fp ==NULL)
     {
         printf("File Pointer is NULL\n");
         return;
     }

    char buf[256];
    IplImage *img = cvCreateImage(cvSize(VGA_IMGWIDTH,VGA_IMGHEIGHT),IPL_DEPTH_8U,1);
    IplImage *rgbimg = cvCreateImage(cvSize(VGA_IMGWIDTH,VGA_IMGHEIGHT),IPL_DEPTH_8U,3);

    Mat blackImg = Mat(480,640,CV_8UC3);
    for(int frames =startFrameIdx;frames <=endFrameIdx;frames+=frameInterval)
    {
        sprintf(buf,"%s/%08d",outputFolder,frames);
        mkdir(buf,0755);

        for(int camIdx=0;camIdx<VGA_CAM_NUM_IN_A_PANEL;camIdx++)
        {
            __int64 memPos;
            memPos = __int64(frames)*VGA_BLOCKSIZE + __int64(camIdx)*VGA_PIXNUM;
            _fseeki64(fp, memPos, SEEK_SET);
            fread(img->imageData, VGA_PIXNUM, 1,fp) ;

            cvCvtColor(img,rgbimg,CV_BayerGR2BGR);
            if(bJpgOut==false)
                sprintf(buf,"%s/%08d/%08d_%02d_%02d.png",outputFolder,frames,frames,panelIdx,camIdx+1);          //camIdx
            else
                sprintf(buf,"%s/%08d/%02d_%02d.jpg",outputFolder,frames,panelIdx,camIdx+1);          //camIdx
            //printf("/%08d/%08d_%02d_%02d\n",frames,frames,panelIdx,camIdx+1);

            if(bUndist)
            {
                if(panelInfo.m_K[camIdx].rows==0 || panelInfo.m_distCoeef[camIdx].rows ==0)
                {
                    //printf("## Error: imageExtraction_vga: calib parameter is not valid. panel %d, cam %d\n",panelIdx,camIdx);
                    printf("## WARNING: imageExtraction_vga: calib parameter is not valid. panel %d, cam %d\n",panelIdx,camIdx+1);
                    imwrite(buf,blackImg);      //just save a black image

                }
                Mat originaImg(rgbimg);
                Mat idealImg;
                UndistortImage(originaImg,idealImg,panelInfo.m_K[camIdx],panelInfo.m_invK[camIdx],panelInfo.m_distCoeef[camIdx]);
                imwrite(buf,idealImg);
            }
            else
                cvSaveImage(buf,rgbimg);
        }\
        //std::cout << "frame" << frames+1 << " have saved" << std::endl;
    }

    cvReleaseImage(&img);
    cvReleaseImage(&rgbimg);
 }

//Made to handle frame drops for vga
//refFrameIdx is for vga frameIdx
//localFrameidx is only for current vga panel
void imageExtraction_vga_withLocalFrameIdx(VGAPanelInfo& panelInfo,const char* outputFolder, int panelIdx,
                          int refFrameIdx,int localFrameIdx
                         ,bool bUndist,bool bJpgOut)

{
    FILE* fp = panelInfo.m_fp;
     if(fp ==NULL)
     {
         printf("File Pointer is NULL\n");
         return;
     }

    char buf[256];
    IplImage *img = cvCreateImage(cvSize(VGA_IMGWIDTH,VGA_IMGHEIGHT),IPL_DEPTH_8U,1);
    IplImage *rgbimg = cvCreateImage(cvSize(VGA_IMGWIDTH,VGA_IMGHEIGHT),IPL_DEPTH_8U,3);

    Mat blackImg = Mat(480,640,CV_8UC3);

    sprintf(buf,"%s/%08d",outputFolder,refFrameIdx);
    mkdir(buf,0755);

    for(int camIdx=0;camIdx<VGA_CAM_NUM_IN_A_PANEL;camIdx++)
    {
        if(localFrameIdx<0)
        {
            if(bJpgOut==false)
                sprintf(buf,"%s/%08d/%08d_%02d_%02d.png",outputFolder,refFrameIdx,refFrameIdx,panelIdx,camIdx+1);          //camIdx
            else
                sprintf(buf,"%s/%08d/%02d_%02d.jpg",outputFolder,refFrameIdx,panelIdx,camIdx+1);          //camIdx

             imwrite(buf,blackImg);
            continue;
        }
        __int64 memPos;
        memPos = __int64(localFrameIdx)*VGA_BLOCKSIZE + __int64(camIdx)*VGA_PIXNUM;
        _fseeki64(fp, memPos, SEEK_SET);
        fread(img->imageData, VGA_PIXNUM, 1,fp) ;

        cvCvtColor(img,rgbimg,CV_BayerGR2BGR);
        if(bJpgOut==false)
            sprintf(buf,"%s/%08d/%08d_%02d_%02d.png",outputFolder,refFrameIdx,refFrameIdx,panelIdx,camIdx+1);          //camIdx
        else
            sprintf(buf,"%s/%08d/%02d_%02d.jpg",outputFolder,refFrameIdx,panelIdx,camIdx+1);          //camIdx
        //printf("/%08d/%08d_%02d_%02d\n",refFrameIdx,refFrameIdx,panelIdx,camIdx+1);

        if(bUndist)
        {
            if(panelInfo.m_K[camIdx].rows==0 || panelInfo.m_distCoeef[camIdx].rows ==0)
            {
                //printf("## Error: imageExtraction_vga: calib parameter is not valid. panel %d, cam %d\n",panelIdx,camIdx);
                printf("## WARNING: imageExtraction_vga: calib parameter is not valid. panel %d, cam %d\n",panelIdx,camIdx+1);
                imwrite(buf,blackImg);      //just save a black image

            }
            Mat originaImg(rgbimg);
            Mat idealImg;
            UndistortImage(originaImg,idealImg,panelInfo.m_K[camIdx],panelInfo.m_invK[camIdx],panelInfo.m_distCoeef[camIdx]);
            imwrite(buf,idealImg);
        }
        else
            cvSaveImage(buf,rgbimg);
    }\
    //std::cout << "refFrameIdx" << refFrameIdx+1 << " have saved" << std::endl;


    cvReleaseImage(&img);
    cvReleaseImage(&rgbimg);
 }


//Assume all folder was made in advance
bool imageExtraction_hd(CameraInfo& camInfo,int frameIdxInRawFile, cv::Mat& outputImg)       //distortion related);
{
    FILE* fp = camInfo.m_fp;
     if(fp ==NULL)
     {
         printf("File Pointer is NULL\n");
         return false;
     }


     char buf[256];
     IplImage *bufImage = cvCreateImage(cvSize(HD_IMGWIDTH*2,HD_IMGHEIGHT),IPL_DEPTH_8U,1);
 //    IplImage *rgbimg = cvCreateImage(cvSize(HD_IMGWIDTH,HD_IMGHEIGHT),IPL_DEPTH_8U,3);

     outputImg = Mat(HD_IMGHEIGHT,HD_IMGWIDTH,CV_8UC3);
     IplImage bufImg = outputImg;
     IplImage* rgbimg = &bufImg;	



     __int64 memPos;
     memPos = __int64(frameIdxInRawFile)*(HD_BLOCKSIZE_WITHHEAD);
     _fseeki64(fp, memPos, SEEK_SET);
     int re=fread(bufImage ->imageData, HD_BLOCKSIZE, 1,fp) ;

     #pragma omp parallel for
     for (int j=0; j<HD_BLOCKSIZE; j+=4)
     {
         int i=j*6/4;
         unsigned char cb = bufImage->imageData[j];
         unsigned char y1 = bufImage->imageData[j+1];
         unsigned char cr = bufImage->imageData[j+2];
         unsigned char y2 = bufImage->imageData[j+3];
         unsigned char r, g, b;
             ycrcb2rgb_HD(y1, cb, cr, &r, &g, &b);
         rgbimg->imageData[i] = r;
             rgbimg->imageData[i+1] = g;
             rgbimg->imageData[i+2] = b;

         ycrcb2rgb_HD(y2, cb, cr, &r, &g, &b);
             rgbimg->imageData[i+3] = r;
             rgbimg->imageData[i+4] = g;
             rgbimg->imageData[i+5] = b;
     }
     //sprintf(buf,"%s/%08d/%08d_%s.png",outputFolder,targetVGAFrameIdx,targetVGAFrameIdx,nameStr);
     /*Mat test(rgbimg);
     imshow("test",test);
     cvWaitKey();*/
     //cvSaveImage(outputFileName,rgbimg);


     /*
     if(bUndist)
     {
         if(camInfo.m_K.rows==0 || camInfo.m_distCoeef.rows ==0)
         {
             printf("## Error: imageExtraction_hd: calib parameter is not valid. outputFileName: %s\n",outputFileName);
             return;
         }
         Mat originalImg(rgbimg);
         Mat idealImg;
         UndistortImage(originalImg,idealImg,camInfo.m_K,camInfo.m_invK,camInfo.m_distCoeef);
         imwrite(outputFileName,idealImg);
     }
     else
         cvSaveImage(outputFileName,rgbimg);
    */
     cvReleaseImage(&bufImage);
//     cvReleaseImage(&rgbimg);
     return true;
 }



#if 0
void imageExtraction_vga_withDropChecker(vector< int  >& frameTimeCodeTable      //frameTimeCodeTable[frameIdx] == expectedTimeCode
                          ,VGAPanelInfo& panelInfo,const char* outputFolder, int panelIdx
                          ,int startFrameIdx,int endFrameIdx, int frameInterval
                          ,bool bUndist,bool bJpgOut,bool makeEvery100Subfolder)

{
    if(frameTimeCodeTable.size()==0)
    {
        printf("## ERROR:: imageExtraction_vga_withDropChecker: frameTimeCodeTable.size()==0");
        return;
    }
    else if(endFrameIdx >= frameTimeCodeTable.size())
    {
        printf("## ERROR:: imageExtraction_vga_withDropChecker: endFrameIdx(%d) >= frameTimeCodeTable.size() (%d)",endFrameIdx,(int)frameTimeCodeTable.size());
        return;
    }

    FILE* fp = panelInfo.m_fp;
     if(fp ==NULL)
     {
         printf("File Pointer is NULL\n");
         return;
     }

    char buf[256];
    IplImage *img = cvCreateImage(cvSize(VGA_IMGWIDTH,VGA_IMGHEIGHT),IPL_DEPTH_8U,1);
    IplImage *rgbimg = cvCreateImage(cvSize(VGA_IMGWIDTH,VGA_IMGHEIGHT),IPL_DEPTH_8U,3);

    char outputFolderFinal[512];
    Mat blackImg = Mat(480,640,CV_8UC3);
    int loadNum = endFrameIdx-startFrameIdx+1;
    int cnt=0;
    for(int frames =startFrameIdx;frames <=endFrameIdx;frames+=frameInterval)
    {
        if(makeEvery100Subfolder)
        {
            sprintf(outputFolderFinal,"%s/%03dXX",outputFolder,int(frames/100));
            mkdir(outputFolderFinal,0755);

            sprintf(outputFolderFinal,"%s/%08d",outputFolderFinal,frames);
            mkdir(outputFolderFinal,0755);
        }
        else
        {
            sprintf(outputFolderFinal,"%s/%08d",outputFolder,frames);
            mkdir(outputFolderFinal,0755);
        }

        int expectedTimeCode = frameTimeCodeTable[frames];
        int tempTimeCode = CSyncManager::ConvFrameToTC_vga(fp,frames);
        int localFrameIdx;      //this is the frame idx in the rawfile. Usually this is same as frames, if there is no frame drop issue
        if(expectedTimeCode==tempTimeCode)
        {
            localFrameIdx = frames;
            //printf("DEBUG:: GOOD : panelIdx %d (expectedTC %d vs tempTC %d)\n",panelIdx,expectedTimeCode,tempTimeCode);
        }
        else //if(expectedTimeCode!=tempTimeCode)
        {
            printf("WARNING:: VGA frameDrop detected (or happend before): panelIdx %d (expectedTC %d vs tempTC %d)\n",panelIdx,expectedTimeCode,tempTimeCode);
            printf("WARNING:: try to find corresponding one\n");

           int searchFrame = frames-1;
           while(true)
           {
               tempTimeCode = CSyncManager::ConvFrameToTC_vga(fp,searchFrame);

               if(tempTimeCode < expectedTimeCode)
               {
                   printf("Warning: VE%02d: frame drop detected for frame %d (expectedTC %d) \n",panelIdx,frames,expectedTimeCode);
                   localFrameIdx = -1;     //put -1 to indicate a frame drop
                   break;
               }
               else if(tempTimeCode==expectedTimeCode)
               {
                   printf("DEBUG: VE%02d: frame drop happend before, but still find corresponding for frame %d (local frameIdx %d) \n",panelIdx,frames,searchFrame);
                   localFrameIdx = searchFrame;
                   break;
               }
               searchFrame--;
           }
            //continue;           //Don't extract anything. This is a best way for me to figure out this problem happen.
        }

        for(int camIdx=0;camIdx<VGA_CAM_NUM_IN_A_PANEL;camIdx++)
        {
            if(bJpgOut==false)
                sprintf(buf,"%s/%08d_%02d_%02d.png",outputFolderFinal,frames,panelIdx,camIdx+1);          //camIdx
            else
                sprintf(buf,"%s/%02d_%02d.jpg",outputFolderFinal,panelIdx,camIdx+1);          //camIdx

            if(localFrameIdx<0)
            {
                imwrite(buf,blackImg);
                continue;
            }
            __int64 memPos;
            //memPos = __int64(frames)*VGA_BLOCKSIZE + __int64(camIdx)*VGA_PIXNUM;
            memPos = __int64(localFrameIdx)*VGA_BLOCKSIZE + __int64(camIdx)*VGA_PIXNUM;     //note that I use localFrameIdx here.
            _fseeki64(fp, memPos, SEEK_SET);
            int re= fread(img->imageData, VGA_PIXNUM, 1,fp) ;

            cvCvtColor(img,rgbimg,CV_BayerGR2BGR);
           /* if(bJpgOut==false)
                sprintf(buf,"%s/%08d_%02d_%02d.png",outputFolderFinal,frames,panelIdx,camIdx+1);          //camIdx
            else
                sprintf(buf,"%s/%02d_%02d.jpg",outputFolderFinal,panelIdx,camIdx+1);          //camIdx*/
            //printf("/%08d/%08d_%02d_%02d\n",frames,frames,panelIdx,camIdx+1);

            if(bUndist)
            {
                if(panelInfo.m_K[camIdx].rows==0 || panelInfo.m_distCoeef[camIdx].rows ==0)
                {
                    //printf("## Error: imageExtraction_vga: calib parameter is not valid. panel %d, cam %d\n",panelIdx,camIdx);
                    //printf("## WARNING: imageExtraction_vga: calib parameter is not valid. panel %d, cam %d\n",panelIdx,camIdx+1);
                    imwrite(buf,blackImg);      //just save a black image

                }
                Mat originaImg(rgbimg);
                Mat idealImg;
                UndistortImage(originaImg,idealImg,panelInfo.m_K[camIdx],panelInfo.m_invK[camIdx],panelInfo.m_distCoeef[camIdx]);
                imwrite(buf,idealImg);
            }
            else
                cvSaveImage(buf,rgbimg);
        }\
        //std::cout << "frame" << frames+1 << " have saved" << std::endl;

        printf("ve%02d: %d/%d=%0.4f%%\n",panelIdx,cnt,loadNum,float(cnt)/loadNum*100.0);
        cnt++;
    }

    cvReleaseImage(&img);
    cvReleaseImage(&rgbimg);
 }
void imageExtraction_kinect(CameraInfo& camInfo,const char* outputFileName,int frameIdxInRawFile
                            ,bool bUndist)       //distortion related);
{
    FILE* fp = camInfo.m_fp;
    if(fp ==NULL)
    {
        printf("File Pointer is NULL\n");
        return;
    }
    char buf[256];
    Mat bufImage(KINECT_IMGHEIGHT,KINECT_IMGWIDTH*2,CV_8UC1);
    Mat rgbImage(KINECT_IMGHEIGHT,KINECT_IMGWIDTH,CV_8UC3);

    off64_t ret = fseeko64(fp, __int64(frameIdxInRawFile)*KINECT_BLOCKSIZE, SEEK_SET);
    int re=fread(bufImage.data, KINECT_BLOCKSIZE, 1,fp) ;
    //cvtColor(bufImage,rgbImage,CV_YUV2GRAY_Y422);
    int RGB_PIXNUM = KINECT_IMGHEIGHT * KINECT_IMGWIDTH* 3;
    #pragma omp parallel for
    for(int i = 0, j=0; i < RGB_PIXNUM; i+=6, j+=4)
    {
        unsigned char y1 = bufImage.data[j];
        unsigned char cb = bufImage.data[j+1];
        unsigned char y2 = bufImage.data[j+2];
        unsigned char cr = bufImage.data[j+3];

        unsigned char r, g, b;
        ycrcb2rgb_yuv422_kinect(y1, cb, cr, &r, &g, &b);
                rgbImage.data[i] = r;
                rgbImage.data[i+1] = g;
                rgbImage.data[i+2] = b;

        ycrcb2rgb_yuv422_kinect(y2, cb, cr, &r, &g, &b);
                rgbImage.data[i+3] = r;
                rgbImage.data[i+4] = g;
                rgbImage.data[i+5] = b;

    }

    //Kinect image was horizontally flipped
    //sprintf(buf,"%s/%08d/%08d_%s.png",outputFolder,targetVGAFrameIdx,targetVGAFrameIdx,nameStr);
    /*Mat test(rgbimg);
    imshow("test",test);
    cvWaitKey();*/
    //cvSaveImage(outputFileName,rgbImage);
    flip(rgbImage,rgbImage,1);
    //imwrite(outputFileName,rgbImage);
    if(bUndist)
    {
        if(camInfo.m_K.rows==0 || camInfo.m_distCoeef.rows ==0)
        {
            printf("## Error: imageExtraction_kinect: calib parameter is not valid. outputFileName: %s\n",outputFileName);
            return;
        }
        Mat idealImg;
        UndistortImage(rgbImage,idealImg,camInfo.m_K,camInfo.m_invK,camInfo.m_distCoeef);
        imwrite(outputFileName,idealImg);
    }
    else
        imwrite(outputFileName,rgbImage);

    /*
    ////////////////////////////////////////////////
    ////    Extract Kinect Frames
    //IplImage *bufImage = cvCreateImage(cvSize(KINECT_IMGWIDTH,KINECT_IMGHEIGHT),IPL_DEPTH_8U,2);
    //IplImage *rgbImage = cvCreateImage(cvSize(KINECT_IMGWIDTH,KINECT_IMGHEIGHT),IPL_DEPTH_8U,3);
    Mat bufImage(KINECT_IMGHEIGHT,KINECT_IMGWIDTH*2,CV_8UC1);
    Mat rgbImage(KINECT_IMGHEIGHT,KINECT_IMGWIDTH,CV_8UC3);
    char kinectFileName[512];
    sprintf(kinectFileName,"%s/kinect/KINECTNODE1/colordata.dat",rootPath);
    FILE* fp_kinect= fopen(kinectFileName,"r");
    if(fp_kinect==NULL)
    {
        printf("I can't find the target file: %s \n",kinectFileName);
        //printf("Please check the /rcapture_hd mount status \n");
        return 0;
    }
    int iframe = 500;
    off64_t ret = fseeko64(fp_kinect, iframe*KINECT_BLOCKSIZE, SEEK_SET);
    int re=fread(bufImage.data, KINECT_BLOCKSIZE, 1,fp_kinect) ;
    //cvtColor(bufImage,rgbImage,CV_YUV2GRAY_Y422);
    int RGB_PIXNUM = KINECT_IMGHEIGHT * KINECT_IMGWIDTH* 3;
    for(int i = 0, j=0; i < RGB_PIXNUM; i+=6, j+=4)
    {
        unsigned char y1 = bufImage.data[j];
        unsigned char cb = bufImage.data[j+1];
        unsigned char y2 = bufImage.data[j+2];
        unsigned char cr = bufImage.data[j+3];

        unsigned char r, g, b;
        ycrcb2rgb_yuv422(y1, cb, cr, &r, &g, &b);
                rgbImage.data[i] = r;
                rgbImage.data[i+1] = g;
                rgbImage.data[i+2] = b;

        ycrcb2rgb_yuv422(y2, cb, cr, &r, &g, &b);
                rgbImage.data[i+3] = r;
                rgbImage.data[i+4] = g;
                rgbImage.data[i+5] = b;

    }
    //    imshow("test",rgbImage);
    imwrite("test2.png",rgbImage);
    cvWaitKey();
    // cvReleaseImage(&bufImage);
    //cvReleaseImage(&rgbImage);
    */
}

#endif
///////////////////////////////////////////////////////////////////////////////////////////////////
//// Image Undistortion Tools
///
//For RGB images
//output: Vec3d which is a double vector
Vec3b BilinearInterpolation(Mat& rgbImage,double x,double y)
{
    int floor_y = (int)floor(y);
    int ceil_y = (int)ceil(y);
    int floor_x = (int)floor(x);
    int ceil_x = (int)ceil(x);

    //interpolation
    Vec3d value_lb = rgbImage.at<Vec3b>(floor_y,floor_x);   //Vec3d: double vector
    Vec3d value_lu = rgbImage.at<Vec3b>(ceil_y,floor_x);
    Vec3d value_rb = rgbImage.at<Vec3b>(floor_y,ceil_x);
    Vec3d value_ru = rgbImage.at<Vec3b>(ceil_y,ceil_x);

    double alpha = y - floor_y;
    Vec3d value_l= (1-alpha) *value_lb + alpha * value_lu;
    Vec3d value_r= (1-alpha) *value_rb + alpha * value_ru;
    double beta = x - floor_x;
    Vec3d finalValue = (1-beta) *value_l + beta * value_r;

    return (Vec3b) finalValue;
}

//Using a single distortion paramters using VisSfM format (from original to Ideal)
Point2f DistortPointR1_VisSfMFormat(Point2f& pt, double k1)
{
        if (k1 == 0)
                return pt;

        if(pt.y==0)
            pt.y=float(1e-5);

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
                return Point2f(float(t18*pt.x/pt.y),float(t18));
        } else {
            printf("## Error: DistortPointR1_VisSfMFormat: this fumction only assume that distortion param is a positive value\n");
                //This should not be
            /*  const double t9 = t3/(t7*t7*4.0);
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
                return Point2f(float(t19.real()*pt.x/pt.y),float(t19.real()));*/
        }
}

//Used function
void UndistortImage_VisSfMFormat(Mat& input,Mat& output,
                                 Mat& K,Mat& kInv,Mat& distCoeef)
{
    /*Mat& K = g_intrinsicVector[calibIdx];
    Mat& kInv = g_intrinsicInvVector[calibIdx];
    Mat& distCoeef = g_distCoefVector[calibIdx];*/
    output = Mat::zeros(input.size(),CV_8UC3);

   if(K.rows==0)       //No calibration data for the camera
        return;

    for(int r=0;r<input.rows;++r)
    {
        #pragma omp parallel for num_threads(5)
        for(int c=0;c<input.cols;++c)
        {
            Mat_<double> homoPt(3,1);
            homoPt(0,0) = c;
            homoPt(1,0) = r;
            homoPt(2,0) = 1;
            homoPt = kInv*homoPt;
            Point2f idealPt(float(homoPt(0,0)/homoPt(2,0)),float(homoPt(1,0)/homoPt(2,0)));
            Point2f  distorted = DistortPointR1_VisSfMFormat(idealPt,distCoeef.at<double>(0,0));

            homoPt(0,0) = distorted.x;
            homoPt(1,0) = distorted.y;
            homoPt(2,0) = 1;
            homoPt = K*homoPt;
            double newX = homoPt(0,0)/homoPt(2,0);
            double newY = homoPt(1,0)/homoPt(2,0);

            //printf("%f %f -> %d %d\n",newX,newY,c,r);
            if(newX>=0 &&newX<input.cols &&  newY>=0 &&newY<input.rows )
            {
                //output.at<Vec3b>(r,c) = input.at<Vec3b>(newY,newX);
                output.at<Vec3b>(r,c) = BilinearInterpolation(input,newX,newY);
            }
            //else  //do nothing
        }
    }
}


Point_<double> ApplyDistort_Gaku(const Point_<double>& pt_d,Mat& K,Mat& invK,Mat& distCoeef)    //from ideal_image coord to image coord
{
    Mat pt_d_Mat = Mat::ones(3,1,CV_64F);
    pt_d_Mat.at<double>(0,0) = pt_d.x;
    pt_d_Mat.at<double>(1,0) = pt_d.y;
    pt_d_Mat = invK * pt_d_Mat;
    pt_d_Mat /= pt_d_Mat.at<double>(2,0);

    double xn = pt_d_Mat.at<double>(0,0);
    double yn = pt_d_Mat.at<double>(1,0);
    double r2 = xn*xn + yn*yn;
    double r4 = r2*r2;
    double r6 = r2*r4;
    double X2 = xn*xn;
    double Y2 = yn*yn;
    double XY = xn*yn;

    double a0 = distCoeef.at<double>(0,0), a1 = distCoeef.at<double>(1,0), a2 = distCoeef.at<double>(2,0);
    double p0 = distCoeef.at<double>(3,0),  p1 = distCoeef.at<double>(4,0);

    double radial       = 1.0 + a0*r2 + a1*r4 + a2*r6;
    double tangential_x = p0*(r2 + 2.0*X2) + 2.0*p1*XY;
    double tangential_y = p1*(r2 + 2.0*Y2) + 2.0*p0*XY;

    Point2d pt_u;
    pt_d_Mat.at<double>(0,0)= radial*xn + tangential_x;
    pt_d_Mat.at<double>(1,0)= radial*yn + tangential_y;
    pt_d_Mat = K*pt_d_Mat;
    pt_d_Mat /= pt_d_Mat.at<double>(2,0);

    pt_u.x = pt_d_Mat.at<double>(0,0);
    pt_u.y = pt_d_Mat.at<double>(1,0);

    return pt_u;
}

//Used function
void UndistortImage_GakuFormat(Mat& input,Mat& output,
                               Mat& K,Mat& kInv,Mat& distCoeef)
{
    //Mat& K = g_intrinsicVector[calibIdx];
    //Mat& kInv = g_intrinsicInvVector[calibIdx];
    //Mat& distCoeef = g_distCoefVector[calibIdx];
    output = Mat::zeros(input.size(),CV_8UC3);

        if(K.rows==0)       //No calibration data for the camera
            return;

    for(int r=0;r<input.rows;++r)
    {
        #pragma omp parallel for num_threads(5)
        for(int c=0;c<input.cols;++c)
        {
                Point2d ptInOriginal = ApplyDistort_Gaku(Point2d(c,r),K,kInv,distCoeef);

            //printf("%f %f -> %d %d\n",newX,newY,c,r);
            if(ptInOriginal.x>=0 && ptInOriginal.x < input.cols &&  ptInOriginal.y >=0 && ptInOriginal.y < input.rows )
            {
                //output.at<Vec3b>(r,c) = input.at<Vec3b>(newY,newX);
                output.at<Vec3b>(r,c) = BilinearInterpolation(input,ptInOriginal.x,ptInOriginal.y);
            }
            //else
        }
    }
}

//Called from outside
void UndistortImage(Mat& input,Mat& output,
                    Mat& K,Mat& kInv,Mat& distCoeef)
{
    if(distCoeef.rows==0)
    {
        output = Mat::zeros(input.size(),CV_8UC3);
        return;
    }
    if(distCoeef.at<double>(0,0) >0)
    {
        UndistortImage_VisSfMFormat(input,output,K,kInv,distCoeef);
    }
    else
    {
        UndistortImage_GakuFormat(input,output,K,kInv,distCoeef);
    }
}



//Used function
using namespace std;
void CameraInfo::LoadCalibFile(char* calibDirPath,int panelIdx, int camIdx)
{
    char buf[256];

    //read K matrix
    char calibParamPath[256];
    sprintf(calibParamPath,"%s/%02d_%02d.txt",calibDirPath,panelIdx,camIdx); //load same K for all cam
    printf("%s\n",calibParamPath);
    ifstream fin(calibParamPath);
    if(fin)
    {
        m_K = Mat::zeros(3,3,CV_64F);
        double* pt = m_K.ptr<double>(0);
        double value;
        for(int t=0;t<9;++t)
        {
            fin >> value;
            pt[t] = value;
    //		printf("%f\n",value);
        }

        m_invK = m_K.inv();
        m_distCoeef = Mat::zeros(5,1,CV_64F);
        pt = m_distCoeef.ptr<double>(0);
        for(int t=0;t<5;++t)  //may have 1 parameters (from Visual SfM) or 5 parameters (Gaku's, similar to opencv but order is a bit different)
        {
            if(fin.eof())
                break;

    //		printf("%f\n",value);
            fin >> value;
            pt[t] = value;
        }
        fin.close();
    }
    else
    {
        printf("Failure: %s\n",calibParamPath);
    }
    //printMatrix(g_intrinsicVector.back(),"K");
    //printMatrix(g_intrinsicInvVector.back(),"invK");
    //printMatrix(g_distCoefVector.back(),"DistCoeff");
}


//load for camIdx 1-24
void VGAPanelInfo::LoadCalibFile(char* calibDirPath,int panelIdx)
{
    //char buf[256];

    m_K.resize(24);
    m_invK.resize(24);
    m_distCoeef.resize(24);

    for(int camIdx =0;camIdx<24;++camIdx)
    {
        //read K matrix
        char calibParamPath[256];
        sprintf(calibParamPath,"%s/%02d_%02d.txt",calibDirPath,panelIdx,camIdx+1); //load same K for all cam
        //printf("%s\n",calibParamPath);
        ifstream fin(calibParamPath);
        if(fin)
        {
            m_K[camIdx] = Mat::zeros(3,3,CV_64F);
            double* pt = m_K[camIdx].ptr<double>(0);
            double value;
            for(int t=0;t<9;++t)
            {
                fin >> value;
                pt[t] = value;
        //		printf("%f\n",value);
            }

            m_invK[camIdx] = m_K[camIdx].inv();
            m_distCoeef[camIdx] = Mat::zeros(5,1,CV_64F);
            pt = m_distCoeef[camIdx].ptr<double>(0);
            for(int t=0;t<5;++t)  //may have 1 parameters (from Visual SfM) or 5 parameters (Gaku's, similar to opencv but order is a bit different)
            {
                if(fin.eof())
                    break;

        //		printf("%f\n",value);
                fin >> value;
                pt[t] = value;
            }
            fin.close();
        }
        else
        {
            printf("Failure: %s\n",calibParamPath);
            continue;
        }
        //printMatrix(g_intrinsicVector.back(),"K");
        //printMatrix(g_intrinsicInvVector.back(),"invK");
        //printMatrix(g_distCoefVector.back(),"DistCoeff");
    }
}

int main()
{
}


