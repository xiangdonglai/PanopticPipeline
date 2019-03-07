#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <vector>



using namespace std;


//Used only for VGA
class VGAPanelInfo
{
public:
    VGAPanelInfo()
    {
        m_fp = NULL;
    }

    FILE *m_fp;

    //For undistortion
    vector<cv::Mat> m_K;
    vector<cv::Mat> m_invK;
    vector<cv::Mat> m_distCoeef;
    void LoadCalibFile(char* calibDirPath,int panelIdx);
};

class CameraInfo
{
public:
    CameraInfo()
    {
        m_fp = NULL;
    }

    FILE *m_fp;

    //For undistortion
    cv::Mat m_K;
    cv::Mat m_invK;
    cv::Mat m_distCoeef;
    void LoadCalibFile(char* calibDirPath,int panelIdx, int camIdx);
};

void UndistortImage(cv::Mat& input,cv::Mat& output,
                    cv::Mat& K,cv::Mat& kInv,cv::Mat& distCoeef);


int ConvFrameToTC_vga(FILE* fp_vga,int vgaFrameIdx);

bool imageExtraction_vga(VGAPanelInfo& panelInfo
                        ,int refFrameIdx
                        ,bool bUndist,cv::Mat& outImg);

//use firstTC

bool imageExtraction_vga_withDropChecker(VGAPanelInfo& panelInfo
                        ,int refFrameIdx,int firstTC
                        ,bool bUndist,vector<int>& camIdxVect,vector<cv::Mat>& outImgVect);

void imageExtraction_vga(VGAPanelInfo& panelInfo,const char* outputFolder, int panelIdx,
                         int startFrameIdx,int endFrameIdx, int frameInterval =1
                        ,bool bUndist=false
                        ,bool bJpgOut=false);

void imageExtraction_vga_withLocalFrameIdx(VGAPanelInfo& panelInfo,const char* outputFolder, int panelIdx,
                          int refFrameIdx,int localFrameIdx
                         ,bool bUndist=false,bool bJpgOut=false);

void imageExtraction_vga_withDropChecker(vector< int  >& frameTimeCodeTable      //frameTimeCodeTable[frameIdx] == expectedTimeCode
                         ,VGAPanelInfo& panelInfo,const char* outputFolder, int panelIdx
                         ,int startFrameIdx,int endFrameIdx, int frameInterval =1
                         ,bool bUndist=false
                         ,bool bJpgOut=false
                         ,bool makeEvery100Subfolder=false);


bool imageExtraction_hd(CameraInfo& camInfo,int frameIdxInRawFile
                         ,cv::Mat& outImg);       //distortion related);

void imageExtraction_kinect(CameraInfo& camInfo,const char* outputFileName,int frameIdxInRawFile
                            ,bool bUndist=false);   //distortion related);


