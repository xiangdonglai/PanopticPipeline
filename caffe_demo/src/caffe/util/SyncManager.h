// Hanbyul Joo
// 12/30/2015
#pragma once


#ifndef SYNCMANAGER_H
#define SYNCMANAGER_H

//#include "TimecodeDecoder.h"
#include <fstream>
#include "pfcmu_config.h"
#include "math.h"
#ifndef _fseeki64
#define _fseeki64 fseeko64
#endif
#ifndef __int64
#define __int64 int64_t
#endif

#include <vector>
#include <map>
#include <utility>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <stdlib.h>

using namespace std;

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
const __int64 KINECT_BLOCKSIZE = KINECT_IMGWIDTH*KINECT_IMGHEIGHT*2;;

int ConvTCstrToTCmsec_hd(const char* timeCodeStr);
void ConvTCmsecToTCstr_hd(int tcmsec, char* timeCodeStr);
bool FileExistCheck(const char* fileName);


#define SENSOR_VGA 0
#define SENSOR_HD 1
#define SENSOR_KINECT 2

#define SENSOR_KINECT_COLOR 3
#define SENSOR_KINECT_DEPTH 4

struct SSensorName
{
    int sensorType;
    int nodeIdx;
    int diskIdx;        //used only for HD


    static int GetCamIdxHD(int nodeI,int distI)
    {
        int camIdx = 2*(nodeI-31) + (distI-1);
        return camIdx;
    }

    int GetCamIdx()
    {
        if(sensorType==SENSOR_VGA)
        {
            //not valid. I only have one instance for VGA (for all panels)
            return -1;
        }
        else if(sensorType==SENSOR_HD)
        {
            int camIdx = 2*(nodeIdx-31) + (diskIdx-1);
            return camIdx;
        }
        else if(sensorType==SENSOR_KINECT)
        {
            return nodeIdx;
        }
    }

    int GetPanelIdx()
    {
        if(sensorType==SENSOR_VGA)
        {
            //not valid. I only have one instance for VGA (for all panels)
            return -1;
        }
        else if(sensorType==SENSOR_HD)
        {
            return 0;
        }
        else if(sensorType==SENSOR_KINECT)
        {
            return 50;
        }
    }

    void GetNameStr(char* str)
    {
        if(sensorType==SENSOR_VGA)
        {
            //sprintf(str,"%02d_%02d",nodeIdx,00);        //not valid. I only have one instance for VGA (for all panels)
        }
        else if(sensorType==SENSOR_HD)
        {
            int camIdx = 2*(nodeIdx-31) + (diskIdx-1);
            sprintf(str,"00_%02d",camIdx);
        }
        else if(sensorType==SENSOR_KINECT)
        {
            sprintf(str,"50_%02d",nodeIdx);
        }
    }
};

struct SUnivSyncUnitForEachSensor
{
    SUnivSyncUnitForEachSensor()
    {
        sensorName.sensorType= sensorName.nodeIdx = sensorName.diskIdx = -1;
    }

    SSensorName sensorName;

    vector< pair<int,double> > frameUnivTimeTable;      //<frameInRawFile, univTime>..
    vector< int> frameTimeCode;      // used to check frame drop (especially in VGA case without using UnivTime) frameTimeCode[frameInRawFile] == TimeCode (in vgaRaw) or LTCTimeCodeMsec (in HD and others)
    std::map<double,int> univTimeFrameMap;            //<univTime, frameInRawFile>
};

struct SKinectSyncUnit
{
    int m_kinectIdx;      //starting from 1
    int m_sensorType;       //SENSOR_KINECT_DEPTH or SENSOR_KINECT_COLOR
    //vector< pair<int,double> > m_HDTCtoIntClock;            //<HDTCmsec, InternalClock>
    std::map<double,int> m_intClockToHDTCmsec;            //<HDTCmsec, InternalClock>
    vector< pair<int, double> > m_frameToIntClock;  //<frameIdx, internalClock>. Only contains valid frames (no dropped frame)
};

class CKinectSyncTableManager
{
public:
    bool loadTable(char* kinectRootPath)
    {
        char kinectSyncTableName[512];
        sprintf(kinectSyncTableName,"%s/finalSyncTable.csv",kinectRootPath);
        vector < vector< string> > allElements;  //outer: each line (a time instance), inner: LTCTimeCode, k1-audio, k1-video, k2-audio...

        ifstream fin(kinectSyncTableName);
        if(fin.is_open()==false)
            return false;
        while(fin)
        {
            string lineStr;
            if (!getline( fin, lineStr )) break;

            allElements.resize(allElements.size() +1);
            vector< string>& lineElements = allElements.back();
            istringstream ss( lineStr );
            string data;

            while (getline( ss, data, ',' ))
            {
                //record.push_back( s );
                //printf("!! %s\n", data.c_str());
                lineElements.push_back( data);
            }
        }

        if(allElements.size()==0)
            return false;
        if(allElements.front().size()<=1)
            return false;

        //Parsing
        int numOfKinects =(allElements.front().size() - 1) /2;
        printf("## Number of Kinects: %d\n",numOfKinects);
        m_kinectUnits.resize(numOfKinects*2);       //color and depth
        //Get Kinect Index (assuming there might be missing kinects)
        for(int i=0;i<numOfKinects;++i)
        {
            int colIdx =  1 + i*2;
            string name = allElements[0][colIdx]; //KINECTNODEX-Audio
            int kinectIdx = name.at(10) - '0';
            if(name.at(11) != '-')      //e.g., KINECTNODE10-Audio
            {
                kinectIdx = 10*kinectIdx  + name.at(11) - '0';
            }
            printf("KinectIdx  %d\n",kinectIdx);
            m_kinectUnits[2*i].m_kinectIdx = kinectIdx;     //color
            m_kinectUnits[2*i].m_sensorType = SENSOR_KINECT_COLOR;     //color
            m_kinectUnits[2*i+1].m_kinectIdx = kinectIdx;   //depth
            m_kinectUnits[2*i+1].m_sensorType = SENSOR_KINECT_DEPTH;     //depth
        }


        int numOfEntry = allElements.size() - 1;        //first line is the header
        //for(int i=0;i<m_kinectUnits.size();++i)
          //  m_kinectUnits[i].m_HDTCtoIntClock.reserve(numOfEntry);

        for(int r=1;r<allElements.size();++r)
        {
            int hdtcmsec =  ConvTCstrToTCmsec_hd(allElements[r][0].c_str());

            for(int i=0;i<m_kinectUnits.size();++i)
            {
                int kinectIdx = m_kinectUnits[i].m_kinectIdx;
                //int colIdx =  1 + i*2;
                int colIdx = 1 + (kinectIdx-1)*2;
               // m_kinectUnits[i].m_HDTCtoIntClock[r-1].first= hdtcmsec;
               // m_kinectUnits[i].m_HDTCtoIntClock[r-1].second= atof(allElements[r][colIdx].c_str());
                double intClock = atof(allElements[r][colIdx].c_str());
                if(intClock<0)
                {
                    printf("%s %f\n",allElements[r][colIdx].c_str(),intClock);
                }
                m_kinectUnits[i].m_intClockToHDTCmsec.insert( std::pair<double,int>(intClock,  hdtcmsec) );
            }
           //printf("%s %s\n",allElements[r][0].c_str(),allElements[i][1].c_str());
        }

        for(int k=0;k<m_kinectUnits.size();++k)
        {
            printf("k: %d\n",k);
            char fileName[512];

            if(m_kinectUnits[k].m_sensorType == SENSOR_KINECT_COLOR)
                sprintf(fileName,"%s/KINECTNODE%d/color.txt",kinectRootPath,m_kinectUnits[k].m_kinectIdx);
            else
                sprintf(fileName,"%s/KINECTNODE%d/depth.txt",kinectRootPath,m_kinectUnits[k].m_kinectIdx);
            LoadKinectTimeCode(m_kinectUnits[k],fileName);
        }
        return true;
    }

    bool LoadKinectTimeCode(SKinectSyncUnit& syncUnit,char* fileName)
    {
        //Load color.txt file
        ifstream fin(fileName);
        if(fin.is_open()==false)
            return false;
        printf("LoadTimeCode: %s\n",fileName);
        int frameIdx=0;
        while(fin)
        {
            string lineStr;
            if (!getline( fin, lineStr )) break;
            if(lineStr.find("Drop") == std::string::npos)       //"Drop" Not found
                syncUnit.m_frameToIntClock.push_back( pair<int,double>(frameIdx, atof(lineStr.c_str())) );           //means drop
            //else  //
            //    syncUnit.m_frameToIntClock.push_back(-1);           //means drop

            frameIdx++;
        }
    }

    vector<SKinectSyncUnit> m_kinectUnits;
};

//Input: VGA-HD sync file (.wav) and
class CSyncManager
{
public:
    CSyncManager()
    {
        m_bInitVgaHDMap = false;
        m_bInitKinectSyncTable = false;
        m_hdStartTimeMsec_for30fps = -1;
        //m_im_vga = new char[VGA_BLOCKSIZE];
    }

    ~CSyncManager()
    {
        //if(m_im_vga!=NULL)
          //  delete[] m_im_vga;
    }

    //To save wave to txt file
    //map<int, UnivTimeUnit>& GetSyncWavMap_VGA() {return m_vgahd_map.GetMapVGA();}
    //map<int, UnivTimeUnit>& GetSyncWavMap_HD() {return m_vgahd_map.GetMapHD();}

    bool InitSyncFile_wavFile(char* syncFileName);
    bool InitSyncFile_kinectTable(char* kinectRootFolder);

    //VGA
    static int ConvFrameToTC_vga(FILE* fp_vga,int vgaFrameIdx);     //FrameIdx starts from 0
    int ConvFrameToTC_vga(int vgaFirstTC,int vgaFrameIdx);   //FrameIdx starts from 0
    double ConvFrameToUnivTime_vga(FILE* fp_vga,int vgaFrameIdx);   //FrameIdx starts from 0
    double ConvFrameToUnivTime_vga(int vgaFirstTC,int vgaFrameIdx); //FrameIdx starts from 0
    double ConvTCToUnivTime_vga(int vgatc_ms);

    //HD
    //HDTCstr is a string as 14:23:46:00 (hour:min:sec:frm).  frm is from 00 to 29 (30hz)
    //HDTCmsec is a millisec as 51826000 (conversion from HDTCstr)
    int ConvFrameToTCmsec_hd(FILE* fp_hd,off64_t hdFrameIdx);    //FrameIdx starts from 0
    double ConvFrameToUnivTime_hd(FILE* fp_hd,int hdFrameIdx);  //FrameIdx starts from 0
    double ConvTCmsecToUnivTime_hd(int hdtc_ms);
    void ConvFrameToTCStr_hd(FILE* fp,off64_t iframe,char *timeCodeStr);
    int GetTCmsecinWavbyIndex_hd(int index);
    double GetUnivTimeinWavbyIndex_hd(int index);     //return -1, if index is not valid
    int GetTCinWavbyIndex_vga(int index);        //return -1, if index is not valid

    //Kinect
    int GenerKinectSyncTable(const char* outputFolder);

    //VGA - HD mapping
    bool FindHDTC_fromVGAFrame(int vgaFirstTC,int vgaFrame,char* hdtcstr,int& offsetMsec);


    ////////////////////////////////////////////////////////////
    //Image extraction related functions (NO wave file is required)////////////////////////
    enum LoadMode { VGAOnly_25 =0,VH_25,VK_25, VHK_25, HDOnly_25, KinectOnly_25,         //HDOnly_25, KinectOnly_25 requires VGA tables
                    HDOnly_30, KinectOnly_30, HK_30,        //30fps always follow HD-based frameIdx definition
                    TimeInfoOnly_25, TimeInfoOnly_30, INDEXMAP_25_30,              //TimeInfo and frame drop only (no image extraction)
                    VGASingleRaw_25,HDSingleRaw_25,HDSingleRaw_30,      //made to run in parallel at each local machine
                    UnKnown};
    static LoadMode GetLoadModeFromStr(const char* str);
    //Load all sync tables
    //These are used to get corresponding frames from raw files (since syncTable is based on frameIdx in raw files)
    void loadAllTables(const char* syncFolder, LoadMode loadMode); //Save them to m_UnivTable_vga, m_UnivTable_hd, m_UnivTable_kinect
    void loadSelectedTables(const char* syncFolder, LoadMode loadMode,int machineIdx=-1,int diskIdx=-1);     //machinhe and diskIdx is needed only for HD
    void GetSensorNames(vector<SSensorName>& outputSensorNames);        //get sensorName order from loaded syncTables
    void findAllCorresFrames_forVGAPanels(vector<FILE*>& vgaPanelFilePts, int vgaFrame,vector< int>& corresVGAFrames);        //only for the VGA. To consider frame drop
    void findAllCorresFrames_fromVGAFrameIdx(int vgaFrame,vector< pair<int,double> >& corresFrames);        //this assumes that the first element if VGA
    void findAllCorresFrames_fromUnivTime(double refUnivTime,vector< pair<int,double> >& corresFrames);     //made for HD_30. (work with VGA case)
    static int findClosetFrameIdx_fromVGA(SUnivSyncUnitForEachSensor& vgaSyncUniv,int vgaFrameIdx,SUnivSyncUnitForEachSensor& targetSyncUnit,double& univTimeDiff);
    static int findClosetFrameIdx_fromUnivTime(double refUnivTime,SUnivSyncUnitForEachSensor& targetSyncUnit,double& univTimeDiff);
    static double findClosetFrameIdx_fromUnivTime_byUnivTimeVect(double refUnivTime,vector<double>& sortedUnivTimeVect,double& univTimeDiff);
    vector< pair<int, double> > m_hdTCTimeTable_for30fps;           //used for 30FPS HD image extractor. m_hdTCTimeTable_for30fps[0] == m_hdStartTimeMsec_for30fps. hdTCTimeTable[HD_frameIdx_30fps] = (HDTCTimeCode (msec), univTime);
    vector<double> m_hdTCTimeTable_for30fps_onlyUnivTime;           //only saved univTime from m_hdTCTimeTable_for30fps. I made this only for HD framedrop checking purpose

    void FrameIdxMapExport_From25VGAfpsTo30HDfps(const char* outFileName);
    void FrameIdxMapExport_From25VGAfpsTo30HDfps_applyOffset(const char* outFileName);

    SUnivSyncUnitForEachSensor* GetVGASyncUnit() { return &m_UnivTable_vga; }
    SUnivSyncUnitForEachSensor* GetFirstHDSyncSyncUnit() {
        if(m_UnivTable_hd.size()>0)
            return &m_UnivTable_hd.front();
        else
            return NULL;
    };
private:
    //TimecodeDecoder m_vgahd_map;
    bool m_bInitVgaHDMap;
    bool m_bInitKinectSyncTable;
    CKinectSyncTableManager m_kinectSyncTable;

    SUnivSyncUnitForEachSensor m_UnivTable_vga;      //only need one instance, because all of them starts exactly same time
    vector<SUnivSyncUnitForEachSensor> m_UnivTable_hd;  //need an instance for each hd
    vector<SUnivSyncUnitForEachSensor> m_UnivTable_kinect;  //need an instance for each kinect
    int m_hdStartTimeMsec_for30fps;           //used for 30FPS HD image extractor


    //char* m_im_vga;
};


#endif // SYNCMANAGER_H
