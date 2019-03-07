#include "SyncManager.h"

bool FileExistCheck(const char* fileName)
{
    ifstream fin(fileName);
    if(fin.is_open())
    {
        fin.close();
        return true;
    }
    else
        return false;
}

int ConvTCstrToTCmsec_hd(const char* timeCodeStr)
{
    char buf[512];
    strcpy(buf,timeCodeStr);
    int curFrame = atoi(&buf[9]);
    buf[8] =0;
    int curSec = atoi(&buf[6]);
    buf[5] =0;
    int curMin = atoi(&buf[3]);
    buf[2] =0;
    int curHour = atoi(&buf[0]);
    int msecValue = 1000*
                      (curFrame/HDfps +
                        (curSec) +
                        60 * (curMin ) +
                        3600* (curHour ) );

    int checker = msecValue%100;
    if(checker ==99 || checker ==32  || checker==65)
        msecValue++;
    if(checker ==1 || checker ==34  || checker==67)
        msecValue--;

    return msecValue;
}

void ConvTCmsecToTCstr_hd(int tcmsec, char* timeCodeStr)
{
    int frame = int((tcmsec % 1000) / 33.33333333 + 0.5);
    int inSec = tcmsec / 1000;
    int ss = inSec % 60;
    int inMin = inSec / 60;
    int mm = inMin % 60;
    int inHour = inMin / 60;
    int hh = inHour % 24;

    sprintf(timeCodeStr,"%02d:%02d:%02d:%02d",hh,mm,ss,frame);
}

bool CSyncManager::InitSyncFile_wavFile(char* syncFileName)
{
    //Load syncFile
//    m_bInitVgaHDMap = m_vgahd_map.read(syncFileName);
 //   return m_bInitVgaHDMap;
    return false;
}

bool CSyncManager::InitSyncFile_kinectTable(char* kinectRootFolder)
{
    //Load kinectSync informationfrom the following files
    //{kinectRootFolder}/finalSyncTable.csv"
    //{kinectRootFolder}/KINECTNODEX/color.txt"
    m_bInitKinectSyncTable = m_kinectSyncTable.loadTable(kinectRootFolder);
    return m_bInitKinectSyncTable;
}

//VGA
int CSyncManager::ConvFrameToTC_vga(FILE* fp_vga,int vgaFrameIdx)     //FrameIdx starts from 0
{
    __int64 memPos = __int64(vgaFrameIdx) * VGA_BLOCKSIZE;
    _fseeki64(fp_vga, memPos, SEEK_SET);
    char im_vga[VGA_BLOCKSIZE];
    //m_im_vga = new char[VGA_BLOCKSIZE];
    int re= fread(im_vga, VGA_TIMECODE_SIZE,1,fp_vga);
    PFCMU::timestamp_t ts = PFCMU::get_timestamp(im_vga);
    return int(ts);
}

int CSyncManager::ConvFrameToTC_vga(int vgaFirstTC,int vgaFrameIdx)   //FrameIdx starts from 0
{
    return vgaFirstTC + vgaFrameIdx * 4;        //4 is an incremental step for VGA time code
}

double CSyncManager::ConvFrameToUnivTime_vga(FILE* fp_vga,int vgaFrameIdx)   //FrameIdx starts from 0
{
    return ConvTCToUnivTime_vga(ConvFrameToTC_vga(fp_vga,vgaFrameIdx));
}

double CSyncManager::ConvFrameToUnivTime_vga(int vgaFirstTC,int vgaFrameIdx) //FrameIdx starts from 0
{
    return ConvTCToUnivTime_vga(ConvFrameToTC_vga(vgaFirstTC,vgaFrameIdx));
}

double CSyncManager::ConvTCToUnivTime_vga(int vgatc_ms)
//int CSyncManager::ConvTCToUnivTime_vga(int vgatc_ms)
{
   // UnivTimeUnit univTime;
//    m_vgahd_map.query(vgatc_ms,1, &univTime);  //query of VGA
    //return int(univTime.start*1000 + 0.5);        //msec
    //return univTime.start*1000;        //msec
   // return univTime.end*1000;        //msec
   return -1;
}

//HD
//HDTCstr is a string as 14:23:46:00 (hour:min:sec:frm).  frm is from 00 to 29 (30hz)
//HDTCmsec is a millisec as 51826000 (conversion from HDTCstr)
int CSyncManager::ConvFrameToTCmsec_hd(FILE* fp_hd,off64_t hdFrameIdx)    //FrameIdx starts from 0
{
    char tcstr[512];
    ConvFrameToTCStr_hd(fp_hd,hdFrameIdx,tcstr);
    return ConvTCstrToTCmsec_hd(tcstr);
}

double CSyncManager::ConvFrameToUnivTime_hd(FILE* fp_hd,int hdFrameIdx)  //FrameIdx starts from 0
{
    return ConvTCmsecToUnivTime_hd(ConvFrameToTCmsec_hd(fp_hd,hdFrameIdx));
}

double CSyncManager::ConvTCmsecToUnivTime_hd(int hdtc_ms)
{
//    UnivTimeUnit univTime;
//    m_vgahd_map.query(hdtc_ms,0, &univTime);  //query of HD
    //return int(univTime.start*1000 + 0.5);
    //return univTime.start*1000;
//    return univTime.end*1000;        //msec
    return -1;
}

void CSyncManager::ConvFrameToTCStr_hd(FILE* fp,off64_t iframe,char *timeCodeStr)
{
    if(fp)
    {
        off64_t ret = fseeko64(fp, iframe*(HD_BLOCKSIZE_WITHHEAD)+HD_BLOCKSIZE, SEEK_SET);
        if(timeCodeStr)
        {
            fread(timeCodeStr, HD_STRBLOCKSIZE, 1, fp);
        }
    }
}

int CSyncManager::GetTCmsecinWavbyIndex_hd(int index)
{
    //printf("HD time code:(%d,%d)\n",index,m_vgahd_map.TCValueByIndex(CHANNEL_HD,index));
    /*
    //verification
    int tcmsec = m_vgahd_map.TCValueByIndex(CHANNEL_HD,index);
    char tcstr[512];
    ConvTCmsecToTCstr_hd(tcmsec,tcstr);
    int tcmsec_ver = ConvTCstrToTCmsec_hd(tcstr);
    //printf("HD time code (veri): str: %s ver %d)\n",tcstr,tcmsec_ver);
    if(tcmsec !=tcmsec_ver)
        printf("ERROR!!!\n");
    */

//    return m_vgahd_map.TCValueByIndex(CHANNEL_HD,index);
    return -1;
}


//int CSyncManager::GetUnivTimeinWavbyIndex_hd(int index)     //return -1, if index is not valid
double CSyncManager::GetUnivTimeinWavbyIndex_hd(int index)     //return -1, if index is not valid
{
//    return m_vgahd_map.UnivTimeByIndex(CHANNEL_HD,index);
    return -1;
}

int CSyncManager::GetTCinWavbyIndex_vga(int index)        //return -1, if index is not valid
{
    //printf("VAG time code:(%d,%d)\n",index,m_vgahd_map.TCValueByIndex(CHANNEL_VGA,index));
//    return m_vgahd_map.TCValueByIndex(CHANNEL_VGA,index);
    return -1;
}

//Kinect
int CSyncManager::GenerKinectSyncTable(const char* outputFolder)
{
    char fileName[512];
    char outputStr[512];
    char tcstr[512];
    for(int k=0;k<m_kinectSyncTable.m_kinectUnits.size();++k)
    {
        SKinectSyncUnit& kinUnit = m_kinectSyncTable.m_kinectUnits[k];
        int num = kinUnit.m_frameToIntClock.size();
        std::map<double,int>::iterator it,itlow,itlargeEqual;

        if(kinUnit.m_sensorType==SENSOR_KINECT_COLOR)
            sprintf(fileName,"%s/syncTable_kinect%d_color.txt",outputFolder,kinUnit.m_kinectIdx);
        else
            sprintf(fileName,"%s/syncTable_kinect%d_depth.txt",outputFolder,kinUnit.m_kinectIdx);

        if(FileExistCheck(fileName) == true)
        {
            printf("File already exists: %s\n",fileName);
            continue;
        }
        ofstream fout(fileName);
        fout <<"FrameIdx(ignoreDroppedFrame)\tUnivTime(ms)\tFrameIdx(includeDroppedFrame)\tClosestLTCTimeCode\tClosestLTCTimeCode(str)\n";

        for(int f=0;f<kinUnit.m_frameToIntClock.size();++f)
        //for(int f=0;f<100;++f)
        {
            int frameIdx = kinUnit.m_frameToIntClock[f].first;  //may not be continuous due to frame drops
            double intClock = kinUnit.m_frameToIntClock[f].second;
            if(intClock <0)     //means frameDrop
                continue;

            it = kinUnit.m_intClockToHDTCmsec.lower_bound(intClock);        //equal or greater
            if(it == kinUnit.m_intClockToHDTCmsec.begin() || it == kinUnit.m_intClockToHDTCmsec.end())
            {
                sprintf(outputStr,"%d\t-1\t-1\t-1\t0\n",f);
                fout <<outputStr;
                continue;
            }
            itlargeEqual = it;
            it--;
            itlow = it;     //strictly low

            //printf("intClock %f, low %f, up %f\n",intClock,itlow->first,itlargeEqual->first);
            double diff_pre = intClock - itlow->first ;     //positive number
            double diff_next = intClock - itlargeEqual->first;      //zero or negative number

            double offset;
            int closestTCmsec;
            if(diff_pre < -diff_next)
            {
                offset = diff_pre;        //positive number
                closestTCmsec = itlow->second;
            }
            else
            {
                offset = diff_next;        //negative number
                closestTCmsec = itlargeEqual->second;
            }

            ConvTCmsecToTCstr_hd(closestTCmsec,tcstr);
            double closestUnivTime = ConvTCmsecToUnivTime_hd(closestTCmsec);
            double univTime = closestUnivTime + offset;  //from closestHDtc to currentTime
            sprintf(outputStr,"%d\t%.3f\t%d\t%d\t%s\n",f,univTime,frameIdx,closestTCmsec,tcstr);
            fout <<outputStr;
        }
        fout.close();
    }
}


//Load all sync tables
//Save them to m_UnivTable_vga, m_UnivTable_hd, m_UnivTable_kinect
void CSyncManager::loadAllTables(const char* syncFolder, LoadMode loadMode)
{
    char tableFileName[512];
    char str[512];

    if(loadMode == VGAOnly_25 || loadMode == VH_25 || loadMode == VHK_25 || loadMode == VK_25
            || loadMode  == HDOnly_25 || loadMode == KinectOnly_25      //Need VGA table anyway
            || loadMode == TimeInfoOnly_25 || loadMode == TimeInfoOnly_30      //TimeInfoOnly_30 may not need VGA table. But just load just in case
            || loadMode == INDEXMAP_25_30)
    {
         //load VGA
         m_UnivTable_vga.sensorName.sensorType = SENSOR_VGA;
         m_UnivTable_vga.sensorName.nodeIdx = 1;  //ve01
         sprintf(tableFileName,"%s/syncTable_vga.txt",syncFolder);
         if(FileExistCheck(tableFileName) == false)
         {
             printf("Can't load vga syncfile: %s\n",tableFileName);
             return;
         }
         ifstream fin(tableFileName);
         string lineStr;
         getline( fin, lineStr );
         int frameIdx;
         double univTime;
         int timeCodeInt;
         while(fin)
         {
             if (!getline( fin, lineStr )) break;

             istringstream ss( lineStr );
             ss >> frameIdx;     //frameIdx
             ss >> univTime;     //univTime
             ss >> timeCodeInt;     //timecode in vga or LTCTimeCodeMsec in HD
             //printf("%d, %f\n",frameIdx,uinvTime);
            m_UnivTable_vga.frameUnivTimeTable.push_back( pair<int,double> (frameIdx,univTime));
            m_UnivTable_vga.univTimeFrameMap.insert( pair<double,int> (univTime,frameIdx));
            m_UnivTable_vga.frameTimeCode.push_back(timeCodeInt);
         }
         printf("Loaded vga table: %lu elments\n",m_UnivTable_vga.frameUnivTimeTable.size());
         fin.close();
    }

     //load HD
     if(loadMode == VH_25 || loadMode == VHK_25  || loadMode  == HDOnly_25
             || loadMode == HDOnly_30 || loadMode ==HK_30   || loadMode ==KinectOnly_30     // HD table is still needed in || loadMode ==HK_30 case,
             || loadMode == TimeInfoOnly_25 || loadMode == TimeInfoOnly_30 )
     for(int m=31;m<=46;++m)
     {
         for(int d=1;d<=2;++d)
         {
             sprintf(tableFileName,"%s/syncTable_hd%02d-%d.txt",syncFolder,m,d);
             if(FileExistCheck(tableFileName) == false)
             {
                 printf("Can't load hd syncfile: %s\n",tableFileName);
                 continue;
             }
             printf("Start load hd syncfile: %s\n",tableFileName);

             m_UnivTable_hd.resize( m_UnivTable_hd.size() +1);
             SUnivSyncUnitForEachSensor& univUnit = m_UnivTable_hd.back();
             univUnit.sensorName.sensorType = SENSOR_HD;
             univUnit.sensorName.nodeIdx = m;
             univUnit.sensorName.diskIdx = d;


             ifstream fin(tableFileName);
             string lineStr;
             getline( fin, lineStr );
             int frameIdx;
             double univTime;
             int timeCodeInt;
             while(fin)
             {
                 if (!getline( fin, lineStr )) break;

                 istringstream ss( lineStr );
                 ss >> frameIdx;     //frameIdx
                 ss >> univTime;     //univTime
                 ss >> timeCodeInt;     //timecode in vga or LTCTimeCodeMsec in HD

                 //printf("%d, %f\n",frameIdx,uinvTime);
                univUnit.frameUnivTimeTable.push_back( pair<int,double> (frameIdx,univTime));
                univUnit.univTimeFrameMap.insert( pair<double,int> (univTime,frameIdx));
                univUnit.frameTimeCode.push_back(timeCodeInt);

             }
             if(univUnit.frameUnivTimeTable.size()==0)
             {
                // printf("Ignore due to the zero size: %d elments\n",univUnit.frameUnivTimeTable.size());
                 m_UnivTable_hd.pop_back();
                 continue;
             }
             printf("Loaded HD table: %lu elments\n",univUnit.frameUnivTimeTable.size());
             fin.close();
         }
     }

     //Read HD Start time and indexTable
     //Just load always. Just in case
     if(true)//loadMode == HDOnly_30 || loadMode ==HK_30 || loadMode ==TimeInfoOnly_30 || loadMode ==KinectOnly_30)
     {
         sprintf(tableFileName,"%s/hdStartPoint.txt",syncFolder);
         if(FileExistCheck(tableFileName) == false)
         {
             printf(" Warning:: Can't load hdStartPoint file: %s\n",tableFileName);
         }
         else
         {
             printf("Start load hdStartPoint: %s\n",tableFileName);
             char str[512];
             double univTime;
             ifstream fin(tableFileName);
             fin >> str >> str >> str;      //headers. HDTimeCode(msec) HDTimeCode(str) UnivTime
             fin >> m_hdStartTimeMsec_for30fps >> str >> univTime;
             printf("HDStartTime: %d, %s, %.3f\n",m_hdStartTimeMsec_for30fps,str,univTime);
             fin.close();
         }
         if(m_hdStartTimeMsec_for30fps>0)
         {
             sprintf(tableFileName,"%s/syncWavToTxt_hd.txt",syncFolder);
             if(FileExistCheck(tableFileName) == false)
             {
                   printf(" Warning:: Can't load hdStartPoint file: %s\n",tableFileName);
             }
             else
             {
                 printf("Start load m_hdTCTimeTable_for30fps: %s\n",tableFileName);
                 char str[512];
                 ifstream fin(tableFileName);
                 fin >> str >> str>> str;   //headers       HDTimeStr       HDTime(ms)      UnivTime(ms)
                 m_hdTCTimeTable_for30fps.reserve(30*60*30);//30min
                 bool bStartToSave =false;
                 double univTime;
                 int timecode_msec;
                 string lineStr;

                 while(fin)
                 {
                     if (!getline( fin, lineStr )) break;

                     istringstream ss( lineStr );
                     ss >> str;     //HDTimeStr
                     ss >> timecode_msec;     //HDTime(ms)
                     ss >> univTime;    //UnivTime(ms)

                    if(bStartToSave==false && m_hdStartTimeMsec_for30fps == timecode_msec)
                    {
                        bStartToSave = true;
                        printf("HDTCTimeTableGen: Find start point!\n");
                    }
                    if(bStartToSave)
                    {
                        m_hdTCTimeTable_for30fps.push_back( make_pair(timecode_msec, univTime ));
                        //printf("idx: %d, %d\n",m_hdTCTimeTable_for30fps.size()-1,m_hdTCTimeTable_for30fps.back());
                    }
                 }
                 fin.close();

                 m_hdTCTimeTable_for30fps_onlyUnivTime.resize(m_hdTCTimeTable_for30fps.size());
                 for(int i=0;i<m_hdTCTimeTable_for30fps.size();++i)
                     m_hdTCTimeTable_for30fps_onlyUnivTime[i] = m_hdTCTimeTable_for30fps[i].second;
             }
         }
     }

     //load Kinects
     if(loadMode == VHK_25 || loadMode == VK_25 || loadMode == KinectOnly_25
             || loadMode == KinectOnly_30 || loadMode ==HK_30
             || loadMode == TimeInfoOnly_25 || loadMode ==TimeInfoOnly_30)
     for(int m=1;m<=10;++m)
     {
         sprintf(tableFileName,"%s/syncTable_kinect%d_color.txt",syncFolder,m);
         if(FileExistCheck(tableFileName) == false)
         {
             printf("Can't load Kinect syncfile: %s\n",tableFileName);
             continue;
         }
         printf("Start load Kinect syncfile: %s\n",tableFileName);

         m_UnivTable_kinect.resize( m_UnivTable_kinect.size() +1);
         SUnivSyncUnitForEachSensor& univUnit = m_UnivTable_kinect.back();
         univUnit.sensorName.sensorType = SENSOR_KINECT;
         univUnit.sensorName.nodeIdx = m;


         ifstream fin(tableFileName);
         string lineStr;
         getline( fin, lineStr );
         int frameIdx;
         double univTime;
         int timeCodeInt;
         while(fin)
         {
             if (!getline( fin, lineStr )) break;

             istringstream ss( lineStr );
             ss >> frameIdx;     //frameIdx
             ss >> univTime;     //univTime
             ss >> timeCodeInt;     // FrameIdx(includeDroppedFrame) --> no need
             ss >> timeCodeInt;     //ClosestLTCTimeCode
             //printf("%d, %f\n",frameIdx,uinvTime);
            univUnit.frameUnivTimeTable.push_back( pair<int,double> (frameIdx,univTime));
            univUnit.univTimeFrameMap.insert( pair<double,int> (univTime,frameIdx));
            univUnit.frameTimeCode.push_back(timeCodeInt);
         }
         if(univUnit.frameUnivTimeTable.size()==0)
         {
            // printf("Ignore due to the zero size: %d elments\n",univUnit.frameUnivTimeTable.size());
             m_UnivTable_kinect.pop_back();
             continue;
         }
         printf("Loaded Kinect table: %lu elments\n",univUnit.frameUnivTimeTable.size());
         fin.close();
     }
}


//Load selected sync tables. Used for singleRaw
//Save them to m_UnivTable_vga, m_UnivTable_hd, m_UnivTable_kinect
//In 25fps mode, anyway we need VGA time table
//In 30fps mode, we only need HD table
void CSyncManager::loadSelectedTables(const char* syncFolder, LoadMode loadMode,int machineIdx,int diskIdx)     //machinhe and diskIdx is needed only for HD
{
    char tableFileName[512];
    char str[512];

    if(loadMode  == VGASingleRaw_25 || loadMode == HDSingleRaw_25)         //SingleRaw modes. Need VGA table anyway
    {
         //load VGA
         m_UnivTable_vga.sensorName.sensorType = SENSOR_VGA;
         m_UnivTable_vga.sensorName.nodeIdx = 1;  //ve01
         sprintf(tableFileName,"%s/syncTable_vga.txt",syncFolder);
         if(FileExistCheck(tableFileName) == false)
         {
             printf("Can't load vga syncfile: %s\n",tableFileName);
             return;
         }
         ifstream fin(tableFileName);
         string lineStr;
         getline( fin, lineStr );
         int frameIdx;
         double univTime;
         int timeCodeInt;
         while(fin)
         {

             if (!getline( fin, lineStr )) break;

             istringstream ss( lineStr );
             ss >> frameIdx;     //frameIdx
             ss >> univTime;     //univTime
             ss >> timeCodeInt;
             //printf("%d, %f\n",frameIdx,uinvTime);
            m_UnivTable_vga.frameUnivTimeTable.push_back( pair<int,double> (frameIdx,univTime));
            m_UnivTable_vga.univTimeFrameMap.insert( pair<double,int> (univTime,frameIdx));
            m_UnivTable_vga.frameTimeCode.push_back(timeCodeInt);
         }
         printf("Loaded vga table: %lu elments\n",m_UnivTable_vga.frameUnivTimeTable.size());
         fin.close();
    }

     //load HD
     if(loadMode  == HDSingleRaw_25 || loadMode == HDSingleRaw_30)         //SingleRaw modes. Need HD anyway
     //for(int m=31;m<=46;++m)
     {
       //  for(int d=1;d<=2;++d)
         {
             int m = machineIdx;
             int d = diskIdx;

             sprintf(tableFileName,"%s/syncTable_hd%02d-%d.txt",syncFolder,m,d);
             if(FileExistCheck(tableFileName) == false)
             {
                 printf("Can't load hd syncfile: %s\n",tableFileName);

             }
             else
             {
                 printf("Start load hd syncfile: %s\n",tableFileName);

                 m_UnivTable_hd.resize( m_UnivTable_hd.size() +1);
                 SUnivSyncUnitForEachSensor& univUnit = m_UnivTable_hd.back();
                 univUnit.sensorName.sensorType = SENSOR_HD;
                 univUnit.sensorName.nodeIdx = m;
                 univUnit.sensorName.diskIdx = d;


                 ifstream fin(tableFileName);
                 string lineStr;
                 getline( fin, lineStr );
                 int frameIdx;
                 double univTime;
                 int timeCodeInt;
                 while(fin)
                 {
                     if (!getline( fin, lineStr )) break;

                     istringstream ss( lineStr );
                     ss >> frameIdx;     //frameIdx
                     ss >> univTime;     //univTime
                     ss >> timeCodeInt;

                     //printf("%d, %f\n",frameIdx,uinvTime);
                    univUnit.frameUnivTimeTable.push_back( pair<int,double> (frameIdx,univTime));
                    univUnit.univTimeFrameMap.insert( pair<double,int> (univTime,frameIdx));
                    univUnit.frameTimeCode.push_back(timeCodeInt);
                 }
                 if(univUnit.frameUnivTimeTable.size()==0)
                 {
                    // printf("Ignore due to the zero size: %d elments\n",univUnit.frameUnivTimeTable.size());
                     m_UnivTable_hd.pop_back();

                 }
                 else
                 {
                     printf("Loaded HD table: %lu elments\n",univUnit.frameUnivTimeTable.size());
                     fin.close();
                 }
             }
         }
     }

     //Read HD Start time and indexTable
     if(loadMode == HDSingleRaw_25 || loadMode == HDSingleRaw_30)
     {
         sprintf(tableFileName,"%s/hdStartPoint.txt",syncFolder);
         if(FileExistCheck(tableFileName) == false)
         {
             printf(" Warning:: Can't load hdStartPoint file: %s\n",tableFileName);
         }
         else
         {
             printf("Start load hdStartPoint: %s\n",tableFileName);
             char str[512];
             double univTime;
             ifstream fin(tableFileName);
             fin >> str >> str >> str;      //headers. HDTimeCode(msec) HDTimeCode(str) UnivTime
             fin >> m_hdStartTimeMsec_for30fps >> str >> univTime;
             printf("HDStartTime: %d, %s, %.3f\n",m_hdStartTimeMsec_for30fps,str,univTime);
             fin.close();
         }
         if(m_hdStartTimeMsec_for30fps>0)
         {
             sprintf(tableFileName,"%s/syncWavToTxt_hd.txt",syncFolder);
             if(FileExistCheck(tableFileName) == false)
             {
                   printf(" Warning:: Can't load hdStartPoint file: %s\n",tableFileName);
             }
             else
             {
                 printf("Start load m_hdTCTimeTable_for30fps: %s\n",tableFileName);
                 char str[512];
                 ifstream fin(tableFileName);
                 fin >> str >> str>> str;   //headers       HDTimeStr       HDTime(ms)      UnivTime(ms)
                 m_hdTCTimeTable_for30fps.reserve(30*60*30);//30min
                 bool bStartToSave =false;
                 double univTime;
                 int timecode_msec;
                 string lineStr;

                 while(fin)
                 {
                     if (!getline( fin, lineStr )) break;

                     istringstream ss( lineStr );
                     ss >> str;     //HDTimeStr
                     ss >> timecode_msec;     //HDTime(ms)
                     ss >> univTime;    //UnivTime(ms)

                    if(bStartToSave==false && m_hdStartTimeMsec_for30fps == timecode_msec)
                    {
                        bStartToSave = true;
                        printf("HDTCTimeTableGen: Find start point!\n");
                    }
                    if(bStartToSave)
                    {
                        m_hdTCTimeTable_for30fps.push_back( make_pair(timecode_msec, univTime ));
                        //printf("idx: %d, %d\n",m_hdTCTimeTable_for30fps.size()-1,m_hdTCTimeTable_for30fps.back());
                    }
                 }
                 fin.close();

                 m_hdTCTimeTable_for30fps_onlyUnivTime.resize(m_hdTCTimeTable_for30fps.size());
                 for(int i=0;i<m_hdTCTimeTable_for30fps.size();++i)
                     m_hdTCTimeTable_for30fps_onlyUnivTime[i] = m_hdTCTimeTable_for30fps[i].second;
             }
         }
     }
     //Kinect is ignored
}

//only for the VGA. To consider frame drop
void CSyncManager::findAllCorresFrames_forVGAPanels(vector<FILE*>& vgaPanelFilePts,int vgaFrame,vector< int>& corresVGAFrames)
{
    int expectedTimeCode = m_UnivTable_vga.frameTimeCode[vgaFrame];

    for(int i=0;i<vgaPanelFilePts.size();++i)
    {

        if(vgaPanelFilePts[i]==NULL)
            continue;

        int panelIdx = i+1;
        int tempTimeCode = CSyncManager::ConvFrameToTC_vga(vgaPanelFilePts[i],vgaFrame);

        if(tempTimeCode == expectedTimeCode)
        {
            corresVGAFrames[i] = vgaFrame;
        }
        else            //Frame drop happened before
        {
            printf("Debug: VE%02d: frame drop happend before. Try to find corresponding frames\n",panelIdx);
            int searchFrame = vgaFrame-1;
            while(true)
            {
                tempTimeCode = CSyncManager::ConvFrameToTC_vga(vgaPanelFilePts[i],searchFrame);

                if(tempTimeCode < expectedTimeCode)
                {
                    printf("Warning: VE%02d: frame drop detected for frame %d (expectedTC %d) \n",panelIdx,vgaFrame,expectedTimeCode);
                    corresVGAFrames[i] = -1;        //put -1 to inform that this is a frame drop
                    break;
                }
                else if(tempTimeCode==expectedTimeCode)
                {
                    corresVGAFrames[i] = searchFrame;
                    printf("DEBUG: VE%02d: frame drop happend before, but still find corresponding for frame %d (local frameIdx %d) \n",panelIdx,vgaFrame,searchFrame);
                    break;
                }

                searchFrame--;
            }
        }
    }
}

//The order of corresFrames should be the same as loadTable order
void CSyncManager::findAllCorresFrames_fromVGAFrameIdx(int vgaFrame,vector< pair<int,double> >& corresFrames)
{
    std::ostringstream oss;

    if(m_UnivTable_vga.frameUnivTimeTable[vgaFrame].first != vgaFrame)
    {
        printf("Frame drop is detected  in vga raw file: frameidx %d\n",vgaFrame);
        return;
    }
    double vgaUnivTime =m_UnivTable_vga.frameUnivTimeTable[vgaFrame].second ;
    //oss << vgaFrame <<"\t" << vgaUnivTime<<"\t";
    corresFrames.push_back( pair<int,double>(vgaFrame,vgaUnivTime) ); //First element is for VGA. this is not used.

    //find closest HDs
    std::map<double,int>::iterator lower,upper;
    for(int i=0;i<m_UnivTable_hd.size();++i)
    {
        lower = m_UnivTable_hd[i].univTimeFrameMap.lower_bound(vgaUnivTime);        //eqaul or greater
        //oss << lower->second <<"\t" << lower->first <<"\t";
        if(lower == m_UnivTable_hd[i].univTimeFrameMap.begin())
            corresFrames.push_back( pair<int,double>(lower->second,lower->first) );
        else
        {
            std::map<double,int>::iterator llower = lower;            //lower
            llower--;
            if(fabs(lower->first - vgaUnivTime) > fabs(llower->first - vgaUnivTime) )       //llower is closer
                corresFrames.push_back( pair<int,double>(llower->second,llower->first) );
            else
                corresFrames.push_back( pair<int,double>(lower->second,lower->first) );
        }
    }

    for(int i=0;i<m_UnivTable_kinect.size();++i)
    {
        lower = m_UnivTable_kinect[i].univTimeFrameMap.lower_bound(vgaUnivTime);        //eqaul or greater
        //oss << lower->second <<"\t" << lower->first <<"\t";
        if(lower == m_UnivTable_kinect[i].univTimeFrameMap.begin())
            corresFrames.push_back( pair<int,double>(lower->second,lower->first) );
        else
        {
            std::map<double,int>::iterator llower = lower;            //lower
            llower--;
            if(fabs(lower->first - vgaUnivTime) > fabs(llower->first - vgaUnivTime) )       //llower is closer
                corresFrames.push_back( pair<int,double>(llower->second,llower->first) );
            else
                corresFrames.push_back( pair<int,double>(lower->second,lower->first) );
        }
    }
    //oss <<"\n";
    //printf("%s",oss.str().c_str());
}


//Return localFrameIdx of targetSyncUnit
int CSyncManager::findClosetFrameIdx_fromVGA(SUnivSyncUnitForEachSensor& vgaSyncUniv,int vgaFrameIdx,SUnivSyncUnitForEachSensor& targetSyncUnit,double& univTimeDiff)
{
    if(vgaFrameIdx >= vgaSyncUniv.frameUnivTimeTable.size())
    {
        printf("## ERROR:: findClosetFrameIdx_fromVGA: vgaFrameIdx >= m_UnivTable_vga.frameUnivTimeTable.size()\n");
        return -1;
    }
    double refUnivTime = vgaSyncUniv.frameUnivTimeTable[vgaFrameIdx].second ;
    std::map<double,int>::iterator lower = targetSyncUnit.univTimeFrameMap.lower_bound(refUnivTime);        //eqaul or greater
    //oss << lower->second <<"\t" << lower->first <<"\t";
    if(lower == targetSyncUnit.univTimeFrameMap.begin())
    {
        univTimeDiff = 1e5;
        return lower->second;
    }
    else
    {
        std::map<double,int>::iterator llower = lower;            //lower
        llower--;
        if(fabs(lower->first - refUnivTime) > fabs(llower->first - refUnivTime) )       //llower is closer
        {
            univTimeDiff = llower->first - refUnivTime;
            return llower->second;
        }
        else
        {
            univTimeDiff = lower->first - refUnivTime;
            return lower->second;

        }
    }
}


//Return localFrameIdx of targetSyncUnit
int CSyncManager::findClosetFrameIdx_fromUnivTime(double refUnivTime,SUnivSyncUnitForEachSensor& targetSyncUnit,double& univTimeDiff)
{
    std::map<double,int>::iterator lower = targetSyncUnit.univTimeFrameMap.lower_bound(refUnivTime);        //eqaul or greater
    //oss << lower->second <<"\t" << lower->first <<"\t";
    if(lower == targetSyncUnit.univTimeFrameMap.begin())
    {
        univTimeDiff = 1e5;
        return lower->second;
    }
    else
    {
        std::map<double,int>::iterator llower = lower;            //lower
        llower--;
        if(fabs(lower->first - refUnivTime) > fabs(llower->first - refUnivTime) )       //llower is closer
        {
            univTimeDiff = llower->first - refUnivTime;
            return llower->second;
        }
        else
        {
            univTimeDiff = lower->first - refUnivTime;
            return lower->second;

        }
    }
}


//Used for frameDropChecker
//Return closet univTime
double CSyncManager::findClosetFrameIdx_fromUnivTime_byUnivTimeVect(double refUnivTime,vector<double>& sortedUnivTimeVect,double& univTimeDiff)
{
    std::vector<double>::iterator lower = lower_bound(sortedUnivTimeVect.begin(),sortedUnivTimeVect.end(),refUnivTime);        //eqaul or greater
    //oss << lower->second <<"\t" << lower->first <<"\t";
    if(lower == sortedUnivTimeVect.begin())
    {
        univTimeDiff = 1e5;
        return -1;
    }
    else
    {
        std::vector<double>::iterator llower = lower;            //lower
        llower--;
        if(fabs(*lower- refUnivTime) > fabs(*llower - refUnivTime) )       //llower is closer
        {
            univTimeDiff = *llower- refUnivTime;
            return *llower;
        }
        else
        {
            univTimeDiff = *lower- refUnivTime;
            return *lower;

        }
    }
}

//The order of corresFrames should be the same as loadTable order
//Made for HD_30 but can used with VGA related options
void CSyncManager::findAllCorresFrames_fromUnivTime(double refUnivTime,vector< pair<int,double> >& corresFrames)
{
    std::map<double,int>::iterator lower;
    if(m_UnivTable_vga.univTimeFrameMap.size()>0)     //VGA is also used
    {
        lower = m_UnivTable_vga.univTimeFrameMap.lower_bound(refUnivTime);        //eqaul or greater
        //oss << lower->second <<"\t" << lower->first <<"\t";
        if(lower == m_UnivTable_vga.univTimeFrameMap.begin())
            corresFrames.push_back( pair<int,double>(lower->second,lower->first) );
        else
        {
            std::map<double,int>::iterator llower = lower;            //lower
            llower--;
            if(fabs(lower->first - refUnivTime) > fabs(llower->first - refUnivTime) )       //llower is closer
                corresFrames.push_back( pair<int,double>(llower->second,llower->first) );
            else
                 corresFrames.push_back( pair<int,double>(lower->second,lower->first) );
        }
    }

    //find closest HDs
    for(int i=0;i<m_UnivTable_hd.size();++i)
    {
        lower = m_UnivTable_hd[i].univTimeFrameMap.lower_bound(refUnivTime);        //eqaul or greater
        //oss << lower->second <<"\t" << lower->first <<"\t";
        if(lower == m_UnivTable_hd[i].univTimeFrameMap.begin())
            corresFrames.push_back( pair<int,double>(lower->second,lower->first) );
        else
        {
            std::map<double,int>::iterator llower = lower;            //lower
            llower--;
            if(fabs(lower->first - refUnivTime) > fabs(llower->first - refUnivTime) )       //llower is closer
                corresFrames.push_back( pair<int,double>(llower->second,llower->first) );
            else
                corresFrames.push_back( pair<int,double>(lower->second,lower->first) );
        }
    }

    for(int i=0;i<m_UnivTable_kinect.size();++i)
    {
        lower = m_UnivTable_kinect[i].univTimeFrameMap.lower_bound(refUnivTime);        //eqaul or greater
        //oss << lower->second <<"\t" << lower->first <<"\t";
        if(lower == m_UnivTable_kinect[i].univTimeFrameMap.begin())
            corresFrames.push_back( pair<int,double>(lower->second,lower->first) );
        else
        {
            std::map<double,int>::iterator llower = lower;            //lower
            llower--;
            if(fabs(lower->first - refUnivTime) > fabs(llower->first - refUnivTime) )       //llower is closer
                corresFrames.push_back( pair<int,double>(llower->second,llower->first) );
            else
                corresFrames.push_back( pair<int,double>(lower->second,lower->first) );
        }
    }

    //oss <<"\n";
    //printf("%s",oss.str().c_str());
}

void CSyncManager::GetSensorNames(vector<SSensorName>& outputSensorNames )
{
    if(m_UnivTable_vga.frameUnivTimeTable.size()>0)
        outputSensorNames.push_back(m_UnivTable_vga.sensorName);

    for(int i=0;i<m_UnivTable_hd.size();++i)
        if(m_UnivTable_hd[i].frameUnivTimeTable.size()>=0)
            outputSensorNames.push_back(m_UnivTable_hd[i].sensorName);

    for(int i=0;i<m_UnivTable_kinect.size();++i)
        if(m_UnivTable_kinect[i].frameUnivTimeTable.size()>=0)
            outputSensorNames.push_back(m_UnivTable_kinect[i].sensorName);
}

//VGA - HD mapping
bool CSyncManager::FindHDTC_fromVGAFrame(int vgaFirstTC,int vgaFrame,char* hdtcstr,int& offsetMsec)
{
    double univTime_vga = ConvFrameToUnivTime_vga(vgaFirstTC,vgaFrame);

    //Find the HDTC with closest univTime from univTime_vga
    //No need to read HDraw file. Wav file has all the information assuming no frame drop
    double hdfirstUnivTime = GetUnivTimeinWavbyIndex_hd(0);

    //printf("hdfirstUnivTime: %f\n",hdfirstUnivTime);
    double diffinUnivTime = univTime_vga - hdfirstUnivTime;
    if(diffinUnivTime<0)
        return -1;

    //printf("univTime_vga %f, diffinUnivTime: %f\n",univTime_vga,diffinUnivTime);

    int estFrame = diffinUnivTime /30;      //truncation
    int estTCmsec = GetTCmsecinWavbyIndex_hd(estFrame);
    double estUnivTime = GetUnivTimeinWavbyIndex_hd(estFrame);

    diffinUnivTime = univTime_vga - estUnivTime;
    int incFrameStep;
    if(diffinUnivTime==0)
    {
        offsetMsec = 0;
        ConvTCmsecToTCstr_hd(estTCmsec,hdtcstr);
        return true;
    }
    else
    {
        incFrameStep = (diffinUnivTime>0) ? 1:-1;
    }

    while(true)
    {
        int prevEstFrame = estFrame;
        double prevEstUnivTime = estUnivTime;

        estFrame += incFrameStep;
        estUnivTime = GetUnivTimeinWavbyIndex_hd(estFrame);
        if(estUnivTime<0)
        {
            offsetMsec = 1e5;
            return false;
        }

        if(univTime_vga==estUnivTime)
        {
            offsetMsec =0;
            estTCmsec = GetTCmsecinWavbyIndex_hd(estFrame);
            ConvTCmsecToTCstr_hd(estTCmsec,hdtcstr);
            return true;
        }
        else
        {
            int newIncFrameStep = (univTime_vga>estUnivTime) ? 1:-1;
            if(newIncFrameStep != incFrameStep)     //sign has been changed
            {
                double diff_cur = fabs(float(estUnivTime - univTime_vga));
                double diff_prev = fabs(float(prevEstUnivTime - univTime_vga));
                if(diff_cur<diff_prev)      //select estFrame
                {
                    //printf("\t comapred %d\n",diff_prev);
                    offsetMsec = diff_cur;
                    estTCmsec = GetTCmsecinWavbyIndex_hd(estFrame);
                    ConvTCmsecToTCstr_hd(estTCmsec,hdtcstr);
                    return true;
                }
                else        //select preEstFrame
                {
                    //printf("\t comapred %d\n",diff_cur);
                    offsetMsec = diff_prev;
                    estTCmsec = GetTCmsecinWavbyIndex_hd(prevEstFrame);
                    ConvTCmsecToTCstr_hd(estTCmsec,hdtcstr);
                    return true;
                }
            }
        }
    }

    //should not be reached here
    offsetMsec = 1e5;
    return false;
}

//VGAOnly_25, VH_25, VK_25, VHK_25, HDOnly_30, KinectOnly_30, HK_30\n");
CSyncManager::LoadMode CSyncManager::GetLoadModeFromStr(const char* str)
{
    CSyncManager::LoadMode loadModeOption;
    if(strcmp("VGAOnly_25",str) ==0)
          loadModeOption = CSyncManager::VGAOnly_25;
    else if(strcmp("VH_25",str) ==0)
          loadModeOption = CSyncManager::VH_25;
    else if(strcmp("VK_25",str) ==0)
          loadModeOption = CSyncManager::VK_25;
    else if(strcmp("VHK_25",str) ==0)
          loadModeOption = CSyncManager::VHK_25;
    else if(strcmp("HDOnly_30",str) ==0)
          loadModeOption = CSyncManager::HDOnly_30;
    else if(strcmp("KinectOnly_30",str) ==0)
          loadModeOption = CSyncManager::KinectOnly_30;
    else if(strcmp("HK_30",str) ==0)
          loadModeOption = CSyncManager::HK_30;

    else if(strcmp("HDOnly_25",str) ==0)
          loadModeOption = CSyncManager::HDOnly_25;
    else if(strcmp("KinectOnly_25",str) ==0)
          loadModeOption = CSyncManager::KinectOnly_25;
    else if(strcmp("VGASingleRaw_25",str) ==0)
          loadModeOption = CSyncManager::VGASingleRaw_25;
    else if(strcmp("HDSingleRaw_25",str) ==0)
          loadModeOption = CSyncManager::HDSingleRaw_25;
    else if(strcmp("HDSingleRaw_30",str) ==0)
          loadModeOption = CSyncManager::HDSingleRaw_30;

    else if(strcmp("TimeInfoOnly_25",str) ==0)
          loadModeOption = CSyncManager::TimeInfoOnly_25;
    else if(strcmp("TimeInfoOnly_30",str) ==0)
          loadModeOption = CSyncManager::TimeInfoOnly_30;

    else if(strcmp("INDEXMAP_25_30",str) ==0)
          loadModeOption = CSyncManager::INDEXMAP_25_30;

    else
    {
        printf("## ERROR: Unidentified loadModeStr: %s\n",str);
        loadModeOption = CSyncManager::UnKnown;
    }
    return loadModeOption;
}

//Assume that m_UnivTable_vga and m_hdTCTimeTable_for30fps_onlyUnivTime is already loaded
void CSyncManager::FrameIdxMapExport_From25VGAfpsTo30HDfps(const char* outFileName)
{
    if(m_UnivTable_vga.frameUnivTimeTable.size()==0 || m_hdTCTimeTable_for30fps_onlyUnivTime.size()==0)
    {
        printf("## ERROR:: (m_UnivTable_vga.frameUnivTimeTable.size()==0 || m_hdTCTimeTable_for30fps_onlyUnivTime.size()==0 \n");
        return;
    }
    std::ostringstream oss;
    char str[512];
    sprintf(str,"vgaFrameIdx\tvgaFrameUnivTime\thdFrameIdx-sameTimeLater\n");

    int hdFrameIdx = 0;     //always pointing equal or later
    for(int vgaFrameIdx=0;vgaFrameIdx < m_UnivTable_vga.frameUnivTimeTable.size() -1;++vgaFrameIdx)
    {
        double vgaUnivTime = m_UnivTable_vga.frameUnivTimeTable[vgaFrameIdx].second;
        sprintf(str,"%d %.3f ",vgaFrameIdx,vgaUnivTime);
        oss << str;

        double vgaUnivTimeNext = m_UnivTable_vga.frameUnivTimeTable[vgaFrameIdx+1].second;

        if(vgaFrameIdx ==0)
        {
            while(vgaUnivTime > m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx])
            {
                hdFrameIdx++;
                if(hdFrameIdx >= m_hdTCTimeTable_for30fps_onlyUnivTime.size())
                {
                    printf("## ERROR:: frameIdxMapFrom25VGAfpsTo30HDfps:: some table is not valid.\n");
                    return;
                }
            }
            //Here, vgaUnivTime <= hdFrameIdx_univTime
        }

        int cnt =0;
        std::ostringstream oss2;
        while(vgaUnivTime <= m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx] && vgaUnivTimeNext >m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx])
        {
            double hdUnivTime = m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx];

            sprintf(str,"%d %.3f ",hdFrameIdx,m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx]);
            oss2 << str;
            cnt++;
            hdFrameIdx++;
        }

        oss << cnt <<" " << oss2.str() <<"\n";
    }

    ofstream fout(outFileName);
    /*oss << "One hundred and one: " << 101;
    std::string s = oss.str();
    std::cout << s << '\n';*/
    fout << oss.str();
    fout.close();

}


//Assume that m_UnivTable_vga and m_hdTCTimeTable_for30fps_onlyUnivTime is already loaded
void CSyncManager::FrameIdxMapExport_From25VGAfpsTo30HDfps_applyOffset(const char* outFileName)
{
    int offset = 1;   //Add this offset to the HD frame index
    if(m_UnivTable_vga.frameUnivTimeTable.size()==0 || m_hdTCTimeTable_for30fps_onlyUnivTime.size()==0)
    {
        printf("## ERROR:: (m_UnivTable_vga.frameUnivTimeTable.size()==0 || m_hdTCTimeTable_for30fps_onlyUnivTime.size()==0 \n");
        return;
    }
    std::ostringstream oss;
    char str[512];
    sprintf(str,"vgaFrameIdx\tvgaFrameUnivTime\thdFrameIdx-sameTimeLater\n");

    int hdFrameIdx = 0;     //always pointing equal or later
    for(int vgaFrameIdx=0;vgaFrameIdx < m_UnivTable_vga.frameUnivTimeTable.size() -1;++vgaFrameIdx)
    {
        double vgaUnivTime = m_UnivTable_vga.frameUnivTimeTable[vgaFrameIdx].second;
        sprintf(str,"%d %.3f ",vgaFrameIdx,vgaUnivTime);
        oss << str;

        double vgaUnivTimeNext = m_UnivTable_vga.frameUnivTimeTable[vgaFrameIdx+1].second;

        if(vgaFrameIdx ==0)
        {
            while(vgaUnivTime > m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx])
            {
                hdFrameIdx++;
                if(hdFrameIdx >= m_hdTCTimeTable_for30fps_onlyUnivTime.size())
                {
                    printf("## ERROR:: frameIdxMapFrom25VGAfpsTo30HDfps:: some table is not valid.\n");
                    return;
                }
            }
            //Here, vgaUnivTime <= hdFrameIdx_univTime
        }

        int cnt =0;
        std::ostringstream oss2;
        while(vgaUnivTime <= m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx] && vgaUnivTimeNext >m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx])
        {
            double hdUnivTime = m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx];

            //sprintf(str,"%d %.3f ",hdFrameIdx,m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx]);
            sprintf(str,"%d %.3f ",hdFrameIdx + offset,m_hdTCTimeTable_for30fps_onlyUnivTime[hdFrameIdx]);
            oss2 << str;
            cnt++;
            hdFrameIdx++;
        }

        oss << cnt <<" " << oss2.str() <<"\n";
    }

    ofstream fout(outFileName);
    /*oss << "One hundred and one: " << 101;
    std::string s = oss.str();
    std::cout << s << '\n';*/
    fout << oss.str();
    fout.close();


}
