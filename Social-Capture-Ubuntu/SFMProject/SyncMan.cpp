// #include "stdafx.h"
#include "SyncMan.h"
#include "Utility.h"

#include "Constants.h"

CSyncMan g_syncMan;

CSyncMan::CSyncMan(void)
{
}

CSyncMan::~CSyncMan(void)
{
}

void CSyncMan::Load(char* mainFolderName,bool bReload)
{
	if(bReload==true)
	{
		m_vga2hd.clear();
		m_hd2vga.clear();
		m_vga2Univtime.clear();
		m_hd2Univtime.clear();
		m_vga2nextHds.clear();
	}
	else
	{
		if(m_vga2Univtime.size()>0)
			return;
	}

	char syncFileName[512];
	//sprintf(syncFileName,"%s/IndexMap25to30.txt",mainFolderName);
	sprintf(syncFileName,"%s/IndexMap25to30_offset.txt",mainFolderName);
	if(IsFileExist(syncFileName)==false)
	{
		printf("## ERROR: cannot find the indexMap file %s\n",syncFileName);
		return;
	}
	ifstream fin(syncFileName);

	m_vga2Univtime.reserve(10*60*25);//
	m_vga2nextHds.reserve(10*60*25);
	m_vga2hd.reserve(10*60*25);

	m_hd2Univtime.reserve(10*60*30);
	m_hd2vga.reserve(10*60*30);

	while(true)
	{
		int vgaIdx,hdNum,hdIdx;
		double vgaUnivTime,hdUnivTime;
		fin >> vgaIdx >> vgaUnivTime;
		if(fin.eof()==true)
			break;

		if(vgaIdx!=m_vga2Univtime.size())
		{
			printf(" ## error: VGA frame index is not continuous\n");
			fin.close();
			return ;
		}
		m_vga2Univtime.push_back(vgaUnivTime);
		m_vga2nextHds.resize(m_vga2Univtime.size());
		fin>> hdNum;
		for(int i=0;i<hdNum;++i)
		{
			fin >> hdIdx >> hdUnivTime;

			if(hdIdx>=m_hd2Univtime.size())
			{
				m_hd2Univtime.resize(hdIdx+1);		//valid for 0~ hdIdx
			}
			
			m_hd2Univtime[hdIdx] = hdUnivTime;
			m_vga2nextHds[vgaIdx].push_back(make_pair(hdIdx,hdUnivTime));
		}
	}
	fin.close();

	m_vga2hd.resize(m_vga2nextHds.size(),-1);
	for(int vIdx=0;vIdx<m_vga2nextHds.size();++vIdx)
	{
		if(m_vga2nextHds[vIdx].size()>0)
		{
			if(vIdx==0 || m_vga2nextHds[vIdx-1].size()==0)
				m_vga2hd[vIdx] = m_vga2nextHds[vIdx].front().first;
			else
			{
				double curUnivTime = m_vga2Univtime[vIdx];
				double prevUnivTime = m_vga2nextHds[vIdx-1].back().second;
				int prevHDidx= m_vga2nextHds[vIdx-1].back().first;
				double nextUnivTime = m_vga2nextHds[vIdx].front().second;
				int nextHDidx= m_vga2nextHds[vIdx].front().first;
				if( abs(curUnivTime- prevUnivTime) < abs(curUnivTime- nextUnivTime) )
				{
					m_vga2hd[vIdx] = prevHDidx;
				}
				else
				{
					m_vga2hd[vIdx] = nextHDidx;
				}
			}
		}
	}

	m_hd2vga.resize(m_hd2Univtime.size(),-1);
	m_hd2neighborVGAs.resize(m_hd2Univtime.size());
	for(int hIdx=0;hIdx<m_hd2vga.size();++hIdx)
	{
		double curUniv = m_hd2Univtime[hIdx];
		std::vector<double>::iterator low,llow;
		low = lower_bound(m_vga2Univtime.begin(),m_vga2Univtime.end(),curUniv);		//equal or greater
		if(low ==m_vga2Univtime.begin() || low ==m_vga2Univtime.end())
		{
			m_hd2vga[hIdx] = -1;
		}
		else
		{
			llow = low;
			llow--;

			double off_after = *low - curUniv;
			int idx_after = int(low-m_vga2Univtime.begin());
			double off_before =  *llow - curUniv;
			int idx_before = int(llow-m_vga2Univtime.begin());
			m_hd2neighborVGAs[hIdx].push_back(make_pair(idx_before,off_before));
			m_hd2neighborVGAs[hIdx].push_back(make_pair(idx_after,off_after));
			if( abs(off_after) < abs(off_before) )
				m_hd2vga[hIdx] = idx_after;
			else
				m_hd2vga[hIdx] = idx_before;
		}
	}
}

int CSyncMan::ClosestVGAfromHD(int hdIdx)
{
	if(m_hd2vga.size()==0)
		Load(g_dataMainFolder,true);

	if(m_hd2vga.size()<=hdIdx)
	{
		printf("Warning: syncTable is not valid. Just return the input index\n");
		return hdIdx;
	//	return -1;
	}
	else
		return m_hd2vga[hdIdx];
}

bool CSyncMan::IsLoaded()
{
	if(m_vga2hd.size()==0)
		Load(g_dataMainFolder,true);
	if(m_vga2hd.size()==0)
		return false;
	else
		return true;
}

int CSyncMan::ClosestHDfromVGA(int vgaIdx)
{
	if(m_vga2hd.size()==0)
		Load(g_dataMainFolder,true);

	if(m_vga2hd.size()<=vgaIdx)
	{
		printf("Warning: syncTable is not valid. Just return the input index\n");
		return vgaIdx;
		//return -1;
	}
	else
		return m_vga2hd[vgaIdx];
}

bool CSyncMan::ClosestHDsWithOffInfofromVGA(int vgaIdx, vector< pair<int,double> >& hdIdxWithOff )
{
	if(m_vga2hd.size()<=vgaIdx)
	{
		printf("Warning: syncTable is not valid. Just return the input index\n");
		return false;
	}

	if(vgaIdx-1>=0)
	{
		if(m_vga2nextHds[vgaIdx-1].size()>0)
		{
			int hdidx = m_vga2nextHds[vgaIdx-1].back().first;
			double univTime = m_vga2nextHds[vgaIdx-1].back().second;
			double off = univTime- m_vga2Univtime[vgaIdx];
			hdIdxWithOff.push_back(make_pair(hdidx,off));
		}
	}
	if(m_vga2nextHds[vgaIdx].size()>0)
	{
		int hdidx = m_vga2nextHds[vgaIdx].front().first;
		double univTime = m_vga2nextHds[vgaIdx].front().second;
		double off = univTime- m_vga2Univtime[vgaIdx];
		hdIdxWithOff.push_back(make_pair(hdidx,off));
	}
	return true;
}

bool CSyncMan::ClosestVGAsWithOffInfofromHD(int hdIdx, vector< pair<int,double> >& vgaIdxWithOff )
{
	if(m_hd2vga.size()<=hdIdx || hdIdx<0)
	{
		printf("Warning: syncTable is not valid. Just return the input index\n");
		return false;
	}

	if(m_hd2neighborVGAs[hdIdx].size()!=2)
		return false;

	else
	{
		vgaIdxWithOff = m_hd2neighborVGAs[hdIdx];
		return true;
	}

	return false;
}

double CSyncMan::GetUnivTimeVGA(int frameIdx)
{
	if(m_vga2Univtime.size()<=frameIdx)
	{
		printf("Warning: syncTable is not valid. Just return -1\n");
		return -1;
	}
	else
		return m_vga2Univtime[frameIdx];
}

double CSyncMan::GetUnivTimeHD(int frameIdx)
{
	if(m_hd2Univtime.size()<=frameIdx)
	{
		printf("Warning: syncTable is not valid. Just return -1\n");
		return -1;
	}
	else
		return m_hd2Univtime[frameIdx];
}