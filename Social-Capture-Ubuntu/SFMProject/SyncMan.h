#pragma once
#include <vector>

using namespace std;

class CSyncMan
{
public:
	CSyncMan(void);
	~CSyncMan(void);

	void Load(char* mainFolderName,bool bReload=false);
	bool IsLoaded();

	int ClosestVGAfromHD(int hdIdx);
	int ClosestHDfromVGA(int vgaIdx);

	bool ClosestHDsWithOffInfofromVGA(int vgaIdx, vector< pair<int,double> >& hdIdxWithOff );
	bool ClosestVGAsWithOffInfofromHD(int hdIdx, vector< pair<int,double> >& vgaIdxWithOff );

	double GetUnivTimeVGA(int frameIdx);
	double GetUnivTimeHD(int frameIdx);
private:

	vector<int> m_vga2hd;		//m_vga2hd[vgaFrame] == closest_hd_frame 
	vector<int> m_hd2vga;		//m_hd2vga[hdframe] == closest_vga_frame
	vector<double> m_vga2Univtime;		//m_vga2Univtime[vgaFrame] == univTime
	vector<double> m_hd2Univtime;		//m_hd2Univtime[hdFrame] == univTime
	vector< vector< pair<int,double> > > m_vga2nextHds;		//m_vga2nextHds[vgaFrame] contains HD frames  where m_vga2Univtime[vgaFrame]  <  univ time for the HD frames  < m_vga2Univtime[vgaFrame+1]
	vector< vector <pair<int,double> > > m_hd2neighborVGAs;		//should have 2 elements (before and after the m_hd2Univtime)
};

extern CSyncMan g_syncMan;