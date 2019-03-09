#pragma once
#include "TrajElement3D.h"
#include "DomeImageManager.h"

#define VOX_BLANK 0
#define VOX_OCCUPIED 1
#define VOX_SURFACE 2


class CVisualHull
{
public:
	CVisualHull()
	{
		m_voxelOccMem =NULL;
		m_voxelColorMem =NULL;

		m_maxProbValue = -1;
		m_minProbValue = -1;
	}
	~CVisualHull()
	{
		if(m_voxelOccMem!=NULL)
			delete[] m_voxelOccMem;
		m_voxelOccMem =NULL;
		if(m_voxelColorMem!=NULL)
			delete[] m_voxelColorMem;
		m_voxelColorMem =NULL;
	}
	void SaveVisualHullIndex(char* outputFileName);

	float GetVoxelSize()
	{
		return m_voxelSize;
	}
	void SetVoxelSize(float v)
	{
		m_voxelSize = v;
	}
	void SetParameters(float xm,float xM,float ym,float yM,float zm,float zM,float vs);
	void GetParameters(float& xm,float& xM,float& ym,float& yM,float& zm,float& zM,float& vs) const
	{
		xm = xMin;
		xM = xMax;
		ym = yMin;
		yM = yMax;
		zm = zMin;
		zM = zMax;
		vs = m_voxelSize;
	}
	cv::Point3f GetVoxelPos(int index);
	int GetVoxelIndex(cv::Point3f pos);
	int GetVoxelIndex(int xIdx,int yIdx,int zIdx)
	{
		return m_xzPlaneCnt*yIdx + zIdx*xWidth + xIdx;
	}
	void GetVoxelXYZIndex(int idx, int& xIdx,int& yIdx,int& zIdx)
	{
		yIdx = idx /m_xzPlaneCnt;
		int remain = idx %m_xzPlaneCnt;
		zIdx = remain /xWidth ;
		xIdx = remain %xWidth ;
	}
	int GetVoxelNum()
	{
		return m_voxelNum;
	}
	void SetVoxelOccupacy(int voxelIndex,char o)
	{
		if(voxelIndex>=m_voxelNum)
			return;

		m_voxelOccMem[voxelIndex] = o;
	}
	void SetVoxelColor(int voxelIndex,cv::Point3f c)
	{
		if(voxelIndex>=m_voxelNum)
			return;

		m_voxelColorMem[voxelIndex] = c;
	}
	char  GetVoxelOccupacy(int voxelIndex)
	{
		return m_voxelOccMem[voxelIndex];
	}
	cv::Point3f GetVoxelColor(int voxelIndex)
	{
		return m_voxelColorMem[voxelIndex];
	}
	bool IsSurfaceVoxel(int index);
	bool GetNeighborSurfVox(int index, vector<int>& neighborVox,cv::Point3f& OuterDirectPt,int searchRange=2);
	bool GetNeighborVoxIdx(int index, vector<int>& neighborVox,int searchRange=2);
	bool IsOccupied(int xIdx,int yIdx,int zIdx);

	void AllocMemory()
	{
		int voxelNum =xWidth*yWidth*zWidth;
		m_voxelOccMem = new char[voxelNum];
		m_voxelColorMem = new cv::Point3f[voxelNum];
		memset(m_voxelOccMem,0,voxelNum);
	}
	void deleteMemory()
	{
		delete[] m_voxelOccMem;
		delete[] m_voxelColorMem;
		m_voxelOccMem = NULL;
		m_voxelColorMem = NULL;
	}
	void ExtractSurfaceVoxel();
	void ExtractSurfaceNormal();

	void Segmentation();
	void ExportBbox(CDomeImageManager& domeImageManager);
	void ColoringSurfVoxBySegmentation();

	//vector< pair <Point3f,Point3f> > surfaceVoxel;  //pos and color
	vector< SurfVoxelUnit > m_surfaceVoxelVis;			//Visualized
	vector< SurfVoxelUnit > m_surfaceVoxelOriginal;	//Original	
	int m_actualFrameIdx;

	vector<Bbox3D> m_segmentBboxVect;

	void GetVoxelDimension(int& x,int& y, int& z)
	{
		x = xWidth;
		y = yWidth;
		z = zWidth;
	}

	//Used for probVolume. 
	float m_maxProbValue;
	float m_minProbValue;

private:
	float xMin,xMax,yMin,yMax,zMin,zMax;
	int xWidth,yWidth,zWidth;
	int m_xzPlaneCnt;
	float m_voxelSize;
	char* m_voxelOccMem;
	cv::Point3f* m_voxelColorMem;
	int m_voxelNum;

};


class CVisualHullManager
{
public:
	CVisualHullManager() 
	{
		m_currentSelectedVisualHullIdx = -1;
		m_imgFramdIdx  =-1;
	}
	vector<CVisualHull> m_visualHullRecon;
	int m_imgFramdIdx;			//used in probability volume for body reconstruction. (Each of this element becomes a data structure for each time instance)
private:
	int m_currentSelectedVisualHullIdx; 
};
