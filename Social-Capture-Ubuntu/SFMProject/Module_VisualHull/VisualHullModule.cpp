#include "Module_VisualHull/VisualHullModule.h"
#include <stack>
// static CClock g_clock;
#define VOXEL_SIZE_CM 1
using namespace cv;

void CVisualHull::SaveVisualHullIndex(char* outputFileName)
{
	ofstream fout(outputFileName,std::ios_base::trunc);
	fout << xMin << "\t" << xMax << "\t" << yMin << "\t" << yMax << "\t" << zMin << "\t" << zMax << "\t" << m_voxelSize <<"\n";
	fout << m_voxelNum <<"\n";

	for(int i=0;i<m_voxelNum;++i)
	{
		if(m_voxelOccMem[i]>VOX_BLANK)
			fout << i <<" ";
	}
	fout.close();
}

void CVisualHull::SetParameters(float xm,float xM,float ym,float yM,float zm,float zM,float vs)
{
	xMin = xm;
	xMax = xM;
	yMin = ym;
	yMax = yM;
	zMin = zm;
	zMax = zM;
	m_voxelSize = vs;

	xWidth = (xMax - xMin)/m_voxelSize;
	yWidth = (yMax - yMin)/m_voxelSize;
	zWidth = (zMax - zMin)/m_voxelSize;
	m_xzPlaneCnt = xWidth * zWidth;
	m_voxelNum =xWidth*yWidth*zWidth;
}

Point3f CVisualHull::GetVoxelPos(int index)
{
	int yIdx = index/m_xzPlaneCnt;
	int residual = index%m_xzPlaneCnt;
	int zIdx = residual / xWidth;
	residual= residual % xWidth;
	int xIdx = residual;

	Point3f pos;
	pos.x = xMin + xIdx *m_voxelSize;
	pos.y = yMin + yIdx *m_voxelSize;
	pos.z = zMin + zIdx *m_voxelSize;

	return pos;
}

int CVisualHull::GetVoxelIndex(Point3f pos)
{
	if(pos.x<xMin || pos.x>xMax || pos.y<yMin || pos.y>yMax || pos.z<zMin || pos.z>zMax)
	{
		return -1;
	}

	int xIdx = (pos.x - xMin) /m_voxelSize;
	int yIdx = (pos.y - yMin) /m_voxelSize;
	int zIdx = (pos.z - zMin) /m_voxelSize;

	return m_xzPlaneCnt*yIdx + zIdx*xWidth + xIdx;  ///start from 0
}

bool CVisualHull::IsSurfaceVoxel(int index)
{
	if(index>=m_voxelNum)
		return false;

	int yIdx = index/m_xzPlaneCnt;
	int residual = index%m_xzPlaneCnt;
	int zIdx = residual / xWidth;
	residual= residual % xWidth;
	int xIdx = residual;


	for(int dy=-1;dy<=1;dy++)
		for(int dz=-1;dz<=1;dz++)
			for(int dx=-1;dx<=1;dx++)
			{
				if(abs(dx) + abs(dy) +abs(dz) !=1)
					continue;
				if(dx ==0 && dy==0 && dz==0)
					continue;
				if(IsOccupied(dx+xIdx,dy+yIdx,dz+zIdx) == false)
					return true;
			}
	return false;
}


bool CVisualHull::GetNeighborVoxIdx(int index, vector<int>& neighborVox,int searchRange)
{
	if(index>=m_voxelNum)
		return false;

	int yIdx = index/m_xzPlaneCnt;
	int residual = index%m_xzPlaneCnt;
	int zIdx = residual / xWidth;
	residual= residual % xWidth;
	int xIdx = residual;

	int blankCnt=0;
	for(int dy=-searchRange;dy<=searchRange;dy++)
		for(int dz=-searchRange;dz<=searchRange;dz++)
			for(int dx=-searchRange;dx<=searchRange;dx++)
			{
				int tempX = dx+xIdx;
				int tempY = dy+yIdx;
				int tempZ = dz+zIdx;
				if(tempX<0 || tempY<0 || tempZ<0)
					continue;
				if(tempX>=xWidth || tempY>=yWidth || tempZ>=zWidth)
					continue;

				int voxIdx = GetVoxelIndex(tempX,tempY,tempZ);
				if(voxIdx>=m_voxelNum)
					continue;
				neighborVox.push_back(voxIdx);
			}


	return true;
}


bool CVisualHull::GetNeighborSurfVox(int index, vector<int>& neighborVox,Point3f& OuterDirectPt,int searchRange)
{
	if(index>=m_voxelNum)
		return false;

	int yIdx = index/m_xzPlaneCnt;
	int residual = index%m_xzPlaneCnt;
	int zIdx = residual / xWidth;
	residual= residual % xWidth;
	int xIdx = residual;

	int blankCnt=0;
	for(int dy=-searchRange;dy<=searchRange;dy++)
		for(int dz=-searchRange;dz<=searchRange;dz++)
			for(int dx=-searchRange;dx<=searchRange;dx++)
			{
				if(dx+xIdx<0 || dy+yIdx<0 || dz+zIdx<0)
					continue;
					//continue;
				//if(abs(dx) + abs(dy) +abs(dz) !=1)
					//continue;
				//if(dx ==0 && dy==0 && dz==0)
					//continue;
				int voxIdx = GetVoxelIndex(dx+xIdx,dy+yIdx,dz+zIdx);
				if(voxIdx>=m_voxelNum)
					continue;
				char occupacyInfo = GetVoxelOccupacy(voxIdx);
				if(occupacyInfo == VOX_SURFACE)
					neighborVox.push_back(voxIdx);
				else if(occupacyInfo == VOX_BLANK)
				{
					OuterDirectPt = OuterDirectPt+ GetVoxelPos(voxIdx);
					blankCnt ++;
				}
			}

	if(blankCnt>0)
	{
		OuterDirectPt.x /=blankCnt;
		OuterDirectPt.y /=blankCnt;
		OuterDirectPt.z /=blankCnt;
	}
	else
	{
		OuterDirectPt = Point3f(0,0,0);
	}
				
	return false;
}
	
bool CVisualHull::IsOccupied(int xIdx,int yIdx,int zIdx)
{
	if(xIdx<0 || xIdx>=xWidth || yIdx<0 || yIdx>=yWidth  || zIdx<0 || zIdx>=zWidth )
		return false;
	
	int index = GetVoxelIndex(xIdx,yIdx,zIdx);
	if(m_voxelOccMem[index]  == VOX_BLANK)
		return false;
	else		// 1 or 2(surface vox)
		return true;
}


void CVisualHull::ExtractSurfaceVoxel()
{
	//Extract Only Surface Voxels
	for(int voxelIdx=0;voxelIdx<m_voxelNum;++voxelIdx)
	{
		if(GetVoxelOccupacy(voxelIdx) && IsSurfaceVoxel(voxelIdx))
		{
			//tempVisualHull.surfaceVoxel.push_back( make_pair(tempVisualHull.GetVoxelPos(voxelIdx),tempVisualHull.GetVoxelColor(voxelIdx)) );
			Point3f voxPos = GetVoxelPos(voxelIdx);
			Point3f voxColor = GetVoxelColor(voxelIdx);
			m_surfaceVoxelVis.push_back( SurfVoxelUnit(voxPos,voxColor));
			m_surfaceVoxelVis.back().voxelIdx = voxelIdx;
		}
	}

	//Set occupancy only for the surface Voxels
	for(size_t i=0;i<m_surfaceVoxelVis.size();++i)
	{
		SetVoxelOccupacy(m_surfaceVoxelVis[i].voxelIdx,VOX_SURFACE);  //set surfave voxel
	}
}

void CVisualHull::ExtractSurfaceNormal()
{
	for(size_t i=0;i<m_surfaceVoxelVis.size();++i)
	{
		vector<int> neighborVox;
		Point3f outerDirectPt;
		GetNeighborSurfVox(m_surfaceVoxelVis[i].voxelIdx,neighborVox,outerDirectPt,4);
		//printf("neighborNum = %d \n",neighborVox.size());
		//Mat vector1 = Mat::zeros(3,1,CV_64F);
		if(neighborVox.size()>0)
		{
			Point3f outside_direct= outerDirectPt - m_surfaceVoxelVis[i].pos;
			/*Point3f direct_1= GetVoxelPos(neighborVox[0]) - m_surfaceVoxelVis[i].pos;
			Point3f direct_2= GetVoxelPos(neighborVox[1]) - m_surfaceVoxelVis[i].pos;
				
			Normalize(direct_1);
			Normalize(direct_2);
			Point3d tempNormal = direct_1.cross(direct_2);
			Normalize(tempNormal);*/
			vector<Point3f> neighborVoxMat;
			for(int t=0;t<neighborVox.size();++t)
			{
				neighborVoxMat.push_back(GetVoxelPos(neighborVox[t]));
			}

			Mat plane = PlaneFitting(neighborVoxMat);;
			Mat plane_ = plane.rowRange(0,3);
			Point3f tempNormal = MatToPoint3d(plane_);
			Normalize(tempNormal);

			if(tempNormal.dot(outside_direct)<0)
				tempNormal = -tempNormal;
			m_surfaceVoxelVis[i].normal = tempNormal;
		}
		else
		{
			m_surfaceVoxelVis[i].normal = outerDirectPt  - GetVoxelPos(m_surfaceVoxelVis[i].voxelIdx);
			Normalize(m_surfaceVoxelVis[i].normal);
		}
	}

	unsigned int* m_surfVoxVolume = new unsigned int[m_voxelNum];
	memset(m_surfVoxVolume,0,sizeof(unsigned int));
	for(unsigned int i=0;i<m_surfaceVoxelVis.size();++i)
	{
		m_surfVoxVolume[m_surfaceVoxelVis[i].voxelIdx] = i +1 ;  //Starts from 1. Here, 0 means blank. 
	}

	//normal smoothing
	for(size_t i=0;i<m_surfaceVoxelVis.size();++i)
	{
		vector<int> neighborVox;
		Point3f dummy(0,0,0);
		GetNeighborSurfVox(m_surfaceVoxelVis[i].voxelIdx,neighborVox,dummy,4);
		//Mat vector1 = Mat::zeros(3,1,CV_64F);
		if(neighborVox.size()>0)
		{
			Point3f averageNormal(0,0,0);
			int cnt=0;
			vector<Point3f> neighborVoxMat;
			for(int t=0;t<neighborVox.size();++t)
			{
				int surfIdx = m_surfVoxVolume[neighborVox[t]];
				if(surfIdx==0)
				{
					assert(false);
					continue;		//should not be this.
				}
				surfIdx--;
				averageNormal  = averageNormal +m_surfaceVoxelVis[surfIdx].normal;
				cnt++;
			}
			averageNormal.x = averageNormal.x/cnt;
			averageNormal.y = averageNormal.y/cnt;
			averageNormal.z = averageNormal.z/cnt;
			Normalize(averageNormal);

			m_surfaceVoxelVis[i].normal = averageNormal;
		}
	}
	delete[] m_surfVoxVolume;
	printf("End\n");
}
