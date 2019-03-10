#include "UtilityGPU.h"
#include <cuda.h>
#include <iostream>

using namespace cv;
using namespace std;

int divUp(int x, int y)
{
	return int(x/float(y) +0.5);
}

void printMatrix2(Mat &M, std::string matrix)
{
    printf("Matrix \"%s\" is %i x %i\n", matrix.c_str(), M.rows, M.cols);
	for( int r=0;r<M.rows;++r)
	{
		for( int c=0;c<M.cols;++c)
		{
			//printf("%.20lf ",M.at<double>(r,c));
			printf("%.6lf ",M.at<float>(r,c));
		}
		cout << endl;
	}
	cout <<endl;
}

void printMatrix2(std::string matrix,Mat &M)
{
    printf("Matrix \"%s\" is %i x %i\n", matrix.c_str(), M.rows, M.cols);
	for( int r=0;r<M.rows;++r)
	{
		for( int c=0;c<M.cols;++c)
		{
			//printf("%.20lf ",M.at<double>(r,c));
			printf("%.6lf ",M.at<float>(r,c));
		}
		cout <<endl;
	}
	cout <<endl;
}

//Device code
//foreGroundParm is floating value
//input: foreGroundParm: 0-255
//result: occupancyOutput: 0 - 1 (average)
__global__ void CalcVolumeCostAverage_hd_float(float* voxelCoordsParm, int voxelNum, float* foreGroundParm, int foreGroundNum, float* projectMatParm,float* occupancyOutput)
{
	//float maxValue = foreGroundNum;
//	int sizeOfOneUnitP = 12*sizeof(float);
    const int voxelIdx = 512* blockIdx.x + threadIdx.x;
	if(voxelIdx<voxelNum)
	{
		float* tempCoord = &voxelCoordsParm[voxelIdx*3];
		int totalCnt=0;
		float foreValue=0;
		for(int i=0;i<foreGroundNum;++i)
		{
			float* tempForeground = &foreGroundParm[2073600*i];  //1920*1080 = 2073600
			float* P = &projectMatParm[12*i];  
			//float* P = projectMatParm+12;//+1 [12*i];  

			//projection
			float z = P[8]*tempCoord[0] +  P[9]*tempCoord[1] + P[10]*tempCoord[2] + P[11] ;					//buf fixed.....it was int. What a bug.
			float x = (P[0]*tempCoord[0] +  P[1]*tempCoord[1] + P[2]*tempCoord[2] + P[3]  )/z;
			float y = (P[4]*tempCoord[0] +  P[5]*tempCoord[1] + P[6]*tempCoord[2] + P[7]  )/z;
			
			//boundary checking
			if(!(x<1 ||x>=1919 || y<1 || y>=1079))
			{
				//bilinear interpolation
				int x_floor = float(x);
				int x_ceil  = x_floor+1;
				int y_floor = float(y);
				int y_ceil = y_floor +1;
				float value_bl = tempForeground[y_floor*1920 + x_floor];		
				float value_br = tempForeground[y_floor*1920+ x_ceil];		
				float value_tl = tempForeground[y_ceil*1920+ x_floor];		
				float value_tr = tempForeground[y_ceil*1920 + x_ceil];		
					
				float alpha = y - y_floor;
				float value_l = (1-alpha) *value_bl + alpha * value_tl;
				float value_r = (1-alpha) *value_br + alpha * value_tr;
					
				float beta = x - x_floor;
				float finalValue = (1-beta) *value_l + beta * value_r;
				foreValue += finalValue;
				totalCnt++;
			}
		}
		
		//occupancyOutput[voxelIdx]  =0.5;
		if(foreValue<=1 || totalCnt<=1)
			occupancyOutput[voxelIdx] = 0;
		else
			occupancyOutput[voxelIdx] = float(foreValue)/float(totalCnt*255);
	}
}

//Device code
//foreGroundParm is floating value
//input: foreGroundParm: 0-255
//result: occupancyOutput: 0 - 1 (average)
__global__ void CalcVolumeCostAverage_hd_float_onlyPositiveCost(float* voxelCoordsParm, int voxelNum, float* foreGroundParm, int foreGroundNum, float* projectMatParm,float* occupancyOutput)
{
	//float maxValue = foreGroundNum;
//	int sizeOfOneUnitP = 12*sizeof(float);
    const int voxelIdx = 512* blockIdx.x + threadIdx.x;
	if(voxelIdx<voxelNum)
	{
		float* tempCoord = &voxelCoordsParm[voxelIdx*3];
		int totalCnt=0;
		float foreValue=0;
		for(int i=0;i<foreGroundNum;++i)
		{
			float* tempForeground = &foreGroundParm[2073600*i];  //1920*1080 = 2073600
			float* P = &projectMatParm[12*i];  
			//float* P = projectMatParm+12;//+1 [12*i];  

			//projection
			float z = P[8]*tempCoord[0] +  P[9]*tempCoord[1] + P[10]*tempCoord[2] + P[11] ;					//buf fixed.....it was int. What a bug.
			float x = (P[0]*tempCoord[0] +  P[1]*tempCoord[1] + P[2]*tempCoord[2] + P[3]  )/z;
			float y = (P[4]*tempCoord[0] +  P[5]*tempCoord[1] + P[6]*tempCoord[2] + P[7]  )/z;
			
			//boundary checking
			if(!(x<1 ||x>=1919 || y<1 || y>=1079))
			{
				//bilinear interpolation
				int x_floor = float(x);
				int x_ceil  = x_floor+1;
				int y_floor = float(y);
				int y_ceil = y_floor +1;
				float value_bl = tempForeground[y_floor*1920 + x_floor];		
				float value_br = tempForeground[y_floor*1920+ x_ceil];		
				float value_tl = tempForeground[y_ceil*1920+ x_floor];		
				float value_tr = tempForeground[y_ceil*1920 + x_ceil];		
					
				float alpha = y - y_floor;
				float value_l = (1-alpha) *value_bl + alpha * value_tl;
				float value_r = (1-alpha) *value_br + alpha * value_tr;
					
				float beta = x - x_floor;
				float finalValue = (1-beta) *value_l + beta * value_r;
				foreValue += finalValue;
				if(finalValue>0)
					totalCnt++;
			}
		}
		
		//occupancyOutput[voxelIdx]  =0.5;
		if(foreValue<=1 || totalCnt<=1) 
			occupancyOutput[voxelIdx] = 0;
		else
			occupancyOutput[voxelIdx] = float(foreValue)/float(totalCnt*255);
	}
}

//Device code
//foreGroundParm is floating value
//input: foreGroundParm: 0-255
//result: occupancyOutput: 0 - 1 (average)
__global__ void CalcVolumeCostAverage_vga_float(float* voxelCoordsParm, int voxelNum, float* foreGroundParm, int foreGroundNum, float* projectMatParm,float* occupancyOutput)
{
	//float maxValue = foreGroundNum;
//	int sizeOfOneUnitP = 12*sizeof(float);
    const int voxelIdx = 512* blockIdx.x + threadIdx.x;
	if(voxelIdx<voxelNum)
	{
		float* tempCoord = &voxelCoordsParm[voxelIdx*3];
		int totalCnt=0;
		float valueSum=0;
		for(int i=0;i<foreGroundNum;++i)
		{
			float* tempForeground = &foreGroundParm[307200*i];  //640*480 = 307200
			float* P = &projectMatParm[12*i];  
			//float* P = projectMatParm+12;//+1 [12*i];  

			//projection
			float z = P[8]*tempCoord[0] +  P[9]*tempCoord[1] + P[10]*tempCoord[2] + P[11] ;					//buf fixed.....it was int. What a bug.
			float x = (P[0]*tempCoord[0] +  P[1]*tempCoord[1] + P[2]*tempCoord[2] + P[3]  )/z;
			float y = (P[4]*tempCoord[0] +  P[5]*tempCoord[1] + P[6]*tempCoord[2] + P[7]  )/z;
			
			//boundary checking
			if(!(x<1 ||x>=639 || y<1 || y>=479))
			{
				//bilinear interpolation
				int x_floor = float(x);
				int x_ceil  = x_floor+1;
				int y_floor = float(y);
				int y_ceil = y_floor +1;
				float value_bl = tempForeground[y_floor*640 + x_floor];		
				float value_br = tempForeground[y_floor*640 + x_ceil];		
				float value_tl = tempForeground[y_ceil*640 + x_floor];		
				float value_tr = tempForeground[y_ceil*640 + x_ceil];		
					
				float alpha = y - y_floor;
				float value_l = (1-alpha) *value_bl + alpha * value_tl;
				float value_r = (1-alpha) *value_br + alpha * value_tr;
					
				float beta = x - x_floor;
				float curValueFinal = (1-beta) *value_l + beta * value_r;

				valueSum += curValueFinal;
				//if(curValueFinal>0)			//ignore non-related views...
					totalCnt++;
			}
		}
		
		//occupancyOutput[voxelIdx]  =0.5;
		if(totalCnt<=1)
			occupancyOutput[voxelIdx] = 0;
		else
			occupancyOutput[voxelIdx] = float(valueSum)/float(totalCnt*255);
	}
}

int bFirstCall=true;
bool DetectionCostVolumeGeneration_float_GPU_average_vga(float* voxelCoords,int voxelNum, vector<Mat_<float> >& foregroundVect,vector<Mat_<float>>& projectMatVect,float* occupancyOutput)
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	if(bFirstCall)
	{
		for (int i = 0; i < nDevices; i++) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			printf("Device Number: %d\n", i);
			printf("  Device name: %s\n", prop.name);
			printf("  Memory Clock Rate (KHz): %d\n",
				   prop.memoryClockRate);
			printf("  Memory Bus Width (bits): %d\n",
				   prop.memoryBusWidth);
			printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
				   2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		}
		printf("## Selected GPU idx: %d\n", g_gpu_device_id);
		bFirstCall =false;
	}
	cudaSetDevice(g_gpu_device_id);

	size_t free,total;
	cudaMemGetInfo(&free,&total);
	//printf("## CUDA:: MemoryChecking ::free%f MB, total %f MB\n",free/1e6,total/1e6);

	assert(foregroundVect.size() == projectMatVect.size());
	
	float* foreGroundParmGPU =NULL;		//forground image data (or detection cost map)
	cudaMalloc((void**) &foreGroundParmGPU, 307200 * foregroundVect.size()*sizeof(float));  //640x480 = 307200
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("####################### CUDA Error 1 : %s ###################\n", cudaGetErrorString(error));
		cudaFree(foreGroundParmGPU);
		return false;
	}
	for(int i=0;i<foregroundVect.size();++i)
	{
		cudaMemcpy(&foreGroundParmGPU[307200*i],foregroundVect[i].data,307200*sizeof(float),cudaMemcpyHostToDevice);
		error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			printf("####################### CUDA Error 2 : %s ###################\n", cudaGetErrorString(error));
			cudaFree(foreGroundParmGPU);
			return false;
		}
	}
	float* projectMatParmGPU=NULL;
	int sizeOfOneUnitP = 12 *sizeof(float);
	cudaMalloc((void**) &projectMatParmGPU, sizeOfOneUnitP* foregroundVect.size());  
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("####################### CUDA Error 3 : %s ###################\n", cudaGetErrorString(error));
		cudaFree(foreGroundParmGPU);
		cudaFree(projectMatParmGPU);
		return false;
	}

	for(int i=0;i<foregroundVect.size();++i)
	{
		cudaMemcpy(&projectMatParmGPU[12*i],projectMatVect[i].data,sizeOfOneUnitP,cudaMemcpyHostToDevice);
		error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			printf("####################### CUDA Error 4 : %s ###################\n", cudaGetErrorString(error));
			cudaFree(foreGroundParmGPU);
			cudaFree(projectMatParmGPU);
			return false;
		}
	}

	////////////////////////////////////////////////////////////////////////////////
	//// Voxel memory allocation
	////////////////////////////////////////////////////////////////////////////////
	int voxelSegNum = 2e8;		//About 200 MB Voxel
	int iterNum = voxelNum/float(voxelSegNum);
	int voxelNuminFinalIter;
	if(voxelNum%(voxelSegNum)>0)
	{
		iterNum++;
		voxelNuminFinalIter = voxelNum%(voxelSegNum);
	}
	if(iterNum ==1)
		voxelSegNum = voxelNum;
	//printf("GPU interation Num %d\n",iterNum);


	//allocate voxel Pos Memory
	float* occupancyOutputGPU=NULL;			//Contains 3D volume costamp
	cudaMalloc((void**) &occupancyOutputGPU, voxelSegNum*sizeof(float));  //voxelSegNum * 4Byte
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("####################### CUDA Error 5 : %s ###################\n", cudaGetErrorString(error));
		cudaFree(foreGroundParmGPU);
		cudaFree(projectMatParmGPU);
		cudaFree(occupancyOutputGPU);
		return false;
	}
	
	float* voxelCoordsGPU=NULL; 
	cudaMalloc((void**) &voxelCoordsGPU, voxelSegNum*sizeof(float)*3);    //voxelSegNum * 12Byte
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("####################### CUDA Error 7 : %s ###################\n", cudaGetErrorString(error));
		cudaFree(foreGroundParmGPU);
		cudaFree(voxelCoordsGPU);
		cudaFree(projectMatParmGPU);
		cudaFree(occupancyOutputGPU);
		return false;
	}

	cudaMemGetInfo(&free,&total);
	//printf("## CUDA:: after :: free%f MB, total %f MB\n",free/1e6,total/1e6);

	for(int i=0;i<iterNum;++i)
	{
		cudaMemset((void*) occupancyOutputGPU, 0, voxelSegNum*sizeof(float));
		error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			printf("####################### CUDA Error 6 : %s ###################\n", cudaGetErrorString(error));
			cudaFree(foreGroundParmGPU);
			cudaFree(voxelCoordsGPU);
			cudaFree(projectMatParmGPU);
			cudaFree(occupancyOutputGPU);
			return false;
		}

		if(i == iterNum-1 && voxelNuminFinalIter>0)		//Final iterlation
		{
			error = cudaGetLastError();
			if(error != cudaSuccess)
			{
				printf("####################### CUDA Error 8 : %s ###################\n", cudaGetErrorString(error));
				cudaFree(foreGroundParmGPU);
				cudaFree(voxelCoordsGPU);
				cudaFree(projectMatParmGPU);
				cudaFree(occupancyOutputGPU);
				return false;
			}

			float* tempVoxelPtr  = &voxelCoords[i*voxelSegNum*3];
			cudaMemcpy(voxelCoordsGPU,tempVoxelPtr,voxelNuminFinalIter*sizeof(float)*3,cudaMemcpyHostToDevice);
			error = cudaGetLastError();
			if(error != cudaSuccess)
			{
				printf("####################### CUDA Error 9 : %s ###################\n", cudaGetErrorString(error));
				cudaFree(foreGroundParmGPU);
				cudaFree(voxelCoordsGPU);
				cudaFree(projectMatParmGPU);
				cudaFree(occupancyOutputGPU);
				return false;
			}

			dim3 block(512,1);
			dim3 grid(divUp(voxelNuminFinalIter,block.x), 1);

			if(grid.x>65535)
			{
				//printf("GPU:: grid size is too big !! (%d,%d)\n",grid.x,grid.y);
				cudaFree(foreGroundParmGPU);
				cudaFree(voxelCoordsGPU);
				cudaFree(projectMatParmGPU);
				cudaFree(occupancyOutputGPU);
				grid.x = 65535;
			}
			//else
				//printf("GPU:: grid size : (%d,%d)\n",grid.x,grid.y);


			CalcVolumeCostAverage_vga_float<<<grid, block>>>(voxelCoordsGPU,voxelNuminFinalIter,foreGroundParmGPU,foregroundVect.size(),projectMatParmGPU,occupancyOutputGPU);
			//CalcVolumeCostAverage_vga_float_onlyPositiveCost<<<grid, block>>>(voxelCoordsGPU,voxelNuminFinalIter,foreGroundParmGPU,foregroundVect.size(),projectMatParmGPU,occupancyOutputGPU);
			//CalcVolumeCostWithOrientation<<<grid, block>>>(voxelCoordsGPU,voxelNuminFinalIter,foreGroundParmGPU,foregroundVect.size(),projectMatParmGPU,(int)camCenters.size(),camCenterParmGPU,occupancyOutputGPU);
			error = cudaGetLastError();
			if(error != cudaSuccess)
			{
				printf("####################### CUDA Error 10 : %s ###################\n", cudaGetErrorString(error));
				cudaFree(foreGroundParmGPU);
				cudaFree(voxelCoordsGPU);
				cudaFree(projectMatParmGPU);
				cudaFree(occupancyOutputGPU);
				return false;
			}

			//dst.download(resultImage);
			//ImageSC2(resultImage);
			cudaDeviceSynchronize();
			cudaMemcpy(&occupancyOutput[i*voxelSegNum],occupancyOutputGPU,voxelNuminFinalIter*sizeof(float),cudaMemcpyDeviceToHost);
			error = cudaGetLastError();
			if(error != cudaSuccess)
			{
				printf("####################### CUDA Error 11 : %s ###################\n", cudaGetErrorString(error));
				cudaFree(foreGroundParmGPU);
				cudaFree(voxelCoordsGPU);
				cudaFree(projectMatParmGPU);
				cudaFree(occupancyOutputGPU);
				return false;
			}
		}
		else
		{
			float* tempVoxelPtr  = &voxelCoords[i*voxelSegNum*3];//i*voxelSegNum];
			cudaMemcpy(voxelCoordsGPU,tempVoxelPtr,voxelSegNum*sizeof(float)*3,cudaMemcpyHostToDevice);

			dim3 block(512,1);
			dim3 grid(divUp(voxelSegNum,block.x), 1);
			if(grid.x>65535)
			{
				//printf("GPU:: grid size is too big !! (%d,%d)\n",grid.x,grid.y);
				grid.x = 65535;
			}
			//else
				//printf("GPU:: grid size : (%d,%d)\n",grid.x,grid.y);

			CalcVolumeCostAverage_vga_float<<<grid, block>>>(voxelCoordsGPU,voxelSegNum,foreGroundParmGPU,foregroundVect.size(),projectMatParmGPU,occupancyOutputGPU);
			cudaGetLastError();
			//dst.download(resultImage);
			//ImageSC2(resultImage);
			cudaDeviceSynchronize();
			cudaMemcpy(&occupancyOutput[i*voxelSegNum],occupancyOutputGPU,voxelSegNum*sizeof(float),cudaMemcpyDeviceToHost);

			cudaError_t error= cudaGetLastError();

			if(error != cudaSuccess)
			{
				// something's gone wrong
				// print out the CUDA error as a string
				printf("CUDA Error: %s\n", cudaGetErrorString(error));
			}
		}
		//printf("CUDA Iteration %d/%d\n",i,iterNum);//
	}

	//delete[] voxelCoords;
	cudaFree(voxelCoordsGPU);
	cudaFree(foreGroundParmGPU);
	cudaFree(projectMatParmGPU);
	cudaFree(occupancyOutputGPU);

	cudaMemGetInfo(&free,&total);
	//printf("## CUDA:: return :: free%f MB, total %f MB\n",free/1e6,total/1e6);

	return true;
}

bool DetectionCostVolumeGeneration_float_GPU_average_hd(float* voxelCoords,int voxelNum, vector<Mat_<float> >& foregroundVect,vector<Mat_<float>>& projectMatVect,float* occupancyOutput,bool bOnlyPositiveValueAvg)
{
	size_t free,total;
	cudaMemGetInfo(&free,&total);
	//printf("## CUDA:: before ::a free%f MB, total %f MB\n",free/1e6,total/1e6);
	assert(foregroundVect.size() == projectMatVect.size());
	
	float* foreGroundParmGPU =NULL;		//forground image data (or detection cost map)
	//cudaMalloc((void**) &foreGroundParmGPU, 307200 * foregroundVect.size()*sizeof(float));  //640x480 = 307200
	cudaMalloc((void**) &foreGroundParmGPU, 2073600 * foregroundVect.size()*sizeof(float));  //1920x1080= 2073600
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("####################### CUDA Error 1 : %s ###################\n", cudaGetErrorString(error));
		cudaFree(foreGroundParmGPU);
		return false;
	}
	for(int i=0;i<foregroundVect.size();++i)
	{
		cudaMemcpy(&foreGroundParmGPU[2073600*i],foregroundVect[i].data,2073600*sizeof(float),cudaMemcpyHostToDevice);
		error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			printf("####################### CUDA Error 2 : %s ###################\n", cudaGetErrorString(error));
			cudaFree(foreGroundParmGPU);
			return false;
		}
	}
	float* projectMatParmGPU=NULL;
	int sizeOfOneUnitP = 12 *sizeof(float);
	cudaMalloc((void**) &projectMatParmGPU, sizeOfOneUnitP* foregroundVect.size());  
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("####################### CUDA Error 3 : %s ###################\n", cudaGetErrorString(error));
		cudaFree(foreGroundParmGPU);
		cudaFree(projectMatParmGPU);
		return false;
	}

	for(int i=0;i<foregroundVect.size();++i)
	{
		cudaMemcpy(&projectMatParmGPU[12*i],projectMatVect[i].data,sizeOfOneUnitP,cudaMemcpyHostToDevice);
		error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			printf("####################### CUDA Error 4 : %s ###################\n", cudaGetErrorString(error));
			cudaFree(foreGroundParmGPU);
			cudaFree(projectMatParmGPU);
			return false;
		}
	}

	////////////////////////////////////////////////////////////////////////////////
	//// Voxel memory allocation
	////////////////////////////////////////////////////////////////////////////////
	int voxelSegNum = 2e8;		//About 200 MB Voxel
	int iterNum = voxelNum/float(voxelSegNum);
	int voxelNuminFinalIter;
	if(voxelNum%(voxelSegNum)>0)
	{
		iterNum++;
		voxelNuminFinalIter = voxelNum%(voxelSegNum);
	}
	if(iterNum ==1)
		voxelSegNum = voxelNum;
	//printf("GPU interation Num %d\n",iterNum);


	//allocate voxel Pos Memory
	float* occupancyOutputGPU=NULL;			//Contains 3D volume costamp
	cudaMalloc((void**) &occupancyOutputGPU, voxelSegNum*sizeof(float));  //voxelSegNum * 4Byte
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("####################### CUDA Error 5 : %s ###################\n", cudaGetErrorString(error));
		cudaFree(foreGroundParmGPU);
		cudaFree(projectMatParmGPU);
		cudaFree(occupancyOutputGPU);
		return false;
	}
	
	float* voxelCoordsGPU=NULL; 
	cudaMalloc((void**) &voxelCoordsGPU, voxelSegNum*sizeof(float)*3);    //voxelSegNum * 12Byte
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("####################### CUDA Error 7 : %s ###################\n", cudaGetErrorString(error));
		cudaFree(foreGroundParmGPU);
		cudaFree(voxelCoordsGPU);
		cudaFree(projectMatParmGPU);
		cudaFree(occupancyOutputGPU);
		return false;
	}

	cudaMemGetInfo(&free,&total);
	//printf("## CUDA:: after :: free%f MB, total %f MB\n",free/1e6,total/1e6);

	for(int i=0;i<iterNum;++i)
	{
		cudaMemset((void*) occupancyOutputGPU, 0, voxelSegNum*sizeof(float));
		error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			printf("####################### CUDA Error 6 : %s ###################\n", cudaGetErrorString(error));
			cudaFree(foreGroundParmGPU);
			cudaFree(voxelCoordsGPU);
			cudaFree(projectMatParmGPU);
			cudaFree(occupancyOutputGPU);
			return false;
		}

		if(i == iterNum-1 && voxelNuminFinalIter>0)		//Final iterlation
		{
			error = cudaGetLastError();
			if(error != cudaSuccess)
			{
				printf("####################### CUDA Error 8 : %s ###################\n", cudaGetErrorString(error));
				cudaFree(foreGroundParmGPU);
				cudaFree(voxelCoordsGPU);
				cudaFree(projectMatParmGPU);
				cudaFree(occupancyOutputGPU);
				return false;
			}

			float* tempVoxelPtr  = &voxelCoords[i*voxelSegNum*3];
			cudaMemcpy(voxelCoordsGPU,tempVoxelPtr,voxelNuminFinalIter*sizeof(float)*3,cudaMemcpyHostToDevice);
			error = cudaGetLastError();
			if(error != cudaSuccess)
			{
				printf("####################### CUDA Error 9 : %s ###################\n", cudaGetErrorString(error));
				cudaFree(foreGroundParmGPU);
				cudaFree(voxelCoordsGPU);
				cudaFree(projectMatParmGPU);
				cudaFree(occupancyOutputGPU);
				return false;
			}

			dim3 block(512,1);
			dim3 grid(divUp(voxelNuminFinalIter,block.x), 1);

			if(grid.x>65535)
			{
				//printf("GPU:: grid size is too big !! (%d,%d)\n",grid.x,grid.y);
				cudaFree(foreGroundParmGPU);
				cudaFree(voxelCoordsGPU);
				cudaFree(projectMatParmGPU);
				cudaFree(occupancyOutputGPU);
				grid.x = 65535;
			}
			//else
				//printf("GPU:: grid size : (%d,%d)\n",grid.x,grid.y);

			if(bOnlyPositiveValueAvg==false)		//avg = valuSum / NumberOfView_ProjectedOnImage
				CalcVolumeCostAverage_hd_float<<<grid, block>>>(voxelCoordsGPU,voxelNuminFinalIter,foreGroundParmGPU,foregroundVect.size(),projectMatParmGPU,occupancyOutputGPU);
			else   //avg = valuSum / (NumberOfView_ProjectedOnImage & positiveValidValue)
				CalcVolumeCostAverage_hd_float_onlyPositiveCost<<<grid, block>>>(voxelCoordsGPU,voxelNuminFinalIter,foreGroundParmGPU,foregroundVect.size(),projectMatParmGPU,occupancyOutputGPU);
			
			//CalcVolumeCostWithOrientation<<<grid, block>>>(voxelCoordsGPU,voxelNuminFinalIter,foreGroundParmGPU,foregroundVect.size(),projectMatParmGPU,(int)camCenters.size(),camCenterParmGPU,occupancyOutputGPU);
			error = cudaGetLastError();
			if(error != cudaSuccess)
			{
				printf("####################### CUDA Error 10 : %s ###################\n", cudaGetErrorString(error));
				cudaFree(foreGroundParmGPU);
				cudaFree(voxelCoordsGPU);
				cudaFree(projectMatParmGPU);
				cudaFree(occupancyOutputGPU);
				return false;
			}

			//dst.download(resultImage);
			//ImageSC2(resultImage);
			cudaDeviceSynchronize();
			cudaMemcpy(&occupancyOutput[i*voxelSegNum],occupancyOutputGPU,voxelNuminFinalIter*sizeof(float),cudaMemcpyDeviceToHost);
			error = cudaGetLastError();
			if(error != cudaSuccess)
			{
				printf("####################### CUDA Error 11 : %s ###################\n", cudaGetErrorString(error));
				cudaFree(foreGroundParmGPU);
				cudaFree(voxelCoordsGPU);
				cudaFree(projectMatParmGPU);
				cudaFree(occupancyOutputGPU);
				return false;
			}
		}
		else
		{
			float* tempVoxelPtr  = &voxelCoords[i*voxelSegNum*3];//i*voxelSegNum];
			cudaMemcpy(voxelCoordsGPU,tempVoxelPtr,voxelSegNum*sizeof(float)*3,cudaMemcpyHostToDevice);

			dim3 block(512,1);
			dim3 grid(divUp(voxelSegNum,block.x), 1);
			if(grid.x>65535)
			{
				//printf("GPU:: grid size is too big !! (%d,%d)\n",grid.x,grid.y);
				grid.x = 65535;
			}
			//else
				//printf("GPU:: grid size : (%d,%d)\n",grid.x,grid.y);

			CalcVolumeCostAverage_hd_float<<<grid, block>>>(voxelCoordsGPU,voxelSegNum,foreGroundParmGPU,foregroundVect.size(),projectMatParmGPU,occupancyOutputGPU);
			cudaGetLastError();
			//dst.download(resultImage);
			//ImageSC2(resultImage);
			cudaDeviceSynchronize();
			cudaMemcpy(&occupancyOutput[i*voxelSegNum],occupancyOutputGPU,voxelSegNum*sizeof(float),cudaMemcpyDeviceToHost);

			cudaError_t error= cudaGetLastError();

			if(error != cudaSuccess)
			{
				// something's gone wrong
				// print out the CUDA error as a string
				printf("CUDA Error: %s\n", cudaGetErrorString(error));
			}
		}
		//printf("CUDA Iteration %d/%d\n",i,iterNum);//
	}

	//delete[] voxelCoords;
	cudaFree(voxelCoordsGPU);
	cudaFree(foreGroundParmGPU);
	cudaFree(projectMatParmGPU);
	cudaFree(occupancyOutputGPU);

	cudaMemGetInfo(&free,&total);
	//printf("## CUDA:: return :: free%f MB, total %f MB\n",free/1e6,total/1e6);

	return true;
}
