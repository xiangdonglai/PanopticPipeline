#pragma once
#include "opencv2/core/core.hpp"
#include "Constants.h"
#include <cuda_runtime.h>
#include <vector>
using namespace std;

bool DetectionCostVolumeGeneration_float_GPU_average_vga(float* voxelCoords,int voxelNum, std::vector<cv::Mat_<float>>& foregroundVect, std::vector<cv::Mat_<float>>& projectMatVect,float* occupancyOutput);
bool DetectionCostVolumeGeneration_float_GPU_average_hd(float* voxelCoords,int voxelNum, std::vector<cv::Mat_<float>>& foregroundVect, std::vector<cv::Mat_<float>>& projectMatVect,float* occupancyOutput,bool bOnlyPositiveValueAvg=false);
