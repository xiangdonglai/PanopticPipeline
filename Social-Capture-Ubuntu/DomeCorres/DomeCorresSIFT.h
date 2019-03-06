#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>

#include <omp.h>

#include <boost/algorithm/string.hpp>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>



// #include <windows.h>
// #include <ImageHlp.h>

#include "BAutil.h"


#ifdef NDEBUG
#define EIGEN_NO_DEBUG
#endif
#include <Eigen/Core>
#include <boost/filesystem.hpp>

void FindCorresSIFT(const std::string projimgfile, const std::vector<std::string> &devicenames, const int nProjectors, const int nDevices, const std::string camfolder, std::vector< std::vector<Eigen::Vector2d> > &PCpts, const int equalize_method = 0);
void drawMatchSIFT(const std::string imgname1, const std::string imgname2, std::vector<cv::KeyPoint> &keys1, std::vector<cv::KeyPoint> &keys2, std::vector<cv::DMatch> matches);
void drawMatchSIFT(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> &keys1, std::vector<cv::KeyPoint> &keys2, std::vector<cv::DMatch> matches);



inline void matchKNN_cross(const cv::Mat &descriptors1, const cv::Mat &descriptors2, std::vector<cv::DMatch> &matches);
void reduceSmallFeaturePoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, float size_threshold);



void removeLessObserevation(const int nProjectors, const int nDevices, const int minViews, std::vector< std::vector<Eigen::Vector2d> > &PCpts);


bool readPCcorres(const std::string filename, const int nDevices, std::vector< std::vector<Eigen::Vector2d> > &PCpts);
bool readPCcorresSparse(const std::string filename, std::vector< std::vector<Eigen::Vector2d> > &PCpts);

bool writePCcorres(const std::string filename, const int nDevices, const std::vector< std::vector<Eigen::Vector2d> > &PCpts);
bool writePCcorresSparse(const std::string filename, const int nDevices, const std::vector< std::vector<Eigen::Vector2d> > &PCpts);

bool writePCcorresTri(const std::string filename, const std::vector< std::vector<Eigen::Vector2d> > &PCpts, const int nPoints, const int nProjectors, const int nVGAs, const int nPanels, const int nHDs, const int nDevices);
bool writeProjectorFeatures(const std::string filename, const std::vector< std::vector<Eigen::Vector2d> > &PCpts, const int nProjectors);

bool writeSIFTformatASCII(const std::string filename, const std::vector< std::vector<Eigen::Vector2d> > &PCpts, const int nPoints, const int devID);
bool writeSIFTformatBinary(const std::string filename, const std::vector< std::vector<Eigen::Vector2d> > &PCpts, const int nPoints, const int devID);
bool writeFeatureMatches(const std::string vsfmfolder, const std::string filename, const std::vector<std::string> &devicenames, const std::vector< std::vector<Eigen::Vector2d> > &PCpts, const int nProjectors, const int nPoints, const int nDevices);

bool writeDeviceName(const std::string filename, const std::string extension, const std::vector<std::string> devicenames);
bool writeNumProjPoints(const std::string filename, const std::vector< std::vector<Eigen::Vector2d> > &PCpts, const std::vector<std::string> devicenames, const int nProjectors);


bool GenerateVisualSFMinput(const std::string imgfolder, const std::vector<std::string> &devicenames, const std::vector< std::vector<Eigen::Vector2d> > &PCpts, const int nProjectors, const int nHDs, const int nVGAs, const int nPanels);
//bool GenerateVisualSFMinput(const std::vector<BA::NVM>& nvmDataVect,const int nHDs, const int nVGAs, const char* outputFolderName);
bool GenerateVisualSFMinput(const std::vector<BA::NVM>& nvmDataVect,const char* outputFolderName);


void updateDetectMap(const int nDevices, const int nAllPoints, const std::vector< std::vector<Eigen::Vector2d> > &PCpts);

void normalizeImg_nonlin(const cv::Mat &src_, cv::Mat &dst_);
void equalizeHist2(const cv::Mat &src_, cv::Mat &dst_);

