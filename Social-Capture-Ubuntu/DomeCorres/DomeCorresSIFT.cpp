#include "DomeCorresSIFT.h"
#include "strlib.h"

typedef long long _Longlong;
typedef unsigned long long _ULonglong;

using Eigen::Vector2d;
using namespace std;

int 
	DISPLAY_W = 1920,
	DISPLAY_H = 1080;

string ZEROS128;

Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> detectMap;
Vector2d initvec(0.0, 0.0);



//#define USE_BUFFER_FOR_OFSTREAM

#define WRITE_SIFT_BINARY
#ifndef WRITE_SIFT_BINARY
#define WRITE_SIFT_ASCII
#endif


#ifdef USE_BUFFER_FOR_OFSTREAM
const int buff_size = 30000000;
#endif



//descriptor1: query, 2: train
void matchKNN_cross(const cv::Mat &descriptors1, const cv::Mat &descriptors2, vector<cv::DMatch> &matches)
{
	matches.clear();

	if (descriptors1.rows == 0 || descriptors2.rows == 0)
		return;


	cv::FlannBasedMatcher matcher;
	vector<vector<cv::DMatch> > matches12, matches21;
	int knn = 1;
	matcher.knnMatch(descriptors1, descriptors2, matches12, knn);
	matcher.knnMatch(descriptors2, descriptors1, matches21, knn);
	

	size_t nMatches12 = matches12.size();
	for (size_t m = 0; m < nMatches12; m++)
	{
		bool findCrossCheck = false;
		for (size_t fk = 0; fk < matches12[m].size(); fk++)
		{
			cv::DMatch forward = matches12[m][fk];

			size_t nMatches21 = matches21[forward.trainIdx].size();
			for (size_t bk = 0; bk < nMatches21; bk++)
			{
				cv::DMatch backward = matches21[forward.trainIdx][bk];
				if (backward.trainIdx == forward.queryIdx)
				{
					matches.push_back(forward);
					findCrossCheck = true;
					break;
				}
			}
			if (findCrossCheck) break;

		}
	}


}

void reduceSmallFeaturePoints(vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, float size_threshold)
{
	size_t nPoints = keypoints.size();
	vector<bool> used(nPoints, false);

	int nUsed = 0;
	vector<cv::KeyPoint> new_keypoints;
	for (size_t ptID = 0; ptID < nPoints; ptID++)
	{
		if (keypoints[ptID].size > size_threshold)
		{
			used[ptID] = true;
			new_keypoints.push_back(keypoints[ptID]);
			nUsed++;
		}
	}

	int row = 0;
	cv::Mat new_descriptors(nUsed, 128, CV_32F);
	for (size_t ptID = 0; ptID < nPoints; ptID++)
	{
		if (used[ptID])
		{
			descriptors.row(ptID).copyTo(new_descriptors.row(row));
			row++;
		}
	}

	keypoints = new_keypoints;
	descriptors = new_descriptors.clone();
}

void FindCorresSIFT(const string projimgfile, const vector<string> &devicenames, const int nProjectors, const int nDevices, const string camfolder, vector< vector<Vector2d> > &PCpts, const int equalize_method)
{
	//SIFT points in Projector image
	const int 
		widthP  = 1280,
		heightP =  800;

	cv::Mat projimg = cv::imread(projimgfile, CV_LOAD_IMAGE_GRAYSCALE);
	vector<cv::KeyPoint> keypoints_proj;
	cv::Mat              descriptors_proj;
	cv::SIFT sift;


	if (projimg.size() != cv::Size(widthP, heightP))
		cv::resize(projimg, projimg, cv::Size(widthP, heightP));


	sift(projimg, cv::Mat(), keypoints_proj, descriptors_proj);
	//reduceSmallFeaturePoints(keypoints_proj, descriptors_proj, 4.f);


	//fill points on the projectors
	int nProjPoints = keypoints_proj.size();
	int nAllPoints  = nProjectors*nProjPoints;
	PCpts.clear();
	PCpts = vector<vector<Vector2d>>(nAllPoints, vector<Vector2d>(nDevices, initvec));//outliers are (-1, -1)

	#pragma omp parallel for
	for (int projID = 0; projID < nProjectors; projID++)
	{
		for (int ptID = 0; ptID < nProjPoints; ptID++)
		{
			int ptID2 = ptID + projID*nProjPoints;
			PCpts[ptID2][projID] = Vector2d(keypoints_proj[ptID].pt.x, keypoints_proj[ptID].pt.y);
		}
	}


	//----find corresponding points-----//
	cout << "Detecting SIFT points..." << endl;

	for (int projID = 0; projID < nProjectors; projID++)
	{
		int offset_ptID = projID*nProjPoints;

		#pragma omp parallel for
		for (int devID = nProjectors; devID < nDevices; devID++)
		{

			double start_time = omp_get_wtime();

			vector<cv::KeyPoint> keypoints_cam;
			cv::Mat              descriptors_cam;

			string  camimgfile = camfolder + strsprintf("/p%d/", projID) + devicenames[devID] + ".png";
			//string  pointsfile = camfolder + strsprintf("/p%d/", projID) + devicenames[devID] + "C.txt";
			cv::Mat camimg = cv::imread(camimgfile, CV_LOAD_IMAGE_GRAYSCALE);

			if (camimg.empty())
			{
				cout << "Warning: Cannot open " << camimgfile << endl;
				continue;
			}
			//ofstream ofs(pointsfile);
			//if (ofs.fail())
			//{
			//	cerr << "Warning: Cannot open " << pointsfile << endl;
			//	continue;
			//}


			switch (equalize_method)
			{
			case 0:
				cv::equalizeHist(camimg, camimg);
				break;
			case 1:
				normalizeImg_nonlin(camimg, camimg);
				break;
			case 2:
				equalizeHist2(camimg, camimg);
				break;
			}


			sift(camimg, cv::Mat(), keypoints_cam, descriptors_cam);

			vector<cv::DMatch> matches;
			matchKNN_cross(descriptors_cam, descriptors_proj, matches);
			//matchKNN_cross(descriptors_proj, descriptors_cam, matches);

			//refineMatchesECC(projimg, camimg, keypoints_proj, keypoints_cam, matches);

			int nMatches = matches.size();

			printf("\t[%d, %s]: %4d matches (%.3f sec)\n", projID, devicenames[devID].c_str(), nMatches, omp_get_wtime() - start_time);

			//drawMatchSIFT(projimgfile, camimgfile, keypoints_proj, keypoints_cam, matches);


			//ofs << fixed << setprecision(8);
			for (int matchID = 0; matchID < nMatches; matchID++)
			{
				int trainID = matches[matchID].trainIdx;
				int queryID = matches[matchID].queryIdx;


				double u = keypoints_cam[queryID].pt.x;
				double v = keypoints_cam[queryID].pt.y;

				PCpts[offset_ptID + trainID][devID] = Vector2d(u, v);
				//ofs << trainID << " " << queryID << " " << u << " " << v << "\n";

			}

		}
	}
	cout << "Done." << endl;

}

void drawMatchSIFT(const std::string imgname1, const std::string imgname2, std::vector<cv::KeyPoint> &keys1, std::vector<cv::KeyPoint> &keys2, std::vector<cv::DMatch> matches)
{
	cv::Mat img1 = cv::imread(imgname1, CV_LOAD_IMAGE_COLOR);
	cv::Mat img2 = cv::imread(imgname2, CV_LOAD_IMAGE_COLOR);

	if (img1.empty())
	{
		cerr << "Failed to open " << imgname1 << endl;
		return;
	}

	if (img2.empty())
	{
		cerr << "Failed to open " << imgname2 << endl;
		return;
	}
	drawMatchSIFT(img1, img2, keys1, keys2, matches);
}


void drawMatchSIFT(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> &keys1, std::vector<cv::KeyPoint> &keys2, std::vector<cv::DMatch> matches)
{
	cv::Mat 
		img1_color, 
		img2_color,
		dst;

	if (img1.channels() == 1)
		img1.convertTo(img1_color, CV_GRAY2BGR);
	else
		img1_color = img1;

	if (img2.channels() == 1)
		img2.convertTo(img2_color, CV_GRAY2BGR);
	else
		img2_color = img2;


	cv::drawMatches(img1_color, keys1, img2_color, keys2, matches, dst);


	if (dst.cols > DISPLAY_W || dst.rows > DISPLAY_H)
	{
		double s = min(DISPLAY_W / (double)(dst.cols), DISPLAY_H / (double)(dst.rows));
		cv::resize(dst, dst, cv::Size(), s, s);
	}

	cv::imshow("OpenCV SIFT match result", dst);
	cv::waitKey(0);

}



bool writeDeviceName(const std::string filename, const std::string extension, const std::vector<std::string> devicenames)
{
	ofstream ofs(filename);
	if (ofs.fail())
	{
		cout << "Cannot open to write: " << filename << endl;
		return false;
	}

	for (size_t i = 0; i < devicenames.size(); i++)
		ofs << devicenames[i] + extension << "\n";


	return true;
}

bool writeNumProjPoints(const std::string filename, const std::vector< std::vector<Eigen::Vector2d> > &PCpts, const vector<string> devicenames, const int nProjectors)
{
	ofstream ofs(filename);
	if (ofs.fail())
	{
		cout << "Cannot open to write: " << filename << endl;
		return false;
	}


	for (int projID = 0; projID < nProjectors; projID++)
	{

		int npoints = 0;
		for (size_t ptID = 0; ptID < PCpts.size(); ptID++)
		{
			Vector2d uv = PCpts[ptID][projID];

			if (uv == initvec)
				continue;

			npoints++;
		}

		ofs << devicenames[projID] <<".jpg "<< npoints << "\n";
	}


	return true;
}


bool writePCcorresSparse(const string filename, const int nDevices, const vector< vector<Vector2d> > &PCpts)
{

	ofstream ofs;

#ifdef USE_BUFFER_FOR_OFSTREAM
	vector<char> buf(buff_size);
	ofs.rdbuf()->pubsetbuf(buf.data(), buff_size);
#endif

	ofs.open(filename);
	if (ofs.fail())
	{
		cout << "Cannot open to write: " << filename << endl;
		return false;
	}
	
	const int nPoints = PCpts.size();


	ofs << setprecision(10);
	ofs << nPoints << " " << nDevices << "\n";

	for (int ptID = 0; ptID < nPoints; ptID++)
	{
		for (int devID = 0; devID < nDevices; devID++)
		{
			double u = PCpts[ptID][devID](0);
			double v = PCpts[ptID][devID](1);
			if (u>0 && v>0)
				ofs << ptID << " " << devID << " " << u << " " << v << "\n";
		}
	}

	return true;
}


bool writePCcorres(const string filename, const int nDevices, const vector< vector<Vector2d> > &PCpts)
{

	ofstream ofs;

#ifdef USE_BUFFER_FOR_OFSTREAM
	vector<char> buf(buff_size);
	ofs.rdbuf()->pubsetbuf(buf.data(), buff_size);
#endif

	ofs.open(filename);
	if (ofs.fail())
	{
		cout << "Cannot open to write: " << filename << endl;
		return false;
	}

	int nPoints = PCpts.size();

	ofs << setprecision(10);
	for (int ptID = 0; ptID < nPoints; ptID++)
	{
		for (int devID = 0; devID < nDevices; devID++)
		{
			ofs << PCpts[ptID][devID](0) << " " << PCpts[ptID][devID](1) << " ";
		}
		ofs << "\n";
	}

	return true;
}

bool writePCcorresTri(const string filename, const vector< vector<Vector2d> > &PCpts, const int nPoints, const int nProjectors, const int nVGAs, const int nPanels, const int nHDs, const int nDevices)
{
	ofstream ofs;

#ifdef USE_BUFFER_FOR_OFSTREAM
	vector<char> buf(buff_size);
	ofs.rdbuf()->pubsetbuf(buf.data(), buff_size);
#endif

	ofs.open(filename);
	if (ofs.fail())
	{
		cout << "Cannot open to write: " << filename << endl;
		return false;
	}


	ofs << nDevices << "\n";
	for (int projID = 0; projID < nProjectors; projID++)
		ofs << strsprintf("p%d.jpg\n", projID);

	for (int panelID = 0; panelID <= nPanels; panelID++)
	{
		int nCameras_on_panel = panelID == 0 ? nHDs : nVGAs;

		for (int camID = 0; camID < nCameras_on_panel; camID++)
		{
			ofs << strsprintf("%02d_%02d.jpg\n", panelID, camID);
		}
	}


	ofs << "\n"
		<< nPoints
		<< "\n";
	//ofs << fixed << setprecision(8);
	ofs << setprecision(10);
	for (int ptID = 0; ptID < nPoints; ptID++)
	{
		for (int devID = 0; devID < nDevices; devID++)
		{
			double u = -1.0, v = -1.0;
			if (detectMap(ptID, devID))
			{
				u = PCpts[ptID][devID](0);
				v = PCpts[ptID][devID](1);
			}

			ofs << u << " " << v << " ";
		}
		ofs << "\n";
	}

	return true;
}

bool writeSIFTformatASCII(const string filename, const vector< vector<Vector2d> > &PCpts, const int nPoints, const int devID)
{
	ofstream ofs;

#ifdef USE_BUFFER_FOR_OFSTREAM
	vector<char> buf(buff_size);
	ofs.rdbuf()->pubsetbuf(buf.data(), buff_size);
#endif

	ofs.open(filename);
	if (ofs.fail())
	{
		cout << "Cannot open to write: " << filename << endl;
		return false;
	}


	//string zeros;
	//for (int i = 0; i < 128; i++)
	//	zeros += "0 ";


	ofs << nPoints << " 128\n";
	//ofs << fixed << setprecision(8);
	ofs << setprecision(10);
	for (int ptID = 0; ptID < nPoints; ptID++)
	{
		ofs << PCpts[ptID][devID](0) << " " << PCpts[ptID][devID](1) << " 0.0 0.0\n"
			<< ZEROS128
			<< "\n";
	}

	return true;
}

bool writeSIFTformatBinary(const string filename, const vector< vector<Vector2d> > &PCpts, const int nPoints, const int devID)
{
	ofstream ofs;

#ifdef USE_BUFFER_FOR_OFSTREAM
	vector<char> buf(buff_size);
	ofs.rdbuf()->pubsetbuf(buf.data(), buff_size);
#endif

	ofs.open(filename, ios::binary);
	if (ofs.fail())
	{
		cout << "Cannot open to write: " << filename << endl;
		return false;
	}

	int name    = ('S' + ('I' << 8) + ('F' << 16) + ('T' << 24));
	int version = ('V' + ('4' << 8) + ('.' << 16) + ('0' << 24));

	int header[5] = {name, version, nPoints, 5, 128};
	for (int i = 0; i < 5; i++)
		ofs.write((char *)&header[i], sizeof(int));

	vector<float> cso(3, 0.0);//color, scale, orientation
	for (int ptID = 0; ptID < nPoints; ptID++)
	{
		float u = (float)PCpts[ptID][devID](0);
		float v = (float)PCpts[ptID][devID](1);

		ofs.write((char *)&u, sizeof(float));
		ofs.write((char *)&v, sizeof(float));
		ofs.write((char *)cso.data(), 3*sizeof(float));
	}

	vector<uchar> zeros(128*nPoints, 0);
	ofs.write((char *)zeros.data(), 128 * nPoints* sizeof(uchar));

	return true;
}

bool writeProjectorFeatures(const string filename, const vector< vector<Vector2d> > &PCpts, const int nProjectors)
{
	ofstream ofs(filename);
	if (ofs.fail())
	{
		cout << "Cannot open to write: " << filename << endl;
		return false;
	}



	int nPoints = PCpts.size() / nProjectors;
	int devID   = 0;

	ofs << fixed << setprecision(12);
	ofs << nPoints << "\n";
	for (int ptID = 0; ptID < nPoints; ptID++)
	{
		double u = PCpts[ptID][devID](0);
		double v = PCpts[ptID][devID](1);

		if (ptID == nPoints - 1)
			ofs << u << ' ' << v;
		else
			ofs << u << ' ' << v << '\n';
		//ofs << u << ' ' << v << string(ptID == nPoints - 1 ? "" : "\n");
	}

}

bool writeProjectorFeatures2(const string filename, const vector< vector<Vector2d> > &PCpts, const int nProjectors)
{

	int nMaxPoints = PCpts.size();

	for (int projID = 0; projID < nProjectors; projID++)
	{
		stringstream ss;
		ss << setprecision(10);

		int npoints = 0;
		for (int ptID = 0; ptID < nMaxPoints; ptID++)
		{
			Vector2d uv = PCpts[ptID][projID];

			if (uv == initvec)
				continue;

			ss << uv[0] << ' ' << uv[1] << '\n';
			npoints++;
		}




		ofstream ofs(filename + to_string((_Longlong)projID) + ".txt");
		if (ofs.fail())
		{
			cout << "Cannot open to write: " << filename << endl;
			return false;
		}
		ofs << npoints << "\n";
		ofs << ss.rdbuf();

	}

	return true;
}


bool writeFeatureMatches(const string vsfmfolder, const string filename, const vector<string> &devicenames, const vector< vector<Vector2d> > &PCpts, const int nProjectors, const int nPoints, const int nDevices)
{
	string matchesfile = vsfmfolder + "/" + filename;

	ofstream ofs;

#ifdef USE_BUFFER_FOR_OFSTREAM
	vector<char> buf(buff_size);
	ofs.rdbuf()->pubsetbuf(buf.data(), buff_size);
#endif

	ofs.open(matchesfile);
	if (ofs.fail())
	{
		cout << "Cannot open to write: " << matchesfile << endl;
		return false;
	}

	for (int dev1 = 0; dev1 < nDevices - 1; dev1++)
	{
		for (int dev2 = dev1 + 1; dev2 < nDevices; dev2++)
		{
			string imgname1 = devicenames[dev1] + ".jpg";
			string imgname2 = devicenames[dev2] + ".jpg";

			string matchIDs;
			int nMatch = 0;

			for (int ptID = 0; ptID < nPoints; ptID++)
			{
				if (detectMap(ptID, dev1) && detectMap(ptID, dev2) )
				{
					matchIDs += to_string((_ULonglong)ptID) + " ";
					nMatch++;
				}
			}

			ofs << imgname1 << " " << imgname2 << " " << nMatch << "\n"
				<< matchIDs << "\n"
				<< matchIDs << "\n\n";
		}
	}
	return true;
}






void updateDetectMap(const int nDevices, const int nAllPoints, const vector< vector<Vector2d> > &PCpts)
{
	detectMap.resize(nAllPoints, nDevices);

	for (int ptID = 0; ptID < nAllPoints; ptID++)
	{
		for (int devID = 0; devID < nDevices; devID++)
		{
			//detectMap(ptID, devID) = (PCpts[ptID][devID](0) > 0 && PCpts[ptID][devID](1) > 0) ? true : false;
			detectMap(ptID, devID) = (PCpts[ptID][devID](0) > 0);
		}
	}
}

void removeLessObserevation(const int nProjectors, const int nDevices, const int minViews, vector< vector<Vector2d> > &PCpts)
{
	
	size_t nPoints = PCpts.size();
	vector< vector<Vector2d> > new_PCpts;
	new_PCpts.reserve(nPoints);


	for (size_t ptID = 0; ptID < nPoints; ptID++)
	{
		int nviews = nDevices - 1 - std::count(PCpts[ptID].begin(), PCpts[ptID].end(), initvec);
		if (nviews >= minViews)
			new_PCpts.push_back(PCpts[ptID]);

	}

	PCpts = std::move(new_PCpts);
}



bool GenerateVisualSFMinput(const string imgfolder, const vector<string> &devicenames, const vector< vector<Vector2d> > &PCpts, const int nProjectors, const int nHDs, const int nVGAs, const int nPanels)
{
	string vsfmfolder = imgfolder + "/vsfm/";
	// CreateDirectory(vsfmfolder.c_str(),NULL);
	boost::filesystem::create_directory(boost::filesystem::path(vsfmfolder));

	int nCameras   = nPanels*nVGAs + nHDs;
	int nDevices   = nProjectors + nCameras;
	int nAllPoints = PCpts.size();

	string filename;
	double start_time;

	updateDetectMap(nDevices, nAllPoints, PCpts);

	//cout << "Writing feature points of the projector image...";
	//start_time = omp_get_wtime();
	//filename = imgfolder + "/ProjPoints.txt";
	//if (!writeProjectorFeatures(filename, PCpts, nProjectors))
	//	return false;
	//cout << "Done." << "(" << omp_get_wtime() - start_time << " sec)" << endl;


	cout << "Writing feature points of the projector image...";
	start_time = omp_get_wtime();
	filename = vsfmfolder + "/ProjPoints";
	if (!writeProjectorFeatures2(filename, PCpts, nProjectors))
		return false;
	cout << "Done." << "(" << omp_get_wtime() - start_time << " sec)" << endl;


	//cout << "Writing PCcorres.txt...";
	//start_time = omp_get_wtime();
	////filename = imgfolder + "/PCcorres.txt";
	////if (!writePCcorres(filename, nDevices, PCpts))
	////	return false;

	// filename = imgfolder + "/PCcorresSparse.txt";
	// if (!writePCcorresSparse(filename, nDevices, PCpts))
		// return false;

	//filename = imgfolder + "/ProjPoints.txt";
	//if (!writeNumProjPoints(filename, PCpts, devicenames, nProjectors))
	//	return false;

	//filename = imgfolder + "/DeviceNames.txt";
	//if (!writeDeviceName(filename, ".jpg" ,devicenames))
	//	return false;
	//cout << "Done." << "(" << omp_get_wtime() - start_time << " sec)" << endl;




	//cout << "Writing PCcorres_Tri.txt...";
	//start_time = omp_get_wtime();
	//filename = imgfolder + "/PCcorres_Tri.txt";
	//if (!writePCcorresTri(filename, PCpts, nAllPoints, nProjectors, nVGAs, nPanels, nHDs, nDevices))
	//	return false;
	//cout << "Done." << "(" << omp_get_wtime() - start_time << " sec)" << endl;




	cout << "Writing points in SIFT format...";
	start_time = omp_get_wtime();

#ifdef WRITE_SIFT_ASCII
	for (int i = 0; i < 128; i++)
		ZEROS128 += "0 ";
#endif

	for (int devID = 0; devID < nDevices; devID++)
	{
		filename = vsfmfolder + "/" + devicenames[devID] + ".sift";
#ifdef WRITE_SIFT_ASCII
		if (!writeSIFTformatASCII(filename, PCpts, nAllPoints, devID))
			return false;
#else
		if (!writeSIFTformatBinary(filename, PCpts, nAllPoints, devID))
			return false;
#endif
	}
	cout << "Done." << "(" << omp_get_wtime() - start_time << " sec)" << endl;


	cout << "Writing FeatureMatches.txt...";
	start_time = omp_get_wtime();
	if (!writeFeatureMatches(vsfmfolder, "FeatureMatches.txt", devicenames, PCpts, nProjectors, nAllPoints, nDevices))
		return false;
	cout << "Done." << "(" << omp_get_wtime() - start_time << " sec)" << endl;



	return true;
}

bool readPCcorres(const string filename, const int nDevices, std::vector< std::vector<Eigen::Vector2d> > &PCpts)
{
	ifstream ifs(filename);
	if (ifs.fail())
	{
		cout << "Cannot open : " << filename << endl;
		return false;
	}


	stringstream ss;
	ss << ifs.rdbuf();
	ifs.close();


	//initialize PCpts
	if (!PCpts.empty())
		for (size_t i = 0; i < PCpts.size(); i++)
			PCpts[i].clear();


	vector<Vector2d> corres(nDevices);
	string str;
	while (getline(ss, str))
	{

		vector<string> str_vec;
		boost::algorithm::split(str_vec, str, boost::is_space());

		if (str_vec.size() / 2 != nDevices)
		{
			cerr << "Error: Num of colums of the PCcorres file does not match nDevices!!" << endl;
			return false;
		}


		for (int devID = 0; devID < nDevices; devID++)
		{
			corres[devID][0] = stod(str_vec[2 * devID]);
			corres[devID][1] = stod(str_vec[2 * devID + 1]);
		}


		PCpts.push_back(corres);
	}


	return true;
}

bool readPCcorresSparse(const string filename, std::vector< std::vector<Eigen::Vector2d> > &PCpts)
{
	ifstream ifs(filename);
	if (ifs.fail())
	{
		cout << "Cannot open : " << filename << endl;
		return false;
	}


	stringstream ss;
	ss << ifs.rdbuf();
	ifs.close();


	//initialize PCpts
	if (!PCpts.empty())
		for (size_t i = 0; i < PCpts.size(); i++)
			PCpts[i].clear();

	int nPoints, nDevices;
	ss >> nPoints >> nDevices;
	ss.get();

	PCpts = vector<vector<Vector2d>>(nPoints, vector<Vector2d>(nDevices, initvec));


	string str;
	while (getline(ss, str))
	{
		vector<string> str_vec;
		boost::algorithm::split(str_vec, str, boost::is_space());
		
		int ptID  = stoi(str_vec[0]),
			devID = stoi(str_vec[1]);

		PCpts[ptID][devID][0] = stod(str_vec[2]);
		PCpts[ptID][devID][1] = stod(str_vec[3]);
	}


	return true;
}

void normalizeImg_nonlin(const cv::Mat &src_, cv::Mat &dst_)
{
	cv::Mat src, dst;
	src_.convertTo(src, CV_64F);


	cv::Scalar mean, std;
	cv::meanStdDev(src, mean, std);

	cv::exp((mean[0] - src) / std[0], dst);
	dst = 255.0 / (1.0 + dst);


	dst.convertTo(dst_, CV_8U);
}

#define round(x) int(x+0.5)
void equalizeHist2(const cv::Mat &src_, cv::Mat &dst_)
{
	cv::Mat src, dst;

	src = src_.clone();
	cv::Mat src_vec = src.reshape(0, 1);
	cv::sort(src_vec, src_vec, CV_SORT_ASCENDING);

	int q0025 = (int)round(0.025*src_vec.cols);
	int q0975 = (int)round(0.975*src_vec.cols);

	double minval = (double)src_vec.at<uchar>(q0025);
	double maxval = (double)src_vec.at<uchar>(q0975);

	src_.convertTo(src, CV_64F);
	dst = (src - minval)*255.0 / (maxval - minval);

	dst.convertTo(dst_, CV_8U);
}
/*
int GetDeviceIdx(int panelIdx,int camIdx)
{
	if(panelIdx>0)
		return (panelIdx-1)*24 + (camIdx-1);
	else
		return 480 + camIdx;
}

void GetDeviceName(int deviceIdx,char* deviceNameString)
{
	int panelIdx,camIdx;
	if(deviceIdx<480)
	{
		panelIdx = deviceIdx /24 + 1;
		camIdx = deviceIdx % 24 + 1;
	}
	else
	{
		panelIdx = 0;
		camIdx = deviceIdx - 480;
	}
	sprintf(deviceNameString,"%02d_%02d",panelIdx,camIdx);
}*/

//bool GenerateVisualSFMinput(const std::vector<BA::NVM>& nvmDataVect,const int nHDs, const int nVGAs, const char* outputFolderName)
bool GenerateVisualSFMinput(const std::vector<BA::NVM>& nvmDataVect,const char* outputFolderName)
{
	//Concat all the device names
	map<string,int> fileNameToDevIdx;
	vector< string > devIdxToFileName;
	for(int nvmIdx=0;nvmIdx<nvmDataVect.size();++nvmIdx)
	{
		for(int i=0;i<nvmDataVect[nvmIdx].filenames.size();++i)
		{
			std::string fileName = nvmDataVect[nvmIdx].filenames[i];
			if(fileNameToDevIdx.find(fileName) == fileNameToDevIdx.end())
			{
				int devIdx = fileNameToDevIdx.size();
				fileNameToDevIdx[fileName] = devIdx;
				std::size_t n = fileName.rfind(".");
				fileName = fileName.substr(0,n);
				//printf("%s\n",fileName.c_str());
				devIdxToFileName.push_back(fileName);  //devIdxToFileName[devIdx] =  fileName. Is devIdxToFileName necessary?? maybe we can get the same info from fileNameToDevIdx. 
			}
		}
	}

	int nDevices = fileNameToDevIdx.size();

	int nAllPoints =0;
	for(int nvmIdx=0;nvmIdx<nvmDataVect.size();++nvmIdx)
		nAllPoints += nvmDataVect[nvmIdx].nInliers;
	Vector2d initvec(-1, -1);
	vector < vector < Vector2d> > PCpts = vector<vector<Vector2d>>(nAllPoints, vector<Vector2d>(nDevices, initvec));//outliers are (-1, -1)		[mergedPtIdx][deviceIdx]
	
	int ptIdxOffset =0;
	for(int nvmIdx=0;nvmIdx<nvmDataVect.size();++nvmIdx)
	{
		int inlierPtIdx =0;
		for(int ptIdx=0;ptIdx < nvmDataVect[nvmIdx].pointdata.size(); ++ptIdx)
		{
			if( nvmDataVect[nvmIdx].pointdata[ptIdx].inlier.size()>0  
				&& nvmDataVect[nvmIdx].pointdata[ptIdx].inlier.front() ==false)		//only consider 3+ points
				continue;

			int inlierPtIdx_withOffset = ptIdxOffset + inlierPtIdx;
			for(int i=0;i<nvmDataVect[nvmIdx].pointdata[ptIdx].camID.size();++i)
			{
				int camIdxInFile = nvmDataVect[nvmIdx].pointdata[ptIdx].camID[i];
				std::string camName = nvmDataVect[nvmIdx].filenames[camIdxInFile];		

				std::string panelStr = camName.substr (0,2);     
				//std::string camStr = camName.substr (3,2);     
				//printf("name %s, panelStr %s, camStr %s\n",camName.c_str(),panelStr.c_str(),camStr.c_str());
				int panelIdx = stoi(panelStr);
				//int camIdx = stoi(camStr);
				//int deviceIdx = GetDeviceIdx(panelIdx,camIdx);
				int deviceIdx = fileNameToDevIdx[camName];

				PCpts[inlierPtIdx_withOffset][deviceIdx] = nvmDataVect[nvmIdx].pointdata[ptIdx].uv[i];

				if(panelIdx==0 || panelIdx==50)			//HD or Kinect
				{
					PCpts[inlierPtIdx_withOffset][deviceIdx](0) += 960;		//1920/2
					PCpts[inlierPtIdx_withOffset][deviceIdx](1) += 540;		//1080/2
				}
				else
				{
					PCpts[inlierPtIdx_withOffset][deviceIdx](0) += 320;
					PCpts[inlierPtIdx_withOffset][deviceIdx](1) += 240;
				}

				 
			}

			inlierPtIdx++;
		}

		ptIdxOffset += inlierPtIdx;
	}

	printf("check: nAllPoints %d, ptIdxOffset %d\n",nAllPoints, ptIdxOffset);

	updateDetectMap(nDevices, nAllPoints, PCpts);

	/*
	cout << "Writing feature points of the projector image...";
	start_time = omp_get_wtime();
	filename = vsfmfolder + "/ProjPoints";
	if (!writeProjectorFeatures2(filename, PCpts, nProjectors))
		return false;
	cout << "Done." << "(" << omp_get_wtime() - start_time << " sec)" << endl;
	*/

	//vector<string> devicenames;
	cout << "Writing points in SIFT format...";
	for (int devID = 0; devID < devIdxToFileName.size(); devID++)
	{
		char filename[512];
		//char camNameStr[512];
		//GetDeviceName(devID,camNameStr);
		//devicenames.push_back(camNameStr);
		sprintf(filename,"%s/%s.sift",outputFolderName,devIdxToFileName[devID].c_str()); 
		printf("fileName: %s\n",filename);
		//if (!writeSIFTformatASCII(filename, PCpts, nAllPoints, devID))
		if (!writeSIFTformatBinary(filename, PCpts, nAllPoints, devID))
			return false;
	}
	/*
	cout << "Done." << "(" << omp_get_wtime() - start_time << " sec)" << endl;
	*/
	cout << "Writing FeatureMatches.txt...";
	//start_time = omp_get_wtime();
	if (!writeFeatureMatches(outputFolderName, "FeatureMatches.txt", devIdxToFileName, PCpts, 0, nAllPoints, nDevices))
		return false;
	//cout << "Done." << "(" << omp_get_wtime() - start_time << " sec)" << endl;
	
	return true;
}