#pragma once
#include <opencv2\opencv.hpp>
#include <vector>
#include "PatchGenerator.h"

using namespace cv;
using namespace std;

class Yape
{
public:
	Yape(int _radius=7, int _threshold=20, int _nOctaves=3,
		int _nViews=1000, double _baseFeatureSize=32, double _clusteringDistance=2);
	
	void operator()(const Mat& image,
		CV_OUT vector<KeyPoint>& keypoints,
		int maxCount = 0, bool scaleCoords = true) const;
	void operator()(const vector<Mat>& pyr,
		CV_OUT vector<KeyPoint>& keypoints,
		int maxCount = 0, bool scaleCoords = true) const;
	
	void getMostStable2D(const Mat& image, CV_OUT vector<KeyPoint>& keypoints,
		int maxCount, const PatchGenerator& patchGenerator) const;
	
	void setVerbose(bool verbose);

	void read(const FileNode& node);
	void write(FileStorage& fs, const String& name = String()) const;

	int radius;
	int threshold;
	int nOctaves;
	int nViews;
	bool verbose;

	double baseFeatureSize;
	double clusteringDistance;
};

