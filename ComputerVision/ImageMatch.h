#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "CImage_Mat.h"

using namespace cv;
using namespace std;

enum Match
{
	SIFT = 0,
	SURF,
	ORB,
	FERNS
};

class ImageMatch {

public:
	static UINT SIFT(CImage& CImage1, CImage& CImage2);//基于SIFT的图像匹配算法

	static UINT SURF(CImage& CImage1, CImage& CImage2);//基于SURF的图像匹配算法

	static UINT ORBMatch(CImage& CImage1, CImage & CImage2);//基于ORB的图像匹配算法

private:

	//RANSAC算法实现 
	static vector<DMatch> ransac(vector<DMatch> matches, vector<KeyPoint> queryKeyPoint, vector<KeyPoint> trainKeyPoint);

};