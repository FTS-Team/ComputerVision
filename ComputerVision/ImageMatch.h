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
	static UINT SIFT(CImage& CImage1, CImage& CImage2);//����SIFT��ͼ��ƥ���㷨

	static UINT SURF(CImage& CImage1, CImage& CImage2);//����SURF��ͼ��ƥ���㷨

	static UINT ORBMatch(CImage& CImage1, CImage & CImage2);//����ORB��ͼ��ƥ���㷨

private:

	//RANSAC�㷨ʵ�� 
	static vector<DMatch> ransac(vector<DMatch> matches, vector<KeyPoint> queryKeyPoint, vector<KeyPoint> trainKeyPoint);

};