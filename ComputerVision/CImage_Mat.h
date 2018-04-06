#pragma once

#include "atlimage.h"

// CImage 与 opencv的Mat 互换
// 仅支持单通道灰度或三通道彩色

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class CImage_Mat
{
public:
	CImage_Mat();
	~CImage_Mat();

	static void CImageToMat(CImage& cimage, cv::Mat& mat);

	static void MatToCImage(cv::Mat& mat, CImage& cimage);

};






