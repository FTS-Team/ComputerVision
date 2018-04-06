#pragma once

#include "atlimage.h"

// CImage �� opencv��Mat ����
// ��֧�ֵ�ͨ���ҶȻ���ͨ����ɫ

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






