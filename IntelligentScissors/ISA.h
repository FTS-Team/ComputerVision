#pragma once
#include <Windows.h>
#include <vector>
using namespace std;

#include "Utility.h"

//#define SPEEDUP

#ifdef SPEEDUP
#define WZ 0.43
#define WG 0.43
#define WD 0.14
#else
#define WZ 0.3
#define WG 0.3
#define WD 0.1
#define WP 0.1
#define WI 0.1
#define WO 0.1

#define NO_TRAINING_WZ ( WZ / ( WZ + WG + WD ))
#define NO_TRAINING_WG ( WG / ( WZ + WG + WD ))
#define NO_TRAINING_WD ( WD / ( WZ + WG + WD ))
#endif // SPEEDUP

#define M 128

class ISA
{
public:
	ISA(CImage* img, int imgType=0);
	~ISA();

	void setSeedPoint(CPoint point);
	void removeSeedPoint();
	vector<CPoint> getShortestPathTo(CPoint currentPoint);
	CImage * getPreprocessedImg();

protected:
	void preprocessImg(Utility::ImageData* imgData, int imgType);
	double* computeFz(Utility::ImageData* img_blur_5x5, Utility::ImageData* img_blur_9x9);
	double* computeFg(int * Gxy_5x5, int * Gxy_9x9, int length);
	Utility::Neighbors<double>* computeFd(Utility::Vector * D, int length, int width);

#ifndef SPEEDUP
	double* computeFp(Utility::ImageData* imgData);
	void computeFio(Utility::ImageData* imgData, int * Gxy_5x5, int * Gxy_9x9, Utility::Vector * D, double **fi, double **fo);
#endif // !SPEEDUP
	
	Utility::Neighbors<int>* computeL(
		int length, int width,
		double *fz, int mz,
		double *fg, int mg,
		Utility::Neighbors<double> *fd, int md,
		double *fp = nullptr, int mp = 0,
		double *fi = nullptr, int mi = 0,
		double *fo = nullptr, int mo = 0
	);

protected:
	int* Fz(Utility::ImageData* laplacianImage);
	int* Gxy(Utility::ImageData* gradientImg_x, Utility::ImageData* gradientImg_y);
	int* getGradient(Utility::ImageData* gradientImg_5x5, Utility::ImageData* gradientImg_9x9);

private:
	Utility::ImageData * m_srcImgData;
	CImage * m_srcImg;
	CImage * m_preprocessedImg;
	
	Utility::Neighbors<int> *m_L;
	vector<Utility::PathMap*> m_pPathMapList;
};

