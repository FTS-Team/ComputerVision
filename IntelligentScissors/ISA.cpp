#include "stdafx.h"
#include "ISA.h"


ISA::ISA(CImage * img, int imgType)
{
	m_srcImg = Utility::CloneImage(img);
	m_srcImgData = Utility::getImageData(img);
	preprocessImg(m_srcImgData, imgType);
}


ISA::~ISA()
{
	m_srcImg->Destroy();
	delete m_srcImgData;
	delete[] m_L;

	for(auto &l : m_pPathMapList)
	{
		delete l;
	}
}

void ISA::setSeedPoint(CPoint point)
{
	// initialize
	int width = m_srcImgData->width;
	int length = m_srcImgData->length;
	Utility::PathMap* path_map = new Utility::PathMap(width, length);
	Utility::PriorityQueue L;
	Utility::PixelNode* seed_point = path_map->getPixelNodeAt(point.x, point.y);
	seed_point->totalCost = 0;
	seed_point->toPixel = nullptr;
	L.push(seed_point);

	// iterate
	while(true)
	{
		Utility::PixelNode* p = L.popMinimum();
		if(p == nullptr)
			break;

		int x = p->position.x;
		int y = p->position.y;

		Utility::Vector neighbor_position[] = {
			{ x - 1, y - 1 },{ x, y - 1 },{ x + 1, y - 1 },
			{ x - 1, y + 0 },             { x + 1, y + 0 },
			{ x - 1, y + 1 },{ x, y + 1 },{ x + 1, y + 1 }
		};

		for(int i = 0; i < 8; i++)
		{
			Utility::PixelNode* q = path_map->getPixelNodeAt(neighbor_position[i]);

			// outside the image or q has been expended
			if(q == nullptr || q->isExpended)
				continue;

			// gtmp = g(p) + l(p,q)
			int tempCost = p->totalCost + m_L[y * width + x].neighbor[i];

			// cost is larger than before
			if(tempCost >= q->totalCost)
				continue;

			// pop q from active list frist, because it has a lower cost now
			L.pop(q);
			q->totalCost = tempCost;
			q->toPixel = p;
			L.push(q);
		}

		p->isExpended = true;
	}

	m_pPathMapList.push_back(path_map);
}

void ISA::removeSeedPoint()
{
	delete m_pPathMapList.back();
	m_pPathMapList.pop_back();
}

vector<CPoint> ISA::getShortestPathTo(CPoint currentPoint)
{
	vector<CPoint> res;
	Utility::PixelNode* currentPixel = m_pPathMapList.back()->getPixelNodeAt(currentPoint.x, currentPoint.y);
	while(currentPixel != nullptr)
	{
		res.push_back({ int(currentPixel->position.x), int(currentPixel->position.y) });
		currentPixel = currentPixel->toPixel;
	}
	return res;
}

CImage * ISA::getPreprocessedImg()
{
	return m_preprocessedImg;
}

void ISA::preprocessImg(Utility::ImageData* imgData, int imgType)
{
	int length = imgData->length;
	int width = imgData->width;

	double *FZ;
	double *FG;
	Utility::Neighbors<double> *FD;
	int MZ = 1;
	int MG = 1;
	int MD = 1;

#ifndef SPEEDUP
	double *FP;
	double *FI;
	double *FO;
	int MP = 1;
	int MI = 1;
	int MO = 1;
#endif // !SPEEDUP


	Utility::ImageData *img_blur_5x5 = Utility::Convolution(imgData, Utility::gaussian_kernel_5x5, 5);
	Utility::ImageData *img_blur_9x9 = Utility::Convolution(imgData, Utility::gaussian_kernel_9x9, 9);

	// Laplacian Zero-Crossing
	FZ = computeFz(img_blur_5x5, img_blur_9x9);

	// Gradient
	Utility::ImageData *img_blur_5x5_gradient_x = Utility::Convolution(img_blur_5x5, Utility::scharr_kernel_3x3_x, 3);
	Utility::ImageData *img_blur_5x5_gradient_y = Utility::Convolution(img_blur_5x5, Utility::scharr_kernel_3x3_y, 3);
	Utility::ImageData *img_blur_9x9_gradient_x = Utility::Convolution(img_blur_9x9, Utility::scharr_kernel_3x3_x, 3);
	Utility::ImageData *img_blur_9x9_gradient_y = Utility::Convolution(img_blur_9x9, Utility::scharr_kernel_3x3_y, 3);

	int *Gxy_5x5 = Gxy(img_blur_5x5_gradient_x, img_blur_5x5_gradient_y);
	int *Gxy_9x9 = Gxy(img_blur_9x9_gradient_x, img_blur_9x9_gradient_y);

	// Multi-Scale Gradient Magnitude
	FG = computeFg(Gxy_5x5, Gxy_9x9, length);
	
	int *Gx = getGradient(img_blur_5x5_gradient_x, img_blur_9x9_gradient_x);
	int *Gy = getGradient(img_blur_5x5_gradient_y, img_blur_9x9_gradient_y);

	Utility::Vector * D = new Utility::Vector[length];
	for(int i = 0; i < length; i++)
	{
		D[i] = { Gx[i], Gy[i] };
		D[i].nomalize();
	}

	// Gradient Direction
	FD = computeFd(D, length, width);

#ifndef SPEEDUP
	// Fp, Fi, Fo
	FP = computeFp(imgData);
	computeFio(imgData, Gxy_5x5, Gxy_9x9, D, &FI, &FO);
#endif // !SPEEDUP

	MZ *= WZ * M;
	MG *= WG * M;
	MD *= WD * M;

#ifndef SPEEDUP
	MP *= WP * M;
	MI *= WI * M;
	MO *= WO * M;
#endif // !SPEEDUP


	// Total link cost
#ifdef SPEEDUP
	m_L = computeL(length, width, FZ, MZ, FG, MG, FD, MD);
#else
	m_L = computeL(length, width, FZ, MZ, FG, MG, FD, MD, FP, MP, FI, MI, FO, MO);
#endif // SPEEDUP

	if(imgType == 0);
	else if(imgType == 1)
	{
		m_preprocessedImg = Utility::createImage(img_blur_5x5, m_srcImg);
	} 
	else if(imgType == 2)
	{
		m_preprocessedImg = Utility::createImage(img_blur_9x9, m_srcImg);
	} 
	else
	{
		for(int i = 0; i < length; i++)
		{
			int x = i % width;
			int y = i / width;

			int color = 0;
			
			switch(imgType)
			{
				case 3:
					color = 255 * FZ[i];
					break;
				case 4:
					color = 255 * FG[i];
					break;
				case 5:
					color = 255 * FD[i].findMin();
					break;
				case 6:
					color = 255 * m_L[i].findMin() / M;
					break;
			}
			
			GET_COLOR_B(imgData->getColorAt(x, y)) = color;
			if(imgData->isRGB)
			{
				GET_COLOR_G(imgData->getColorAt(x, y)) = color;
				GET_COLOR_R(imgData->getColorAt(x, y)) = color;
			}
		}
		m_preprocessedImg = Utility::createImage(imgData, m_srcImg);
	}

	delete img_blur_5x5;
	delete img_blur_9x9;
	delete img_blur_5x5_gradient_x;
	delete img_blur_5x5_gradient_y;
	delete img_blur_9x9_gradient_x;
	delete img_blur_9x9_gradient_y;
	delete[] Gxy_5x5;
	delete[] Gxy_9x9;
	delete[] Gx;
	delete[] Gy;
	delete[] D;
	delete[] FZ;
	delete[] FG;
	delete[] FD;
#ifndef SPEEDUP
	delete[] FP;
	delete[] FI;
	delete[] FO;
#endif // !SPEEDUP
}

double * ISA::computeFz(Utility::ImageData* img_blur_5x5, Utility::ImageData* img_blur_9x9)
{
	Utility::ImageData * img_laplacian_blur_5x5 = Utility::Convolution(img_blur_5x5, Utility::laplacian_kernel_3x3_4n, 3);
	Utility::ImageData * img_laplacian_blur_9x9 = Utility::Convolution(img_blur_9x9, Utility::laplacian_kernel_3x3_4n, 3);

	int *fz_5x5 = Fz(img_laplacian_blur_5x5);
	int *fz_9x9 = Fz(img_laplacian_blur_9x9);

	double *fz = new double[img_laplacian_blur_5x5->length];
	for(int i = 0; i < img_laplacian_blur_5x5->length; i++)
	{
		fz[i] = 0.45 * fz_5x5[i] + 0.55 * fz_9x9[i];
	}

	delete img_laplacian_blur_5x5;
	delete img_laplacian_blur_9x9;
	delete[] fz_5x5;
	delete[] fz_9x9;
	return fz;
}

double * ISA::computeFg(int * Gxy_5x5, int * Gxy_9x9, int length)
{
	// compute G, min(G)
	int Gxy_min = 999999;
	for(int i = 0; i < length; i++)
	{
		if(Gxy_9x9[i] > Gxy_5x5[i])
		{
			Gxy_5x5[i] = Gxy_9x9[i];
		}
		if(Gxy_5x5[i] < Gxy_min)
		{
			Gxy_min = Gxy_5x5[i];
		}
	}

	// compute G' = G - min(G), max(G')
	int Gxy_max = -1;
	for(int i = 0; i < length; i++)
	{
		Gxy_5x5[i] -= Gxy_min;
		if(Gxy_5x5[i] > Gxy_max)
		{
			Gxy_max = Gxy_5x5[i];
		}
	}

	// fg = 1 - G' / max(G')
	double *fg = new double[length];
	for(int i = 0; i < length; i++)
	{
		fg[i] = 1 - (double) Gxy_5x5[i] / Gxy_max;
	}

	return fg;
}

Utility::Neighbors<double> * ISA::computeFd(Utility::Vector * D, int length, int width)
{
	Utility::Neighbors<double> * fd = new Utility::Neighbors<double>[length];
	for(int i = 0; i < length; i++)
	{
		int x = i % width;
		int y = i / width;

		// p
		Utility::Vector current_position(x, y);

		// q[8]
		Utility::Vector neighbor_position[] = {
			{ x - 1, y - 1 },{ x, y - 1 },{ x + 1, y - 1 },
			{ x - 1, y + 0 },             { x + 1, y + 0 },
			{ x - 1, y + 1 },{ x, y + 1 },{ x + 1, y + 1 }
		};

		// D'(p)
		Utility::Vector D_p = ~D[i];

		// compute fd
		for(int j = 0; j < 8; j++)
		{
			if(neighbor_position[j].x < 0 || neighbor_position[j].x >= width ||
				neighbor_position[j].y < 0 || neighbor_position[j].y >= length / width)
			{
				fd[i].neighbor[j] = 1;
				fd[i].hasNeighbor[j] = false;
				continue;
			}

			// pq = q - p
			Utility::Vector pq = neighbor_position[j] - current_position;

			if(D_p * pq < 0)
				pq = -pq;

			Utility::Vector Lpq = (1 / pq.length()) * pq;
			int q_index = neighbor_position[j].y * width + neighbor_position[j].x;

			// dp(p,q), dq(p,q)
			double dp = D_p * Lpq;
			double dq = Lpq * (~D[q_index]);

#ifdef SPEEDUP
			fd[i].neighbor[j] = (1.0 / PI) * (acos(dp) + acos(dq));
#else
			fd[i].neighbor[j] = (2.0 / 3.0 / PI) * (acos(dp) + acos(dq));
#endif // SPEEDUP

			fd[i].hasNeighbor[j] = true;
		}
	}
	
	return fd;
}

#ifndef SPEEDUP
double * ISA::computeFp(Utility::ImageData * imgData)
{
	double *fp = new double[imgData->length];
	for(int i = 0; i < imgData->length; i++)
	{
		int x = i % imgData->width;
		int y = i / imgData->width;

		fp[i] = imgData->getBrightnessAt(x, y) / 255.0;
	}
	return fp;
}

void ISA::computeFio(Utility::ImageData * imgData, int * Gxy_5x5, int * Gxy_9x9, Utility::Vector * D, double **fi, double **fo)
{
	*fi = new double[imgData->length];
	*fo = new double[imgData->length];

	for(int i = 0; i < imgData->length; i++)
	{
		Utility::Vector currentPixel(i % imgData->width, i / imgData->width);

		int k = Gxy_5x5[i] > Gxy_9x9[i] ? 3 : 5;
		Utility::Vector I = currentPixel + k * D[i];
		Utility::Vector O = currentPixel - k * D[i];

		(*fi)[i] = imgData->getBrightnessAt(I.x, I.y) / 255.0;
		(*fo)[i] = imgData->getBrightnessAt(O.x, O.y) / 255.0;
	}
}
#endif // !SPEEDUP

Utility::Neighbors<int> * ISA::computeL(
	int length, int width, 
	double *fz, int mz,
	double *fg, int mg,
	Utility::Neighbors<double> *fd, int md,
	double *fp, int mp,
	double *fi, int mi,
	double *fo,	int mo
)
{
	Utility::Neighbors<int> * l = new Utility::Neighbors<int>[length];

	double wn[] = {
		     1,      1 / sqrt(2),      1,
		1 / sqrt(2),              1 / sqrt(2),
		     1,      1 / sqrt(2),      1
	};

	for(int i = 0; i < length; i++)
	{
		int x = i % width;
		int y = i / width;

		// p
		Utility::Vector current_position(x, y);

		// q[8]
		Utility::Vector neighbor_position[] = {
			{ x - 1, y - 1 },{ x, y - 1 },{ x + 1, y - 1 },
			{ x - 1, y + 0 },             { x + 1, y + 0 },
			{ x - 1, y + 1 },{ x, y + 1 },{ x + 1, y + 1 }
		};

		for(int j = 0; j < 8; j++)
		{
			l[i].hasNeighbor[j] = fd[i].hasNeighbor[j];

			if(!fd[i].hasNeighbor[j])
			{
				l[i].neighbor[j] = 1;
				continue;
			}

			int q_index = neighbor_position[j].y * width + neighbor_position[j].x;

			// ls(p,q) = floor(MZ * FZ(q) + 0.5) + floor(MD * FD(p,q) + 0.5)
			int ls_pq = floor(mz * fz[q_index] + 0.5) + floor(md * fd[i].neighbor[j] + 0.5);
			
			// l(p,q) = ls(p,q) + wn(p,q) * floor(MG * FG(p,q) + 0.5) + floor(MP * FP(p,q) + 0.5) + floor(MI * FI(p,q) + 0.5) + floor(MO * FO(p,q) + 0.5)
#ifdef SPEEDUP
			l[i].neighbor[j] = ls_pq + wn[j] * floor(mg * fg[q_index] + 0.5);
#else
			l[i].neighbor[j] = ls_pq + wn[j] * floor(mg * fg[q_index] + 0.5) + floor(mp * fp[q_index] + 0.5) + floor(mi * fi[q_index] + 0.5) + floor(mo * fo[q_index] + 0.5);
#endif // SPEEDUP
		}
	}
	return l;
}

int * ISA::Fz(Utility::ImageData * laplacianImage)
{
	int * fz = new int[laplacianImage->length];

	for(int i = 0; i < laplacianImage->length; i++)
	{
		int x = i % laplacianImage->width;
		int y = i / laplacianImage->width;

		int* currPixelColor  = laplacianImage->getColorAt(x, y);
		int* neighbor_top    = laplacianImage->getColorAt(x, y - 1);
		int* neighbor_right  = laplacianImage->getColorAt(x + 1, y);
		int* neighbor_bottom = laplacianImage->getColorAt(x, y + 1);
		int* neighbor_left   = laplacianImage->getColorAt(x - 1, y);
		int temp[3] = { 0 };

#pragma region rgb_color_B channel
		if(GET_COLOR_B(currPixelColor) != 0)
		{
			int curr_x_top    = 1;
			int curr_x_right  = 1;
			int curr_x_bottom = 1;
			int curr_x_left   = 1;

			if(neighbor_top != nullptr)
				curr_x_top = GET_COLOR_B(neighbor_top) * GET_COLOR_B(currPixelColor);
			if(neighbor_right != nullptr)
				curr_x_right = GET_COLOR_B(neighbor_right) * GET_COLOR_B(currPixelColor);
			if(neighbor_bottom != nullptr)
				curr_x_bottom = GET_COLOR_B(neighbor_bottom) * GET_COLOR_B(currPixelColor);
			if(neighbor_left != nullptr)
				curr_x_left = GET_COLOR_B(neighbor_left) * GET_COLOR_B(currPixelColor);

			if(
				(curr_x_top < 0 && abs(GET_COLOR_B(currPixelColor)) <= abs(GET_COLOR_B(neighbor_top))) ||
				(curr_x_right < 0 && abs(GET_COLOR_B(currPixelColor)) <= abs(GET_COLOR_B(neighbor_right))) ||
				(curr_x_bottom < 0 && abs(GET_COLOR_B(currPixelColor)) <= abs(GET_COLOR_B(neighbor_bottom))) ||
				(curr_x_left < 0 && abs(GET_COLOR_B(currPixelColor)) <= abs(GET_COLOR_B(neighbor_left))))
			{
				temp[B] = 0;
			} else
			{
				temp[B] = 1;
			}
		}
#pragma endregion

		if(laplacianImage->isRGB)
		{
#pragma region rgb_color_G channel
			if(GET_COLOR_G(currPixelColor) != 0)
			{
				int curr_x_top = 1;
				int curr_x_right = 1;
				int curr_x_bottom = 1;
				int curr_x_left = 1;

				if(neighbor_top != nullptr)
					curr_x_top = GET_COLOR_G(neighbor_top) * GET_COLOR_G(currPixelColor);
				if(neighbor_right != nullptr)
					curr_x_right = GET_COLOR_G(neighbor_right) * GET_COLOR_G(currPixelColor);
				if(neighbor_bottom != nullptr)
					curr_x_bottom = GET_COLOR_G(neighbor_bottom) * GET_COLOR_G(currPixelColor);
				if(neighbor_left != nullptr)
					curr_x_left = GET_COLOR_G(neighbor_left) * GET_COLOR_G(currPixelColor);

				if(
					(curr_x_top < 0 && abs(GET_COLOR_G(currPixelColor)) <= abs(GET_COLOR_G(neighbor_top))) ||
					(curr_x_right < 0 && abs(GET_COLOR_G(currPixelColor)) <= abs(GET_COLOR_G(neighbor_right))) ||
					(curr_x_bottom < 0 && abs(GET_COLOR_G(currPixelColor)) <= abs(GET_COLOR_G(neighbor_bottom))) ||
					(curr_x_left < 0 && abs(GET_COLOR_G(currPixelColor)) <= abs(GET_COLOR_G(neighbor_left))))
				{
					temp[G] = 0;
				} else
				{
					temp[G] = 1;
				}
			}
#pragma endregion
#pragma region rgb_color_R channel
			if(GET_COLOR_R(currPixelColor) != 0)
			{
				int curr_x_top = 1;
				int curr_x_right = 1;
				int curr_x_bottom = 1;
				int curr_x_left = 1;

				if(neighbor_top != nullptr)
					curr_x_top = GET_COLOR_R(neighbor_top) * GET_COLOR_R(currPixelColor);
				if(neighbor_right != nullptr)
					curr_x_right = GET_COLOR_R(neighbor_right) * GET_COLOR_R(currPixelColor);
				if(neighbor_bottom != nullptr)
					curr_x_bottom = GET_COLOR_R(neighbor_bottom) * GET_COLOR_R(currPixelColor);
				if(neighbor_left != nullptr)
					curr_x_left = GET_COLOR_R(neighbor_left) * GET_COLOR_R(currPixelColor);

				if(
					(curr_x_top < 0 && abs(GET_COLOR_R(currPixelColor)) <= abs(GET_COLOR_R(neighbor_top))) ||
					(curr_x_right < 0 && abs(GET_COLOR_R(currPixelColor)) <= abs(GET_COLOR_R(neighbor_right))) ||
					(curr_x_bottom < 0 && abs(GET_COLOR_R(currPixelColor)) <= abs(GET_COLOR_R(neighbor_bottom))) ||
					(curr_x_left < 0 && abs(GET_COLOR_R(currPixelColor)) <= abs(GET_COLOR_R(neighbor_left))))
				{
					temp[R] = 0;
				} else
				{
					temp[R] = 1;
				}
			}
#pragma endregion
		}

		fz[i] = temp[0] | temp[1] | temp[2];
	}

	return fz;
}

int * ISA::Gxy(Utility::ImageData * gradientImg_x, Utility::ImageData * gradientImg_y)
{
	int *gradient = new int[gradientImg_x->length];

	for(int i = 0; i < gradientImg_x->length; i++)
	{
		int x = i % gradientImg_x->width;
		int y = i / gradientImg_x->width;

		int *Gx = gradientImg_x->getColorAt(x, y);
		int *Gy = gradientImg_y->getColorAt(x, y);

		int Gx_max = abs(GET_COLOR_B(Gx));
		int Gy_max = abs(GET_COLOR_B(Gy));
		if(gradientImg_x->isRGB)
		{
			Gx_max = max(Gx_max, max(abs(GET_COLOR_G(Gx)), abs(GET_COLOR_R(Gx))));
			Gy_max = max(Gy_max, max(abs(GET_COLOR_G(Gy)), abs(GET_COLOR_R(Gy))));
		}
		gradient[i] = Gx_max + Gy_max;
		
		/*int Gxy[3] = { 0 };
		Gxy[B] = abs(GET_COLOR_B(Gx)) + abs(GET_COLOR_B(Gy));
		if(gradientImg_x->isRGB)
		{
			Gxy[G] = abs(GET_COLOR_G(Gx)) + abs(GET_COLOR_G(Gy));
			Gxy[R] = abs(GET_COLOR_R(Gx)) + abs(GET_COLOR_R(Gy));
		}
		gradient[i] = max(Gxy[B], max(Gxy[G], Gxy[R]));*/
	}

	return gradient;
}

int * ISA::getGradient(Utility::ImageData * gradientImg_5x5, Utility::ImageData * gradientImg_9x9)
{
	int *gradient = new int[gradientImg_5x5->length];

	for(int i = 0; i < gradientImg_5x5->length; i++)
	{
		int x = i % gradientImg_5x5->width;
		int y = i / gradientImg_5x5->width;

		int *G_5x5 = gradientImg_5x5->getColorAt(x, y);
		int *G_9x9 = gradientImg_5x5->getColorAt(x, y);

		int G_max = max(abs(GET_COLOR_B(G_5x5)), abs(GET_COLOR_B(G_9x9)));
		if(gradientImg_5x5->isRGB)
		{
			int G_max_G = max(abs(GET_COLOR_G(G_5x5)), abs(GET_COLOR_G(G_9x9)));
			int G_max_R = max(abs(GET_COLOR_R(G_5x5)), abs(GET_COLOR_R(G_9x9)));

			G_max = max(G_max, max(G_max_G, G_max_R));
		}
		gradient[i] = G_max;
	}

	return gradient;
}