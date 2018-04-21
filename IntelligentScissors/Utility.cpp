#include "stdafx.h"
#include "Utility.h"

const double Utility::gaussian_kernel_5x5[] = {
	0.000252,	0.00352,	0.008344,	0.00352,	0.000252,
	0.00352,	0.049081,	0.11634,	0.049081,	0.00352,
	0.008344,	0.11634,	0.275768,	0.11634,	0.008344,
	0.00352,	0.049081,	0.11634,	0.049081,	0.00352,
	0.000252,	0.00352,	0.008344,	0.00352,	0.000252
};

const double Utility::gaussian_kernel_9x9[] = {
	0.000031,	0.000172,	0.000584,	0.001217,	0.001554,	0.001217,	0.000584,	0.000172,	0.000031,
	0.000172,	0.000955,	0.003247,	0.006761,	0.008634,	0.006761,	0.003247,	0.000955,	0.000172,
	0.000584,	0.003247,	0.011036,	0.022985,	0.02935,	0.022985,	0.011036,	0.003247,	0.000584,
	0.001217,	0.006761,	0.022985,	0.047869,	0.061126,	0.047869,	0.022985,	0.006761,	0.001217,
	0.001554,	0.008634,	0.02935,	0.061126,	0.078053,	0.061126,	0.02935,	0.008634,	0.001554,
	0.001217,	0.006761,	0.022985,	0.047869,	0.061126,	0.047869,	0.022985,	0.006761,	0.001217,
	0.000584,	0.003247,	0.011036,	0.022985,	0.02935,	0.022985,	0.011036,	0.003247,	0.000584,
	0.000172,	0.000955,	0.003247,	0.006761,	0.008634,	0.006761,	0.003247,	0.000955,	0.000172,
	0.000031,	0.000172,	0.000584,	0.001217,	0.001554,	0.001217,	0.000584,	0.000172,	0.000031
};

const double Utility::laplacian_kernel_3x3_4n[] = {
	 0, -1,  0,
	-1,  4, -1,
	 0, -1,  0
};

const double Utility::sobel_kernel_3x3_x[] = {
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};

const double Utility::sobel_kernel_3x3_y[] = {
	 1,  2,  1,
	 0,  0,  0,
	-1, -2, -1
};

const double Utility::scharr_kernel_3x3_x[] = {
	 3,  0,  -3,
	10,  0, -10,
	 3,  0,  -3
};

const double Utility::scharr_kernel_3x3_y[] = {
	 3,  10,  3,
	 0,   0,  0,
	-3, -10, -3
};

int * Utility::ImageData::getColorAt(int x, int y)
{
	if(x >= 0 && x < this->width && y >= 0 && y < this->length / this->width)
		return COLOR_2D_GET_PTR(this->color, x, y, this->width, this->offset);
	return nullptr;
}

int Utility::ImageData::getBrightnessAt(int x, int y)
{
	int *color = getColorAt(x, y);
	if(color == nullptr)
		return 0;

	int V = GET_COLOR_B(color);
	if(this->isRGB)
	{
		V = max(V, max(GET_COLOR_G(color), GET_COLOR_R(color)));
	}
	return V;
}

CImage * Utility::CloneImage(CImage * srcImage)
{
	if(srcImage == nullptr)
		return nullptr;

	int srcImageBitsCount = srcImage->GetBPP();
	int srcImageWidth = srcImage->GetWidth();
	int srcImageHeight = srcImage->GetHeight();

	CImage * destImage = new CImage();
	destImage->Create(srcImageWidth, srcImageHeight, srcImageBitsCount, srcImageBitsCount == 32 ? 1 : 0);

	srcImage->BitBlt(destImage->GetDC(), 0, 0);
	destImage->ReleaseDC();

	return destImage;
}

Utility::ImageData * Utility::getImageData(CImage * srcImage)
{
	if(srcImage == nullptr)
		return nullptr;

	int srcImageWidth = srcImage->GetWidth();
	int srcImageHeight = srcImage->GetHeight();

	byte* srcImageBits = (byte*) srcImage->GetBits();
	int srcImageBitsCount = srcImage->GetBPP() / 8;
	int srcImagePitch = srcImage->GetPitch();

	ImageData *image_data = new ImageData();
	image_data->isRGB = srcImageBitsCount != 1;
	image_data->length = srcImageWidth * srcImageHeight;
	image_data->width = srcImageWidth;

	image_data->offset = image_data->isRGB ? 3 : 1;
	image_data->color = new int[image_data->length * image_data->offset];

	for(int y = 0; y < srcImageHeight; y++)
	{
		for(int x = 0; x < srcImageWidth; x++)
		{
			int *imageColor = COLOR_2D_GET_PTR(image_data->color, x, y, srcImageWidth, image_data->offset);

			GET_COLOR_B(imageColor) = GET_CIMAGE_COLOR_B(srcImageBits, x, y, srcImageBitsCount, srcImagePitch);
			if(image_data->isRGB)
			{
				GET_COLOR_G(imageColor) = GET_CIMAGE_COLOR_G(srcImageBits, x, y, srcImageBitsCount, srcImagePitch);
				GET_COLOR_R(imageColor) = GET_CIMAGE_COLOR_R(srcImageBits, x, y, srcImageBitsCount, srcImagePitch);
			}
		}
	}

	return image_data;
}

CImage * Utility::createImage(ImageData * imageData, CImage * baseImage)
{
	if(baseImage == nullptr)
		return nullptr;

	int baseImageBitsCount = baseImage->GetBPP() / 8;
	int baseImagePitch = baseImage->GetPitch();
	int baseImageWidth = baseImage->GetWidth();
	int baseImageHeight = baseImage->GetHeight();

	CImage * newImage = new CImage();
	newImage->Create(baseImageWidth, baseImageHeight, baseImageBitsCount*8, baseImageBitsCount == 4 ? 1 : 0);
	byte* newImageBits = (byte*) newImage->GetBits();

	for(int y = 0; y < baseImageHeight; y++)
	{
		for(int x = 0; x < baseImageWidth; x++)
		{
			int *imageColor = COLOR_2D_GET_PTR(imageData->color, x, y, baseImageWidth, imageData->offset);

			GET_CIMAGE_COLOR_B(newImageBits, x, y, baseImageBitsCount, baseImagePitch) = GET_COLOR_B(imageColor);
			if(imageData->isRGB)
			{
				GET_CIMAGE_COLOR_G(newImageBits, x, y, baseImageBitsCount, baseImagePitch) = GET_COLOR_G(imageColor);
				GET_CIMAGE_COLOR_R(newImageBits, x, y, baseImageBitsCount, baseImagePitch) = GET_COLOR_R(imageColor);
			}
		}
	}

	return newImage;
}

Utility::ImageData * Utility::Convolution(ImageData * srcImageData, const double * kernel, int kernelSize)
{
	int kernelRadius = kernelSize / 2;

	ImageData *image_data = new ImageData();
	image_data->isRGB = srcImageData->isRGB;
	image_data->length = srcImageData->length;
	image_data->width = srcImageData->width;

	image_data->offset = image_data->isRGB ? 3 : 1;
	image_data->color = new int[image_data->length * image_data->offset];

	int imageHeight = srcImageData->length / srcImageData->width;
	int imageWidth = srcImageData->width;

	for(int l = 0; l < srcImageData->length; l++)
	{
		int x = l % imageWidth;
		int y = l / imageWidth;

		double res[3] = { 0 };

		// convolution
		for(int m = -kernelRadius; m <= kernelRadius; m++)
		{
			if(y + m < 0)
				continue;
			if(y + m >= imageHeight)
				break;

			for(int n = -kernelRadius; n <= kernelRadius; n++)
			{
				if(x + n < 0)
					continue;
				if(x + n >= imageWidth)
					break;

				int *srcImageColor = COLOR_2D_GET_PTR(srcImageData->color, x + n, y + m, imageWidth, image_data->offset);

				res[B] += ARRAY_2D_GET_VALUE(kernel, n + kernelRadius, m + kernelRadius, kernelSize) * GET_COLOR_B(srcImageColor);
				if(srcImageData->isRGB)
				{
					res[G] += ARRAY_2D_GET_VALUE(kernel, n + kernelRadius, m + kernelRadius, kernelSize) * GET_COLOR_G(srcImageColor);
					res[R] += ARRAY_2D_GET_VALUE(kernel, n + kernelRadius, m + kernelRadius, kernelSize) * GET_COLOR_R(srcImageColor);
				}
			}
		}
		int *imageColor = COLOR_2D_GET_PTR(image_data->color, x, y, imageWidth, image_data->offset);

		GET_COLOR_B(imageColor) = res[B];
		if(image_data->isRGB)
		{
			GET_COLOR_G(imageColor) = res[G];
			GET_COLOR_R(imageColor) = res[R];
		}
	}

	return image_data;
}

Utility::PixelNode::PixelNode(int x, int y)
	: position({ x,y })
{
	init();
}

Utility::PixelNode::PixelNode(Vector vec)
	: position(vec)
{
	init();
}

Utility::PixelNode::~PixelNode()
{
	next = nullptr;
	toPixel = nullptr;
}

void Utility::PixelNode::init()
{
	next = nullptr;
	toPixel = nullptr;

	isInList = false;
	isExpended = false;
	totalCost = 999999;
}

Utility::PathMap::PathMap(int width, int length)
	: width(width)
	, length(length)
{
	image = new PixelNode*[length];

	for(int i = 0; i < length; i++)
	{
		image[i] = new PixelNode(i % width, i / width);
	}
}

Utility::PathMap::~PathMap()
{
	for(int i = 0; i < length; i++)
	{
		delete image[i];
	}
	delete[] image;
}

Utility::PixelNode * Utility::PathMap::getPixelNodeAt(int x, int y)
{
	if(x < 0 || x >= width || y < 0 || y >= length / width)
		return nullptr;

	return image[y * width + x];
}

Utility::PixelNode * Utility::PathMap::getPixelNodeAt(Vector & vec)
{
	return getPixelNodeAt(vec.x, vec.y);
}

Utility::PriorityQueue::PriorityQueue()
	: m_pHead(nullptr)
{
}

Utility::PixelNode * Utility::PriorityQueue::popMinimum()
{
	if(m_pHead == nullptr)
		return nullptr;

	PixelNode *first = m_pHead;
	m_pHead = first->next;

	first->next = nullptr;
	first->isInList = false;
	return first;
}

void Utility::PriorityQueue::pop(PixelNode * pixel)
{
	if(!pixel->isInList)
		return;

	pixel->isInList = false;

	if(m_pHead == pixel)
	{
		m_pHead = pixel->next;
		pixel->next = nullptr;
		return;
	}

	PixelNode *temp = m_pHead;
	while(temp != nullptr && temp->next != pixel)
	{
		temp = temp->next;
	}

	temp->next = pixel->next;
	pixel->next = nullptr;
}

void Utility::PriorityQueue::push(PixelNode * pixel)
{
	pixel->isInList = true;

	if(m_pHead == nullptr)
	{
		m_pHead = pixel;
		return;
	}

	PixelNode *temp_s = m_pHead;
	PixelNode *temp_q = m_pHead->next;
	while(temp_q != nullptr && temp_q->totalCost < pixel->totalCost )
	{
		temp_s = temp_q;
		temp_q = temp_q->next;
	}

	temp_s->next = pixel;
	pixel->next = temp_q;
}

