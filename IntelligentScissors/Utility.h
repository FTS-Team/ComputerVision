#pragma once
#include <Windows.h>

#define PI 3.14159265358979323846

// get value from 2D array
#define ARRAY_2D_GET_VALUE(arr, x, y, width) \
	(*((arr) + (y) * (width) + (x)))

// if offset is 3, return a BGR_PTR which is point to [b, g, r]
#define COLOR_2D_GET_PTR(colorArray, x, y, width, offset) \
	((colorArray) + ((y) * (width) + (x)) * (offset))

#define B 0
#define G 1
#define R 2

// get rgb value from [b, g, r] array
#define GET_COLOR_B(BGR_PTR) (*((BGR_PTR) + B))
#define GET_COLOR_G(BGR_PTR) (*((BGR_PTR) + G))
#define GET_COLOR_R(BGR_PTR) (*((BGR_PTR) + R))

// get rgb value from CImage bits
#define GET_CIMAGE_COLOR_B(bits, x, y, bpp, pitch) (*((bits) + (pitch) * (y) + (x) * (bpp) + B))
#define GET_CIMAGE_COLOR_G(bits, x, y, bpp, pitch) (*((bits) + (pitch) * (y) + (x) * (bpp) + G))
#define GET_CIMAGE_COLOR_R(bits, x, y, bpp, pitch) (*((bits) + (pitch) * (y) + (x) * (bpp) + R))

class Utility
{
public:
	// sigma: 0.7
	static const double gaussian_kernel_5x5[];
	// sigma: 1.4
	static const double gaussian_kernel_9x9[];

	// 4 neighbors
	static const double laplacian_kernel_3x3_4n[];

	// gradient kernel
	static const double sobel_kernel_3x3_x[];
	static const double sobel_kernel_3x3_y[];
	static const double scharr_kernel_3x3_x[]; 
	static const double scharr_kernel_3x3_y[]; 

public:
	struct ImageData
	{
		bool isRGB;
		int offset;

		// rgb data: [b, g, r]
		// not rgb data: [b]
		int* color;

		// single channel length
		int length;

		// image width
		int width;

		ImageData() : color(nullptr) {}
		~ImageData() { delete[] color; }

		int* getColorAt(int x, int y);
		int getBrightnessAt(int x, int y);

#ifdef DEBUG
		bool operator== (ImageData *y)
		{
			if(this->isRGB == y->isRGB && this->length == y->length && this->width == y->width)
			{
				int total = this->isRGB ? 3 * this->length : this->length;
				for(int i = 0; i < total; i++)
				{
					if(*(this->color + i) != *(y->color + i))
						return false;
				}
				return true;
			}
			return false;
		}
#endif // DEBUG
	};

	struct Vector
	{
		double x;
		double y;

		Vector(double x=0, double y=0) : x(x), y(y) {}
		Vector(int x, int y) : x(double(x)), y(double(y)) {}

		double operator*(const Vector &vec)
		{
			return this->x * vec.x + this->y * vec.y;
		}

		friend Vector operator*(double scale, const Vector &vec)
		{
			return Vector(scale * vec.x, scale * vec.y);
		}

		friend Vector operator+(const Vector &vec1, const Vector &vec2)
		{
			return Vector(vec1.x + vec2.x, vec1.y + vec2.y);
		}

		friend Vector operator-(const Vector &vec1, const Vector &vec2)
		{
			return Vector(vec1.x - vec2.x, vec1.y - vec2.y);
		}

		friend Vector operator-(const Vector &vec)
		{
			return Vector(-vec.x, -vec.y);
		}

		// return the prependicular vector of the input vector
		friend Vector operator~(const Vector &vec)
		{
			return Vector(vec.y, -vec.x);
		}

		void nomalize()
		{
			double temp = this->length();
			x /= temp;
			y /= temp;
		}

		double length()
		{
			return sqrt((this->x * this->x) + (this->y * this->y));
		}
	};

	template<typename T>
	struct Neighbors
	{
		// 0 1 2
		// 3   4
		// 5 6 7
		T neighbor[8];
		bool hasNeighbor[8];

		T findMax()
		{
			T max = -999999;
			for(int i = 0; i < 8; i++)
			{
				if(hasNeighbor[i] && neighbor[i] > max)
					max = neighbor[i];
			}
			return max;
		}

		T findMin()
		{
			T min = 999999;
			for(int i = 0; i < 8; i++)
			{
				if(hasNeighbor[i] && neighbor[i] < min)
					min = neighbor[i];
			}
			return min;
		}
	};

	struct PixelNode
	{
		// next pointer just use in active list
		PixelNode *next;
		bool isInList;

		bool isExpended;
		Vector position;
		int totalCost;
		// used to find the shortest path
		PixelNode *toPixel;

		PixelNode(int x, int y);
		PixelNode(Vector vec);
		~PixelNode();

		void init();
	};

	// used to initialize the path map
	// use link list to sort cost
	class PriorityQueue
	{
	public:
		PriorityQueue();

		PixelNode * popMinimum();
		void pop(PixelNode * pixel);
		void push(PixelNode * pixel);

	private:
		PixelNode *m_pHead;
	};

	struct PathMap
	{
		PixelNode **image;
		int width;
		int length;

		PathMap(int width, int length);
		~PathMap();

		PixelNode * getPixelNodeAt(int x, int y);
		PixelNode * getPixelNodeAt(Vector &vec);
	};
public:
	// clone an image from CImage
	static CImage * CloneImage(CImage* srcImage);
	static ImageData * getImageData(CImage* srcImage);
	static CImage * createImage(ImageData* imageData, CImage* baseImage);

public:
	static ImageData* Convolution(ImageData* srcImageData, const double* kernel, int kernelSize);
};

