#include "stdafx.h"
#include "Yape.h"
#include "General.h"



Yape::Yape(
	int _radius, int _threshold, int _nOctaves, int _nViews,
	double _baseFeatureSize, double _clusteringDistance
)
	: radius(_radius)
	, threshold(_threshold)
	, nOctaves(_nOctaves)
	, nViews(_nViews)
	, verbose(false)
	, baseFeatureSize(_baseFeatureSize)
	, clusteringDistance(_clusteringDistance)
{
}

static void getDiscreteCircle(int R, vector<Point>& circle, vector<int>& filledHCircle)
{
	int x = R, y = 0;
	for(;; y++)
	{
		x = cvRound(std::sqrt((double) R*R - y * y));
		if(x < y)
			break;
		circle.push_back(Point(x, y));
		if(x == y)
			break;
	}

	int i, n8 = (int) circle.size() - (x == y), n8_ = n8 - (x != y), n4 = n8 + n8_, n = n4 * 4;
	CV_Assert(n8 > 0);
	circle.resize(n);

	for(i = 0; i < n8; i++)
	{
		Point p = circle[i];
		circle[i + n4] = Point(-p.y, p.x);
		circle[i + n4 * 2] = Point(-p.x, -p.y);
		circle[i + n4 * 3] = Point(p.y, -p.x);
	}

	for(i = n8; i < n4; i++)
	{
		Point p = circle[n4 - i], q = Point(p.y, p.x);
		circle[i] = q;
		circle[i + n4] = Point(-q.y, q.x);
		circle[i + n4 * 2] = Point(-q.x, -q.y);
		circle[i + n4 * 3] = Point(q.y, -q.x);
	}

	// the filled upper half of the circle is encoded as sequence of integers,
	// i-th element is the coordinate of right-most circle point in each horizontal line y=i.
	// the left-most point will be -filledHCircle[i].
	for(i = 0, y = -1; i < n4; i++)
	{
		Point p = circle[i];
		if(p.y != y)
		{
			filledHCircle.push_back(p.x);
			y = p.y;
			if(y == R)
				break;
		}
	}
}


struct CmpKeypointScores
{
	bool operator ()(const KeyPoint& a, const KeyPoint& b) const { return std::abs(a.response) > std::abs(b.response); }
};


void Yape::getMostStable2D(const Mat& image, vector<KeyPoint>& keypoints,
	int maxPoints, const PatchGenerator& _patchGenerator) const
{
	PatchGenerator patchGenerator = _patchGenerator;
	patchGenerator.backgroundMin = patchGenerator.backgroundMax = 128;

	Mat warpbuf, warped;
	Mat matM(2, 3, CV_64F), _iM(2, 3, CV_64F);
	double *M = (double*) matM.data, *iM = (double*) _iM.data;
	RNG& rng = theRNG();
	int i, k;
	vector<KeyPoint> tempKeypoints;
	double d2 = clusteringDistance * clusteringDistance;
	keypoints.clear();

	// TODO: this loop can be run in parallel, for that we need
	// a separate accumulator keypoint lists for different threads.
	for(i = 0; i < nViews; i++)
	{
		// 1. generate random transform
		// 2. map the source image corners and compute the ROI in canvas
		// 3. select the ROI in canvas, adjust the transformation matrix
		// 4. apply the transformation
		// 5. run keypoint detector in pyramids
		// 6. map each point back and update the lists of most stable points

		/*if(verbose && (i + 1)*General::progressBarSize / nViews != i * General::progressBarSize / nViews)
			putchar('.');*/
		if(verbose)
			General::displayRateOfProgress(i, nViews);
		if(i > 0)
			patchGenerator.generateRandomTransform(Point2f(), Point2f(), matM, rng);
		else
		{
			// identity transformation
			M[0] = M[4] = 1;
			M[1] = M[3] = M[2] = M[5] = 0;
		}

		patchGenerator.warpWholeImage(image, matM, warpbuf, warped, cvCeil(baseFeatureSize*0.5 + radius), rng);
		(*this)(warped, tempKeypoints, maxPoints * 3);
		invertAffineTransform(matM, _iM);

		int j, sz0 = (int) tempKeypoints.size(), sz1;
		for(j = 0; j < sz0; j++)
		{
			KeyPoint kpt1 = tempKeypoints[j];
			KeyPoint kpt0((float) (iM[0] * kpt1.pt.x + iM[1] * kpt1.pt.y + iM[2]),
				(float) (iM[3] * kpt1.pt.x + iM[4] * kpt1.pt.y + iM[5]),
				kpt1.size, -1.f, 1.f, kpt1.octave);
			float r = kpt1.size*0.5f;
			if(kpt0.pt.x < r || kpt0.pt.x >= image.cols - r ||
				kpt0.pt.y < r || kpt0.pt.y >= image.rows - r)
				continue;

			sz1 = (int) keypoints.size();
			for(k = 0; k < sz1; k++)
			{
				KeyPoint kpt = keypoints[k];
				if(kpt.octave != kpt0.octave)
					continue;
				double dx = kpt.pt.x - kpt0.pt.x, dy = kpt.pt.y - kpt0.pt.y;
				if(dx*dx + dy * dy <= d2 * (1 << kpt.octave * 2))
				{
					keypoints[k] = KeyPoint((kpt.pt.x*kpt.response + kpt0.pt.x) / (kpt.response + 1),
						(kpt.pt.y*kpt.response + kpt0.pt.y) / (kpt.response + 1),
						kpt.size, -1.f, kpt.response + 1, kpt.octave);
					break;
				}
			}
			if(k == sz1)
				keypoints.push_back(kpt0);
		}
	}

	/*if(verbose)
		putchar('\n');*/

	if((int) keypoints.size() > maxPoints)
	{
		General::sort(keypoints, CmpKeypointScores());
		keypoints.resize(maxPoints);
	}
}


static inline int computeLResponse(const uchar* ptr, const int* cdata, int csize)
{
	int i, csize2 = csize / 2, sum = -ptr[0] * csize;
	for(i = 0; i < csize2; i++)
	{
		int ofs = cdata[i];
		sum += ptr[ofs] + ptr[-ofs];
	}
	return sum;
}


static Point2f adjustCorner(const float* fval, float& fvaln)
{
	double bx = (fval[3] - fval[5])*0.5;
	double by = (fval[2] - fval[7])*0.5;
	double Axx = fval[3] - fval[4] * 2 + fval[5];
	double Axy = (fval[0] - fval[2] - fval[6] + fval[8])*0.25;
	double Ayy = fval[1] - fval[4] * 2 + fval[7];
	double D = Axx * Ayy - Axy * Axy;
	D = D != 0 ? 1. / D : 0;
	double dx = (bx*Ayy - by * Axy)*D;
	double dy = (by*Axx - bx * Axy)*D;
	dx = std::min(std::max(dx, -1.), 1.);
	dy = std::min(std::max(dy, -1.), 1.);
	fvaln = (float) (fval[4] + (bx*dx + by * dy)*0.5);
	if(fvaln*fval[4] < 0 || std::abs(fvaln) < std::abs(fval[4]))
		fvaln = fval[4];

	return Point2f((float) dx, (float) dy);
}

void Yape::operator()(const Mat& image, vector<KeyPoint>& keypoints, int maxCount, bool scaleCoords) const
{
	vector<Mat> pyr;
	buildPyramid(image, pyr, std::max(nOctaves - 1, 0));
	(*this)(pyr, keypoints, maxCount, scaleCoords);
}

void Yape::operator()(const vector<Mat>& pyr, vector<KeyPoint>& keypoints, int maxCount, bool scaleCoords) const
{
	const int lthreshold = 3;
	int L, x, y, i, j, k, tau = lthreshold;
	Mat scoreBuf(pyr[0].size(), CV_16S), maskBuf(pyr[0].size(), CV_8U);
	int scoreElSize = (int) scoreBuf.elemSize();
	vector<Point> circle0;
	vector<int> fhcircle0, circle, fcircle_s, fcircle;
	getDiscreteCircle(radius, circle0, fhcircle0);
	CV_Assert(fhcircle0.size() == (size_t) (radius + 1) && circle0.size() % 2 == 0);
	keypoints.clear();

	for(L = 0; L < nOctaves; L++)
	{
		//  Pyramidal keypoint detector body:
		//    1. build next pyramid layer
		//    2. scan points, check the circular neighborhood, compute the score
		//    3. do non-maxima suppression
		//    4. adjust the corners (sub-pix)
		double cscale = scaleCoords ? 1 << L : 1;
		Size layerSize = pyr[L].size();
		if(layerSize.width < radius * 2 + 3 || layerSize.height < radius * 2 + 3)
			break;
		Mat scoreLayer(layerSize, scoreBuf.type(), scoreBuf.data);
		Mat maskLayer(layerSize, maskBuf.type(), maskBuf.data);
		const Mat& pyrLayer = pyr[L];
		int sstep = (int) (scoreLayer.step / sizeof(short));
		int mstep = (int) maskLayer.step;

		int csize = (int) circle0.size(), csize2 = csize / 2;
		circle.resize(csize * 3);
		for(i = 0; i < csize; i++)
			circle[i] = circle[i + csize] = circle[i + csize * 2] = (int) ((-circle0[i].y)*pyrLayer.step + circle0[i].x);
		fcircle.clear();
		fcircle_s.clear();
		for(i = -radius; i <= radius; i++)
		{
			x = fhcircle0[std::abs(i)];
			for(j = -x; j <= x; j++)
			{
				fcircle_s.push_back(i*sstep + j);
				fcircle.push_back((int) (i*pyrLayer.step + j));
			}
		}
		int nsize = (int) fcircle.size();
		const int* cdata = &circle[0];
		const int* ndata = &fcircle[0];
		const int* ndata_s = &fcircle_s[0];

		for(y = 0; y < radius; y++)
		{
			memset(scoreLayer.ptr<short>(y), 0, layerSize.width*scoreElSize);
			memset(scoreLayer.ptr<short>(layerSize.height - y - 1), 0, layerSize.width*scoreElSize);
			memset(maskLayer.ptr<uchar>(y), 0, layerSize.width);
			memset(maskLayer.ptr<uchar>(layerSize.height - y - 1), 0, layerSize.width);
		}

		int vradius = (int) (radius*pyrLayer.step);

		for(y = radius; y < layerSize.height - radius; y++)
		{
			const uchar* img = pyrLayer.ptr<uchar>(y) + radius;
			short* scores = scoreLayer.ptr<short>(y);
			uchar* mask = maskLayer.ptr<uchar>(y);

			for(x = 0; x < radius; x++)
			{
				scores[x] = scores[layerSize.width - 1 - x] = 0;
				mask[x] = mask[layerSize.width - 1 - x] = 0;
			}

			for(x = radius; x < layerSize.width - radius; x++, img++)
			{
				int val0 = *img;
				if((std::abs(val0 - img[radius]) < tau && std::abs(val0 - img[-radius]) < tau) ||
					(std::abs(val0 - img[vradius]) < tau && std::abs(val0 - img[-vradius]) < tau))
				{
					scores[x] = 0;
					mask[x] = 0;
					continue;
				}

				for(k = 0; k < csize; k++)
				{
					if(std::abs(val0 - img[cdata[k]]) < tau &&
						(std::abs(val0 - img[cdata[k + csize2]]) < tau ||
							std::abs(val0 - img[cdata[k + csize2 - 1]]) < tau ||
							std::abs(val0 - img[cdata[k + csize2 + 1]]) < tau ||
							std::abs(val0 - img[cdata[k + csize2 - 2]]) < tau ||
							std::abs(val0 - img[cdata[k + csize2 + 2]]) < tau/* ||
																			 std::abs(val0 - img[cdata[k + csize2 - 3]]) < tau ||
																			 std::abs(val0 - img[cdata[k + csize2 + 3]]) < tau*/))
						break;
				}

				if(k < csize)
				{
					scores[x] = 0;
					mask[x] = 0;
				} else
				{
					scores[x] = (short) computeLResponse(img, cdata, csize);
					mask[x] = 1;
				}
			}
		}

		for(y = radius + 1; y < layerSize.height - radius - 1; y++)
		{
			const uchar* img = pyrLayer.ptr<uchar>(y) + radius + 1;
			short* scores = scoreLayer.ptr<short>(y) + radius + 1;
			const uchar* mask = maskLayer.ptr<uchar>(y) + radius + 1;

			for(x = radius + 1; x < layerSize.width - radius - 1; x++, img++, scores++, mask++)
			{
				int val0 = *scores;
				if(!*mask || std::abs(val0) < lthreshold ||
					(mask[-1] + mask[1] + mask[-mstep - 1] + mask[-mstep] + mask[-mstep + 1] +
						mask[mstep - 1] + mask[mstep] + mask[mstep + 1] < 3))
					continue;
				bool recomputeZeroScores = radius * 2 < y && y < layerSize.height - radius * 2 &&
					radius * 2 < x && x < layerSize.width - radius * 2;

				if(val0 > 0)
				{
					for(k = 0; k < nsize; k++)
					{
						int val = scores[ndata_s[k]];
						if(val == 0 && recomputeZeroScores)
							scores[ndata_s[k]] = (short) (val =
								computeLResponse(img + ndata[k], cdata, csize));
						if(val0 < val)
							break;
					}
				} else
				{
					for(k = 0; k < nsize; k++)
					{
						int val = scores[ndata_s[k]];
						if(val == 0 && recomputeZeroScores)
							scores[ndata_s[k]] = (short) (val =
								computeLResponse(img + ndata[k], cdata, csize));
						if(val0 > val)
							break;
					}
				}
				if(k < nsize)
					continue;
				float fval[9], fvaln = 0;
				for(int i1 = -1; i1 <= 1; i1++)
					for(int j1 = -1; j1 <= 1; j1++)
					{
						fval[(i1 + 1) * 3 + j1 + 1] = (float) (scores[sstep*i1 + j1] ? scores[sstep*i1 + j1] :
							computeLResponse(img + pyrLayer.step*i1 + j1, cdata, csize));
					}
				Point2f pt = adjustCorner(fval, fvaln);
				pt.x += x;
				pt.y += y;
				keypoints.push_back(KeyPoint((float) (pt.x*cscale), (float) (pt.y*cscale),
					(float) (baseFeatureSize*cscale), -1, fvaln, L));
			}
		}
	}

	if(maxCount > 0 && keypoints.size() > (size_t) maxCount)
	{
		General::sort(keypoints, CmpKeypointScores());
		keypoints.resize(maxCount);
	}
}

void Yape::read(const FileNode& objnode)
{
	radius = (int) objnode["radius"];
	threshold = (int) objnode["threshold"];
	nOctaves = (int) objnode["noctaves"];
	nViews = (int) objnode["nviews"];
	baseFeatureSize = (int) objnode["base-feature-size"];
	clusteringDistance = (int) objnode["clustering-distance"];
}

void Yape::write(FileStorage& fs, const String& name) const
{
	internal::WriteStructContext ws(fs, name, CV_NODE_MAP);

	fs << "radius" << radius
		<< "threshold" << threshold
		<< "noctaves" << nOctaves
		<< "nviews" << nViews
		<< "base-feature-size" << baseFeatureSize
		<< "clustering-distance" << clusteringDistance;
}

void Yape::setVerbose(bool _verbose)
{
	verbose = _verbose;
}