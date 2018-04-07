#include "stdafx.h"
#include "FernClassifier.h"
#include "General.h"


FernClassifier::FernClassifier()
{
	verbose = false;
	clear();
}


FernClassifier::FernClassifier(const FileNode& node)
{
	verbose = false;
	clear();
	read(node);
}

FernClassifier::~FernClassifier()
{
}


int FernClassifier::getClassCount() const
{
	return nclasses;
}


int FernClassifier::getStructCount() const
{
	return nstructs;
}


int FernClassifier::getStructSize() const
{
	return structSize;
}


int FernClassifier::getSignatureSize() const
{
	return signatureSize;
}


int FernClassifier::getCompressionMethod() const
{
	return compressionMethod;
}


Size FernClassifier::getPatchSize() const
{
	return patchSize;
}


FernClassifier::FernClassifier(const vector<vector<Point2f> >& points,
	const vector<Mat>& refimgs,
	const vector<vector<int> >& labels,
	int _nclasses, int _patchSize,
	int _signatureSize, int _nstructs,
	int _structSize, int _nviews, int _compressionMethod,
	const PatchGenerator& patchGenerator)
{
	verbose = false;
	clear();
	train(points, refimgs, labels, _nclasses, _patchSize,
		_signatureSize, _nstructs, _structSize, _nviews,
		_compressionMethod, patchGenerator);
}


void FernClassifier::write(FileStorage& fs, const String& objname) const
{
	internal::WriteStructContext ws(fs, objname, CV_NODE_MAP);

	cv::write(fs, "nstructs", nstructs);
	cv::write(fs, "struct-size", structSize);
	cv::write(fs, "nclasses", nclasses);
	cv::write(fs, "signature-size", signatureSize);
	cv::write(fs, "compression-method", compressionMethod);
	cv::write(fs, "patch-size", patchSize.width);
	{
		internal::WriteStructContext wsf(fs, "features", CV_NODE_SEQ + CV_NODE_FLOW);
		int i, nfeatures = (int) features.size();
		for(i = 0; i < nfeatures; i++)
		{
			cv::write(fs, features[i].y1*patchSize.width + features[i].x1);
			cv::write(fs, features[i].y2*patchSize.width + features[i].x2);
		}
	}
	{
		internal::WriteStructContext wsp(fs, "posteriors", CV_NODE_SEQ + CV_NODE_FLOW);
		cv::write(fs, posteriors);
	}
}


void FernClassifier::read(const FileNode& objnode)
{
	clear();

	nstructs = (int) objnode["nstructs"];
	structSize = (int) objnode["struct-size"];
	nclasses = (int) objnode["nclasses"];
	signatureSize = (int) objnode["signature-size"];
	compressionMethod = (int) objnode["compression-method"];
	patchSize.width = patchSize.height = (int) objnode["patch-size"];
	leavesPerStruct = 1 << structSize;

	FileNode _nodes = objnode["features"];
	int i, nfeatures = structSize * nstructs;
	features.resize(nfeatures);
	FileNodeIterator it = _nodes.begin(), it_end = _nodes.end();
	for(i = 0; i < nfeatures && it != it_end; i++)
	{
		int ofs1, ofs2;
		it >> ofs1 >> ofs2;
		features[i] = Feature(ofs1%patchSize.width, ofs1 / patchSize.width,
			ofs2%patchSize.width, ofs2 / patchSize.width);
	}

	FileNode _posteriors = objnode["posteriors"];
	int psz = leavesPerStruct * nstructs*signatureSize;
	posteriors.reserve(psz);
	_posteriors >> posteriors;
}


void FernClassifier::clear()
{
	signatureSize = nclasses = nstructs = structSize = compressionMethod = leavesPerStruct = 0;
	vector<Feature>().swap(features);
	vector<float>().swap(posteriors);
}

bool FernClassifier::empty() const
{
	return features.empty();
}

int FernClassifier::getLeaf(int fern, const Mat& _patch) const
{
	assert(0 <= fern && fern < nstructs);
	size_t fofs = fern * structSize, idx = 0;
	const Mat_<uchar>& patch = (const Mat_<uchar>&)_patch;

	for(int i = 0; i < structSize; i++)
	{
		const Feature& f = features[fofs + i];
		idx = (idx << 1) + f(patch);
	}

	return (int) (fern*leavesPerStruct + idx);
}


void FernClassifier::prepare(int _nclasses, int _patchSize, int _signatureSize,
	int _nstructs, int _structSize,
	int _nviews, int _compressionMethod)
{
	clear();

	CV_Assert(_nclasses > 1 && _patchSize >= 5 && _nstructs > 0 &&
		_nviews > 0 && _structSize > 0 &&
		(_compressionMethod == COMPRESSION_NONE ||
			_compressionMethod == COMPRESSION_RANDOM_PROJ ||
			_compressionMethod == COMPRESSION_PCA));

	nclasses = _nclasses;
	patchSize = Size(_patchSize, _patchSize);
	nstructs = _nstructs;
	structSize = _structSize;
	signatureSize = _compressionMethod == COMPRESSION_NONE ? nclasses : std::min(_signatureSize, nclasses);
	compressionMethod = signatureSize == nclasses ? COMPRESSION_NONE : _compressionMethod;

	leavesPerStruct = 1 << structSize;

	int i, nfeatures = structSize * nstructs;

	features = vector<Feature>(nfeatures);
	posteriors = vector<float>(leavesPerStruct*nstructs*nclasses, 1.f);
	classCounters = vector<int>(nclasses, leavesPerStruct);

	CV_Assert(patchSize.width <= 256 && patchSize.height <= 256);
	RNG& rng = theRNG();

	for(i = 0; i < nfeatures; i++)
	{
		int x1 = (unsigned) rng % patchSize.width;
		int y1 = (unsigned) rng % patchSize.height;
		int x2 = (unsigned) rng % patchSize.width;
		int y2 = (unsigned) rng % patchSize.height;
		features[i] = Feature(x1, y1, x2, y2);
	}
}

static int calcNumPoints(const vector<vector<Point2f> >& points)
{
	size_t count = 0;
	for(size_t i = 0; i < points.size(); i++)
		count += points[i].size();
	return (int) count;
}

void FernClassifier::train(const vector<vector<Point2f> >& points,
	const vector<Mat>& refimgs,
	const vector<vector<int> >& labels,
	int _nclasses, int _patchSize,
	int _signatureSize, int _nstructs,
	int _structSize, int _nviews, int _compressionMethod,
	const PatchGenerator& patchGenerator)
{
	CV_Assert(points.size() == refimgs.size());
	int numPoints = calcNumPoints(points);
	_nclasses = (!labels.empty() && _nclasses>0) ? _nclasses : numPoints;
	CV_Assert(labels.empty() || labels.size() == points.size());


	prepare(_nclasses, _patchSize, _signatureSize, _nstructs,
		_structSize, _nviews, _compressionMethod);

	// pass all the views of all the samples through the generated trees and accumulate
	// the statistics (posterior probabilities) in leaves.
	Mat patch;
	RNG& rng = theRNG();

	int globalPointIdx = 0;
	for(size_t imgIdx = 0; imgIdx < points.size(); imgIdx++)
	{
		const Point2f* imgPoints = &points[imgIdx][0];
		const int* imgLabels = labels.empty() ? 0 : &labels[imgIdx][0];
		for(size_t pointIdx = 0; pointIdx < points[imgIdx].size(); pointIdx++, globalPointIdx++)
		{
			Point2f pt = imgPoints[pointIdx];
			const Mat& src = refimgs[imgIdx];
			int classId = imgLabels == 0 ? globalPointIdx : imgLabels[pointIdx];
			/*if(verbose && (globalPointIdx + 1)*General::progressBarSize / numPoints != globalPointIdx * General::progressBarSize / numPoints)
				putchar('.');*/
			if(verbose)
				General::displayRateOfProgress(globalPointIdx, numPoints);
			CV_Assert(0 <= classId && classId < nclasses);
			classCounters[classId] += _nviews;
			for(int v = 0; v < _nviews; v++)
			{
				patchGenerator(src, pt, patch, patchSize, rng);
				for(int f = 0; f < nstructs; f++)
					posteriors[getLeaf(f, patch)*nclasses + classId]++;
			}
		}
	}
	/*if(verbose)
		putchar('\n');*/

	finalize(rng);
}


void FernClassifier::trainFromSingleView(const Mat& image,
	const vector<KeyPoint>& keypoints,
	int _patchSize, int _signatureSize,
	int _nstructs, int _structSize,
	int _nviews, int _compressionMethod,
	const PatchGenerator& patchGenerator)
{
	prepare((int) keypoints.size(), _patchSize, _signatureSize, _nstructs,
		_structSize, _nviews, _compressionMethod);
	int i, j, k, nsamples = (int) keypoints.size(), maxOctave = 0;
	for(i = 0; i < nsamples; i++)
	{
		classCounters[i] = _nviews;
		maxOctave = std::max(maxOctave, keypoints[i].octave);
	}

	double maxScale = patchGenerator.lambdaMax * 2;
	Mat canvas(cvRound(std::max(image.cols, image.rows)*maxScale + patchSize.width * 2 + 10),
		cvRound(std::max(image.cols, image.rows)*maxScale + patchSize.width * 2 + 10), image.type());
	Mat noisebuf;
	vector<Mat> pyrbuf(maxOctave + 1), pyr(maxOctave + 1);
	Point2f center0((image.cols - 1)*0.5f, (image.rows - 1)*0.5f),
		center1((canvas.cols - 1)*0.5f, (canvas.rows - 1)*0.5f);
	Mat matM(2, 3, CV_64F);
	double *M = (double*) matM.data;
	RNG& rng = theRNG();

	Mat patch(patchSize, CV_8U);

	for(i = 0; i < _nviews; i++)
	{
		patchGenerator.generateRandomTransform(center0, center1, matM, rng);

		CV_Assert(matM.type() == CV_64F);
		Rect roi(INT_MAX, INT_MAX, INT_MIN, INT_MIN);

		for(k = 0; k < 4; k++)
		{
			Point2f pt0, pt1;
			pt0.x = (float) (k == 0 || k == 3 ? 0 : image.cols);
			pt0.y = (float) (k < 2 ? 0 : image.rows);
			pt1.x = (float) (M[0] * pt0.x + M[1] * pt0.y + M[2]);
			pt1.y = (float) (M[3] * pt0.x + M[4] * pt0.y + M[5]);

			roi.x = std::min(roi.x, cvFloor(pt1.x));
			roi.y = std::min(roi.y, cvFloor(pt1.y));
			roi.width = std::max(roi.width, cvCeil(pt1.x));
			roi.height = std::max(roi.height, cvCeil(pt1.y));
		}

		roi.width -= roi.x + 1;
		roi.height -= roi.y + 1;

		Mat canvas_roi(canvas, roi);
		M[2] -= roi.x;
		M[5] -= roi.y;

		Size size = canvas_roi.size();
		rng.fill(canvas_roi, RNG::UNIFORM, Scalar::all(0), Scalar::all(256));
		warpAffine(image, canvas_roi, matM, size, INTER_LINEAR, BORDER_TRANSPARENT);

		pyr[0] = canvas_roi;
		for(j = 1; j <= maxOctave; j++)
		{
			size = Size((size.width + 1) / 2, (size.height + 1) / 2);
			if(pyrbuf[j].cols < size.width*size.height)
				pyrbuf[j].create(1, size.width*size.height, image.type());
			pyr[j] = Mat(size, image.type(), pyrbuf[j].data);
			pyrDown(pyr[j - 1], pyr[j]);
		}

		if(patchGenerator.noiseRange > 0)
		{
			const int noiseDelta = 128;
			if(noisebuf.cols < pyr[0].cols*pyr[0].rows)
				noisebuf.create(1, pyr[0].cols*pyr[0].rows, image.type());
			for(j = 0; j <= maxOctave; j++)
			{
				Mat noise(pyr[j].size(), image.type(), noisebuf.data);
				rng.fill(noise, RNG::UNIFORM, Scalar::all(-patchGenerator.noiseRange + noiseDelta),
					Scalar::all(patchGenerator.noiseRange + noiseDelta));
				addWeighted(pyr[j], 1, noise, 1, -noiseDelta, pyr[j]);
			}
		}

		for(j = 0; j < nsamples; j++)
		{
			KeyPoint kpt = keypoints[j];
			float scale = 1.f / (1 << kpt.octave);
			Point2f pt((float) ((M[0] * kpt.pt.x + M[1] * kpt.pt.y + M[2])*scale),
				(float) ((M[3] * kpt.pt.x + M[4] * kpt.pt.y + M[5])*scale));
			getRectSubPix(pyr[kpt.octave], patchSize, pt, patch, patch.type());
			for(int f = 0; f < nstructs; f++)
				posteriors[getLeaf(f, patch)*nclasses + j]++;
		}

		/*if(verbose && (i + 1)*General::progressBarSize / _nviews != i * General::progressBarSize / _nviews)
			putchar('.');*/
		if(verbose)
			General::displayRateOfProgress(i, _nviews);
	}
	/*if(verbose)
		putchar('\n');*/

	finalize(rng);
}


int FernClassifier::operator()(const Mat& img, Point2f pt, vector<float>& signature) const
{
	Mat patch;
	getRectSubPix(img, patchSize, pt, patch, img.type());
	return (*this)(patch, signature);
}


int FernClassifier::operator()(const Mat& patch, vector<float>& signature) const
{
	if(posteriors.empty())
		CV_Error(CV_StsNullPtr,
			"The descriptor has not been trained or "
			"the floating-point posteriors have been deleted");
	CV_Assert(patch.size() == patchSize);

	int i, j, sz = signatureSize;
	signature.resize(sz);
	float* s = &signature[0];

	for(j = 0; j < sz; j++)
		s[j] = 0;

	for(i = 0; i < nstructs; i++)
	{
		int lf = getLeaf(i, patch);
		const float* ldata = &posteriors[lf*signatureSize];
		for(j = 0; j <= sz - 4; j += 4)
		{
			float t0 = s[j] + ldata[j];
			float t1 = s[j + 1] + ldata[j + 1];
			s[j] = t0; s[j + 1] = t1;
			t0 = s[j + 2] + ldata[j + 2];
			t1 = s[j + 3] + ldata[j + 3];
			s[j + 2] = t0; s[j + 3] = t1;
		}
		for(; j < sz; j++)
			s[j] += ldata[j];
	}

	j = 0;
	if(signatureSize == nclasses && compressionMethod == COMPRESSION_NONE)
	{
		for(i = 1; i < nclasses; i++)
			if(s[j] < s[i])
				j = i;
	}
	return j;
}


void FernClassifier::finalize(RNG&)
{
	int i, j, k, n = nclasses;
	vector<double> invClassCounters(n);
	Mat_<double> _temp(1, n);
	double* temp = &_temp(0, 0);

	for(i = 0; i < n; i++)
		invClassCounters[i] = 1. / classCounters[i];

	for(i = 0; i < nstructs; i++)
	{
		for(j = 0; j < leavesPerStruct; j++)
		{
			float* P = &posteriors[(i*leavesPerStruct + j)*nclasses];
			double sum = 0;
			for(k = 0; k < n; k++)
				sum += P[k] * invClassCounters[k];
			sum = 1. / sum;
			for(k = 0; k < n; k++)
				temp[k] = P[k] * invClassCounters[k] * sum;
			log(_temp, _temp);
			for(k = 0; k < n; k++)
				P[k] = (float) temp[k];
		}
	}
}

void FernClassifier::setVerbose(bool _verbose)
{
	verbose = _verbose;
}