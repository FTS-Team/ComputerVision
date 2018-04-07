#include "stdafx.h"
#include "PlanarObjectDetector.h"



PlanarObjectDetector::PlanarObjectDetector()
{
}

PlanarObjectDetector::PlanarObjectDetector(const FileNode& node)
{
	read(node);
}

PlanarObjectDetector::PlanarObjectDetector(const vector<Mat>& pyr, int npoints,
	int patchSize, int nstructs, int structSize,
	int nviews, const Yape& detector,
	const PatchGenerator& patchGenerator)
{
	train(pyr, npoints, patchSize, nstructs,
		structSize, nviews, detector, patchGenerator);
}

PlanarObjectDetector::~PlanarObjectDetector()
{
}

vector<KeyPoint> PlanarObjectDetector::getModelPoints() const
{
	return modelPoints;
}

void PlanarObjectDetector::train(const vector<Mat>& pyr, int npoints,
	int patchSize, int nstructs, int structSize,
	int nviews, const Yape& detector,
	const PatchGenerator& patchGenerator)
{
	modelROI = Rect(0, 0, pyr[0].cols, pyr[0].rows);
	ldetector = detector;
	ldetector.setVerbose(verbose);
	ldetector.getMostStable2D(pyr[0], modelPoints, npoints, patchGenerator);

	npoints = (int) modelPoints.size();
	fernClassifier.setVerbose(verbose);
	fernClassifier.trainFromSingleView(pyr[0], modelPoints,
		patchSize, (int) modelPoints.size(), nstructs, structSize, nviews,
		FernClassifier::COMPRESSION_NONE, patchGenerator);
}

void PlanarObjectDetector::train(const vector<Mat>& pyr, const vector<KeyPoint>& keypoints,
	int patchSize, int nstructs, int structSize,
	int nviews, const Yape& detector,
	const PatchGenerator& patchGenerator)
{
	modelROI = Rect(0, 0, pyr[0].cols, pyr[0].rows);
	ldetector = detector;
	ldetector.setVerbose(verbose);
	modelPoints.resize(keypoints.size());
	std::copy(keypoints.begin(), keypoints.end(), modelPoints.begin());

	fernClassifier.setVerbose(verbose);
	fernClassifier.trainFromSingleView(pyr[0], modelPoints,
		patchSize, (int) modelPoints.size(), nstructs, structSize, nviews,
		FernClassifier::COMPRESSION_NONE, patchGenerator);
}

void PlanarObjectDetector::read(const FileNode& node)
{
	FileNodeIterator it = node["model-roi"].begin(), it_end;
	it >> modelROI.x >> modelROI.y >> modelROI.width >> modelROI.height;
	ldetector.read(node["detector"]);
	fernClassifier.read(node["fern-classifier"]);
	cv::read(node["model-points"], modelPoints);
	CV_Assert(modelPoints.size() == (size_t) fernClassifier.getClassCount());
}


void PlanarObjectDetector::write(FileStorage& fs, const String& objname) const
{
	internal::WriteStructContext ws(fs, objname, CV_NODE_MAP);

	{
		internal::WriteStructContext wsroi(fs, "model-roi", CV_NODE_SEQ + CV_NODE_FLOW);
		cv::write(fs, modelROI.x);
		cv::write(fs, modelROI.y);
		cv::write(fs, modelROI.width);
		cv::write(fs, modelROI.height);
	}
	ldetector.write(fs, "detector");
	cv::write(fs, "model-points", modelPoints);
	fernClassifier.write(fs, "fern-classifier");
}


bool PlanarObjectDetector::operator()(const Mat& image, Mat& H, vector<Point2f>& corners) const
{
	vector<Mat> pyr;
	buildPyramid(image, pyr, ldetector.nOctaves - 1);
	vector<KeyPoint> keypoints;
	ldetector(pyr, keypoints);

	return (*this)(pyr, keypoints, H, corners);
}

bool PlanarObjectDetector::operator()(const vector<Mat>& pyr, const vector<KeyPoint>& keypoints,
	Mat& matH, vector<Point2f>& corners, vector<int>* pairs) const
{
	int i, j, m = (int) modelPoints.size(), n = (int) keypoints.size();
	vector<int> bestMatches(m, -1);
	vector<float> maxLogProb(m, -FLT_MAX);
	vector<float> signature;
	vector<Point2f> fromPt, toPt;

	for(i = 0; i < n; i++)
	{
		KeyPoint kpt = keypoints[i];
		CV_Assert(0 <= kpt.octave && kpt.octave < (int) pyr.size());
		kpt.pt.x /= (float) (1 << kpt.octave);
		kpt.pt.y /= (float) (1 << kpt.octave);
		int k = fernClassifier(pyr[kpt.octave], kpt.pt, signature);
		if(k >= 0 && (bestMatches[k] < 0 || signature[k] > maxLogProb[k]))
		{
			maxLogProb[k] = signature[k];
			bestMatches[k] = i;
		}
	}

	if(pairs)
		pairs->resize(0);

	for(i = 0; i < m; i++)
		if(bestMatches[i] >= 0)
		{
			fromPt.push_back(modelPoints[i].pt);
			toPt.push_back(keypoints[bestMatches[i]].pt);
		}

	if(fromPt.size() < 4)
		return false;

	vector<uchar> mask;
	matH = findHomography(fromPt, toPt, RANSAC, 10, mask);
	if(matH.data)
	{
		const Mat_<double>& H = matH;
		corners.resize(4);
		for(i = 0; i < 4; i++)
		{
			Point2f pt((float) (modelROI.x + (i == 0 || i == 3 ? 0 : modelROI.width)),
				(float) (modelROI.y + (i <= 1 ? 0 : modelROI.height)));
			double w = 1. / (H(2, 0)*pt.x + H(2, 1)*pt.y + H(2, 2));
			corners[i] = Point2f((float) ((H(0, 0)*pt.x + H(0, 1)*pt.y + H(0, 2))*w),
				(float) ((H(1, 0)*pt.x + H(1, 1)*pt.y + H(1, 2))*w));
		}
	}

	if(pairs)
	{
		for(i = j = 0; i < m; i++)
			if(bestMatches[i] >= 0 && mask[j++])
			{
				pairs->push_back(i);
				pairs->push_back(bestMatches[i]);
			}
	}

	return matH.data != 0;
}


void PlanarObjectDetector::setVerbose(bool _verbose)
{
	verbose = _verbose;
}

