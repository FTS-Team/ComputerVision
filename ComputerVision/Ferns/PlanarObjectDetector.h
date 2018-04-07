#pragma once
#include "PatchGenerator.h"
#include "Yape.h"
#include "FernClassifier.h"

#include <vector>
using namespace std;

class PlanarObjectDetector
{
public:
	PlanarObjectDetector();
	PlanarObjectDetector(const FileNode& node);
	PlanarObjectDetector(const vector<Mat>& pyr, int _npoints = 300,
		int _patchSize = FernClassifier::PATCH_SIZE,
		int _nstructs = FernClassifier::DEFAULT_STRUCTS,
		int _structSize = FernClassifier::DEFAULT_STRUCT_SIZE,
		int _nviews = FernClassifier::DEFAULT_VIEWS,
		const Yape& detector = Yape(),
		const PatchGenerator& patchGenerator = PatchGenerator());
	virtual ~PlanarObjectDetector();
	virtual void train(const vector<Mat>& pyr, int _npoints = 300,
		int _patchSize = FernClassifier::PATCH_SIZE,
		int _nstructs = FernClassifier::DEFAULT_STRUCTS,
		int _structSize = FernClassifier::DEFAULT_STRUCT_SIZE,
		int _nviews = FernClassifier::DEFAULT_VIEWS,
		const Yape& detector = Yape(),
		const PatchGenerator& patchGenerator = PatchGenerator());
	virtual void train(const vector<Mat>& pyr, const vector<KeyPoint>& keypoints,
		int _patchSize = FernClassifier::PATCH_SIZE,
		int _nstructs = FernClassifier::DEFAULT_STRUCTS,
		int _structSize = FernClassifier::DEFAULT_STRUCT_SIZE,
		int _nviews = FernClassifier::DEFAULT_VIEWS,
		const Yape& detector = Yape(),
		const PatchGenerator& patchGenerator = PatchGenerator());
	Rect getModelROI() const;
	vector<KeyPoint> getModelPoints() const;
	const Yape& getDetector() const;
	const FernClassifier& getClassifier() const;
	void setVerbose(bool verbose);

	void read(const FileNode& node);
	void write(FileStorage& fs, const String& name = String()) const;
	bool operator()(const Mat& image, CV_OUT Mat& H, CV_OUT vector<Point2f>& corners) const;
	bool operator()(const vector<Mat>& pyr, const vector<KeyPoint>& keypoints,
		CV_OUT Mat& H, CV_OUT vector<Point2f>& corners,
		CV_OUT vector<int>* pairs = 0) const;

protected:
	bool verbose;
	Rect modelROI;
	vector<KeyPoint> modelPoints;
	Yape ldetector;
	FernClassifier fernClassifier;
};

