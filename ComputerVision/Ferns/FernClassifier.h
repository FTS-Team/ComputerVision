#pragma once
#include "PatchGenerator.h"
#include <opencv2\opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

class FernClassifier
{
public:
	FernClassifier();
	FernClassifier(const FileNode& node);
	FernClassifier(const vector<vector<Point2f> >& points,
		const vector<Mat>& refimgs,
		const vector<vector<int> >& labels = vector<vector<int> >(),
		int _nclasses = 0, int _patchSize = PATCH_SIZE,
		int _signatureSize = DEFAULT_SIGNATURE_SIZE,
		int _nstructs = DEFAULT_STRUCTS,
		int _structSize = DEFAULT_STRUCT_SIZE,
		int _nviews = DEFAULT_VIEWS,
		int _compressionMethod = COMPRESSION_NONE,
		const PatchGenerator& patchGenerator = PatchGenerator());
	virtual ~FernClassifier();
	virtual void read(const FileNode& n);
	virtual void write(FileStorage& fs, const String& name = String()) const;
	virtual void trainFromSingleView(const Mat& image,
		const vector<KeyPoint>& keypoints,
		int _patchSize = PATCH_SIZE,
		int _signatureSize = DEFAULT_SIGNATURE_SIZE,
		int _nstructs = DEFAULT_STRUCTS,
		int _structSize = DEFAULT_STRUCT_SIZE,
		int _nviews = DEFAULT_VIEWS,
		int _compressionMethod = COMPRESSION_NONE,
		const PatchGenerator& patchGenerator = PatchGenerator());
	virtual void train(const vector<vector<Point2f> >& points,
		const vector<Mat>& refimgs,
		const vector<vector<int> >& labels = vector<vector<int> >(),
		int _nclasses = 0, int _patchSize = PATCH_SIZE,
		int _signatureSize = DEFAULT_SIGNATURE_SIZE,
		int _nstructs = DEFAULT_STRUCTS,
		int _structSize = DEFAULT_STRUCT_SIZE,
		int _nviews = DEFAULT_VIEWS,
		int _compressionMethod = COMPRESSION_NONE,
		const PatchGenerator& patchGenerator = PatchGenerator());
	virtual int operator()(const Mat& img, Point2f kpt, vector<float>& signature) const;
	virtual int operator()(const Mat& patch, vector<float>& signature) const;
	virtual void clear();
	virtual bool empty() const;
	void setVerbose(bool verbose);

	int getClassCount() const;
	int getStructCount() const;
	int getStructSize() const;
	int getSignatureSize() const;
	int getCompressionMethod() const;
	Size getPatchSize() const;

	struct Feature
	{
		uchar x1, y1, x2, y2;
		Feature() : x1(0), y1(0), x2(0), y2(0) {}
		Feature(int _x1, int _y1, int _x2, int _y2)
			: x1((uchar) _x1), y1((uchar) _y1), x2((uchar) _x2), y2((uchar) _y2)
		{
		}
		template<typename _Tp> bool operator ()(const Mat_<_Tp>& patch) const
		{
			return patch(y1, x1) > patch(y2, x2);
		}
	};

	enum
	{
		PATCH_SIZE = 31,
		DEFAULT_STRUCTS = 50,
		DEFAULT_STRUCT_SIZE = 9,
		DEFAULT_VIEWS = 5000,
		DEFAULT_SIGNATURE_SIZE = 176,
		COMPRESSION_NONE = 0,
		COMPRESSION_RANDOM_PROJ = 1,
		COMPRESSION_PCA = 2,
		DEFAULT_COMPRESSION_METHOD = COMPRESSION_NONE
	};

protected:
	virtual void prepare(int _nclasses, int _patchSize, int _signatureSize,
		int _nstructs, int _structSize,
		int _nviews, int _compressionMethod);
	virtual void finalize(RNG& rng);
	virtual int getLeaf(int fidx, const Mat& patch) const;

	bool verbose;
	int nstructs;
	int structSize;
	int nclasses;
	int signatureSize;
	int compressionMethod;
	int leavesPerStruct;
	Size patchSize;
	vector<Feature> features;
	vector<int> classCounters;
	vector<float> posteriors;
};

