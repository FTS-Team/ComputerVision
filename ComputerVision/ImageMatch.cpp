#include "stdafx.h"

#include "ImageMatch.h"
#include "Ferns\Yape.h"
#include "Ferns\PatchGenerator.h"
#include "Ferns\PlanarObjectDetector.h"
#include <conio.h>


UINT ImageMatch::SIFT(CImage& CImage1, CImage& CImage2) {


	//����ת��
	Mat image1;
	Mat image2;
	CImage_Mat::CImageToMat(CImage1, image1);
	CImage_Mat::CImageToMat(CImage2, image2);

	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
	vector<KeyPoint> kp1, kp2;//����������

    //����sift��������
	sift->detect(image1, kp1);
	sift->detect(image2, kp2);

	//�ڴ��л���������
	Mat res1, res2;
	drawKeypoints(image1, kp1, res1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(image2, kp2, res2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


	namedWindow("KeyPoints of image1", WINDOW_GUI_EXPANDED);//��������
	namedWindow("KeyPoints of image2", WINDOW_GUI_EXPANDED);//��������
	imshow("KeyPoints of image1", res1); // ����ͼ��
	imshow("KeyPoints of image2", res2); // ����ͼ��


	//������������
	Mat des1, des2;
	sift->compute(image1, kp1, des1);
	sift->compute(image2, kp2, des2);


	
	//ͼ��ƥ��(����ھӷ�)
	//BFMatcher matcher(NORM_L2, false);
	//vector<DMatch> matches;//ɸѡ���ƥ���
	//vector<vector<DMatch>> knnMatches;//����һ����������װ����ڵ�ʹν��ڵ�
	//matcher.knnMatch(des1, des2, knnMatches, 2);
	//const int ratio = 0.5;

	//for (int n = 0; n < knnMatches.size(); n++) {
	//	DMatch& bestmatch = knnMatches[n][0];
	//	DMatch& bettermatch = knnMatches[n][1];
	//	if (bestmatch.distance < ratio*bettermatch.distance)//ɸѡ�����������ĵ�
	//	{
	//		matches.push_back(bestmatch);//�����������ĵ㱣����matches
	//	}
	//}


	//RANSACƥ��
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);
	vector<DMatch> goodMatch = ransac(matches, kp1, kp2);

	//����ƥ����ͼ��
	Mat imgMatch;
	drawMatches(image1, kp1, image2, kp2, goodMatch , imgMatch);
	namedWindow("matches", WINDOW_GUI_EXPANDED);//��������
	imshow("matches", imgMatch);
	

	return 0;
}



UINT ImageMatch::SURF(CImage& CImage1, CImage& CImage2)
{
	//CImage to Mat
	Mat img_1;
	Mat img_2;
	CImage_Mat::CImageToMat(CImage1, img_1);
	CImage_Mat::CImageToMat(CImage2, img_2);

	//Create SURF class pointer
	Ptr<Feature2D> surf = xfeatures2d::SURF::create();
	//Detect the keypoints
	vector<KeyPoint> keypoints_1, keypoints_2;
	surf->detect(img_1, keypoints_1);
	surf->detect(img_2, keypoints_2);
	//Draw keypoints
	Mat img_keypoints_1, img_keypoints_2;
	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	namedWindow("surf_keypoints_1", WINDOW_GUI_EXPANDED);
	namedWindow("surf_keypoints_2", WINDOW_GUI_EXPANDED);
	imshow("surf_keypoints_1", img_keypoints_1);
	imshow("surf_keypoints_2", img_keypoints_2);

	//Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	surf->compute(img_1, keypoints_1, descriptors_1);
	surf->compute(img_2, keypoints_2, descriptors_2);


	//Matching descriptor vector using BFMatcher
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	//����ƥ����Ĺؼ���
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
	namedWindow("��ƥ������ǰmatchͼ", WINDOW_GUI_EXPANDED);
	imshow("��ƥ������ǰmatchͼ", img_matches);

	//����RANSAC����������ƥ���,������ƥ����Ĺؼ���
	Mat img_matches_after;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, ransac(matches, keypoints_1, keypoints_2), img_matches_after);
	namedWindow("��ƥ��������matchͼ", WINDOW_GUI_EXPANDED);
	imshow("��ƥ��������matchͼ", img_matches_after);

	return 0;
}



UINT ImageMatch::ORBMatch(CImage& CImage1, CImage & CImage2) {


	Mat img_1;
	Mat img_2;
	CImage_Mat::CImageToMat(CImage1, img_1);
	CImage_Mat::CImageToMat(CImage2, img_2);

	// -- Step 1: Detect the keypoints using STAR Detector 
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	int nkeypoint = 50;//���������
	Ptr<cv::ORB> orb = cv::ORB::create(nkeypoint);

	orb->detect(img_1, keypoints_1);
	orb->detect(img_2, keypoints_2);

	Mat res1, res2;
	drawKeypoints(img_1, keypoints_1, res1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img_2, keypoints_2, res2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


	namedWindow("KeyPoints of image1", WINDOW_GUI_EXPANDED);
	namedWindow("KeyPoints of image2", WINDOW_GUI_EXPANDED);
	imshow("KeyPoints of image1", res1); 
	imshow("KeyPoints of image2", res2); 

	// -- Stpe 2: Calculate descriptors (feature vectors) 
	Mat descriptors_1, descriptors_2;
	orb->compute(img_1, keypoints_1, descriptors_1);
	orb->compute(img_2, keypoints_2, descriptors_2);

	//-- Step 3: Matching descriptor vectors with a ransac matcher 
	BFMatcher matcher(NORM_HAMMING);
	std::vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);



	// -- dwaw matches 
	Mat img_mathes;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, ransac(matches, keypoints_1, keypoints_2), img_mathes);
	// -- show 
	namedWindow("OPENCV_ORB", WINDOW_GUI_EXPANDED);
	imshow("OPENCV_ORB", img_mathes);

	return 0;

}

UINT ImageMatch::FERNS(CString img1_path, CString img2_path)
{
	// ��CStringת��Ϊascii����
	CT2A img1_path_char(img1_path);
	CT2A img2_path_char(img2_path);

#pragma region pattern image
	// ����ͼƬ
	Mat pattern_origin, pattern_gray;
	pattern_origin = imread(img1_path_char.m_psz);
	//Mat object = imread(img1_path_char.m_psz, CV_LOAD_IMAGE_GRAYSCALE);
	if(!pattern_origin.data)
	{
		_cprintf("Can not load %s \n"
			"Usage: find_obj [<object_filename> ]\n",
			img1_path_char.m_psz);
		return 1;
	}
	cvtColor(pattern_origin, pattern_gray, CV_BGR2GRAY);

	//Size patchSize(32, 32);
	Yape yape_detector(7, 20, 4, 5000, 32, 2);
	yape_detector.setVerbose(true);
	PlanarObjectDetector detector;

	vector<Mat> pattern_pyramid;
	int blurKSize = 7;
	double sigma = 0;
	GaussianBlur(pattern_gray, pattern_gray, Size(blurKSize, blurKSize), sigma, sigma);

	vector<KeyPoint> pattern_keypoints;
	PatchGenerator gen(0, 256, 5);

	// ���Դ��ļ����ط�����
	CString img1_name = img1_path.Right(img1_path.GetLength() - img1_path.ReverseFind('\\') - 1);
	CT2A img1_name_char(img1_name);
	string model_filename = format(".\\Ferns Classifier Datas\\%s_model.xml.gz", img1_name_char.m_psz);
	_cprintf("Trying to load %s ...\n", model_filename.c_str());
	FileStorage fs(model_filename, FileStorage::READ);
	if(fs.isOpened())
	{
		// ���ļ��ж�ȡ����������
		detector.read(fs.getFirstTopLevelNode());
		_cprintf("Successfully loaded %s.\n", model_filename.c_str());
		pattern_keypoints = detector.getModelPoints();
	} else
	{
		// ��ͼ�����ѵ������÷�����
		_cprintf("The file not found and can not be read. Let's train the model.\n");
		_cprintf("Step 1. Finding the robust keypoints ...\n");
		buildPyramid(pattern_gray, pattern_pyramid, yape_detector.nOctaves - 1);
		yape_detector.setVerbose(true);
		yape_detector.getMostStable2D(pattern_gray, pattern_keypoints, 400, gen);
		_cprintf("Done.\nStep 2. Training ferns-based planar object detector ...\n");
		detector.setVerbose(true);

		detector.train(pattern_pyramid, pattern_keypoints, 32, 30, 11, 10000, yape_detector, gen);
		_cprintf("Done.\nStep 3. Saving the model to %s ...\n", model_filename.c_str());
		if(fs.open(model_filename, FileStorage::WRITE))
			detector.write(fs, "ferns_model");
	}
	_cprintf("Now find the keypoints in the image, try recognize them and compute the homography matrix\n");
	fs.release();
	_cprintf("Object keypoints: %d\n", pattern_keypoints.size());

	// ��ͼ1�л���������
	namedWindow("Object", 1);
	Mat objectColor;
	cvtColor(pattern_gray, objectColor, CV_GRAY2BGR);
	for(int i = 0; i < (int)pattern_keypoints.size(); i++)
	{
		circle(objectColor, pattern_keypoints[i].pt, 2, Scalar(0, 0, 255), -1);
		circle(objectColor, pattern_keypoints[i].pt, (1 << pattern_keypoints[i].octave) * 15, Scalar(0, 255, 0), 1);
	}
	imshow("Object", objectColor);
#pragma endregion

	// ��ȡͼ2��Ϊƥ������
	Mat image;
	Mat frame, frame_gray;
	frame = imread(img2_path_char.m_psz);
	cvtColor(frame, frame_gray, CV_BGR2GRAY);

	// Ѱ��ͼ2��������
	double imgscale = 1;
	resize(frame_gray, image, Size(), 1. / imgscale, 1. / imgscale, INTER_CUBIC);
	GaussianBlur(image, image, Size(blurKSize, blurKSize), sigma, sigma);

	vector<Mat> imgpyr;
	buildPyramid(image, imgpyr, yape_detector.nOctaves - 1);

	Mat correspond(pattern_gray.rows + image.rows, std::max(pattern_gray.cols, image.cols), CV_8UC3);
	correspond = Scalar(0.);
	Mat part(correspond, Rect(0, 0, pattern_gray.cols, pattern_gray.rows));
	cvtColor(pattern_gray, part, CV_GRAY2BGR);
	part = Mat(correspond, Rect(0, pattern_gray.rows, image.cols, image.rows));
	cvtColor(image, part, CV_GRAY2BGR);

	// ����ͼ2��������
	vector<KeyPoint> imgKeypoints;
	yape_detector(imgpyr, imgKeypoints, 1000);
	_cprintf("Image keypoints: %d\n", imgKeypoints.size());

	Mat imageColor;
	cvtColor(image, imageColor, CV_GRAY2BGR);
	for(int i = 0; i < (int) imgKeypoints.size(); i++)
	{
		circle(imageColor, imgKeypoints[i].pt, 2, Scalar(0, 0, 255), -1);
		circle(imageColor, imgKeypoints[i].pt, (1 << imgKeypoints[i].octave) * 15, Scalar(0, 255, 0), 1);
	}
	namedWindow("Image", 1);
	imshow("Image", imageColor);

	// ƥ��
	vector<Point2f> dst_corners;
	vector<int> pairs;
	Mat H;
	bool found = detector(imgpyr, imgKeypoints, H, dst_corners, &pairs);

	vector<DMatch> matches;
	for (int i = 0; i < (int)pairs.size(); i += 2)
	{
		matches.push_back({ pairs[i + 1], pairs[i], 1 });
	}

	// ����ƥ����
	Mat outImg;
	drawMatches
	(
		frame,
		imgKeypoints,
		pattern_origin,
		pattern_keypoints,
		matches,
		outImg,
		Scalar(0, 200, 0, 255),
		Scalar::all(-1),
		vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
	);
	imshow("Object Correspondence", outImg);
}

vector<DMatch> ImageMatch::ransac(vector<DMatch> matches, vector<KeyPoint> queryKeyPoint, vector<KeyPoint> trainKeyPoint)
{
	//���屣��ƥ��������  
	vector<Point2f> srcPoints(matches.size()), dstPoints(matches.size());
	//����ӹؼ�������ȡ����ƥ���Ե�����  
	for (int i = 0; i<matches.size(); i++)
	{
		srcPoints[i] = queryKeyPoint[matches[i].queryIdx].pt;
		dstPoints[i] = trainKeyPoint[matches[i].trainIdx].pt;
	}
	//�������ĵ�Ӧ�Ծ���  
	Mat homography;
	//�������Ƿ����ı�־  
	vector<unsigned char> inliersMask(srcPoints.size());
	//ƥ���Խ���RANSAC����  
	homography = findHomography(srcPoints, dstPoints, CV_RANSAC, 10, inliersMask);
	
	//RANSAC���˺�ĵ��ƥ����Ϣ  
	vector<DMatch> matches_ransac;
	//�ֶ��ı���RANSAC���˺��ƥ����  
	for (int i = 0; i<inliersMask.size(); i++)
	{
		if (inliersMask[i])
		{
			matches_ransac.push_back(matches[i]);
		}
	}
	//����RANSAC���˺�ĵ��ƥ����Ϣ  
	return matches_ransac;
}