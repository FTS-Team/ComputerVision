#include "stdafx.h"

#include "ImageMatch.h"



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