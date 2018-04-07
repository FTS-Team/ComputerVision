#include "stdafx.h"

#include "ImageMatch.h"



UINT ImageMatch::SIFT(CImage& CImage1, CImage& CImage2) {


	//类型转换
	Mat image1;
	Mat image2;
	CImage_Mat::CImageToMat(CImage1, image1);
	CImage_Mat::CImageToMat(CImage2, image2);

	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
	vector<KeyPoint> kp1, kp2;//特征点容器

    //创建sift特征点检测
	sift->detect(image1, kp1);
	sift->detect(image2, kp2);

	//内存中画出特征点
	Mat res1, res2;
	drawKeypoints(image1, kp1, res1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(image2, kp2, res2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


	namedWindow("KeyPoints of image1", WINDOW_GUI_EXPANDED);//创建窗口
	namedWindow("KeyPoints of image2", WINDOW_GUI_EXPANDED);//创建窗口
	imshow("KeyPoints of image1", res1); // 呈现图像
	imshow("KeyPoints of image2", res2); // 呈现图像


	//计算特征向量
	Mat des1, des2;
	sift->compute(image1, kp1, des1);
	sift->compute(image2, kp2, des2);


	
	//图像匹配(最近邻居法)
	//BFMatcher matcher(NORM_L2, false);
	//vector<DMatch> matches;//筛选后的匹配对
	//vector<vector<DMatch>> knnMatches;//定义一个容器用来装最近邻点和次近邻点
	//matcher.knnMatch(des1, des2, knnMatches, 2);
	//const int ratio = 0.5;

	//for (int n = 0; n < knnMatches.size(); n++) {
	//	DMatch& bestmatch = knnMatches[n][0];
	//	DMatch& bettermatch = knnMatches[n][1];
	//	if (bestmatch.distance < ratio*bettermatch.distance)//筛选出符合条件的点
	//	{
	//		matches.push_back(bestmatch);//将符合条件的点保存在matches
	//	}
	//}


	//RANSAC匹配
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);
	vector<DMatch> goodMatch = ransac(matches, kp1, kp2);

	//呈现匹配后的图像
	Mat imgMatch;
	drawMatches(image1, kp1, image2, kp2, goodMatch , imgMatch);
	namedWindow("matches", WINDOW_GUI_EXPANDED);//创建窗口
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

	//绘制匹配出的关键点
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
	namedWindow("误匹配消除前match图", WINDOW_GUI_EXPANDED);
	imshow("误匹配消除前match图", img_matches);

	//利用RANSAC进行消除无匹配点,并绘制匹配出的关键点
	Mat img_matches_after;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, ransac(matches, keypoints_1, keypoints_2), img_matches_after);
	namedWindow("误匹配消除后match图", WINDOW_GUI_EXPANDED);
	imshow("误匹配消除后match图", img_matches_after);

	return 0;
}



UINT ImageMatch::ORBMatch(CImage& CImage1, CImage & CImage2) {


	Mat img_1;
	Mat img_2;
	CImage_Mat::CImageToMat(CImage1, img_1);
	CImage_Mat::CImageToMat(CImage2, img_2);

	// -- Step 1: Detect the keypoints using STAR Detector 
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	int nkeypoint = 50;//特征点个数
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
	//定义保存匹配点对坐标  
	vector<Point2f> srcPoints(matches.size()), dstPoints(matches.size());
	//保存从关键点中提取到的匹配点对的坐标  
	for (int i = 0; i<matches.size(); i++)
	{
		srcPoints[i] = queryKeyPoint[matches[i].queryIdx].pt;
		dstPoints[i] = trainKeyPoint[matches[i].trainIdx].pt;
	}
	//保存计算的单应性矩阵  
	Mat homography;
	//保存点对是否保留的标志  
	vector<unsigned char> inliersMask(srcPoints.size());
	//匹配点对进行RANSAC过滤  
	homography = findHomography(srcPoints, dstPoints, CV_RANSAC, 10, inliersMask);
	
	//RANSAC过滤后的点对匹配信息  
	vector<DMatch> matches_ransac;
	//手动的保留RANSAC过滤后的匹配点对  
	for (int i = 0; i<inliersMask.size(); i++)
	{
		if (inliersMask[i])
		{
			matches_ransac.push_back(matches[i]);
		}
	}
	//返回RANSAC过滤后的点对匹配信息  
	return matches_ransac;
}