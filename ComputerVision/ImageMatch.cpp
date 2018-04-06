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

	
	//�����ʼ��
	//CvFont font;
	//cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 1, 1, 0, 5);
	//

	//IplImage* transimg1 = cvCloneImage(&(IplImage)res1);
	//IplImage* transimg2 = cvCloneImage(&(IplImage)res2);

	//char str1[20], str2[20];
	//sprintf_s(str1, "%zd", kp1.size());
	//sprintf_s(str2, "%zd", kp2.size());

	//cvPutText(transimg1, str1, cvPoint(280, 230), &font, CV_RGB(255, 0, 0));//��ͼƬ������ַ�   
	//cvPutText(transimg2, str2, cvPoint(280, 230), &font, CV_RGB(255, 0, 0));//��ͼƬ������ַ�   
	//cvShowImage("descriptor1", transimg1);


	namedWindow("KeyPoints of image1", WINDOW_GUI_EXPANDED);//��������
	imshow("KeyPoints of image1", res1); // ����ͼ��

	//������������
	Mat des1, des2;
	sift->compute(image1, kp1, des1);
	sift->compute(image2, kp2, des2);
	
	return 0;
}